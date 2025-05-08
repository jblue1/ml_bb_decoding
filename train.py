import argparse
from copy import deepcopy
import math
from time import perf_counter

import lightning as pl
from lightning.fabric.wrappers import _FabricModule, _FabricOptimizer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np


from qec_data import QECData
from model import ContModel
from utils import create_weighted_src_mask, count_errors


def lr_function(current_step: int, num_warmup_steps: int) -> float:
    if current_step <= num_warmup_steps:
        return current_step / num_warmup_steps
    else:
        return 1 / math.sqrt(current_step - num_warmup_steps)

    return 0.5 * (
        1 + np.cos(np.pi * (current_step - num_warmup_steps) / num_warmup_steps)
    )


def validate(
    fabric: pl.Fabric,
    model: ContModel | _FabricModule,
    validation_dataloader: DataLoader[QECData],
    validation_dataset_size: int,
    num_rounds: int,
    num_ct_rounds: int,
    num_cts: int,
    N_D_total: int,
    criterion: nn.BCEWithLogitsLoss,
    src_mask: torch.Tensor | None,
) -> tuple[float, torch.Tensor]:
    model.eval()
    iters = len(validation_dataloader)
    with torch.no_grad():
        total_loss = 0
        total_errors = torch.zeros(num_rounds + 1 - num_ct_rounds)
        for inputs, targets in validation_dataloader:
            model_outputs = model(
                inputs[:, :N_D_total],
                inputs[:, N_D_total:],
                src_mask,
                num_ct_rounds,
                num_cts,
            )
            loss = criterion(
                model_outputs.reshape(-1),
                targets[:, model.N_L * num_ct_rounds :].reshape(-1),
            )
            total_loss += loss.item()
            predictions = model.generate_predictions(
                inputs[:, :N_D_total],
                src_mask,
                num_ct_rounds,
                num_cts,
            )
            for i in range(num_rounds + 1 - num_ct_rounds):
                total_errors[i] += count_errors(
                    predictions[:, i, :],
                    targets[
                        :,
                        (i + num_ct_rounds) * model.N_L : (i + num_ct_rounds + 1)
                        * model.N_L,
                    ],
                ).item()
    return total_loss / iters, total_errors / validation_dataset_size


def train(
    fname_prefix: str,
    fabric: pl.Fabric,
    model: _FabricModule,
    optimizer: torch.optim.Optimizer | _FabricOptimizer,
    dataset: QECData,
    dataloader: DataLoader,
    validation_dataloader: DataLoader,
    validation_dataset_size: int,
    num_epochs: int,
    generate_data_every: int,
    src_mask: torch.Tensor | None,
    N_D_total: int,
    num_rounds: int,
    num_ct_rounds: int,
    num_cts: int,
    criterion: nn.BCEWithLogitsLoss,
    validate_every: int,
    save_every: int,
    gradient_accumulation_steps: int,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    cosine_annealing: bool = False,
    saved_dataset: str | None = None,
) -> None:
    print(f"World size: {fabric.world_size}")
    model.train()
    best_model = deepcopy(model)
    if scheduler is not None:
        state = {
            "model": model,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "best_model": best_model,
        }
    else:
        state = {
            "model": model,
            "optimizer": optimizer,
            "best_model": best_model,
        }
    losses = []
    validation_losses = []
    validation_lers = []
    best_validation_loss = 100

    # initial validation before training
    val_loss, val_lers = validate(
        fabric,
        model,
        validation_dataloader,
        validation_dataset_size,
        num_rounds,
        num_ct_rounds,
        num_cts,
        N_D_total,
        criterion,
        src_mask,
    )
    validation_losses.append(val_loss)
    validation_lers.append(val_lers)
    
    for epoch in range(num_epochs):
        epoch += 1
        print("Starting epoch:", epoch)
        epoch_start_time = perf_counter()
        dataset_generation_time = 0
        start_size_losses = len(losses)
        if epoch % generate_data_every == 0 and saved_dataset is None:
            dataset_start_time = perf_counter()
            dataset.generate_training_data()
            dataset_end_time = perf_counter()
            dataset_generation_time = dataset_end_time - dataset_start_time
            print(f"\tTime to generate data: {dataset_generation_time:.3f}s")
        iters = len(dataloader)
        train_start_time = perf_counter()
        for step, (inputs, targets) in enumerate(dataloader):
            is_accumulating = (step + 1) % gradient_accumulation_steps != 0
            # https://lightning.ai/docs/fabric/stable/advanced/gradient_accumulation.html
            with fabric.no_backward_sync(model, enabled=is_accumulating):
                model_outputs = model(
                    inputs[:, :N_D_total],
                    inputs[:, N_D_total:],
                    src_mask,
                    num_ct_rounds,
                    num_cts,
                )
                loss = criterion(
                    model_outputs.reshape(-1),
                    targets[:, model.N_L * num_ct_rounds :].reshape(-1),
                )
                losses.append(loss.item())
                fabric.backward(loss)
            if not is_accumulating:
                optimizer.step()
                optimizer.zero_grad()
            if scheduler is not None:
                if cosine_annealing:
                    # cosine annealing scheduler is weird and requires a float input
                    scheduler.step(epoch - 1 + step / iters)  # type: ignore
                else:
                    scheduler.step()
        train_end_time = perf_counter()
        print(f"\tTime to train: {train_end_time - train_start_time:.3f}s")
        avg_epoch_loss = np.mean(losses[start_size_losses:])
        print(f"\tAvg loss for epoch: {avg_epoch_loss:.3f}")
        if scheduler is not None:
            print(f"\tLearning rate: {scheduler.get_last_lr()}")

        eval_time = 0
        if epoch % validate_every == 0:
            eval_start = perf_counter()
            val_loss, val_lers = validate(
                fabric,
                model,
                validation_dataloader,
                validation_dataset_size,
                num_rounds,
                num_ct_rounds,
                num_cts,
                N_D_total,
                criterion,
                src_mask,
            )
            validation_losses.append(val_loss)
            validation_lers.append(val_lers)

            if val_loss < best_validation_loss:
                best_validation_loss = val_loss
                best_model = deepcopy(model)
                state["best_model"] = best_model
            eval_end = perf_counter()
            print(f"\tValidation loss: {val_loss:.3f}")
            print(f"\tValidation lers: {val_lers}")
            eval_time = eval_end - eval_start
            print(f"\tTime to evaluate: {eval_time:.3f}s")
        save_time = 0
        if epoch % save_every == 0 and save_every > 0:
            save_start = perf_counter()
            fabric.save(f"{fname_prefix}.ckpt", state)
            np.save(fname_prefix + "update_training_losses.npy", losses)
            np.save(fname_prefix + "update_validation_losses.npy", validation_losses)
            np.save(fname_prefix + "update_validation_lers.npy", validation_lers)
            save_end = perf_counter()
            save_time = save_end - save_start
            print(f"\tTime to save: {save_time:.3f}s")
        print(
            f"\tTotal epoch time: {perf_counter() - epoch_start_time:.3f}", flush=True
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fname_prefix", default="cont_model_test")
    parser.add_argument("--code_name", "-cn", default="bbc-72-12-6")
    parser.add_argument("--batch_size", "-bs", type=int, default=256)
    parser.add_argument("--physical_batch_size", "-pbs", type=int, default=-1)
    parser.add_argument("--error_rate", "-er", type=float, default=0.006)
    parser.add_argument("--num_encoder_layers", "-nel", type=int, default=3)
    parser.add_argument("--num_decoder_layers", "-ndl", type=int, default=3)
    parser.add_argument("--nhead", "-nh", type=int, default=4)
    parser.add_argument("--d_model", "-dm", type=int, default=128)
    parser.add_argument("--dim_feedforward", "-df", type=int, default=256)
    parser.add_argument("--dropout", "-dr", type=float, default=0.1)
    parser.add_argument("--num_epochs", "-ne", type=int, default=10)
    parser.add_argument("--learning_rate", "-lr", type=float, default=1e-4)
    parser.add_argument(
        "--num_rounds",
        "-nr",
        type=int,
        default=6,
        help="Number of rounds of noisy syndrome measurement to train on. If using data qubit noise, set this to 1",
    )
    parser.add_argument(
        "--num_ct_rounds",
        type=int,
        default=0,
        help="Number of rounds in which the decoder makes latent space predictions"
    )
    parser.add_argument(
        "--num_cts",
        type=int,
        default=1,
        help="The number of latent space predictions made per round"
    )
    parser.add_argument(
        "--validate_every",
        "-ve",
        type=int,
        default=5,
        help="How often to evaluate the model on the validation data set",
    )
    parser.add_argument(
        "--generate_data_every",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "--measure_both",
        "-mb",
        action="store_true",
        help="If true, use a syndrome measurement circuit that measures both X and Z checks. If false, only measure the same kind of check as the logical operator being measured.",
    )
    parser.add_argument("--dataset_size", type=int, default=16384)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--matmul_precision", default="highest")
    parser.add_argument("--load_path")
    parser.add_argument("--load_optimizer", action="store_true")
    parser.add_argument("--load_scheduler", action="store_true")
    parser.add_argument("--lr_scheduler")
    parser.add_argument("--num_warmup_steps", type=int, default=1000)
    parser.add_argument("--accelerator", default="cuda")
    parser.add_argument("--t0", type=int, default=50)
    parser.add_argument("--validation_dataset_size", type=int, default=-1)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--use_mask", action="store_true")
    parser.add_argument("--saved_dataset")
    args = parser.parse_args()
    print("Args:")
    print(args, flush=True)

    torch.set_float32_matmul_precision(args.matmul_precision)

    if args.physical_batch_size > 0:
        batch_size = args.physical_batch_size
        gradient_accumulation_steps = args.batch_size // batch_size
    else:
        batch_size = args.batch_size
        gradient_accumulation_steps = 1
    dataset = QECData(
        args.code_name,
        "X",
        "circuit",
        args.dataset_size,
        args.error_rate,
        args.num_rounds,
        args.measure_both,
        True,
    )
    N_D = dataset.N_D // (args.num_rounds + 1)
    N_L = dataset.N_L
    N_D_total = dataset.N_D

    src_mask = None
    if args.use_mask:
        src_mask = create_weighted_src_mask(torch.from_numpy(dataset.decoding_matrix))

    if args.validation_dataset_size == -1:
        validation_dataset_size = args.dataset_size // 8
    else:
        validation_dataset_size = args.validation_dataset_size
    validation_dataset = QECData(
        args.code_name,
        "X",
        "circuit",
        validation_dataset_size,
        args.error_rate,
        args.num_rounds,
        args.measure_both,
        True,
    )
    validation_dataset.generate_training_data()
    if args.saved_dataset is not None:
        dataset.generate_training_data()
        saved_dataset = np.load(args.saved_dataset)
        dataset.X = saved_dataset["x"][:512, :]
        dataset.Y = saved_dataset["y"][:512, :]
        validation_dataset.X = saved_dataset["x"][512:, :]
        validation_dataset.Y = saved_dataset["y"][512:, :]
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size)

    model = ContModel(
        N_D,
        N_L,
        args.d_model,
        args.nhead,
        args.dim_feedforward,
        args.num_encoder_layers,
        args.num_decoder_layers,
        args.dropout,
    )
    if args.accelerator == "cuda":
        print("Num gpus =", args.num_gpus)
        fabric = pl.Fabric(
            accelerator="gpu",
            devices=args.num_gpus,
            num_nodes=args.num_nodes,
        )
    else:
        fabric = pl.Fabric()
    # if args.num_gpus > 1:
    fabric.launch()

    total_params = sum(param.numel() for param in model.parameters(recurse=True))
    print("Total number of parameters:", total_params, flush=True)

    if args.load_path is not None:
        try:
            checkpoint = fabric.load(args.load_path)
            model.load_state_dict(checkpoint["model"])
        except:
            checkpoint = torch.load(args.load_path)
            model.load_state_dict(checkpoint["best_model_state_dict"])
    model = fabric.setup_module(model)
    if args.weight_decay > 0:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = None
    if args.lr_scheduler == "cosine_annealing_wr":
        num_batches_per_epoch = args.dataset_size // args.batch_size
        print("Num batches per epoch:", num_batches_per_epoch, flush=True)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=args.t0
        )
    elif args.lr_scheduler == "lin_warmup_sqrt_decrease":
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda current_step: lr_function(
                current_step, args.num_warmup_steps
            ),
        )
    if args.load_path is not None and args.load_optimizer:
        checkpoint = fabric.load(args.load_path)
        optimizer.load_state_dict(checkpoint["optimizer"])
    wrapped_optimizer: _FabricOptimizer = fabric.setup_optimizers(optimizer)  # type: ignore

    model.mark_forward_method("generate_predictions")
    dataloader = DataLoader(dataset, batch_size=batch_size)
    dataloader, validation_dataloader = fabric.setup_dataloaders(
        dataloader, validation_dataloader, use_distributed_sampler=False
    )
    criterion = nn.BCEWithLogitsLoss()

    train(
        args.fname_prefix,
        fabric,
        model,
        wrapped_optimizer,
        dataset,
        dataloader,
        validation_dataloader,
        validation_dataset_size,
        args.num_epochs,
        args.generate_data_every,
        src_mask,
        N_D_total,
        args.num_rounds,
        args.num_ct_rounds,
        args.num_cts,
        criterion,
        args.validate_every,
        args.save_every,
        gradient_accumulation_steps,
        scheduler,
        args.lr_scheduler == "cosine_annealing_wr",
        args.saved_dataset,
    )


if __name__ == "__main__":
    main()
