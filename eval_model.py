import argparse
from collections.abc import Sequence
import os
from time import perf_counter
import sys

import torch
from torch.utils.data import DataLoader
import pandas as pd


from utils import count_errors, create_weighted_src_mask
from model import ContModel
from qec_data import QECData


def parse_args_string(args_str: str) -> dict:
    if args_str == "":
        sys.exit("Could not find args string in provided log file")
    args = dict()
    first_index = args_str.find("(")
    for arg_pair in args_str[first_index + 1 : -2].split(","):
        arg_pair_list = arg_pair.split("=")
        args[arg_pair_list[0].strip()] = arg_pair_list[1].strip("'")
    return args


def eval_model(
    model: ContModel,
    dataset: QECData,
    device: torch.device,
    shots: int,
    src_mask: torch.Tensor | None,
    num_ct_rounds: int,
    c: int,
) -> tuple[float, float]:
    time = 0
    total_errors = 0

    with torch.no_grad():
        model.eval()
        device = torch.device("cuda")
        dataset.shots = shots
        dataset.generate_training_data()
        dataloader = DataLoader(dataset, batch_size=512)
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            start = perf_counter()
            preds = model.generate_predictions(
                inputs[:, : dataset.N_D],
                src_mask,
                num_ct_rounds,
                c,
            )
            end = perf_counter()
            time += end - start
            total_errors += count_errors(
                preds[:, 0, -dataset.N_L :],
                targets[:, -dataset.N_L :],
            ).item()
        return total_errors, time


def get_logical_error_count(
    args,
    model_args,
    error_rates: Sequence[float],
    num_shots: Sequence[int],
    num_rounds: int,
    num_ct_rounds: int,
    num_cts: int,
    gpu_id: int = 0,
    desired_errors: int = 100,
    max_iters: int = 1,
) -> tuple[list[int], list[float], list[int]]:
    print("Num shots =", num_shots)
    dummy_dataset = QECData(
        "bbc-72-12-6",
        "X",
        "circuit",
        1,
        0.1,
        num_rounds,
        measure_both=True,
        load_saved_logical_ops=True,
    )

    src_mask = create_weighted_src_mask(torch.from_numpy(dummy_dataset.decoding_matrix))
    N_D = dummy_dataset.N_D // (num_rounds + 1)
    print("N_D =", N_D, flush=True)
    N_L = dummy_dataset.N_L
    model = ContModel(
        N_D,
        N_L,
        int(model_args["d_model"]),
        int(model_args["nhead"]),
        int(model_args["dim_feedforward"]),
        int(model_args["num_encoder_layers"]),
        int(model_args["num_decoder_layers"]),
        float(model_args["dropout"]),
    )
    print("Initialized model")
    checkpoint = torch.load(args.model_file)
    try:
        model.load_state_dict(checkpoint["best_model"])
    except:
        model.load_state_dict(checkpoint["best_model_state_dict"])
    device = torch.device(gpu_id)
    model = model.to(device)
    src_mask = src_mask.to(device)
    print("Loaded model and moved to gpu")
    times: list[float] = []
    logical_errors: list[int] = []
    shots_list: list[int] = []
    for error_rate, shots in zip(error_rates, num_shots):
        dataset = QECData(
            "bbc-72-12-6",
            "X",
            "circuit",
            shots,
            error_rate,
            num_rounds,
            measure_both=True,
            load_saved_logical_ops=True,
        )
        total_errors = 0
        total_time = 0.0
        iters = 0
        total_shots = 0
        while total_errors < desired_errors and iters < max_iters:
            num_errors, time = eval_model(
                model, dataset, device, shots, src_mask, num_ct_rounds, num_cts
            )
            iters += 1
            total_errors += num_errors
            total_time += time
            total_shots += shots
        print("Num errors =", total_errors)
        print("Time =", total_time)
        print("Shots =", total_shots)
        logical_errors.append(int(total_errors))
        times.append(total_time)
        shots_list.append(total_shots)
    return logical_errors, times, shots_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_file")
    parser.add_argument("--log_file")
    parser.add_argument("--shots", "-s", type=int, nargs="*")
    parser.add_argument("--error_rates", "-er", type=float, nargs="*")
    parser.add_argument("--save_data", action="store_true")
    parser.add_argument("--save_dir")
    parser.add_argument("--job_id")
    parser.add_argument("--eval_num")
    parser.add_argument("--num_ct_rounds", type=int)
    parser.add_argument("--num_cts", type=int)
    parser.add_argument("--num_rounds", type=int, default=-1)
    parser.add_argument("--desired_errors", type=int, default=100)
    parser.add_argument("--max_iters", type=int, default=1)
    args = parser.parse_args()
    if args.save_data:
        assert args.save_dir is not None
        assert args.job_id is not None

    model_arg_string = ""
    with open(args.log_file, "r") as f:
        for line in f:
            if "Namespace" in line:
                model_arg_string = line
                break
    model_args = parse_args_string(model_arg_string)

    num_rounds = args.num_rounds
    if num_rounds < 0:
        num_rounds = model_args["num_rounds"]

    logical_errors = [0 for _ in range(len(args.error_rates))]
    total_shots = [0 for _ in range(len(args.error_rates))]
    times = [0.0 for _ in range(len(args.error_rates))]
    num_jobs = torch.cuda.device_count()
    num_jobs = 1
    print("Num jobs =", num_jobs)
    jobs = []
    for i in range(num_jobs):
        jobs.append(
            [
                args,
                model_args,
                args.error_rates,
                args.shots,
                num_rounds,
                args.num_ct_rounds,
                args.num_cts,
                i,
                args.desired_errors,
                args.max_iters,
            ]
        )

    res = get_logical_error_count(
        args,
        model_args,
        args.error_rates,
        args.shots,
        num_rounds,
        args.num_ct_rounds,
        args.num_cts,
        0,
        args.desired_errors,
        args.max_iters,
    )
    for i in range(len(times)):
        logical_errors[i] += res[0][i]
        times[i] += res[1][i]
        total_shots[i] += res[2][i]
    df = pd.DataFrame(
        {
            "error_rate": args.error_rates,
            "num_shots": total_shots,
            "logical_errors": logical_errors,
            "time": times,
        }
    )
    if args.save_data:
        if args.eval_num is not None:
            eval_num = args.eval_num
        else:
            eval_num = 0
        df.to_csv(os.path.join(args.save_dir, f"results_{args.job_id}-{eval_num}.csv"))
    print(df)


if __name__ == "__main__":
    main()
