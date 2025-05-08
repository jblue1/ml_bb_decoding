import argparse
from time import perf_counter

import numpy as np
from train_autoregressive import QECData
from utils import get_bp_osd_decoder



def generate_data_and_decode_batch(dataset: QECData, decoder) -> list[float]:
    dataset.generate_training_data()
    times = []
    predicted_errors = np.empty((dataset.shots, dataset.N_E), dtype=np.uint8)
    for i in range(dataset.shots):
        start = perf_counter()
        predicted_errors[i, :] = decoder.decode(
            np.array(dataset.X[:, : dataset.N_D])[i, :]
        )
        end = perf_counter()
        times.append(end - start)
    return times


def run_specific_error_rate(
    physical_error_rate: float,
    num_rounds: int,
    num_shots: int,
    osd_order: int,
) -> list[float]:
    dataset = QECData(
        "bbc-144-12-12",
        "X",
        "circuit",
        num_shots,
        physical_error_rate,
        num_rounds,
        False,
        True,
    )
    decoder = get_bp_osd_decoder(
        np.array(dataset.decoding_matrix).astype(np.bool_),
        np.array(dataset.probabilities),
        osd_order,
    )
    return generate_data_and_decode_batch(dataset, decoder)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobid", type=int, default=0)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--num_rounds", type=int, default=6)
    parser.add_argument("--num_shots", type=int, default=100)
    parser.add_argument("--osd_order", type=int, default=3)
    parser.add_argument("--physical_error_rate", type=float, default=0.001)
    args = parser.parse_args()
    times = run_specific_error_rate(
        args.physical_error_rate, args.num_rounds, args.num_shots, args.osd_order
    )
    np.save(
        "data/bposd_timing_data/"
        +f"times_bp_osd_144-12-12_loX_ntcircuit_nr{args.num_rounds}_osd_order"
        +f"{args.osd_order}_ji{args.jobid}_rank{args.rank}_p0.003.npy",
        times,
    )


if __name__ == "__main__":
    main()
