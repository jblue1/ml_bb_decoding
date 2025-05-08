from typing import Literal

import numpy as np
from torch.utils.data import Dataset

from gym import QECGym


class QECData(Dataset):
    """
    Create QEC dataset for training a neural networks.
    """

    def __init__(
        self,
        code_name: str,
        logical_operator: Literal["X", "Z"],
        noise_type: Literal["data", "circuit"],
        shots: int,
        error_rate: float,
        num_rounds: int = -1,
        measure_both: bool = False,
        load_saved_logical_ops: bool = False,
    ):
        # print("Initializing QECData")
        self.gym = QECGym(
            code_name,
            logical_operator,
            noise_type,
            error_rate,
            num_rounds,
            measure_both=measure_both,
            load_saved_logical_ops=load_saved_logical_ops,
        )
        self.shots = shots
        self.compute_decoder()

    def __len__(self):
        return self.shots

    def __getitem__(self, idx):
        return self.X[idx, :], self.Y[idx, :]

    def compute_decoder(self):
        """
        Compute the decoding matrix and logical decoding matrix for the code.
        """
        self.decoding_matrix = self.gym.get_spacetime_parity_check_matrix()
        self.probabilities = self.gym.get_channel_probabalities()
        self.logical_decoding_matrix = self.gym.get_logical_decoding_matrix()
        self.N_D = self.decoding_matrix.shape[0]
        self.N_E = self.decoding_matrix.shape[1]
        self.N_L = self.logical_decoding_matrix.shape[0]

    def generate_training_data(self):
        """
        Generate training data (X,Y)

        Y has shape (batch_size, N_L) and type float because the loss function
        expect floats.
        X has shape (batch_size, num_detectors + N_L) and type int because the embedding layer
        expects ints.
        """
        # print("Generating dataset")
        (
            syndromes,
            measured_observables,
            actual_errors,
        ) = self.gym.get_decoding_instances(
            shots=self.shots,
        )
        num_rounds = self.gym.num_rounds
        self.X = np.ones(
            (self.shots, self.N_D + (num_rounds + 1) * (self.N_L + 1)), dtype=np.int32
        )
        self.X *= 2
        locs_per_round = actual_errors.shape[1] // num_rounds
        self.Y = np.zeros((self.shots, num_rounds * self.N_L), dtype=np.float32)
        # the code below is doing the same thing as calculating the following:
        # >>> for i in range(num_rounds):
        # >>>     tmp_errors = np.zeros_like(actual_errors).astype(np.uint8)
        # >>>     tmp_errors[:, : (i + 1) * locs_per_round] = actual_errors[
        # >>>         :, : (i + 1) * locs_per_round
        # >>>     ]
        # >>>     tmp_l = (tmp_errors @ self.logical_decoding_matrix.astype(np.uint8).T) % 2
        # >>>     self.Y[:, i * self.N_L : (i + 1) * self.N_L] = tmp_l.astype(np.float32)
        # just ~10x as fast for the sizes being considered
        for i in range(num_rounds):
            new_errors = (
                actual_errors[:, i * locs_per_round : (i + 1) * locs_per_round].astype(
                    np.float32
                )
                @ self.logical_decoding_matrix.astype(np.float32)[
                    :, i * locs_per_round : (i + 1) * locs_per_round
                ].T
            ) % 2
            if i == 0:
                self.Y[:, : self.N_L] = new_errors
            else:
                self.Y[:, i * self.N_L : (i + 1) * self.N_L] = (
                    self.Y[:, (i - 1) * self.N_L : i * self.N_L] + new_errors
                )
        self.Y %= 2
        self.Y = np.hstack((self.Y, self.Y[:, -self.N_L :]))
        self.X[:, : self.N_D] = syndromes.astype(np.int32)
        for i in range(num_rounds + 1):
            self.X[
                :,
                self.N_D + i * (self.N_L + 1) + 1 : self.N_D
                + i * (self.N_L + 1)
                + 1
                + self.N_L,
            ] = self.Y[:, i * self.N_L : (i + 1) * self.N_L]
