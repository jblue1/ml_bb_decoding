import sys
from typing import Tuple
from random import randint

import numpy as np
import stim

from generate_sc_circuits import generate_sc_circuit
from circuit_gen import get_72_12_6_code, get_144_12_12_code, get_98_6_12_code, GF2


# TODO: See if its faster to do this with bool arrays with numba/cython
def f2_matmul(m1: np.ndarray, m2: np.ndarray):
    """
    Multiply two matrices over the field F_2.
    """
    return ((m1.astype(np.uint8) @ m2.astype(np.uint8)) % 2).astype(np.bool_)


def parse_dem_to_decoding_matrices(
    dem: stim.DetectorErrorModel,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert a detector error model into a tuple of decoding matrices, H, L, and P.

    NOTE: The detector error model must be flattened, i.e. have no repeat blocks.

    Let n be the number of error mechanisms in a circuit, l be the number of logical operators,
    and d be the number of detectors.
    - H: spacetime parity matrix, a (d x n) binary matrix where H(i, j) = 1 if error mechanism j flips detector i.
    - L: logical matrix, an (l x n) binary matrix where L(i, j) = 1 if error mechanism j flips logical operator i.
    - P: probabilities vector, an (n x 1) matrix where P[i] is the probability of error mechanism i.
    """
    decoding_matrix = np.zeros((dem.num_detectors, dem.num_errors), dtype=np.bool_)
    logical_decoding_matrix = np.zeros(
        (dem.num_observables, dem.num_errors), dtype=np.bool_
    )
    probabilities = np.zeros((dem.num_errors), dtype=np.float32)

    error_index = 0
    for dem_instruction in dem:
        type = dem_instruction.type
        if type == "error":
            probabilities[error_index] = dem_instruction.args_copy()[0]
            for target in dem_instruction.targets_copy():
                col = target.val
                if target.is_relative_detector_id():
                    decoding_matrix[col, error_index] = True
                elif target.is_logical_observable_id:
                    logical_decoding_matrix[col, error_index] = True
            error_index += 1

    return (decoding_matrix, logical_decoding_matrix, probabilities)


def adjust_error_rate(fname: str, error_rate: float = 0.01):
    """
    Load a stim circuit from a file and adjust the error rate of
    all DEPOLARIZE, X_ERROR, and Z_ERROR by rewriting the lines
    involved. This is a hacky way to do this, but it works for now.
    Parse the circuit and return a stim circuit object.
    """
    sys.exit(
        "This function is depracated. To generate a surface code circuit in stim,"
        " use the function 'generate_sc_circuit' from generate_sc_circuits.py"
    )
    with open(fname, "r") as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        # TODO Fix this to handle measurement errors
        if "DEPOLARIZE" in line or "X_ERROR" in line or "Z_ERROR" in line:
            start = line.find("(")
            end = line.find(")")
            lines[i] = line[: start + 1] + str(error_rate) + line[end:]

    return stim.Circuit("".join(lines))


class QECGym:
    """
    An environment for training/evaluating ML QEC decoders inspired by reinforcement
    learning gyms.
    """

    def __init__(
        self,
        code_name: str,
        logical_operator: str,
        noise_type: str,
        physical_error_rate: float,
        num_rounds: int = -1,
        skip_redundant_checks: bool = False,
        measure_both=False,
        load_saved_logical_ops=False,
    ) -> None:
        """
        Create a QEC gym.

        Args:
            code_name: The kind of quantum error correction code to use. Options are
            - 'rotated_surface_code-<distance>' (e.g. 'rotated_surface_code-3') where
                distance is an odd number greater than or equal to three
            - 'bbc-72-12-6' ('bbc' stands for bivariate bicycle code)
            - 'bbc-144-12-12'
            - 'bbc-98-6-12'

            logical_operator: The logical operator to measure. Options are 'X' or 'Z'.
            noise_type: The kind of noise to use. Options are 'data' or 'circuit'.
            physical_error_rate: The physical error rate being used.
        """
        self.code_name = code_name
        self.logical_operator = logical_operator
        self.noise_type = noise_type
        self.phyical_error_rate = physical_error_rate
        self.num_rounds = num_rounds
        if logical_operator == "X":
            x_detectors = True
            z_detectors = False
        elif logical_operator == "Z":
            z_detectors = True
            x_detectors = False
        if measure_both:
            z_detectors = True
            x_detectors = True
        if "rotated_surface_code" in code_name:
            distance = int(code_name[len("rotated_surface_code-") :])
            if noise_type == "data":
                num_rounds = 1
            elif noise_type == "circuit":
                if num_rounds < 0:
                    num_rounds = distance
            else:
                sys.exit(
                    (
                        f"'{noise_type}' is not a supported noise type."
                        "Please choose either 'data' or 'circuit'"
                    )
                )
            circuit = generate_sc_circuit(
                distance,
                num_rounds,
                x_detectors,
                z_detectors,
                logical_operator,
                physical_error_rate,
            )
        elif "bbc" in code_name:
            if code_name == "bbc-72-12-6":
                code = get_72_12_6_code()
                distance = 6
                if load_saved_logical_ops:
                    code.logical_X_ops = GF2(
                        np.load(
                            "data/logical_ops/bbc-72-12-6_logical_X_ops.npy"
                        )
                    )
            elif code_name == "bbc-98-6-12":
                code = get_98_6_12_code()
                distance = 12
                if load_saved_logical_ops:
                    code.logical_X_ops = GF2(
                        np.load("data/logical_ops/98_6_12_logical_X_ops.npy")
                    )  # Todo Change Path
            elif code_name == "bbc-144-12-12":
                code = get_144_12_12_code()
                distance = 12
                if load_saved_logical_ops:
                    code.logical_X_ops = GF2(
                        np.load(
                            "data/logical_ops/bbc-144-12-12_logical_X_ops.npy"
                        )
                    )

            if noise_type == "data":
                # circuit = code.create_syndrome_measurement_circuit_data_noise(
                #     logical_operator, physical_error_rate
                # )
                circuit = code.create_syndrome_measurement_circuit(
                    x_detectors,
                    z_detectors,
                    0,
                    logical_operator,
                    physical_error_rate,
                    skip_redundant_checks,
                )
            elif noise_type == "circuit":
                if num_rounds < 0:
                    num_rounds = distance
                circuit = code.create_syndrome_measurement_circuit(
                    x_detectors,
                    z_detectors,
                    num_rounds,
                    logical_operator,
                    physical_error_rate,
                    skip_redundant_checks,
                )
        else:
            raise ValueError("Invalid code name.")
        self._dem = circuit.detector_error_model()
        stim_rand_seed = randint(0, 2**64)
        print("Using random seed for stim:", stim_rand_seed)
        self._dem_sampler = self._dem.compile_sampler(seed=stim_rand_seed)
        (
            self._spacetime_H,
            self._logical_decoding_matrix,
            self._channel_probabilities,
        ) = parse_dem_to_decoding_matrices(self._dem.flattened())

        self._det_data: np.ndarray | None = None
        self._actual_errors: np.ndarray | None = None
        self._measured_observables: np.ndarray | None = None
        self._circuit = circuit
        self.N_E = self._spacetime_H.shape[1]

    def get_detector_error_model(self) -> stim.DetectorErrorModel:
        """Get the detector error model describing a circuit.

        A detector error model (DEM) is the data structure used by stim to describe the
        effect of errors in a stabilizer circuit. Each line of the DEM contains the probability
        of an error occuring, which detectors the error flips, and which logical operators the error flips.

        Note that multiple independent errors can be merged into the same entry in the
        DEM if the errors have the same effect on the detectors and logical operators in
        a circuit. Consider the following example from the stim documentation where
        we measure both qubits of a bell state.

        >>> circuit = stim.Circuit('''
        ...     H 0
        ...     TICK
        ...     CX 0 1
        ...     X_ERROR(0.2) 0 1
        ...     TICK
        ...     M 0 1
        ...     DETECTOR rec[-1] rec[-2]
        ... ''')
        >>> dem = circuit.detector_error_model()
        >>> print(dem)
        error(0.3200000000000000622) D0

        While an X error on qubit 1 and an X error on qubit 2 are technically different
        errors, they both flip the detector in this circuit. The probability that there
        is an X error on one of the qubits is 2 * (0.2 * 0.8) = 0.32.

        More information about detector error models can be found at
        https://github.com/quantumlib/Stim/blob/main/doc/file_format_dem_detector_error_model.md
        """
        return self._dem

    def get_spacetime_parity_check_matrix(self) -> np.ndarray:
        return self._spacetime_H

    def get_channel_probabalities(self) -> np.ndarray:
        return self._channel_probabilities

    def get_logical_decoding_matrix(self) -> np.ndarray:
        return self._logical_decoding_matrix

    def get_decoding_instances(
        self, shots: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate a given number of runs of the QEC circuit, and return the measured
        detector results.

        Note that we do not actually do a simulation of the circuit - instead, we
        directly sample from the detector error model that was created when the Gym was
        initialized, which is much faster.

        Args:
            shots: Number of decoding instances to return

        Returns:
            det_data: A shots x num_detectors numpy array. det_data[i, j] is True if on
                shot i, detector j was flipped by an error in the circuit.
            measured_observables: A shots x k numpy array. measured_observables[i, j] is
                True if on shot i, logical measurement j was measured to be -1.
            actual_errors: A shots x num_errors numpy array. actual_errors[i, j] is True
                if on shot i, error_mechanism j occured.
        """
        (
            self._det_data,
            self._measured_observables,
            self._actual_errors,
        ) = self._dem_sampler.sample(shots=shots, return_errors=True)
        return self._det_data, self._measured_observables, self._actual_errors

    def evaluate_predictions(
        self,
        predicted_errors: np.ndarray,
        return_errors: bool = False,
        allow_inconsistent_predictions: bool = False,
        use_canonical_errors: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray | None]:
        """
        Evaluate the predictions of errors based on syndrome data from a previous
        `get_decoding_instances` call.

        Args:
            predicted_errors: A shots x num_errors ndarray of np.bool_.
            allow_inconsistent_predictions: If `False`, a predicted error that is
                inconsistent with the syndrome will raise an error. If `True`, such a
                predicted error will be counted as incorrect for all logical operators.

        Returns:
            prediction_errors: A `shots` x k numpy array of np.bool_, where k is the
                number of logical qubits in the code. The value prediction_errors[i, j]
                is `True` if on shot i, a logical error that anti-commutes with logical
                operator j occured.
            actual_errors: If `return_errors` is True, then this is a shots x num_errors
                ndarray of np.bool_ of the actual errors that produced the observed
                syndromes. TODO: If `use_canonical_errors` is `True`, then these errors are
                brought into a canonical form.
        """
        if use_canonical_errors:
            raise NotImplementedError
        predicted_syndromes = f2_matmul(predicted_errors, self._spacetime_H.T)
        differences = np.logical_xor(predicted_syndromes, self._det_data)
        # TODO: better name for this variable
        inconsistencies = differences.sum(axis=1)
        if not allow_inconsistent_predictions and inconsistencies.any():
            raise ValueError("Predicted errors are not consistent with the syndrome")
        inconsistencies = inconsistencies != np.zeros_like(inconsistencies)
        inconsistencies = np.expand_dims(inconsistencies, axis=1)

        predicted_observables = f2_matmul(
            predicted_errors, self._logical_decoding_matrix.T
        )
        observable_difference = np.logical_xor(
            self._measured_observables, predicted_observables
        )
        if return_errors:
            # + is logical OR for numpy arrays of np.bool_
            return observable_difference + inconsistencies, self._actual_errors
        else:
            return observable_difference, None

    def __repr__(self) -> str:
        return (
            f"QECGym(code_name={self.code_name},"
            f"logical_operator={self.logical_operator},"
            f"noise_type={self.noise_type},"
            f"physical_error_rate={self.phyical_error_rate})"
        )
