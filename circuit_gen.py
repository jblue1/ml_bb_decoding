"""Functions for generating stim circuits to implement bivariate bicycle codes"""

import sys
from typing import List, Tuple, Literal, Sequence

import numpy as np

import stim
import galois
from galois import FieldArray

from utils import find_dependent_row_vectors

GF2 = galois.GF(2)


class Monomial:
    def __init__(self, l, m, x_power, y_power) -> None:
        self.l = l
        self.m = m
        self.x_power = x_power
        self.y_power = y_power

    def to_matrix(self) -> FieldArray:
        return GF2(
            np.kron(
                np.linalg.matrix_power(shift_matrix(self.l), self.x_power),
                np.linalg.matrix_power(shift_matrix(self.m), self.y_power),
            )
        )

    def to_vector(self) -> FieldArray:
        first = np.zeros(self.l, dtype=np.uint8)
        second = np.zeros(self.m, dtype=np.uint8)
        first[self.x_power] = 1
        second[self.y_power] = 1
        return GF2(np.kron(first, second))

    def __str__(self) -> str:
        return f"x^{self.x_power} y^{self.y_power}"

    def __repr__(self) -> str:
        return f"Monomial({self.l}, {self.m}, {self.x_power}, {self.y_power})"

    def __mul__(self, other):
        assert self.l == other.l
        assert self.m == other.m
        return Monomial(
            self.l,
            self.m,
            (self.x_power + other.x_power) % self.l,
            (self.y_power + other.y_power) % self.m,
        )

    def __eq__(self, other):
        if (
            self.l != other.l
            or self.m != other.m
            or self.x_power != other.x_power
            or self.y_power != other.y_power
        ):
            return False
        return True

    def inverse(self):
        return Monomial(
            self.l,
            self.m,
            (self.l - self.x_power) % self.l,
            (self.m - self.y_power) % self.m,
        )


class Polynomial:
    def __init__(
        self,
        monomials: Sequence[Monomial],
        l: int = 0,
        m: int = 0,
        vec: FieldArray | None = None,
    ) -> None:
        if len(monomials) == 0:
            assert l != 0
            assert m != 0
            self.l = l
            self.m = m
        else:
            self.l = monomials[0].l
            self.m = monomials[0].m
        if vec is None:
            self.vec = GF2(np.zeros(self.l * self.m, dtype=np.uint8))
            self.monomials: Sequence[Monomial] = monomials
            for i in monomials:
                self.vec[i.x_power * self.m + i.y_power] = 1
        else:
            self.vec = vec
            if len(monomials) == 0:
                self.monomials = []
                nonzero_vals = vec.nonzero()[0]
                for i in nonzero_vals:
                    x_power = i // m
                    y_power = i - m * x_power
                    self.monomials.append(Monomial(l, m, x_power, y_power))

    def __str__(self) -> str:
        monomial_strings = [str(m) for m in self.monomials]
        return "Polynomial: " + " + ".join(monomial_strings)

    def __mul__(self, other):
        vec = GF2(np.zeros(self.l * self.m, dtype=np.uint8))
        for m in self.monomials:
            for o in other.monomials:
                x_power = (m.x_power + o.x_power) % self.l
                y_power = (m.y_power + o.y_power) % self.m
                vec[x_power * self.m + y_power] += GF2(1)
        return Polynomial([], self.l, self.m, vec)

    def __add__(self, other):
        return Polynomial([], self.l, self.m, self.vec + other.vec)

    def __eq__(self, other):
        if (
            self.l != other.l
            or self.m != other.m
            or not np.array_equal(self.vec, other.vec)
            or self.monomials != other.monomials
        ):
            return False

        return True

    def T(self):
        if len(self.monomials) == 0:
            return self
        return Polynomial([m.inverse() for m in self.monomials])


class Code:
    def __init__(
        self,
        A_polynomial: Polynomial,
        B_polynomial: Polynomial,
        f: Polynomial | None = None,
    ) -> None:
        self.l = A_polynomial.monomials[0].l
        self.m = A_polynomial.monomials[0].m
        self.A_polynomial = A_polynomial
        self.B_polynomial = B_polynomial
        self.Hx = GF2(np.zeros((self.l * self.m, 2 * self.l * self.m), dtype=np.uint8))
        self.Hz = GF2(np.zeros((self.l * self.m, 2 * self.l * self.m), dtype=np.uint8))
        for i in A_polynomial.monomials:
            self.Hx[:, : self.l * self.m] += i.to_matrix()
            self.Hz[:, self.l * self.m :] += i.to_matrix().T
        for i in B_polynomial.monomials:
            self.Hx[:, self.l * self.m :] += i.to_matrix()
            self.Hz[:, : self.l * self.m] += i.to_matrix().T

        assert self.Hx.shape[0] - np.linalg.matrix_rank(self.Hx) == self.Hz.shape[
            0
        ] - np.linalg.matrix_rank(self.Hz)
        self.k = 2 * (self.Hx.shape[0] - np.linalg.matrix_rank(self.Hx))
        self.logical_X_ops, self.logical_Z_ops = self.get_logical_operators(f)

    def check_X_op_commutes(self, poly_1: Polynomial, poly_2: Polynomial) -> bool:
        vec1 = poly_1.vec
        vec2 = poly_2.vec
        for i in range(self.l * self.m):
            left_parity = int(np.dot(self.Hz[i, : self.l * self.m], vec1))
            right_parity = int(np.dot(self.Hz[i, self.l * self.m :], vec2))
            if (left_parity + right_parity) % 2 != 0:
                return False
        return True

    def check_Z_op_commutes(self, poly_1: Polynomial, poly_2: Polynomial) -> bool:
        vec1 = poly_1.vec
        vec2 = poly_2.vec
        for i in range(self.l * self.m):
            left_parity = int(np.dot(self.Hx[i, : self.l * self.m], vec1))
            right_parity = int(np.dot(self.Hx[i, self.l * self.m :], vec2))
            if (left_parity + right_parity) % 2 != 0:
                return False
        return True

    def _find_f_spanning_sets(
        self, f_cands: FieldArray, f_cand: Polynomial | None
    ) -> Tuple[list[FieldArray], list[FieldArray]]:
        """Find a set of logical operators of the form {X(αf, 0) | α ∈ M},
        and {Z(0, αf^T) | α ∈ M} that span k/2 logical qubits each"""
        Hx_rank = np.linalg.matrix_rank(self.Hx)
        Hz_rank = np.linalg.matrix_rank(self.Hz)
        for i in range(f_cands.shape[0]):
            if f_cand is None:
                f_cand = Polynomial([], self.l, self.m, GF2(f_cands[i, :]))
            assert f_cand * self.B_polynomial == Polynomial([], self.l, self.m)
            assert self.check_X_op_commutes(f_cand, Polynomial([], self.l, self.m))
            assert self.check_Z_op_commutes(Polynomial([], self.l, self.m), f_cand.T())
            all_f_ops = GF2(
                np.zeros((self.l * self.m, f_cands.shape[1]), dtype=np.uint8)
            )
            all_f_T_ops = GF2(np.zeros_like(all_f_ops))
            for ii, alpha in enumerate(
                [
                    Monomial(self.l, self.m, x_pow, y_pow)
                    for x_pow in range(self.l)
                    for y_pow in range(self.m)
                ]
            ):
                all_f_ops[ii, :] = (Polynomial([alpha]) * f_cand).vec
                all_f_T_ops[ii, :] = (Polynomial([alpha]) * f_cand.T()).vec
            big_X_matrix = GF2(
                np.vstack((self.Hx, np.hstack((all_f_ops, np.zeros_like(all_f_ops)))))
            )
            big_Z_matrix = GF2(
                np.vstack(
                    (self.Hz, np.hstack((np.zeros_like(all_f_T_ops), all_f_T_ops)))
                )
            )
            # make sure that there are enough independent ones up to the stabilizer
            if (
                np.linalg.matrix_rank(big_X_matrix) - Hx_rank == self.k // 2
                and np.linalg.matrix_rank(big_Z_matrix) - Hz_rank == self.k // 2
            ):
                X1_ops = [
                    GF2(big_X_matrix[i + self.l * self.m, :])
                    for i in range(self.l * self.m)
                ]
                Z2_ops = [
                    GF2(big_Z_matrix[i + self.l * self.m, :])
                    for i in range(self.l * self.m)
                ]
                return X1_ops, Z2_ops
        sys.exit("Wasn't able to find f candidate")

    def _find_gh_spanning_sets(self, gh_cands):
        """Find sets of logical operators of the form {X(αg, αh) | α ∈ M} and
        {Z(αh^T, αg^T) | α ∈ M} that each span k/2 logical qubits."""
        Hx_rank = np.linalg.matrix_rank(self.Hx)
        Hz_rank = np.linalg.matrix_rank(self.Hz)
        for j in range(gh_cands.shape[0]):
            g_cand = Polynomial([], self.l, self.m, gh_cands[j, : self.l * self.m])
            h_cand = Polynomial([], self.l, self.m, gh_cands[j, self.l * self.m :])
            assert self.check_X_op_commutes(g_cand, h_cand)
            assert (
                g_cand * self.B_polynomial + h_cand * self.A_polynomial
                == Polynomial([], self.l, self.m)
            )
            assert self.check_Z_op_commutes(h_cand.T(), g_cand.T())
            all_gh_ops = GF2(
                np.zeros((self.l * self.m, gh_cands.shape[1]), dtype=np.uint8)
            )
            all_hg_T_ops = GF2(np.zeros_like(all_gh_ops))
            for jj, alpha in enumerate(
                [
                    Monomial(self.l, self.m, x_pow, y_pow)
                    for x_pow in range(self.l)
                    for y_pow in range(self.m)
                ]
            ):
                all_gh_ops[jj, :] = GF2(
                    np.hstack(
                        (
                            (Polynomial([alpha]) * g_cand).vec,
                            (Polynomial([alpha]) * h_cand).vec,
                        )
                    )
                )
                all_hg_T_ops[jj, :] = GF2(
                    np.hstack(
                        (
                            (Polynomial([alpha]) * h_cand.T()).vec,
                            (Polynomial([alpha]) * g_cand.T()).vec,
                        )
                    )
                )
            big_X_matrix = GF2(np.vstack((self.Hx, all_gh_ops)))
            big_Z_matrix = GF2(np.vstack((self.Hz, all_hg_T_ops)))
            if (
                np.linalg.matrix_rank(big_X_matrix) - Hx_rank == self.k // 2
                and np.linalg.matrix_rank(big_Z_matrix) - Hz_rank == self.k // 2
            ):
                X2_ops = [GF2(big_X_matrix[-i - 1, :]) for i in range(self.l * self.m)]
                Z1_ops = [GF2(big_Z_matrix[-i - 1, :]) for i in range(self.l * self.m)]
                return X2_ops, Z1_ops
        sys.exit(1)

    def find_logical_qubits(
        self,
        X_ops: List[FieldArray],
        Z_ops: List[FieldArray],
        num_qubits: int,
        logical_ops: List[Tuple[FieldArray, FieldArray]],
    ) -> List[Tuple[FieldArray, FieldArray]]:
        """Given a list of X operators and Z operators that span `num_qubits` logical
        qubits, return a set of logical operators that identify the logical qubits."""
        if num_qubits == 0:
            return logical_ops
        X1 = X_ops.pop(0)
        Z1_index = -1
        for i, Z_op in enumerate(Z_ops):
            if np.dot(X1, Z_op) == GF2(1):
                if Z1_index == -1:
                    Z1_index = i
                else:
                    Z_ops[i] = Z_op + Z_ops[Z1_index]
        if Z1_index == -1:
            # this X op is in the stabilizer, remove it and try again
            aug_mat = np.vstack((self.Hx, X1))
            assert np.linalg.matrix_rank(aug_mat) == np.linalg.matrix_rank(self.Hx)
            return self.find_logical_qubits(X_ops, Z_ops, num_qubits, logical_ops)
        Z1 = Z_ops.pop(Z1_index)
        for i, X_op in enumerate(X_ops):
            if np.dot(X_op, Z1) == GF2(1):
                X_ops[i] = X_op + X1

        logical_ops.append((X1, Z1))
        return self.find_logical_qubits(X_ops, Z_ops, num_qubits - 1, logical_ops)

    def get_logical_operators(
        self,
        f_cand: Polynomial | None = None,
    ) -> Tuple[FieldArray, FieldArray]:
        f_cands, gh_cands = self._get_logical_op_polynomial_cands()
        X1_ops, Z2_ops = self._find_f_spanning_sets(f_cands, f_cand)
        X2_ops, Z1_ops = self._find_gh_spanning_sets(gh_cands)
        unprimed_ops = self.find_logical_qubits(X1_ops, Z1_ops, self.k // 2, [])
        primed_ops = self.find_logical_qubits(X2_ops, Z2_ops, self.k // 2, [])
        assert len(unprimed_ops) == self.k // 2 and len(primed_ops) == self.k // 2
        logical_X_ops = np.vstack(
            [ops[0] for ops in unprimed_ops] + [ops[0] for ops in primed_ops]
        )
        logical_Z_ops = np.vstack(
            [ops[1] for ops in unprimed_ops] + [ops[1] for ops in primed_ops]
        )
        return GF2(logical_X_ops), GF2(logical_Z_ops)

    def _get_logical_op_polynomial_cands(self) -> Tuple[FieldArray, FieldArray]:
        A = self.Hx[:, : self.l * self.m]
        B = GF2(self.Hx[:, self.l * self.m :])
        f_cands = B.T.null_space()
        tmp = GF2(np.hstack((B.T, A.T)))
        gh_cands = tmp.null_space()
        return f_cands, gh_cands

    def create_syndrome_measurement_circuit_data_noise(
        self,
        logical_error: Literal["X", "Z"],
        error_rate: float,
    ) -> stim.Circuit:
        A1 = self.A_polynomial.monomials[0].to_matrix()
        A2 = self.A_polynomial.monomials[1].to_matrix()
        A3 = self.A_polynomial.monomials[2].to_matrix()
        B1 = self.B_polynomial.monomials[0].to_matrix()
        B2 = self.B_polynomial.monomials[1].to_matrix()
        B3 = self.B_polynomial.monomials[2].to_matrix()
        circuit = stim.Circuit()
        qubits_per_block = self.l * self.m
        L_block = [i for i in range(qubits_per_block)]
        R_block = [i + qubits_per_block for i in range(qubits_per_block)]
        X_block = [i + 2 * qubits_per_block for i in range(qubits_per_block)]
        Z_block = [i + 3 * qubits_per_block for i in range(qubits_per_block)]

        if logical_error == "Z":
            circuit.append("RX", L_block + R_block)
        elif logical_error == "X":
            circuit.append("RZ", L_block + R_block)

        for i in range(2):
            if i == 1:
                circuit.append("DEPOLARIZE1", L_block + R_block, error_rate)
            # initialize ancillas
            circuit.append("RZ", Z_block)
            circuit.append("RX", X_block)

            circuit = self._add_gates_round1(circuit, A1, R_block, L_block, Z_block, 0)
            circuit = self._add_cnot_gates(
                circuit,
                A2,
                A3.T,
                L_block,
                R_block,
                X_block,
                Z_block,
                0,
            )
            circuit = self._add_cnot_gates(
                circuit,
                B2,
                B1.T,
                R_block,
                L_block,
                X_block,
                Z_block,
                0,
            )
            circuit = self._add_cnot_gates(
                circuit,
                B1,
                B2.T,
                R_block,
                L_block,
                X_block,
                Z_block,
                0,
            )
            circuit = self._add_cnot_gates(
                circuit,
                B3,
                B3.T,
                R_block,
                L_block,
                X_block,
                Z_block,
                0,
            )
            circuit = self._add_cnot_gates(
                circuit,
                A1,
                A2.T,
                L_block,
                R_block,
                X_block,
                Z_block,
                0,
            )
            circuit = self._add_gates_round7(circuit, A3, R_block, L_block, X_block, 0)
            # measure ancillas
            circuit.append("MZ", Z_block)
            circuit.append("MX", X_block)
        measurement_offset = 0
        if logical_error == "X":
            measurement_offset = self.l * self.m
        for j in range(self.l * self.m):
            j += measurement_offset
            circuit.append(
                "DETECTOR",
                [
                    stim.target_rec(-1 - j),
                    stim.target_rec(-1 - j - 2 * self.l * self.m),
                ],
            )

        # measure off all data qubits
        if logical_error == "Z":
            circuit.append("MX", L_block + R_block)
            ops = self.logical_X_ops
        else:
            circuit.append("MZ", L_block + R_block)
            ops = self.logical_Z_ops
        for i in range(self.k):
            support = ops[i, :].nonzero()[0]
            targets = [2 * self.l * self.m - s for s in support]
            circuit.append(
                "OBSERVABLE_INCLUDE", [stim.target_rec(-t) for t in targets], i
            )
        return circuit

    def create_syndrome_measurement_circuit(
        self,
        x_detectors: bool,
        z_detectors: bool,
        num_rounds: int,
        logical_operator: Literal["X", "Z"],
        error_rate: float,
        skip_redundant_checks: bool = False,
    ) -> stim.Circuit:
        A1 = self.A_polynomial.monomials[0].to_matrix()
        A2 = self.A_polynomial.monomials[1].to_matrix()
        A3 = self.A_polynomial.monomials[2].to_matrix()
        B1 = self.B_polynomial.monomials[0].to_matrix()
        B2 = self.B_polynomial.monomials[1].to_matrix()
        B3 = self.B_polynomial.monomials[2].to_matrix()
        circuit = stim.Circuit()
        qubits_per_block = self.l * self.m
        L_block = [i for i in range(qubits_per_block)]
        R_block = [i + qubits_per_block for i in range(qubits_per_block)]
        X_block = [i + 2 * qubits_per_block for i in range(qubits_per_block)]
        Z_block = [i + 3 * qubits_per_block for i in range(qubits_per_block)]

        redundant_X_checks = None
        redundant_Z_checks = None
        if skip_redundant_checks:
            redundant_X_checks, redundant_Z_checks = self._get_redundant_stabilizers()

        if logical_operator == "Z":
            circuit.append("RZ", L_block + R_block)
        elif logical_operator == "X":
            circuit.append("RX", L_block + R_block)

        circuit += self._syndrome_measurement_rounds(
            False,
            False,
            1,
            A1,
            A2,
            A3,
            B1,
            B2,
            B3,
            R_block,
            L_block,
            X_block,
            Z_block,
            0,
            skip_redundant_checks,
            redundant_X_checks,
            redundant_Z_checks,
        )
        if num_rounds != 0:
            circuit += self._syndrome_measurement_rounds(
                x_detectors,
                z_detectors,
                num_rounds,
                A1,
                A2,
                A3,
                B1,
                B2,
                B3,
                R_block,
                L_block,
                X_block,
                Z_block,
                error_rate,
                skip_redundant_checks,
                redundant_X_checks,
                redundant_Z_checks,
            )
        else:
            circuit.append("DEPOLARIZE1", L_block + R_block, error_rate)
        circuit += self._syndrome_measurement_rounds(
            x_detectors,
            z_detectors,
            1,
            A1,
            A2,
            A3,
            B1,
            B2,
            B3,
            R_block,
            L_block,
            X_block,
            Z_block,
            0,
            skip_redundant_checks,
            redundant_X_checks,
            redundant_Z_checks,
        )
        # measure off all data qubits
        if logical_operator == "X":
            circuit.append("MX", L_block + R_block)
            ops = self.logical_X_ops
        else:
            circuit.append("MZ", L_block + R_block)
            ops = self.logical_Z_ops
        for i in range(self.k):
            support = ops[i, :].nonzero()[0]
            targets = [2 * self.l * self.m - s for s in support]
            circuit.append(
                "OBSERVABLE_INCLUDE", [stim.target_rec(-t) for t in targets], i
            )

        return circuit

    def _add_gates_round1(
        self,
        circuit: stim.Circuit,
        A1: np.ndarray,
        R_block: List[int],
        L_block: List[int],
        Z_block: List[int],
        error_rate: float = 0.0,
    ) -> stim.Circuit:
        cnot_qubits = []
        cnot_control_indices = A1.T.nonzero()[1]
        for i in range(self.l * self.m):
            cnot_qubits.append(R_block[cnot_control_indices[i]])
            cnot_qubits.append(Z_block[i])
        circuit.append("CNOT", cnot_qubits)
        if error_rate != 0:
            circuit.append("DEPOLARIZE2", cnot_qubits, error_rate)
            # idle qubit errors
            circuit.append("DEPOLARIZE1", L_block, error_rate)
        circuit.append("TICK")
        return circuit

    def _add_gates_round1_skip_redundant_checks(
        self,
        circuit: stim.Circuit,
        A1: np.ndarray,
        R_block: list[int],
        L_block: list[int],
        Z_block: list[int],
        redundant_Z_checks: list[int],
        error_rate: float = 0.0,
    ) -> stim.Circuit:
        cnot_qubits = []
        cnot_control_indices = A1.T.nonzero()[1]
        for i in range(self.l * self.m):
            # NOTE: Change list to set if this becomes performance bottleneck
            if i not in redundant_Z_checks:
                cnot_qubits.append(R_block[cnot_control_indices[i]])
                cnot_qubits.append(Z_block[i])
        circuit.append("CNOT", cnot_qubits)
        if error_rate != 0:
            circuit.append("DEPOLARIZE2", cnot_qubits, error_rate)
            # idle qubit errors
            circuit.append("DEPOLARIZE1", L_block, error_rate)
        circuit.append("TICK")
        return circuit

    def _add_cnot_gates(
        self,
        circuit: stim.Circuit,
        X_block_target_indices: np.ndarray,
        Z_block_control_indices: np.ndarray,
        X_block_targets: list[int],
        Z_block_controls: list[int],
        X_block: list[int],
        Z_block: list[int],
        error_rate: float = 0.0,
    ) -> stim.Circuit:
        nonzero_X_block_target_indices = X_block_target_indices.nonzero()[1]
        nonzero_Z_block_control_indices = Z_block_control_indices.nonzero()[1]
        cnot_qubits = []
        for i in range(self.l * self.m):
            cnot_qubits.append(X_block[i])
            cnot_qubits.append(X_block_targets[nonzero_X_block_target_indices[i]])
            cnot_qubits.append(Z_block_controls[nonzero_Z_block_control_indices[i]])
            cnot_qubits.append(Z_block[i])
        circuit.append("CNOT", cnot_qubits)
        if error_rate != 0:
            circuit.append("DEPOLARIZE2", cnot_qubits, error_rate)
        circuit.append("TICK")
        return circuit

    def _add_cnot_gates_skip_redundant_checks(
        self,
        circuit: stim.Circuit,
        X_block_target_indices: np.ndarray,
        Z_block_control_indices: np.ndarray,
        X_block_targets: list[int],
        Z_block_controls: list[int],
        X_block: list[int],
        Z_block: list[int],
        redundant_X_checks: list[int],
        redundant_Z_checks: list[int],
        error_rate: float = 0.0,
    ) -> stim.Circuit:
        nonzero_X_block_target_indices = X_block_target_indices.nonzero()[1]
        nonzero_Z_block_control_indices = Z_block_control_indices.nonzero()[1]
        cnot_qubits = []
        for i in range(self.l * self.m):
            if i not in redundant_X_checks:
                cnot_qubits.append(X_block[i])
                cnot_qubits.append(X_block_targets[nonzero_X_block_target_indices[i]])
            if i not in redundant_Z_checks:
                cnot_qubits.append(Z_block_controls[nonzero_Z_block_control_indices[i]])
                cnot_qubits.append(Z_block[i])
        circuit.append("CNOT", cnot_qubits)
        if error_rate != 0:
            circuit.append("DEPOLARIZE2", cnot_qubits, error_rate)
        circuit.append("TICK")
        return circuit

    def _add_gates_round7(
        self,
        circuit: stim.Circuit,
        A3: np.ndarray,
        R_block: list[int],
        L_block: list[int],
        X_block: list[int],
        error_rate: float = 0.0,
    ) -> stim.Circuit:
        nonzero_X_block_target_indices = A3.nonzero()[1]
        cnot_qubits = []
        for i in range(self.l * self.m):
            cnot_qubits.append(X_block[i])
            cnot_qubits.append(L_block[nonzero_X_block_target_indices[i]])
        circuit.append("CNOT", cnot_qubits)
        if error_rate != 0:
            circuit.append("DEPOLARIZE2", cnot_qubits, error_rate)
            # idle qubit errors
            circuit.append("DEPOLARIZE1", R_block, error_rate)
        return circuit

    def _add_gates_round7_skip_redundant_checks(
        self,
        circuit: stim.Circuit,
        A3: np.ndarray,
        R_block: list[int],
        L_block: list[int],
        X_block: list[int],
        redundant_X_checks: list[int],
        error_rate: float = 0.0,
    ) -> stim.Circuit:
        nonzero_X_block_target_indices = A3.nonzero()[1]
        cnot_qubits = []
        for i in range(self.l * self.m):
            if i not in redundant_X_checks:
                cnot_qubits.append(X_block[i])
                cnot_qubits.append(L_block[nonzero_X_block_target_indices[i]])
        circuit.append("CNOT", cnot_qubits)
        if error_rate != 0:
            circuit.append("DEPOLARIZE2", cnot_qubits, error_rate)
            # idle qubit errors
            circuit.append("DEPOLARIZE1", R_block, error_rate)
        return circuit

    def _syndrome_measurement_rounds(
        self,
        x_detectors: bool,
        z_detectors: bool,
        rounds: int,
        A1: list[int],
        A2: list[int],
        A3: list[int],
        B1: list[int],
        B2: list[int],
        B3: list[int],
        R_block: list[int],
        L_block: list[int],
        X_block: list[int],
        Z_block: list[int],
        error_rate: float,
        skip_redundant_checks: bool = False,
        redundant_X_checks: list[int] | None = None,
        redundant_Z_checks: list[int] | None = None,
    ) -> stim.CircuitRepeatBlock:
        X_block_measured = X_block.copy()
        Z_block_measured = Z_block.copy()
        if skip_redundant_checks:
            x_offset = X_block_measured[0]
            z_offset = Z_block_measured[0]
            for i in redundant_X_checks:
                X_block_measured.remove(i + x_offset)
            for i in redundant_Z_checks:
                Z_block_measured.remove(i + z_offset)
        circuit_block = stim.Circuit()
        circuit_block.append("RZ", Z_block_measured)
        if error_rate > 0:
            circuit_block.append("X_ERROR", Z_block_measured, error_rate)
        circuit_block.append("TICK")
        circuit_block.append("RX", X_block_measured)
        if error_rate > 0:
            circuit_block.append("Z_ERROR", X_block_measured, error_rate)
        if skip_redundant_checks:
            circuit_block = self._add_gates_round1_skip_redundant_checks(
                circuit_block,
                A1,
                R_block,
                L_block,
                Z_block,
                redundant_Z_checks,
                error_rate,
            )
            circuit_block = self._add_cnot_gates_skip_redundant_checks(
                circuit_block,
                A2,
                A3.T,
                L_block,
                R_block,
                X_block,
                Z_block,
                redundant_X_checks,
                redundant_Z_checks,
                error_rate,
            )
            circuit_block = self._add_cnot_gates_skip_redundant_checks(
                circuit_block,
                B2,
                B1.T,
                R_block,
                L_block,
                X_block,
                Z_block,
                redundant_X_checks,
                redundant_Z_checks,
                error_rate,
            )
            circuit_block = self._add_cnot_gates_skip_redundant_checks(
                circuit_block,
                B1,
                B2.T,
                R_block,
                L_block,
                X_block,
                Z_block,
                redundant_X_checks,
                redundant_Z_checks,
                error_rate,
            )
            circuit_block = self._add_cnot_gates_skip_redundant_checks(
                circuit_block,
                B3,
                B3.T,
                R_block,
                L_block,
                X_block,
                Z_block,
                redundant_X_checks,
                redundant_Z_checks,
                error_rate,
            )
            circuit_block = self._add_cnot_gates_skip_redundant_checks(
                circuit_block,
                A1,
                A2.T,
                L_block,
                R_block,
                X_block,
                Z_block,
                redundant_X_checks,
                redundant_Z_checks,
                error_rate,
            )
            circuit_block = self._add_gates_round7_skip_redundant_checks(
                circuit_block,
                A3,
                R_block,
                L_block,
                X_block,
                redundant_X_checks,
                error_rate,
            )
        else:
            circuit_block = self._add_gates_round1(
                circuit_block, A1, R_block, L_block, Z_block, error_rate
            )
            circuit_block = self._add_cnot_gates(
                circuit_block,
                A2,
                A3.T,
                L_block,
                R_block,
                X_block,
                Z_block,
                error_rate,
            )
            circuit_block = self._add_cnot_gates(
                circuit_block,
                B2,
                B1.T,
                R_block,
                L_block,
                X_block,
                Z_block,
                error_rate,
            )
            circuit_block = self._add_cnot_gates(
                circuit_block,
                B1,
                B2.T,
                R_block,
                L_block,
                X_block,
                Z_block,
                error_rate,
            )
            circuit_block = self._add_cnot_gates(
                circuit_block,
                B3,
                B3.T,
                R_block,
                L_block,
                X_block,
                Z_block,
                error_rate,
            )
            circuit_block = self._add_cnot_gates(
                circuit_block,
                A1,
                A2.T,
                L_block,
                R_block,
                X_block,
                Z_block,
                error_rate,
            )
            circuit_block = self._add_gates_round7(
                circuit_block, A3, R_block, L_block, X_block, error_rate
            )
        if error_rate > 0:
            circuit_block.append("MZ", Z_block_measured, error_rate)
        else:
            circuit_block.append("MZ", Z_block_measured)
        circuit_block.append("TICK")
        if error_rate > 0:
            circuit_block.append("MX", X_block_measured, error_rate)
            circuit_block.append("DEPOLARIZE1", R_block + L_block, error_rate)
        else:
            circuit_block.append("MX", X_block_measured)

        if x_detectors or z_detectors:
            circuit_block.append_from_stim_program_text("SHIFT_COORDS(0, 1)")
            offset = len(Z_block_measured) + len(X_block_measured)
            # for now, ignoring the actual layout of qubits on a torus, and labelling the detectors
            # with one coordinate as position on a "line", and one coordinate to indicate the round
            if z_detectors:
                for i in range(len(Z_block_measured)):
                    circuit_block.append(
                        "DETECTOR",
                        [
                            stim.target_rec(-offset + i),
                            stim.target_rec(-2 * offset + i),
                        ],
                        (i, 0),
                    )
            if x_detectors:
                coord_offset = 0
                if z_detectors:
                    coord_offset = len(Z_block_measured)
                for i in range(len(X_block_measured)):
                    circuit_block.append(
                        "DETECTOR",
                        [
                            stim.target_rec(-len(X_block_measured) + i),
                            stim.target_rec(-len(X_block_measured) - offset + i),
                        ],
                        (i + coord_offset, 0),
                    )

        return circuit_block * rounds

    def _get_redundant_stabilizers(self) -> tuple[list[int], list[int]]:
        dependent_X_checks = find_dependent_row_vectors(
            np.array(self.Hx).astype(np.uint8)
        )
        dependent_Z_checks = find_dependent_row_vectors(
            np.array(self.Hz).astype(np.uint8)
        )
        return dependent_X_checks, dependent_Z_checks


########################
# Standalone functions #
########################


def shift_matrix(l: int) -> FieldArray:
    """Generate a cyclic shift matrix of size lxl. For example,
    >>> shift_matrix(3)
    array([[0, 1, 0],
           [0, 0, 1],
           [1, 0, 0]])
    """
    s = np.diag(np.ones(l - 1, dtype=np.uint8), 1)
    s[l - 1, 0] = 1
    return GF2(s)


def get_72_12_6_code(use_paper_f=False):
    a1 = Monomial(6, 6, 3, 0)
    a2 = Monomial(6, 6, 0, 1)
    a3 = Monomial(6, 6, 0, 2)
    A_poly = Polynomial([a1, a2, a3])

    b1 = Monomial(6, 6, 0, 3)
    b2 = Monomial(6, 6, 1, 0)
    b3 = Monomial(6, 6, 2, 0)
    B_poly = Polynomial([b1, b2, b3])

    if use_paper_f:
        f1 = Monomial(6, 6, 0, 3)
        f2 = Monomial(6, 6, 2, 0)
        f3 = Monomial(6, 6, 3, 3)
        f4 = Monomial(6, 6, 4, 0)
        f5 = Monomial(6, 6, 4, 3)
        f6 = Monomial(6, 6, 5, 3)
        f = Polynomial([f1, f2, f3, f4, f5, f6])
    else:
        f = None

    return Code(A_poly, B_poly, f)


def get_144_12_12_code():
    l = 12
    m = 6

    a1 = Monomial(l, m, 3, 0)
    a2 = Monomial(l, m, 0, 1)
    a3 = Monomial(l, m, 0, 2)
    A_poly = Polynomial([a1, a2, a3])

    b1 = Monomial(l, m, 0, 3)
    b2 = Monomial(l, m, 1, 0)
    b3 = Monomial(l, m, 2, 0)
    B_poly = Polynomial([b1, b2, b3])
    return Code(A_poly, B_poly)


def get_98_6_12_code():
    """From http://arxiv.org/abs/2407.03973v1"""
    l = 7
    m = 7
    a1 = Monomial(l, m, 1, 0)
    a2 = Monomial(l, m, 0, 3)
    a3 = Monomial(l, m, 0, 4)
    A_poly = Polynomial([a1, a2, a3])

    b1 = Monomial(l, m, 0, 1)
    b2 = Monomial(l, m, 3, 0)
    b3 = Monomial(l, m, 4, 0)
    B_poly = Polynomial([b1, b2, b3])
    return Code(A_poly, B_poly)
