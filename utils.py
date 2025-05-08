import subprocess
from math import comb

import numpy as np
import torch
from numba import jit
from ldpc import bposd_decoder, bp_decoder
import galois
from galois import FieldArray

GF2 = galois.GF(2)


def create_weighted_src_mask(D: torch.Tensor, max_distance: int = 2) -> torch.Tensor:
    """Create a weighted src mask.

    The original src mask unmasks a position if two detectors are flipped by a common
    error. This version does the same thing, but also adds the log of the number of
    errors that flip the two detectors.
    """
    weight_matrix = D.float() @ D.float().T
    return torch.log(weight_matrix)


def count_errors(
    outputs: torch.Tensor, measured_observables: torch.Tensor
) -> torch.Tensor:
    outputs = outputs.float()
    results = (outputs + measured_observables) % 2
    results = torch.sum(results, 1)
    results = (results > 0).int()
    return torch.sum(results)


def get_git_hash() -> str:
    """Adopted from https://stackoverflow.com/a/21901260"""
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()


def total_error_rate(p, n: int):
    """Given n qubits with error rate p, return the probability that one or more
    have an error"""
    res = np.zeros_like(p)
    for i in range(1, n + 1):
        res += comb(n, i) * p**i * (1 - p) ** (12 - i)
    return res


def get_bp_osd_decoder(H, channel_probs, osd_order: int = 3):
    bpd = bposd_decoder(
        H,
        error_rate=0.01,
        channel_probs=channel_probs,
        max_iter=10000,
        bp_method="ms",
        ms_scaling_factor=0,
        osd_method="osd_cs",
        osd_order=osd_order,
    )
    return bpd


def get_bp_decoder(H, channel_probs):
    bpd = bp_decoder(
        H,
        error_rate=0.01,
        channel_probs=channel_probs,
        max_iter=10000,
        bp_method="ms",
        ms_scaling_factor=0,
    )
    return bpd


@jit(nopython=True)
def rref_f2_int(A: np.ndarray) -> np.ndarray:
    n_rows, n_cols = A.shape
    i = 0
    j = 0
    M = A.copy()
    while i < n_rows and j < n_cols:
        p = get_first_odd_entry(M[i:, j])
        if p < 0:
            j += 1
        else:
            p += i
            if p != i:
                M[np.array([p, i]), :] = M[np.array([i, p]), :]
            for ii in range(n_rows):
                if ii != i and M[ii, j] % 2 == 1:
                    M[ii, :] += M[i, :]
            j += 1
            i += 1
    return M % 2


def rref_f2_int_record(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_rows, n_cols = A.shape
    i = 0
    j = 0
    M = A.copy()
    record = np.eye(n_rows, dtype=np.uint8)
    while i < n_rows and j < n_cols:
        p = get_first_odd_entry(M[i:, j])
        if p < 0:
            j += 1
        else:
            p += i
            if p != i:
                M[np.array([p, i]), :] = M[np.array([i, p]), :]
                record[np.array([p, i]), :] = record[np.array([i, p]), :]
            for ii in range(n_rows):
                if ii != i and M[ii, j] % 2 == 1:
                    M[ii, :] += M[i, :]
                    record[ii, :] += record[i, :]

            j += 1
            i += 1
    return M % 2, record % 2


def find_dependent_row_vectors(A: np.ndarray) -> list[int]:
    M, record = rref_f2_int_record(A)
    sums = np.sum(M, axis=1)
    num_dependent_vecs = (sums == 0).sum()
    A_sets = []
    for i in range(num_dependent_vecs):
        i += 1
        vecs = record[-i, :].nonzero()[0]
        A = set()
        for vec in vecs:
            A.add(vec)
        A_sets.append(A)
    dependent_vec_indices = []
    for _ in range(num_dependent_vecs):
        A = A_sets.pop(0)
        A_others = set()
        for A_other in A_sets:
            for elt in A_other:
                A_others.add(elt)
        assert len(A - A_others) > 0
        dependent_vec_indices.append((A - A_others).pop())
        A_sets.append(A)

    return dependent_vec_indices


@jit(nopython=True)
def get_first_odd_entry(v: np.ndarray) -> int:
    for i in range(v.shape[0]):
        if v[i] % 2 == 1:
            return i
    return -1


@jit(nopython=True)
def rref_f2_bool(A: np.ndarray) -> np.ndarray:
    """
    Find the reduced row echelon form of A over F2
    """
    n_rows, n_cols = A.shape
    i = 0
    j = 0
    M = A.copy()
    while i < n_rows and j < n_cols:
        p = get_first_true_entry(M[i:, j])
        if p < 0:
            j += 1
        else:
            p += i
            if p != i:
                M[np.array([p, i]), :] = M[np.array([i, p]), :]
            for ii in range(n_rows):
                if ii != i and M[ii, j]:
                    M[ii, :] = np.logical_xor(M[ii, :], M[i, :])
            j += 1
            i += 1
    return M


@jit(nopython=True)
def get_first_true_entry(v: np.ndarray) -> int:
    """Assumes v is 1d"""
    for i in range(v.shape[0]):
        if v[i]:
            return i
    return -1


@jit(nopython=True)
def check_in_rowspace_f2(M: np.ndarray, v: np.ndarray) -> np.ndarray | None:
    """Check if 1d array v is in rowspace of M. If it is, return which vectors it is the sum of.
    If it isn't, return None.

    NOTE: This function assumes the rows of M are LI."""
    v_2d = v[:, np.newaxis]
    aug_mat = np.hstack((M.T, v_2d))
    rref = rref_f2_int(aug_mat)
    sol = np.zeros(aug_mat.shape[1] - 1)
    for i in range(rref.shape[0] - 1, -1, -1):
        if rref[-i, -1] == 1:
            x = get_first_odd_entry(rref[-i, :])
            if x == rref.shape[1] - 1:
                return None
            else:
                sol[x] = 1
    return sol


@jit(nopython=True)
def moving_average(arr, window_size):
    length = len(arr)
    avgs = np.zeros_like(arr)
    for i in range(length):
        avgs[i] = np.mean(arr[i : min(i + window_size, length)])
    return avgs


def compute_intersection_f2(A: FieldArray, B: FieldArray) -> FieldArray:
    len_a = A.shape[1]
    C: FieldArray = np.hstack((A, B))
    null_space_C = C.null_space()
    a = null_space_C[:, :len_a].T
    output1 = A @ a
    return output1


def find_complement_f2(A: FieldArray) -> FieldArray:
    rng = np.random.default_rng()
    found_vectors = 0
    n, d = A.shape
    B = GF2(np.zeros((n, n - d), dtype=np.uint8))
    A_aug = np.hstack((A, B)).T
    num_vectors_tested = 0
    while found_vectors != n - d:
        num_vectors_tested += 1
        A_aug[d + found_vectors, :] = GF2(rng.integers(low=0, high=2, size=n))
        # TODO: Instead of computing the rank each time (O(n^3)), keep it in row_echelon,
        # so that adding another vector at the bottom and seeing if it's in the row
        # space is cheap
        if (
            np.linalg.matrix_rank(A_aug[: d + found_vectors + 1, :])
            == d + found_vectors + 1
        ):
            found_vectors += 1
    print("Num vectors tested:", num_vectors_tested)
    return A_aug[d:, :].T
