from time import perf_counter
import numpy as np

import galois

from utils import (
    rref_f2_bool,
    rref_f2_int,
    check_in_rowspace_f2,
    rref_f2_int_record,
    find_dependent_row_vectors,
)

GF2 = galois.GF(2)


def test_rref_f2_1():
    arr = np.array(
        [[1, 1], [1, 0]],
        dtype=np.bool_,
    )
    arr_gf2 = GF2(arr.astype(np.uint8))
    rref1 = arr_gf2.row_reduce()
    rref2 = rref_f2_bool(arr)
    rref3 = rref_f2_int(arr.astype(np.uint8))
    assert np.array_equal(rref1, rref2)
    assert np.array_equal(rref1, rref3)


def test_rref_f2_2():
    arr = np.array(
        [
            [1, 0, 0, 1, 0, 1],
            [0, 1, 1, 0, 1, 0],
            [0, 1, 0, 0, 0, 1],
            [1, 1, 1, 0, 1, 0],
        ],
        dtype=np.bool_,
    )
    arr_gf2 = GF2(arr.astype(np.uint8))
    rref1 = arr_gf2.row_reduce()
    rref2 = rref_f2_bool(arr)
    rref3 = rref_f2_int(arr.astype(np.uint8))
    assert np.array_equal(rref1, rref2)
    assert np.array_equal(rref1, rref3)


def test_rref_f2_3():
    arr = np.array(
        [
            [1, 0, 0, 1, 0, 1],
            [0, 1, 1, 0, 1, 0],
            [0, 1, 0, 0, 0, 1],
            [1, 1, 1, 0, 1, 0],
        ],
        dtype=np.bool_,
    )
    arr_gf2 = GF2(arr.astype(np.uint8))
    rref1 = arr_gf2.row_reduce()
    rref2 = rref_f2_bool(arr)
    rref3 = rref_f2_int(arr.astype(np.uint8))
    assert np.array_equal(rref1, rref2)
    assert np.array_equal(rref1, rref3)


def test_rref_f2_4():
    arr = np.array(
        [
            [1, 1, 1, 1, 1, 1],
            [0, 0, 1, 0, 1, 0],
            [0, 0, 1, 1, 1, 1],
            [1, 1, 1, 0, 0, 1],
        ],
        dtype=np.bool_,
    )
    arr_gf2 = GF2(arr.astype(np.uint8))
    rref1 = arr_gf2.row_reduce()
    rref2 = rref_f2_bool(arr)
    rref3 = rref_f2_int(arr.astype(np.uint8))
    assert np.array_equal(rref1, rref2)
    assert np.array_equal(rref1, rref3)


def test_rref_f2_5():
    rng = np.random.default_rng()
    arr = rng.integers(0, 2, (1000, 1000)).astype(np.uint8)
    int_start = perf_counter()
    rref3 = rref_f2_int(arr)
    int_end = perf_counter()
    arr_gf2 = GF2(arr.astype(np.uint8))
    arr = arr.astype(np.bool_)
    t1 = perf_counter()
    rref1 = arr_gf2.row_reduce()
    t2 = perf_counter()
    rref2 = rref_f2_bool(arr)
    t3 = perf_counter()
    # print(
    #     "Time galois:",
    #     t2 - t1,
    #     "s. Time numba bool:",
    #     t3 - t2,
    #     "s. Time numba int:",
    #     int_end - int_start,
    #     "s.",
    # )
    assert np.array_equal(rref1, rref2)
    assert np.array_equal(rref2, rref3)
    assert np.array_equal(rref1, rref3)


def test_check_in_rowspace1():
    M = np.array(
        [[1, 1, 1, 1, 1, 1], [0, 0, 1, 0, 1, 0], [0, 0, 1, 1, 1, 1], [1, 1, 1, 0, 0, 1]]
    )
    v = np.array([1, 1, 0, 0, 1, 1])
    ans = np.array([0, 1, 0, 1])
    assert np.array_equal(check_in_rowspace_f2(M, v), ans)


def test_check_in_rowspace2():
    M = np.array(
        [[1, 1, 1, 1, 1, 1], [0, 0, 1, 0, 1, 0], [0, 0, 1, 1, 1, 1], [1, 1, 1, 0, 0, 1]]
    )
    v = np.array([1, 1, 1, 0, 1, 1])
    assert check_in_rowspace_f2(M, v) is None


def test_rref_f2_record():
    arr = np.array(
        [
            [1, 0, 0, 1, 0, 1],
            [0, 1, 1, 0, 1, 0],
            [0, 1, 0, 0, 0, 1],
            [1, 1, 1, 0, 1, 0],
        ]
    )
    rref, record = rref_f2_int_record(arr)
    print("rref:\n", rref)
    print("record:\n", record)
    for i in range(rref.shape[0]):
        res = np.zeros(rref.shape[1])
        for j in range(rref.shape[0]):
            if record[i, j] == 1:
                res += arr[j, :]
        assert np.array_equal(rref[i, :], res % 2)


def test_find_dependent_row_vectors():
    indep_arr = np.array(
        [
            [1, 0, 0, 1, 0, 1],
            [0, 1, 1, 0, 1, 0],
            [0, 1, 0, 0, 0, 1],
            [1, 1, 1, 0, 1, 0],
        ]
    )

    depend_arr = np.zeros((6, 6), dtype=np.uint8)
    depend_arr[0, :] = indep_arr[0, :]
    depend_arr[1, :] = indep_arr[1, :] + indep_arr[3, :]
    depend_arr[2, :] = indep_arr[1, :]
    depend_arr[3, :] = indep_arr[2, :]
    depend_arr[4, :] = indep_arr[0, :] + indep_arr[2, :]
    depend_arr[5, :] = indep_arr[3, :]

    dependent_vec_indices = find_dependent_row_vectors(depend_arr)
    assert len(dependent_vec_indices) == 2
    dependent_vecs = depend_arr[dependent_vec_indices, :]
    independent_vecs = np.delete(depend_arr, dependent_vec_indices, axis=0)
    for i in range(len(dependent_vec_indices)):
        assert check_in_rowspace_f2(independent_vecs, dependent_vecs[i, :]) is not None
