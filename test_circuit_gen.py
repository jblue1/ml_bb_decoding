from circuit_gen import *


def check_logical_operators(code: Code, print_X_unprimed_ops=False):
    X_ops = code.logical_X_ops
    Z_ops = code.logical_Z_ops

    k = X_ops.shape[0]
    if print_X_unprimed_ops:
        for i in range(k // 2):
            print(Polynomial([], code.l, code.m, X_ops[i, : code.l * code.m]))
    # check that the operators commute with the stabilizer
    X_ops_output = code.Hz @ X_ops.T
    Z_ops_output = code.Hx @ Z_ops.T
    assert not X_ops_output.any()
    assert not Z_ops_output.any()
    # check that the operators anticommute/commute in the correct ways
    for i in range(k):
        for j in range(k):
            dot_prod = np.inner(X_ops[i, :], Z_ops[j, :])
            if i == j:
                assert dot_prod == GF2(1)
            else:
                assert dot_prod == GF2(0)
    # check that the operators are not in the stabilizer
    Hx_rank = np.linalg.matrix_rank(code.Hx)
    Hz_rank = np.linalg.matrix_rank(code.Hz)
    for i in range(k):
        aug_X_matrix = np.vstack((code.Hx, X_ops[i, :]))
        aug_Z_matrix = np.vstack((code.Hz, Z_ops[i, :]))
        assert np.linalg.matrix_rank(aug_X_matrix) == Hx_rank + 1
        assert np.linalg.matrix_rank(aug_Z_matrix) == Hz_rank + 1


def test_72_12_6_logical_ops_given_f():
    a1 = Monomial(6, 6, 3, 0)
    a2 = Monomial(6, 6, 0, 1)
    a3 = Monomial(6, 6, 0, 2)
    A_poly = Polynomial([a1, a2, a3])

    b1 = Monomial(6, 6, 0, 3)
    b2 = Monomial(6, 6, 1, 0)
    b3 = Monomial(6, 6, 2, 0)
    B_poly = Polynomial([b1, b2, b3])

    f1 = Monomial(6, 6, 0, 3)
    f2 = Monomial(6, 6, 2, 0)
    f3 = Monomial(6, 6, 3, 3)
    f4 = Monomial(6, 6, 4, 0)
    f5 = Monomial(6, 6, 4, 3)
    f6 = Monomial(6, 6, 5, 3)
    f = Polynomial([f1, f2, f3, f4, f5, f6])

    code = Code(A_poly, B_poly, f)

    check_logical_operators(code, True)


def test_72_12_6_logical_ops():
    a1 = Monomial(6, 6, 3, 0)
    a2 = Monomial(6, 6, 0, 1)
    a3 = Monomial(6, 6, 0, 2)
    A_poly = Polynomial([a1, a2, a3])

    b1 = Monomial(6, 6, 0, 3)
    b2 = Monomial(6, 6, 1, 0)
    b3 = Monomial(6, 6, 2, 0)
    B_poly = Polynomial([b1, b2, b3])

    code = Code(A_poly, B_poly)
    check_logical_operators(code)


def test_144_12_12_logical_ops():
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
    code = Code(A_poly, B_poly)
    check_logical_operators(code)
