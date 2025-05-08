import stim

from generate_sc_circuits import generate_sc_circuit


def test_distance_3_data_noise_X_checks():
    """Compare with hard coded circuit. Measuring X checks, logical X operator"""
    correct_circuit = stim.Circuit(
        """
        QUBIT_COORDS(1, 1) 1
        QUBIT_COORDS(2, 0) 2
        QUBIT_COORDS(3, 1) 3
        QUBIT_COORDS(5, 1) 5
        QUBIT_COORDS(1, 3) 8
        QUBIT_COORDS(2, 2) 9
        QUBIT_COORDS(3, 3) 10
        QUBIT_COORDS(4, 2) 11
        QUBIT_COORDS(5, 3) 12
        QUBIT_COORDS(6, 2) 13
        QUBIT_COORDS(0, 4) 14
        QUBIT_COORDS(1, 5) 15
        QUBIT_COORDS(2, 4) 16
        QUBIT_COORDS(3, 5) 17
        QUBIT_COORDS(4, 4) 18
        QUBIT_COORDS(5, 5) 19
        QUBIT_COORDS(4, 6) 25
        R 9 13 14 18
        RX 1 3 5 8 10 12 15 17 19 2 11 16 25
        TICK
        CX 2 3 11 12 16 17 10 9 15 14 19 18
        TICK
        CX 2 1 11 10 16 15 3 9 8 14 12 18
        TICK
        CX 11 5 16 10 25 19 8 9 12 13 17 18
        TICK
        CX 11 3 16 8 25 17 1 9 5 13 10 18
        TICK
        M 9 13 14 18
        MX 2 11 16 25
        TICK
        DEPOLARIZE1(0.1) 1 3 5 8 10 12 15 17 19
        R 9 13 14 18
        RX 2 11 16 25
        TICK
        CX 2 3 11 12 16 17 10 9 15 14 19 18
        TICK
        CX 2 1 11 10 16 15 3 9 8 14 12 18
        TICK
        CX 11 5 16 10 25 19 8 9 12 13 17 18
        TICK
        CX 11 3 16 8 25 17 1 9 5 13 10 18
        TICK
        M 9 13 14 18
        MX 2 11 16 25
        TICK
        SHIFT_COORDS(0, 0, 1)
        DETECTOR(2, 0, 0) rec[-4] rec[-12]
        DETECTOR(4, 2, 0) rec[-3] rec[-11]
        DETECTOR(2, 4, 0) rec[-2] rec[-10]
        DETECTOR(4, 6, 0) rec[-1] rec[-9]
        MX 1 8 15
        OBSERVABLE_INCLUDE(0) rec[-1] rec[-2] rec[-3]
        """
    )
    circuit = generate_sc_circuit(3, 1, True, False, "X", 0.1)
    assert circuit == correct_circuit


def test_distance_3_data_noise_Z_checks():
    """Compare with hard coded circuit. Measuring Z checks, logical Z operator"""
    correct_circuit = stim.Circuit(
        """
        QUBIT_COORDS(1, 1) 1
        QUBIT_COORDS(2, 0) 2
        QUBIT_COORDS(3, 1) 3
        QUBIT_COORDS(5, 1) 5
        QUBIT_COORDS(1, 3) 8
        QUBIT_COORDS(2, 2) 9
        QUBIT_COORDS(3, 3) 10
        QUBIT_COORDS(4, 2) 11
        QUBIT_COORDS(5, 3) 12
        QUBIT_COORDS(6, 2) 13
        QUBIT_COORDS(0, 4) 14
        QUBIT_COORDS(1, 5) 15
        QUBIT_COORDS(2, 4) 16
        QUBIT_COORDS(3, 5) 17
        QUBIT_COORDS(4, 4) 18
        QUBIT_COORDS(5, 5) 19
        QUBIT_COORDS(4, 6) 25
        R 1 3 5 8 10 12 15 17 19 9 13 14 18
        RX 2 11 16 25
        TICK
        CX 2 3 11 12 16 17 10 9 15 14 19 18
        TICK
        CX 2 1 11 10 16 15 3 9 8 14 12 18
        TICK
        CX 11 5 16 10 25 19 8 9 12 13 17 18
        TICK
        CX 11 3 16 8 25 17 1 9 5 13 10 18
        TICK
        M 9 13 14 18
        MX 2 11 16 25
        TICK
        DEPOLARIZE1(0.1) 1 3 5 8 10 12 15 17 19
        R 9 13 14 18
        RX 2 11 16 25
        TICK
        CX 2 3 11 12 16 17 10 9 15 14 19 18
        TICK
        CX 2 1 11 10 16 15 3 9 8 14 12 18
        TICK
        CX 11 5 16 10 25 19 8 9 12 13 17 18
        TICK
        CX 11 3 16 8 25 17 1 9 5 13 10 18
        TICK
        M 9 13 14 18
        MX 2 11 16 25
        TICK
        SHIFT_COORDS(0, 0, 1)
        DETECTOR(2, 2, 0) rec[-8] rec[-16]
        DETECTOR(6, 2, 0) rec[-7] rec[-15]
        DETECTOR(0, 4, 0) rec[-6] rec[-14]
        DETECTOR(4, 4, 0) rec[-5] rec[-13]
        M 1 3 5
        OBSERVABLE_INCLUDE(0) rec[-1] rec[-2] rec[-3]
        """
    )
    circuit = generate_sc_circuit(3, 1, False, True, "Z", 0.1)
    assert circuit == correct_circuit


def test_distance_3_circuit_noise_all_checks():
    correct_circuit = stim.Circuit(
        """
        QUBIT_COORDS(1, 1) 1
        QUBIT_COORDS(2, 0) 2
        QUBIT_COORDS(3, 1) 3
        QUBIT_COORDS(5, 1) 5
        QUBIT_COORDS(1, 3) 8
        QUBIT_COORDS(2, 2) 9
        QUBIT_COORDS(3, 3) 10
        QUBIT_COORDS(4, 2) 11
        QUBIT_COORDS(5, 3) 12
        QUBIT_COORDS(6, 2) 13
        QUBIT_COORDS(0, 4) 14
        QUBIT_COORDS(1, 5) 15
        QUBIT_COORDS(2, 4) 16
        QUBIT_COORDS(3, 5) 17
        QUBIT_COORDS(4, 4) 18
        QUBIT_COORDS(5, 5) 19
        QUBIT_COORDS(4, 6) 25
        R 1 3 5 8 10 12 15 17 19 9 13 14 18
        RX 2 11 16 25
        TICK
        CX 2 3 11 12 16 17 10 9 15 14 19 18
        TICK
        CX 2 1 11 10 16 15 3 9 8 14 12 18
        TICK
        CX 11 5 16 10 25 19 8 9 12 13 17 18
        TICK
        CX 11 3 16 8 25 17 1 9 5 13 10 18
        TICK
        M 9 13 14 18
        MX 2 11 16 25
        TICK
        REPEAT 3 {
            R 9 13 14 18
            RX 2 11 16 25
            DEPOLARIZE1(0.1) 1 3 5 8 10 12 15 17 19
            X_ERROR(0.1) 9 13 14 18
            Z_ERROR(0.1) 2 11 16 25
            TICK
            CX 2 3 11 12 16 17 10 9 15 14 19 18
            DEPOLARIZE2(0.1) 2 3 11 12 16 17 10 9 15 14 19 18
            DEPOLARIZE1(0.1) 1 5 8
            TICK
            CX 2 1 11 10 16 15 3 9 8 14 12 18
            DEPOLARIZE2(0.1) 2 1 11 10 16 15 3 9 8 14 12 18
            DEPOLARIZE1(0.1) 5 17 19
            TICK
            CX 11 5 16 10 25 19 8 9 12 13 17 18
            DEPOLARIZE2(0.1) 11 5 16 10 25 19 8 9 12 13 17 18
            DEPOLARIZE1(0.1) 1 3 15
            TICK
            CX 11 3 16 8 25 17 1 9 5 13 10 18
            DEPOLARIZE2(0.1) 11 3 16 8 25 17 1 9 5 13 10 18
            DEPOLARIZE1(0.1) 12 15 19
            TICK
            DEPOLARIZE1(0.1) 1 3 5 8 10 12 15 17 19
            X_ERROR(0.1) 9 13 14 18
            Z_ERROR(0.1) 2 11 16 25
            M 9 13 14 18
            MX 2 11 16 25
            TICK
            SHIFT_COORDS(0, 0, 1)
            DETECTOR(2, 2, 0) rec[-8] rec[-16]
            DETECTOR(6, 2, 0) rec[-7] rec[-15]
            DETECTOR(0, 4, 0) rec[-6] rec[-14]
            DETECTOR(4, 4, 0) rec[-5] rec[-13]
            DETECTOR(2, 0, 0) rec[-4] rec[-12]
            DETECTOR(4, 2, 0) rec[-3] rec[-11]
            DETECTOR(2, 4, 0) rec[-2] rec[-10]
            DETECTOR(4, 6, 0) rec[-1] rec[-9]
        }
        R 9 13 14 18
        RX 2 11 16 25
        TICK
        CX 2 3 11 12 16 17 10 9 15 14 19 18
        TICK
        CX 2 1 11 10 16 15 3 9 8 14 12 18
        TICK
        CX 11 5 16 10 25 19 8 9 12 13 17 18
        TICK
        CX 11 3 16 8 25 17 1 9 5 13 10 18
        TICK
        M 9 13 14 18
        MX 2 11 16 25
        TICK
        SHIFT_COORDS(0, 0, 1)
        DETECTOR(2, 2, 0) rec[-8] rec[-16]
        DETECTOR(6, 2, 0) rec[-7] rec[-15]
        DETECTOR(0, 4, 0) rec[-6] rec[-14]
        DETECTOR(4, 4, 0) rec[-5] rec[-13]
        DETECTOR(2, 0, 0) rec[-4] rec[-12]
        DETECTOR(4, 2, 0) rec[-3] rec[-11]
        DETECTOR(2, 4, 0) rec[-2] rec[-10]
        DETECTOR(4, 6, 0) rec[-1] rec[-9]
        M 1 3 5
        OBSERVABLE_INCLUDE(0) rec[-1] rec[-2] rec[-3]
        """
    )
    circuit = generate_sc_circuit(3, 3, True, True, "Z", 0.1)
    assert circuit == correct_circuit


def test_distance_5_circuit_noise():
    correct_circuit = stim.Circuit(
        """
        QUBIT_COORDS(1, 1) 1
        QUBIT_COORDS(2, 0) 2
        QUBIT_COORDS(3, 1) 3
        QUBIT_COORDS(5, 1) 5
        QUBIT_COORDS(6, 0) 6
        QUBIT_COORDS(7, 1) 7
        QUBIT_COORDS(9, 1) 9
        QUBIT_COORDS(1, 3) 12
        QUBIT_COORDS(2, 2) 13
        QUBIT_COORDS(3, 3) 14
        QUBIT_COORDS(4, 2) 15
        QUBIT_COORDS(5, 3) 16
        QUBIT_COORDS(6, 2) 17
        QUBIT_COORDS(7, 3) 18
        QUBIT_COORDS(8, 2) 19
        QUBIT_COORDS(9, 3) 20
        QUBIT_COORDS(10, 2) 21
        QUBIT_COORDS(0, 4) 22
        QUBIT_COORDS(1, 5) 23
        QUBIT_COORDS(2, 4) 24
        QUBIT_COORDS(3, 5) 25
        QUBIT_COORDS(4, 4) 26
        QUBIT_COORDS(5, 5) 27
        QUBIT_COORDS(6, 4) 28
        QUBIT_COORDS(7, 5) 29
        QUBIT_COORDS(8, 4) 30
        QUBIT_COORDS(9, 5) 31
        QUBIT_COORDS(1, 7) 34
        QUBIT_COORDS(2, 6) 35
        QUBIT_COORDS(3, 7) 36
        QUBIT_COORDS(4, 6) 37
        QUBIT_COORDS(5, 7) 38
        QUBIT_COORDS(6, 6) 39
        QUBIT_COORDS(7, 7) 40
        QUBIT_COORDS(8, 6) 41
        QUBIT_COORDS(9, 7) 42
        QUBIT_COORDS(10, 6) 43
        QUBIT_COORDS(0, 8) 44
        QUBIT_COORDS(1, 9) 45
        QUBIT_COORDS(2, 8) 46
        QUBIT_COORDS(3, 9) 47
        QUBIT_COORDS(4, 8) 48
        QUBIT_COORDS(5, 9) 49
        QUBIT_COORDS(6, 8) 50
        QUBIT_COORDS(7, 9) 51
        QUBIT_COORDS(8, 8) 52
        QUBIT_COORDS(9, 9) 53
        QUBIT_COORDS(4, 10) 59
        QUBIT_COORDS(8, 10) 63
        R 1 3 5 7 9 12 14 16 18 20 23 25 27 29 31 34 36 38 40 42 45 47 49 51 53 13 17 21 22 26 30 35 39 43 44 48 52
        RX 2 6 15 19 24 28 37 41 46 50 59 63
        TICK
        CX 2 3 6 7 15 16 19 20 24 25 28 29 37 38 41 42 46 47 50 51 14 13 18 17 23 22 27 26 31 30 36 35 40 39 45 44 49 48 53 52
        TICK
        CX 2 1 6 5 15 14 19 18 24 23 28 27 37 36 41 40 46 45 50 49 3 13 7 17 12 22 16 26 20 30 25 35 29 39 34 44 38 48 42 52
        TICK
        CX 15 5 19 9 24 14 28 18 37 27 41 31 46 36 50 40 59 49 63 53 12 13 16 17 20 21 25 26 29 30 34 35 38 39 42 43 47 48 51 52
        TICK
        CX 15 3 19 7 24 12 28 16 37 25 41 29 46 34 50 38 59 47 63 51 1 13 5 17 9 21 14 26 18 30 23 35 27 39 31 43 36 48 40 52
        TICK
        M 13 17 21 22 26 30 35 39 43 44 48 52
        MX 2 6 15 19 24 28 37 41 46 50 59 63
        TICK
        DEPOLARIZE1(0.1) 1 3 5 7 9 12 14 16 18 20 23 25 27 29 31 34 36 38 40 42 45 47 49 51 53
        R 13 17 21 22 26 30 35 39 43 44 48 52
        RX 2 6 15 19 24 28 37 41 46 50 59 63
        TICK
        CX 2 3 6 7 15 16 19 20 24 25 28 29 37 38 41 42 46 47 50 51 14 13 18 17 23 22 27 26 31 30 36 35 40 39 45 44 49 48 53 52
        TICK
        CX 2 1 6 5 15 14 19 18 24 23 28 27 37 36 41 40 46 45 50 49 3 13 7 17 12 22 16 26 20 30 25 35 29 39 34 44 38 48 42 52
        TICK
        CX 15 5 19 9 24 14 28 18 37 27 41 31 46 36 50 40 59 49 63 53 12 13 16 17 20 21 25 26 29 30 34 35 38 39 42 43 47 48 51 52
        TICK
        CX 15 3 19 7 24 12 28 16 37 25 41 29 46 34 50 38 59 47 63 51 1 13 5 17 9 21 14 26 18 30 23 35 27 39 31 43 36 48 40 52
        TICK
        M 13 17 21 22 26 30 35 39 43 44 48 52
        MX 2 6 15 19 24 28 37 41 46 50 59 63
        TICK
        SHIFT_COORDS(0, 0, 1)
        DETECTOR(2, 2, 0) rec[-24] rec[-48]
        DETECTOR(6, 2, 0) rec[-23] rec[-47]
        DETECTOR(10, 2, 0) rec[-22] rec[-46]
        DETECTOR(0, 4, 0) rec[-21] rec[-45]
        DETECTOR(4, 4, 0) rec[-20] rec[-44]
        DETECTOR(8, 4, 0) rec[-19] rec[-43]
        DETECTOR(2, 6, 0) rec[-18] rec[-42]
        DETECTOR(6, 6, 0) rec[-17] rec[-41]
        DETECTOR(10, 6, 0) rec[-16] rec[-40]
        DETECTOR(0, 8, 0) rec[-15] rec[-39]
        DETECTOR(4, 8, 0) rec[-14] rec[-38]
        DETECTOR(8, 8, 0) rec[-13] rec[-37]
        DETECTOR(2, 0, 0) rec[-12] rec[-36]
        DETECTOR(6, 0, 0) rec[-11] rec[-35]
        DETECTOR(4, 2, 0) rec[-10] rec[-34]
        DETECTOR(8, 2, 0) rec[-9] rec[-33]
        DETECTOR(2, 4, 0) rec[-8] rec[-32]
        DETECTOR(6, 4, 0) rec[-7] rec[-31]
        DETECTOR(4, 6, 0) rec[-6] rec[-30]
        DETECTOR(8, 6, 0) rec[-5] rec[-29]
        DETECTOR(2, 8, 0) rec[-4] rec[-28]
        DETECTOR(6, 8, 0) rec[-3] rec[-27]
        DETECTOR(4, 10, 0) rec[-2] rec[-26]
        DETECTOR(8, 10, 0) rec[-1] rec[-25]
        M 1 3 5 7 9
        OBSERVABLE_INCLUDE(0) rec[-1] rec[-2] rec[-3] rec[-4] rec[-5]
        """
    )
    circuit = generate_sc_circuit(5, 1, True, True, "Z", 0.1)
    assert circuit == correct_circuit
