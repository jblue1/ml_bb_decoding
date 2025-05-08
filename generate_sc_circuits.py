from dataclasses import dataclass
from typing import List, Dict

import stim


@dataclass
class coord:
    r: int
    c: int


def get_index(r, c, d):
    return r + (c // 2) * (2 * d + 1)


def syndrome_measurement_rounds(
    distance: int,
    rounds: int,
    data_qubits: List[int],
    x_ancillas: List[int],
    z_ancillas: List[int],
    index_to_coords: Dict[int, coord],
    max_qubit: int,
    is_qubit_array: List[bool],
    x_detectors: bool,
    z_detectors: bool,
    error_rate: float = 0.0,
    initialize_ancillas=True,
):
    circuit_block = stim.Circuit()
    x_order = [(1, 1), (-1, 1), (1, -1), (-1, -1)]
    z_order = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    data_qubits_set = set(data_qubits)
    if initialize_ancillas:
        circuit_block.append("R", z_ancillas)
        circuit_block.append("RX", x_ancillas)
        if error_rate != 0:
            circuit_block.append("DEPOLARIZE1", data_qubits, error_rate)
            circuit_block.append("X_ERROR", z_ancillas, error_rate)
            circuit_block.append("Z_ERROR", x_ancillas, error_rate)
        circuit_block.append("TICK")
    for i in range(4):
        cx_qubits = []
        data_qubits_this_step_set = set()
        for index in x_ancillas:
            ancilla_coords = index_to_coords[index]
            step = x_order[i]
            data_qubit = get_index(
                step[0] + ancilla_coords.r, step[1] + ancilla_coords.c, distance
            )
            if data_qubit <= max_qubit:
                if is_qubit_array[data_qubit]:
                    cx_qubits += [index, data_qubit]
                    data_qubits_this_step_set.add(data_qubit)
        for index in z_ancillas:
            ancilla_coords = index_to_coords[index]
            step = z_order[i]
            data_coords = coord(step[0] + ancilla_coords.r, step[1] + ancilla_coords.c)
            data_qubit = get_index(data_coords.r, data_coords.c, distance)
            if data_qubit <= max_qubit:
                if is_qubit_array[data_qubit]:
                    if data_coords == index_to_coords[data_qubit]:
                        cx_qubits += [data_qubit, index]
                        data_qubits_this_step_set.add(data_qubit)

        circuit_block.append("CX", cx_qubits)
        if error_rate != 0:
            circuit_block.append("DEPOLARIZE2", cx_qubits, error_rate)
            # converting to a list + sorting for ease of testing (the order of a set will typically not match numerical order)
            idle_qubits = list(data_qubits_set - data_qubits_this_step_set)
            idle_qubits.sort()
            circuit_block.append("DEPOLARIZE1", idle_qubits, error_rate)
        circuit_block.append("TICK")
    if error_rate != 0:
        circuit_block.append("DEPOLARIZE1", data_qubits, error_rate)
        circuit_block.append("X_ERROR", z_ancillas, error_rate)
        circuit_block.append("Z_ERROR", x_ancillas, error_rate)
    circuit_block.append("M", z_ancillas)
    circuit_block.append("MX", x_ancillas)
    circuit_block.append("TICK")
    if x_detectors or z_detectors:
        circuit_block.append_from_stim_program_text("SHIFT_COORDS(0, 0, 1)")
        offset = len(z_ancillas) + len(x_ancillas)
        if z_detectors:
            for i, q in enumerate(z_ancillas):
                coords = index_to_coords[q]
                circuit_block.append(
                    "DETECTOR",
                    [stim.target_rec(-offset + i), stim.target_rec(-2 * offset + i)],
                    (coords.r, coords.c, 0),
                )
        if x_detectors:
            for i, q in enumerate(x_ancillas):
                coords = index_to_coords[q]
                circuit_block.append(
                    "DETECTOR",
                    [
                        stim.target_rec(-len(x_ancillas) + i),
                        stim.target_rec(-len(x_ancillas) - offset + i),
                    ],
                    (coords.r, coords.c, 0),
                )

    return circuit_block * rounds


def generate_sc_circuit(
    distance: int,
    rounds: int,
    x_detectors: bool,
    z_detectors: bool,
    logical_operator: str,
    error_rate: float = 0.0,
) -> stim.Circuit:
    """Generate a syndrome measurement circuit for the rotated surface code.

    NOTE: The faulty rounds of syndrome measurment are followed by a round of perfect
    syndrome measurement, and then a measurement of the X or Z logical operators.

    Args:
        distance: The code distance. Must be an odd integer greater than or equal to three.
        rounds: The number of faulty syndrome measurement rounds to perform.
        x_detectors: Whether or not to include detectors for X checks
        z_detectors: Whether or not to include detectors for Z checks
        logical_opeerator: Which logical operator to measure
        error_rate: The physical error rate for a depolarizing noise model
    """
    circuit = stim.Circuit()
    (
        data_qubits,
        index_to_coord,
        x_ancillas,
        z_ancillas,
        x_op_qubits,
        z_op_qubits,
    ) = get_qubit_info(distance)
    all_qubits = data_qubits + x_ancillas + z_ancillas
    all_qubits.sort()
    max_qubit = all_qubits[-1]
    is_qubit_array = [False] * (max_qubit + 1)
    for q in all_qubits:
        is_qubit_array[q] = True
        coords = index_to_coord[q]
        circuit.append_from_stim_program_text(
            f"QUBIT_COORDS({coords.r}, {coords.c}) {q}"
        )
    if logical_operator == "Z":
        init_zero_qubits = data_qubits + z_ancillas
        init_plus_qubits = x_ancillas
    elif logical_operator == "X":
        init_zero_qubits = z_ancillas
        init_plus_qubits = data_qubits + x_ancillas

    circuit.append("R", init_zero_qubits)
    circuit.append("RX", init_plus_qubits)
    circuit.append("TICK")
    circuit += syndrome_measurement_rounds(
        distance,
        1,
        data_qubits,
        x_ancillas,
        z_ancillas,
        index_to_coord,
        max_qubit,
        is_qubit_array,
        False,
        False,
        initialize_ancillas=False,
    )

    if rounds == 1:
        circuit.append("DEPOLARIZE1", data_qubits, error_rate)
        error_rate = 0
    circuit += syndrome_measurement_rounds(
        distance,
        rounds,
        data_qubits,
        x_ancillas,
        z_ancillas,
        index_to_coord,
        max_qubit,
        is_qubit_array,
        x_detectors,
        z_detectors,
        error_rate=error_rate,
    )
    if rounds != 1:
        circuit += syndrome_measurement_rounds(
            distance,
            1,
            data_qubits,
            x_ancillas,
            z_ancillas,
            index_to_coord,
            max_qubit,
            is_qubit_array,
            x_detectors,
            z_detectors,
        )

    if logical_operator == "Z":
        circuit.append("M", z_op_qubits)
    elif logical_operator == "X":
        circuit.append("MX", x_op_qubits)
    circuit.append(
        "OBSERVABLE_INCLUDE", [stim.target_rec(-i - 1) for i in range(distance)], 0
    )
    return circuit


def get_qubit_info(distance):
    index_to_coord = dict()
    x_op = []
    z_op = []
    x_ancillas = []
    z_ancillas = []
    data_qubits = []
    for c in range(distance):
        for r in range(distance):
            index = get_index(2 * r + 1, 2 * c + 1, distance)
            data_qubits.append(index)
            index_to_coord[index] = coord(2 * r + 1, 2 * c + 1)
            if r == 0:
                z_op.append(index)
            if c == 0:
                x_op.append(index)
    for c in range(distance + 1):
        for r in range(distance + 1):
            top_boundary = r == 0
            left_boundary = c == 0
            right_boundary = c == distance
            bottom_boundary = r == distance
            if top_boundary and left_boundary:
                continue
            if top_boundary and c % 2 != 0:
                continue
            if right_boundary and r % 2 != 0:
                continue
            if left_boundary and (r - 1) % 2 != 0:
                continue
            if bottom_boundary and (c - 1) % 2 != 0:
                continue
            index = get_index(2 * r, 2 * c, distance)
            index_to_coord[index] = coord(2 * r, 2 * c)
            if r % 2 != c % 2:
                x_ancillas.append(index)
            else:
                z_ancillas.append(index)
    data_qubits.sort()
    return data_qubits, index_to_coord, x_ancillas, z_ancillas, z_op, x_op
