import numpy as np
import pytest

from gym import QECGym


def test_rotated_surface_code_data_noise_get_decoding_instances():
    gym = QECGym("rotated_surface_code-3", "X", "data", 0.1)
    dem = gym.get_detector_error_model()
    num_detectors = dem.num_detectors
    num_observables = dem.num_observables
    assert num_detectors == 4
    assert num_observables == 1

    shots = 5
    syndromes, _, _ = gym.get_decoding_instances(shots)
    assert syndromes.shape == (shots, num_detectors)


def test_rotated_surface_code_data_noise_evaluate_predictions():
    """Since stim makes no gaurantees about runs producing the same results on different
    platforms, even with the same seed, have a test where the results of the shots are
    hardcoded"""
    gym = QECGym("rotated_surface_code-3", "Z", "data", 0.1)

    gym._det_data = np.array(
        [
            [True, True, True, True],
            [True, False, False, False],
            [False, True, True, True],
        ]
    )

    gym._measured_observables = np.array(
        [
            [False],
            [True],
            [True],
        ]
    )
    gym._actual_errors = np.array(
        [
            [True, False, False, True, False, False, False],
            [False, False, True, False, False, False, False],
            [False, False, False, False, True, True, True],
        ]
    )

    assert np.array_equal(gym._actual_errors @ gym._spacetime_H.T, gym._det_data)

    # first prediction is correct up to the stabilizer, second is not consistent with
    # the syndrome, and third results in logical error
    predicted_errors = np.array(
        [
            [False, True, False, True, False, True, True],
            [True, False, True, False, False, False, False],
            [True, False, True, False, True, False, True],
        ]
    )

    with pytest.raises(ValueError):
        prediction_errors, actual_errors = gym.evaluate_predictions(
            predicted_errors, True, False
        )

    prediction_errors, actual_errors = gym.evaluate_predictions(
        predicted_errors, True, True
    )
    assert np.array_equal(prediction_errors, np.array([[False], [True], [True]]))
