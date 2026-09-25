import numpy as np
import pytest
from benchmark_g1_cricket_model_grouping import check_arrays


def test_grouping_parity_checks_every_row_substep_channel_and_rejects_nonfinite():
    expected = (np.zeros((8, 4, 3)), np.ones((8, 4, 5)))
    check_arrays(expected, expected)
    for part in range(2):
        for row in range(8):
            for value in (0.25, np.nan, np.inf):
                actual = tuple(array.copy() for array in expected)
                actual[part][row, -1, -1] = value
                with pytest.raises(AssertionError):
                    check_arrays(actual, expected)
    with pytest.raises(AssertionError):
        check_arrays(tuple(array.astype(np.float32) for array in expected), expected)
