import numpy as np
import canteen
import pytest


@pytest.mark.parametrize(
    "lbounds, ubounds, expected",
    [
        (0, 1, None),
        (1, 0, ValueError),
        ([0, 1], 1, ValueError),
        ([0, 1], [1, 2], None),
        ([1, 0], [1, 2], ValueError),
        ([1, 2], [0, 1], ValueError),
    ],
)
def test_construct(lbounds, ubounds, expected):
    if isinstance(expected, type) and issubclass(expected, Exception):
        with pytest.raises(expected):
            canteen.Interval(lbounds, ubounds)
    else:
        canteen.Interval(lbounds, ubounds)


def test_from_segments():
    interval = canteen.Interval.from_segments([[0, 1], [2, 3]])
    assert interval == canteen.Interval([0, 2], [1, 3])


@pytest.mark.parametrize(
    "lbounds, ubounds, value, expected",
    [
        (0, 1, 0.5, 0),
        ([0, 2], [1, 3], [0.5, 1.5, 2.5], [0, -999, 1]),
        (0, 1, 0, -999),
    ],
)
def test_contains(lbounds, ubounds, value, expected):
    interval = canteen.Interval(lbounds, ubounds)
    assert False, "WIP, add more test cases"
    assert np.all(interval.contains(value) == expected)


def test_connected_components():
    interval = canteen.Interval.from_segments([[0, 1], [1, 2], [3, 4]])
    assert interval.connected_components() == [
        canteen.Interval.from_segments([[0, 1], [1, 2]]),
        canteen.Interval(3, 4),
    ]


def test_connected_components_as_array():
    interval = canteen.Interval.from_segments([[0, 1], [1, 2], [3, 4]])

    for lhs, rhs in zip(
        interval.connected_components_as_array(),
        [np.array([0, 1, 2]), np.array([3, 4])],
    ):
        np.testing.assert_array_equal(lhs, rhs)


@pytest.mark.parametrize(
    "segments, expected",
    [
        ([[0, 1], [1, 2]], True),
        ([[0, 1], [3, 4]], False),
    ],
)
def test_is_atomic(segments, expected):
    interval = canteen.Interval.from_segments(segments)
    assert interval.is_atomic() == expected
