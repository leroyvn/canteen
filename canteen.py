from __future__ import annotations

import attrs
import numpy as np


@attrs.define
class Interval:
    """
    A data structure describing a union of intervals.
    """

    lbounds: np.ndarray = attrs.field(
        converter=np.atleast_1d, eq=attrs.cmp_using(eq=np.array_equal)
    )
    ubounds: np.ndarray = attrs.field(
        converter=np.atleast_1d, eq=attrs.cmp_using(eq=np.array_equal)
    )

    def __attrs_post_init__(self):
        if self.lbounds.shape != self.ubounds.shape:
            raise ValueError("The lbounds and ubounds arrays must have the same shape")

        if not np.all(self.lbounds < self.ubounds):
            raise ValueError(
                "Some elements of ubounds are lower or equal to the "
                "corresponding elements of lbounds"
            )

        if (self.lbounds.size > 1) and not np.all(np.diff(self.lbounds) > 0):
            raise ValueError("Intervals must be ordered in ascending order")

    @classmethod
    def from_segments(cls, segments):
        segments = np.asarray(segments)
        return cls(
            lbounds=np.squeeze(segments[:, 0]), ubounds=np.squeeze(segments[:, 1])
        )

    def contains(self, value: np.ndarray) -> np.ndarray:
        """
        Return an array containing the indexes of the intervals containing the searched
        value.
        """
        sel_lower = np.searchsorted(self.lbounds, value)
        sel_upper = np.searchsorted(self.ubounds, value) + 1
        hit = sel_lower == sel_upper  # Mask where values triggered an interval hit

        # Map values to selected interval (index -999 means no hit)
        interval_index = np.where(hit, sel_lower - 1, np.full_like(value, -999)).astype(
            "int"
        )

        return interval_index

    def connected_components(self) -> list[Interval]:
        components = np.concatenate(((False,), (self.lbounds[1:] == self.ubounds[:-1])))
        assert np.shape(components) == np.shape(self.lbounds)

        component_start = np.where(components == False)[0]
        component_end = np.concatenate((component_start[1:], (len(components),)))

        return [
            Interval(self.lbounds[start:end], self.ubounds[start:end])
            for start, end in zip(component_start, component_end)
        ]

    def connected_components_as_array(self) -> list[np.ndarray]:
        return [
            np.unique((component.lbounds, component.ubounds))
            for component in self.connected_components()
        ]

    def is_atomic(self):
        return len(self.connected_components()) == 1
