"""Oriented elbow geometry for the G1's single-axis elbow hinge."""

import numpy as np


def signed_elbow_angle(shoulder, elbow, wrist, axis):
    axis = np.asarray(axis, dtype=float)
    length = np.linalg.norm(axis)
    if not np.isfinite(length) or length <= 1e-8:
        raise ValueError("degenerate elbow axis")
    axis = axis / length
    upper, forearm = np.asarray(shoulder) - elbow, np.asarray(wrist) - elbow
    upper = upper - axis * np.dot(upper, axis)
    forearm = forearm - axis * np.dot(forearm, axis)
    lengths = np.linalg.norm(upper), np.linalg.norm(forearm)
    if not np.isfinite(lengths).all() or min(lengths) <= 1e-8:
        raise ValueError("degenerate projected elbow geometry")
    return float(
        np.arctan2(np.dot(axis, np.cross(upper, forearm)), np.dot(upper, forearm)) % (2 * np.pi)
    )
