"""Point-to-plane information matrix calculations and finite-difference checks."""

from __future__ import annotations

import numpy as np

try:
    from .geometry import apply_transform, se3_exp, skew
except ImportError:
    from geometry import apply_transform, se3_exp, skew


def point_to_plane_jacobian(points_camera: np.ndarray, normals_camera: np.ndarray) -> np.ndarray:
    """Return one 1x6 point-to-plane Jacobian row per point.

    The pose perturbation is left-multiplied as ``T' = Exp(delta) T`` and
    ``delta = [translation, rotation]``. Under this convention

        J_i = n_i^T [I, -[p_i]_x].
    """

    points_camera = np.asarray(points_camera, dtype=float)
    normals_camera = np.asarray(normals_camera, dtype=float)
    rotation_part = np.cross(points_camera, normals_camera)
    return np.concatenate([normals_camera, rotation_part], axis=1)


def build_fim(
    jacobian: np.ndarray,
    weights: np.ndarray,
    noise_variance: float | np.ndarray = 1e-4,
) -> np.ndarray:
    """Accumulate weighted scalar point-to-plane information."""

    jacobian = np.asarray(jacobian, dtype=float)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    if np.isscalar(noise_variance):
        inverse_variance = np.full(len(weights), 1.0 / float(noise_variance))
    else:
        variance = np.asarray(noise_variance, dtype=float).reshape(-1)
        if len(variance) != len(weights):
            raise ValueError("noise_variance must have one value per observation")
        inverse_variance = 1.0 / variance
    information = weights * inverse_variance
    fim = jacobian.T @ (information[:, None] * jacobian)
    return 0.5 * (fim + fim.T)


def finite_difference_jacobian(
    points_object: np.ndarray,
    t_co: np.ndarray,
    normals_camera: np.ndarray,
    step: float = 1e-6,
) -> np.ndarray:
    """Numerically differentiate nominal point-to-plane residuals.

    The correspondence point is fixed at the nominal transformed point and
    the nominal normal is kept fixed, matching the point-to-plane residual
    used by the analytic Jacobian.
    """

    nominal_points = apply_transform(t_co, points_object)
    numerical = np.zeros((len(points_object), 6), dtype=float)
    for coordinate in range(6):
        delta = np.zeros(6, dtype=float)
        delta[coordinate] = step
        plus_points = apply_transform(se3_exp(delta) @ t_co, points_object)
        delta[coordinate] = -step
        minus_points = apply_transform(se3_exp(delta) @ t_co, points_object)
        plus_residual = np.sum(normals_camera * (plus_points - nominal_points), axis=1)
        minus_residual = np.sum(normals_camera * (minus_points - nominal_points), axis=1)
        numerical[:, coordinate] = (plus_residual - minus_residual) / (2.0 * step)
    return numerical


def relative_frobenius_error(analytic: np.ndarray, numeric: np.ndarray, epsilon: float = 1e-12) -> float:
    denominator = float(np.linalg.norm(numeric, ord="fro")) + epsilon
    return float(np.linalg.norm(analytic - numeric, ord="fro") / denominator)


def eigensystem(fim: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ascending eigenvalues and corresponding eigenvectors."""

    return np.linalg.eigh(0.5 * (fim + fim.T))


def skew_residual_check() -> float:
    """Check the vector identity used by the rotational Jacobian block."""

    point = np.array([0.3, -0.2, 1.1])
    normal = np.array([-0.4, 0.8, 0.2])
    left = normal @ (-skew(point))
    right = np.cross(point, normal)
    return float(np.max(np.abs(left - right)))
