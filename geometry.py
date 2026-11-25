"""Parameterised rigid-body geometry and small SE(3) utilities."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, inf, pi, sin, sqrt

import numpy as np


def skew(vector: np.ndarray) -> np.ndarray:
    """Return the 3x3 cross-product matrix for a 3-vector."""

    x, y, z = np.asarray(vector, dtype=float)
    return np.array(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=float
    )


def so3_exp(rotation_vector: np.ndarray) -> np.ndarray:
    """Compute a rotation matrix with Rodrigues' formula."""

    rotation_vector = np.asarray(rotation_vector, dtype=float)
    theta = float(np.linalg.norm(rotation_vector))
    omega_hat = skew(rotation_vector)
    if theta < 1e-8:
        return np.eye(3) + omega_hat + 0.5 * omega_hat @ omega_hat

    a = sin(theta) / theta
    b = (1.0 - cos(theta)) / (theta * theta)
    return np.eye(3) + a * omega_hat + b * (omega_hat @ omega_hat)


def se3_exp(delta: np.ndarray) -> np.ndarray:
    """Compute Exp(delta) for delta=[translation, rotation]."""

    delta = np.asarray(delta, dtype=float)
    translation = delta[:3]
    rotation_vector = delta[3:]
    theta = float(np.linalg.norm(rotation_vector))
    omega_hat = skew(rotation_vector)

    if theta < 1e-8:
        b = 0.5
        c = 1.0 / 6.0
    else:
        b = (1.0 - cos(theta)) / (theta * theta)
        c = (1.0 - sin(theta) / theta) / (theta * theta)

    v_matrix = np.eye(3) + b * omega_hat + c * (omega_hat @ omega_hat)
    transform = np.eye(4)
    transform[:3, :3] = so3_exp(rotation_vector)
    transform[:3, 3] = v_matrix @ translation
    return transform


def apply_transform(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a 4x4 rigid transform to an Nx3 array."""

    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    points = np.asarray(points, dtype=float)
    return points @ rotation.T + translation


def apply_rotation(rotation: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """Apply a 3x3 rotation to an Nx3 array."""

    return np.asarray(vectors, dtype=float) @ np.asarray(rotation, dtype=float).T


@dataclass(frozen=True)
class SurfaceCloud:
    points: np.ndarray
    normals: np.ndarray
    face_ids: np.ndarray


@dataclass(frozen=True)
class Cuboid:
    size: tuple[float, float, float]

    def sample_surface(self, resolution: int = 12) -> SurfaceCloud:
        """Sample all six faces of a cuboid centred at the object origin."""

        sx, sy, sz = (float(value) for value in self.size)
        hx, hy, hz = sx / 2.0, sy / 2.0, sz / 2.0
        count = max(2, int(resolution))
        u = np.linspace(-1.0, 1.0, count)
        points: list[np.ndarray] = []
        normals: list[np.ndarray] = []
        face_ids: list[np.ndarray] = []

        def add_face(face_points: np.ndarray, normal: tuple[float, float, float], face_id: int) -> None:
            points.append(face_points.reshape(-1, 3))
            normals.append(np.tile(np.asarray(normal, dtype=float), (face_points.shape[0] * face_points.shape[1], 1)))
            face_ids.append(np.full(face_points.shape[0] * face_points.shape[1], face_id, dtype=int))

        grid_a, grid_b = np.meshgrid(u, u, indexing="ij")
        add_face(np.stack([np.full_like(grid_a, hx), grid_a * hy, grid_b * hz], axis=-1), (1, 0, 0), 0)
        add_face(np.stack([np.full_like(grid_a, -hx), grid_a * hy, grid_b * hz], axis=-1), (-1, 0, 0), 1)
        add_face(np.stack([grid_a * hx, np.full_like(grid_a, hy), grid_b * hz], axis=-1), (0, 1, 0), 2)
        add_face(np.stack([grid_a * hx, np.full_like(grid_a, -hy), grid_b * hz], axis=-1), (0, -1, 0), 3)
        add_face(np.stack([grid_a * hx, grid_b * hy, np.full_like(grid_a, hz)], axis=-1), (0, 0, 1), 4)
        add_face(np.stack([grid_a * hx, grid_b * hy, np.full_like(grid_a, -hz)], axis=-1), (0, 0, -1), 5)
        return SurfaceCloud(np.concatenate(points), np.concatenate(normals), np.concatenate(face_ids))

    def ray_intersection(self, origin: np.ndarray, direction: np.ndarray) -> float:
        """Return the first non-negative ray hit with the cuboid."""

        half_sizes = np.asarray(self.size, dtype=float) / 2.0
        t_min, t_max = -inf, inf
        for coordinate, ray_component, half_size in zip(origin, direction, half_sizes):
            if abs(ray_component) < 1e-12:
                if coordinate < -half_size or coordinate > half_size:
                    return inf
                continue
            first = (-half_size - coordinate) / ray_component
            second = (half_size - coordinate) / ray_component
            first, second = min(first, second), max(first, second)
            t_min, t_max = max(t_min, first), min(t_max, second)
            if t_min > t_max:
                return inf

        if t_max < 0.0:
            return inf
        return max(0.0, t_min)


@dataclass(frozen=True)
class Cylinder:
    radius: float
    height: float

    def sample_surface(self, resolution: int = 24) -> SurfaceCloud:
        """Sample the side and two caps of a cylinder around the object z-axis."""

        count = max(8, int(resolution))
        z_count = max(2, count // 2)
        theta = np.linspace(0.0, 2.0 * pi, count, endpoint=False)
        z_values = np.linspace(-self.height / 2.0, self.height / 2.0, z_count)
        theta_grid, z_grid = np.meshgrid(theta, z_values, indexing="ij")
        side_points = np.stack(
            [self.radius * np.cos(theta_grid), self.radius * np.sin(theta_grid), z_grid],
            axis=-1,
        ).reshape(-1, 3)
        side_normals = np.stack(
            [np.cos(theta_grid), np.sin(theta_grid), np.zeros_like(theta_grid)], axis=-1
        ).reshape(-1, 3)
        side_ids = np.zeros(len(side_points), dtype=int)

        radial_count = max(3, count // 4)
        radii = np.linspace(0.0, self.radius, radial_count)
        cap_theta, cap_radius = np.meshgrid(theta, radii, indexing="ij")
        cap_x = cap_radius * np.cos(cap_theta)
        cap_y = cap_radius * np.sin(cap_theta)
        top_points = np.stack([cap_x, cap_y, np.full_like(cap_x, self.height / 2.0)], axis=-1).reshape(-1, 3)
        bottom_points = np.stack([cap_x, cap_y, np.full_like(cap_x, -self.height / 2.0)], axis=-1).reshape(-1, 3)
        top_normals = np.tile(np.array([0.0, 0.0, 1.0]), (len(top_points), 1))
        bottom_normals = np.tile(np.array([0.0, 0.0, -1.0]), (len(bottom_points), 1))
        return SurfaceCloud(
            np.concatenate([side_points, top_points, bottom_points]),
            np.concatenate([side_normals, top_normals, bottom_normals]),
            np.concatenate([
                side_ids,
                np.ones(len(top_points), dtype=int),
                np.full(len(bottom_points), 2, dtype=int),
            ]),
        )

    def ray_intersection(self, origin: np.ndarray, direction: np.ndarray) -> float:
        """Return the first non-negative ray hit with a closed cylinder."""

        hits: list[float] = []
        ox, oy, oz = origin
        dx, dy, dz = direction
        a = dx * dx + dy * dy
        b = 2.0 * (ox * dx + oy * dy)
        c = ox * ox + oy * oy - self.radius * self.radius
        if a > 1e-12:
            discriminant = b * b - 4.0 * a * c
            if discriminant >= 0.0:
                root = sqrt(discriminant)
                for candidate in ((-b - root) / (2.0 * a), (-b + root) / (2.0 * a)):
                    if candidate >= 0.0:
                        z_hit = oz + candidate * dz
                        if -self.height / 2.0 - 1e-10 <= z_hit <= self.height / 2.0 + 1e-10:
                            hits.append(candidate)

        if abs(dz) > 1e-12:
            for cap_z in (-self.height / 2.0, self.height / 2.0):
                candidate = (cap_z - oz) / dz
                if candidate >= 0.0:
                    x_hit = ox + candidate * dx
                    y_hit = oy + candidate * dy
                    if x_hit * x_hit + y_hit * y_hit <= self.radius * self.radius + 1e-10:
                        hits.append(candidate)

        return min(hits) if hits else inf


def shape_from_name(
    name: str,
    cuboid_size: tuple[float, float, float] | None = None,
    cylinder_radius: float | None = None,
    cylinder_height: float | None = None,
) -> Cuboid | Cylinder:
    """Return a parameterised shape, using the experiment defaults when omitted."""

    if name == "cuboid":
        return Cuboid(cuboid_size or (0.42, 0.32, 0.52))
    if name == "cylinder":
        return Cylinder(
            radius=0.20 if cylinder_radius is None else cylinder_radius,
            height=0.52 if cylinder_height is None else cylinder_height,
        )
    raise ValueError(f"Unsupported shape {name!r}")
