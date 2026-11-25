"""Camera projection, analytic visibility, and optional external occlusion."""

from __future__ import annotations

from dataclasses import dataclass
from math import inf

import numpy as np

try:
    from .geometry import Cuboid, Cylinder
except ImportError:
    from geometry import Cuboid, Cylinder


Shape = Cuboid | Cylinder


@dataclass(frozen=True)
class CameraIntrinsics:
    width: int = 640
    height: int = 480
    fx: float = 520.0
    fy: float = 520.0
    cx: float = 320.0
    cy: float = 240.0


@dataclass(frozen=True)
class BoxOccluder:
    """Axis-aligned box in the camera coordinate frame."""

    center: tuple[float, float, float]
    size: tuple[float, float, float]

    def ray_intersection(self, origin: np.ndarray, direction: np.ndarray) -> float:
        half_sizes = np.asarray(self.size, dtype=float) / 2.0
        box_min = np.asarray(self.center, dtype=float) - half_sizes
        box_max = np.asarray(self.center, dtype=float) + half_sizes
        t_min, t_max = -inf, inf
        for coordinate, ray_component, lower, upper in zip(origin, direction, box_min, box_max):
            if abs(ray_component) < 1e-12:
                if coordinate < lower or coordinate > upper:
                    return inf
                continue
            first = (lower - coordinate) / ray_component
            second = (upper - coordinate) / ray_component
            first, second = min(first, second), max(first, second)
            t_min, t_max = max(t_min, first), min(t_max, second)
            if t_min > t_max:
                return inf
        if t_max < 0.0:
            return inf
        return max(0.0, t_min)


@dataclass(frozen=True)
class VisibilityResult:
    points_camera: np.ndarray
    normals_camera: np.ndarray
    pixels: np.ndarray
    depths: np.ndarray
    weights: np.ndarray
    in_front_facing: np.ndarray
    in_field_of_view: np.ndarray
    self_visible: np.ndarray
    external_visible: np.ndarray


def project_points(points_camera: np.ndarray, camera: CameraIntrinsics) -> tuple[np.ndarray, np.ndarray]:
    """Project camera-frame points and return pixels plus positive-depth mask."""

    points_camera = np.asarray(points_camera, dtype=float)
    depths = points_camera[:, 2]
    pixels = np.full((len(points_camera), 2), np.nan, dtype=float)
    positive = depths > 1e-9
    pixels[positive, 0] = camera.fx * points_camera[positive, 0] / depths[positive] + camera.cx
    pixels[positive, 1] = camera.fy * points_camera[positive, 1] / depths[positive] + camera.cy
    return pixels, depths


def _fov_mask(pixels: np.ndarray, depths: np.ndarray, camera: CameraIntrinsics) -> np.ndarray:
    return (
        (depths > 1e-9)
        & (pixels[:, 0] >= 0.0)
        & (pixels[:, 0] < camera.width)
        & (pixels[:, 1] >= 0.0)
        & (pixels[:, 1] < camera.height)
    )


def compute_visibility(
    shape: Shape,
    points_object: np.ndarray,
    normals_object: np.ndarray,
    t_co: np.ndarray,
    camera: CameraIntrinsics,
    occluder: BoxOccluder | None = None,
    front_facing_epsilon: float = 1e-8,
) -> VisibilityResult:
    """Compute binary visibility from a camera at the origin.

    Self-occlusion is evaluated by intersecting each camera ray with the
    parameterised shape. The external occluder, when supplied, is an
    axis-aligned box in camera coordinates.
    """

    rotation = t_co[:3, :3]
    translation = t_co[:3, 3]
    points_object = np.asarray(points_object, dtype=float)
    normals_object = np.asarray(normals_object, dtype=float)
    points_camera = points_object @ rotation.T + translation
    normals_camera = normals_object @ rotation.T
    pixels, depths = project_points(points_camera, camera)
    fov = _fov_mask(pixels, depths, camera)

    view_to_camera = -points_camera
    front_facing = np.sum(normals_camera * view_to_camera, axis=1) > front_facing_epsilon

    inverse_rotation = rotation.T
    origin_object = -inverse_rotation @ translation
    self_visible = np.zeros(len(points_object), dtype=bool)
    external_visible = np.ones(len(points_object), dtype=bool)

    for index, point_camera in enumerate(points_camera):
        if not (fov[index] and front_facing[index]):
            continue
        distance = float(np.linalg.norm(point_camera))
        direction_camera = point_camera / distance
        direction_object = inverse_rotation @ direction_camera
        hit = shape.ray_intersection(origin_object, direction_object)
        tolerance = max(2e-5, 2e-4 * distance)
        self_visible[index] = abs(hit - distance) <= tolerance

        if occluder is not None:
            occluder_hit = occluder.ray_intersection(np.zeros(3), direction_camera)
            external_visible[index] = not (occluder_hit < distance - tolerance)

    visible = fov & front_facing & self_visible & external_visible
    return VisibilityResult(
        points_camera=points_camera,
        normals_camera=normals_camera,
        pixels=pixels,
        depths=depths,
        weights=visible.astype(float),
        in_front_facing=front_facing,
        in_field_of_view=fov,
        self_visible=self_visible,
        external_visible=external_visible,
    )
