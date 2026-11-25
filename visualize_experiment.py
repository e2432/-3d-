"""Render camera-view and 3D visualizations for the parameterised experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

try:
    from .fim import build_fim, eigensystem, point_to_plane_jacobian
    from .geometry import shape_from_name
    from .run_experiment import make_object_pose, make_occluder
    from .visibility import CameraIntrinsics, compute_visibility, project_points
except ImportError:
    from fim import build_fim, eigensystem, point_to_plane_jacobian
    from geometry import shape_from_name
    from run_experiment import make_object_pose, make_occluder
    from visibility import CameraIntrinsics, compute_visibility, project_points


def convex_hull(points: np.ndarray) -> np.ndarray:
    """Return the 2D monotonic-chain convex hull in counter-clockwise order."""

    points = np.unique(np.asarray(points, dtype=float), axis=0)
    if len(points) <= 2:
        return points

    ordered = sorted((float(point[0]), float(point[1])) for point in points)

    def cross(origin, first, second):
        return (first[0] - origin[0]) * (second[1] - origin[1]) - (first[1] - origin[1]) * (second[0] - origin[0])

    lower = []
    for point in ordered:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)
    upper = []
    for point in reversed(ordered):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1], dtype=float)


def box_corners(occluder) -> np.ndarray:
    center = np.asarray(occluder.center, dtype=float)
    half_size = np.asarray(occluder.size, dtype=float) / 2.0
    return np.asarray(
        [
            center + [sx * half_size[0], sy * half_size[1], sz * half_size[2]]
            for sx in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
            for sz in (-1.0, 1.0)
        ],
        dtype=float,
    )


def set_axes_equal_3d(axis, points: np.ndarray) -> None:
    minimum = np.min(points, axis=0)
    maximum = np.max(points, axis=0)
    centre = (minimum + maximum) / 2.0
    radius = max(float(np.max(maximum - minimum)) / 2.0, 1e-3)
    axis.set_xlim(centre[0] - radius, centre[0] + radius)
    axis.set_ylim(centre[1] - radius, centre[1] + radius)
    axis.set_zlim(centre[2] - radius, centre[2] + radius)


def render(
    shape_name: str,
    resolution: int,
    output_dir: Path,
    cuboid_size: tuple[float, float, float],
    cylinder_radius: float,
    cylinder_height: float,
    disable_occluder: bool,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Polygon

    shape = shape_from_name(
        shape_name,
        cuboid_size=cuboid_size,
        cylinder_radius=cylinder_radius,
        cylinder_height=cylinder_height,
    )
    surface = shape.sample_surface(resolution)
    t_co = make_object_pose()
    camera = CameraIntrinsics()
    occluder = None if disable_occluder else make_occluder()
    full = compute_visibility(shape, surface.points, surface.normals, t_co, camera, occluder=None)
    after_occlusion = compute_visibility(shape, surface.points, surface.normals, t_co, camera, occluder=occluder)

    output_dir.mkdir(parents=True, exist_ok=True)
    full_visible = full.weights > 0.5
    occluded_visible = after_occlusion.weights > 0.5
    blocked_by_external = full_visible & ~occluded_visible
    in_view = full.in_field_of_view

    figure, axis = plt.subplots(figsize=(10, 7), dpi=160)
    axis.scatter(
        full.pixels[in_view, 0],
        full.pixels[in_view, 1],
        s=8,
        color="#cbd5e1",
        alpha=0.75,
        label="in camera field of view",
    )
    axis.scatter(
        full.pixels[occluded_visible, 0],
        full.pixels[occluded_visible, 1],
        s=18,
        color="#2563eb",
        label="visible after occlusion",
    )
    axis.scatter(
        full.pixels[blocked_by_external, 0],
        full.pixels[blocked_by_external, 1],
        s=24,
        color="#dc2626",
        label="blocked by external occluder",
    )

    if occluder is not None:
        corners = box_corners(occluder)
        corner_pixels, corner_depths = project_points(corners, camera)
        valid_corners = corner_pixels[corner_depths > 0.0]
        if len(valid_corners) >= 3:
            hull = convex_hull(valid_corners)
            axis.add_patch(Polygon(hull, closed=True, facecolor="#fca5a5", edgecolor="#b91c1c", alpha=0.22))

    axis.set_xlim(0.0, camera.width)
    axis.set_ylim(camera.height, 0.0)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("image u / pixel")
    axis.set_ylabel("image v / pixel")
    axis.set_title(
        f"Camera view | {shape_name} | after occlusion {int(occluded_visible.sum())} points"
    )
    axis.legend(loc="upper right")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_dir / "camera_view.png", bbox_inches="tight")
    plt.close(figure)

    figure = plt.figure(figsize=(10, 8), dpi=160)
    axis = figure.add_subplot(111, projection="3d")
    axis.scatter(
        full.points_camera[:, 0],
        full.points_camera[:, 1],
        full.points_camera[:, 2],
        s=5,
        color="#cbd5e1",
        alpha=0.28,
    )
    axis.scatter(
        full.points_camera[occluded_visible, 0],
        full.points_camera[occluded_visible, 1],
        full.points_camera[occluded_visible, 2],
        s=15,
        color="#2563eb",
        label="visible after occlusion",
    )
    axis.scatter(
        full.points_camera[blocked_by_external, 0],
        full.points_camera[blocked_by_external, 1],
        full.points_camera[blocked_by_external, 2],
        s=18,
        color="#dc2626",
        label="blocked by external occluder",
    )
    scene_points = [full.points_camera]
    if occluder is not None:
        corners = box_corners(occluder)
        scene_points.append(corners)
        edges = (
            (0, 1),
            (0, 2),
            (0, 4),
            (1, 3),
            (1, 5),
            (2, 3),
            (2, 6),
            (3, 7),
            (4, 5),
            (4, 6),
            (5, 7),
            (6, 7),
        )
        for first, second in edges:
            axis.plot(
                corners[[first, second], 0],
                corners[[first, second], 1],
                corners[[first, second], 2],
                color="#b91c1c",
                linewidth=1.2,
            )
    axis.scatter([0.0], [0.0], [0.0], color="#111827", s=35, marker="^", label="camera")
    scene_points.append(np.zeros((1, 3)))
    set_axes_equal_3d(axis, np.concatenate(scene_points))
    axis.set_xlabel("camera X / m")
    axis.set_ylabel("camera Y / m")
    axis.set_zlabel("camera Z / m")
    axis.set_title(f"3D camera-frame scene | {shape_name}")
    axis.view_init(elev=18.0, azim=-68.0)
    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2563eb", markersize=7, label="visible after occlusion"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#dc2626", markersize=7, label="blocked by external occluder"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#111827", markersize=7, label="camera"),
    ]
    axis.legend(handles=legend_handles, loc="upper left")
    figure.tight_layout()
    figure.savefig(output_dir / "scene_3d.png", bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 5), dpi=160)
    fim_path = output_dir / "fim.npz"
    if fim_path.exists():
        with np.load(fim_path) as arrays:
            full_eigenvalues = arrays["full_eigenvalues"]
            occluded_eigenvalues = arrays["occluded_eigenvalues"]
    else:
        jacobian = point_to_plane_jacobian(full.points_camera, full.normals_camera)
        full_eigenvalues, _ = eigensystem(build_fim(jacobian, full.weights))
        occluded_eigenvalues, _ = eigensystem(build_fim(jacobian, after_occlusion.weights))
    x = np.arange(6)
    width = 0.36
    axis.bar(x - width / 2.0, np.maximum(full_eigenvalues, 1e-12), width, label="full visibility", color="#16a34a")
    axis.bar(x + width / 2.0, np.maximum(occluded_eigenvalues, 1e-12), width, label="after occlusion", color="#dc2626")
    axis.set_yscale("log")
    axis.set_xticks(x, ["λ1", "λ2", "λ3", "λ4", "λ5", "λ6"])
    axis.set_ylabel("FIM eigenvalue, log scale")
    axis.set_title(f"Geometric information spectrum | {shape_name}")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "fim_eigenvalues.png", bbox_inches="tight")
    plt.close(figure)

    print(f"saved camera view: {output_dir / 'camera_view.png'}")
    print(f"saved 3D scene: {output_dir / 'scene_3d.png'}")
    print(f"saved FIM spectrum: {output_dir / 'fim_eigenvalues.png'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", choices=("cuboid", "cylinder"), default="cuboid")
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--output", type=Path, default=Path("outputs/visualization"))
    parser.add_argument("--no-occluder", action="store_true")
    parser.add_argument("--cuboid-size", nargs=3, type=float, default=(0.42, 0.32, 0.52), metavar=("SX", "SY", "SZ"))
    parser.add_argument("--cylinder-radius", type=float, default=0.20)
    parser.add_argument("--cylinder-height", type=float, default=0.52)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    resolution = args.resolution or (12 if args.shape == "cuboid" else 32)
    render(
        args.shape,
        resolution,
        args.output,
        tuple(args.cuboid_size),
        args.cylinder_radius,
        args.cylinder_height,
        args.no_occluder,
    )


if __name__ == "__main__":
    main()

