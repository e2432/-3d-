"""Run the parameterised visibility-geometry FIM experiment."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

try:
    from .fim import (
        build_fim,
        eigensystem,
        finite_difference_jacobian,
        point_to_plane_jacobian,
        relative_frobenius_error,
        skew_residual_check,
    )
    from .geometry import shape_from_name, so3_exp
    from .visibility import BoxOccluder, CameraIntrinsics, compute_visibility
except ImportError:
    from fim import (
        build_fim,
        eigensystem,
        finite_difference_jacobian,
        point_to_plane_jacobian,
        relative_frobenius_error,
        skew_residual_check,
    )
    from geometry import shape_from_name, so3_exp
    from visibility import BoxOccluder, CameraIntrinsics, compute_visibility


def make_object_pose() -> np.ndarray:
    """Return a fixed, non-degenerate object pose in the camera frame."""

    transform = np.eye(4)
    transform[:3, :3] = so3_exp(np.array([0.18, -0.28, 0.12], dtype=float))
    transform[:3, 3] = np.array([0.0, 0.0, 1.15], dtype=float)
    return transform


def make_occluder() -> BoxOccluder:
    """Return a box between the camera and the object."""

    return BoxOccluder(center=(0.04, 0.02, 0.72), size=(0.22, 0.62, 0.30))


def save_surface_csv(
    path: Path,
    points_object: np.ndarray,
    normals_object: np.ndarray,
    full_visibility,
    occluded_visibility,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "px_object",
                "py_object",
                "pz_object",
                "nx_object",
                "ny_object",
                "nz_object",
                "px_camera",
                "py_camera",
                "pz_camera",
                "nx_camera",
                "ny_camera",
                "nz_camera",
                "u",
                "v",
                "depth",
                "visible_full",
                "visible_occluded",
            ]
        )
        for point_object, normal_object, point_camera, normal_camera, pixel, depth, full_weight, occ_weight in zip(
            points_object,
            normals_object,
            full_visibility.points_camera,
            full_visibility.normals_camera,
            full_visibility.pixels,
            full_visibility.depths,
            full_visibility.weights,
            occluded_visibility.weights,
        ):
            writer.writerow(
                [
                    *point_object.tolist(),
                    *normal_object.tolist(),
                    *point_camera.tolist(),
                    *normal_camera.tolist(),
                    *pixel.tolist(),
                    float(depth),
                    int(full_weight > 0.0),
                    int(occ_weight > 0.0),
                ]
            )


def matrix_to_list(matrix: np.ndarray) -> list[list[float]]:
    return np.asarray(matrix, dtype=float).round(12).tolist()


def run(
    shape_name: str,
    resolution: int,
    noise_std: float,
    output_dir: Path,
    disable_occluder: bool,
    cuboid_size: tuple[float, float, float] = (0.42, 0.32, 0.52),
    cylinder_radius: float = 0.20,
    cylinder_height: float = 0.52,
) -> dict:
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

    visibility_full = compute_visibility(
        shape,
        surface.points,
        surface.normals,
        t_co,
        camera,
        occluder=None,
    )
    visibility_occluded = compute_visibility(
        shape,
        surface.points,
        surface.normals,
        t_co,
        camera,
        occluder=occluder,
    )

    jacobian_analytic = point_to_plane_jacobian(
        visibility_full.points_camera, visibility_full.normals_camera
    )
    jacobian_numeric = finite_difference_jacobian(
        surface.points,
        t_co,
        visibility_full.normals_camera,
    )
    jacobian_max_abs_error = float(np.max(np.abs(jacobian_analytic - jacobian_numeric)))
    variance = float(noise_std * noise_std)
    fim_full_analytic = build_fim(jacobian_analytic, visibility_full.weights, variance)
    fim_occ_analytic = build_fim(jacobian_analytic, visibility_occluded.weights, variance)
    fim_full_numeric = build_fim(jacobian_numeric, visibility_full.weights, variance)
    fim_occ_numeric = build_fim(jacobian_numeric, visibility_occluded.weights, variance)
    full_eigenvalues, full_eigenvectors = eigensystem(fim_full_analytic)
    occ_eigenvalues, occ_eigenvectors = eigensystem(fim_occ_analytic)

    output_dir.mkdir(parents=True, exist_ok=True)
    save_surface_csv(output_dir / "surface_points.csv", surface.points, surface.normals, visibility_full, visibility_occluded)
    np.savez_compressed(
        output_dir / "fim.npz",
        fim_full=fim_full_analytic,
        fim_occluded=fim_occ_analytic,
        fim_full_numeric=fim_full_numeric,
        fim_occluded_numeric=fim_occ_numeric,
        full_eigenvalues=full_eigenvalues,
        full_eigenvectors=full_eigenvectors,
        occluded_eigenvalues=occ_eigenvalues,
        occluded_eigenvectors=occ_eigenvectors,
        object_pose=t_co,
    )

    summary = {
        "shape": shape_name,
        "surface_sample_count": int(len(surface.points)),
        "full_visible_count": int(np.count_nonzero(visibility_full.weights)),
        "occluded_visible_count": int(np.count_nonzero(visibility_occluded.weights)),
        "noise_std_m": noise_std,
        "jacobian_max_abs_error": jacobian_max_abs_error,
        "fim_relative_error_full": relative_frobenius_error(fim_full_analytic, fim_full_numeric),
        "fim_relative_error_occluded": relative_frobenius_error(fim_occ_analytic, fim_occ_numeric),
        "skew_identity_max_abs_error": skew_residual_check(),
        "full_eigenvalues": full_eigenvalues.tolist(),
        "occluded_eigenvalues": occ_eigenvalues.tolist(),
        "full_fim": matrix_to_list(fim_full_analytic),
        "occluded_fim": matrix_to_list(fim_occ_analytic),
        "object_pose_object_to_camera": matrix_to_list(t_co),
        "camera": {
            "width": camera.width,
            "height": camera.height,
            "fx": camera.fx,
            "fy": camera.fy,
            "cx": camera.cx,
            "cy": camera.cy,
        },
        "occluder_enabled": occluder is not None,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", choices=("cuboid", "cylinder"), default="cuboid")
    parser.add_argument("--resolution", type=int, default=None, help="Surface sampling resolution")
    parser.add_argument("--noise-std", type=float, default=0.005, help="Point-to-plane noise standard deviation in metres")
    parser.add_argument("--output", type=Path, default=Path("outputs/cuboid"))
    parser.add_argument("--no-occluder", action="store_true", help="Skip the external box occluder")
    parser.add_argument("--cuboid-size", nargs=3, type=float, default=(0.42, 0.32, 0.52), metavar=("SX", "SY", "SZ"))
    parser.add_argument("--cylinder-radius", type=float, default=0.20)
    parser.add_argument("--cylinder-height", type=float, default=0.52)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    resolution = args.resolution
    if resolution is None:
        resolution = 12 if args.shape == "cuboid" else 32
    summary = run(
        args.shape,
        resolution,
        args.noise_std,
        args.output,
        args.no_occluder,
        cuboid_size=tuple(args.cuboid_size),
        cylinder_radius=args.cylinder_radius,
        cylinder_height=args.cylinder_height,
    )
    print(f"shape: {summary['shape']}")
    print(f"surface samples: {summary['surface_sample_count']}")
    print(f"visible points, full: {summary['full_visible_count']}")
    print(f"visible points, occluded: {summary['occluded_visible_count']}")
    print(f"max analytic-vs-numeric Jacobian error: {summary['jacobian_max_abs_error']:.3e}")
    print(f"full FIM relative error: {summary['fim_relative_error_full']:.3e}")
    print(f"occluded FIM relative error: {summary['fim_relative_error_occluded']:.3e}")
    print("full FIM eigenvalues:", np.array(summary["full_eigenvalues"]))
    print("occluded FIM eigenvalues:", np.array(summary["occluded_eigenvalues"]))
    print(f"saved to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
