#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from trimesh.transformations import quaternion_matrix


def se3_to_matrix(transform: list[float]) -> np.ndarray:
    """Convert [tx, ty, tz, qw, qx, qy, qz] into a 4x4 transform matrix."""
    transform = np.asarray(transform, dtype=np.float64)
    matrix = quaternion_matrix(transform[3:])
    matrix[:3, 3] = transform[:3]
    return matrix


def find_sample_dirs(
    input_path: Path,
    gt_name: str,
    json_name: str,
) -> list[Path]:
    if input_path.is_file():
        raise ValueError("Please pass a directory, not a file.")

    if (input_path / gt_name).is_file() and (input_path / json_name).is_file():
        return [input_path]

    assembly_root = input_path / "assembly_results"
    search_root = assembly_root if assembly_root.is_dir() else input_path

    sample_dirs = sorted(
        path.parent for path in search_root.rglob(json_name) if (path.parent / gt_name).is_file()
    )
    if not sample_dirs:
        raise FileNotFoundError(
            f"Could not find any sample directories under {input_path} containing "
            f"{gt_name} and {json_name}."
        )
    return sample_dirs


def export_start_pose(
    sample_dir: Path,
    gt_name: str,
    json_name: str,
    output_name: str,
) -> Path:
    gt_path = sample_dir / gt_name
    json_path = sample_dir / json_name
    output_path = sample_dir / output_name

    with json_path.open() as f:
        assembly_data = json.load(f)

    gt_transforms = assembly_data["gt_transform"]
    scene_gt = trimesh.load(gt_path, force="scene")
    geometry_nodes = list(scene_gt.graph.nodes_geometry)

    if len(geometry_nodes) != len(gt_transforms):
        raise ValueError(
            f"{sample_dir}: found {len(geometry_nodes)} mesh nodes in {gt_name}, "
            f"but {len(gt_transforms)} gt transforms in {json_name}."
        )

    scene_start = trimesh.Scene()
    for node_name, gt_transform in zip(geometry_nodes, gt_transforms):
        current_transform, geom_name = scene_gt.graph[node_name]
        start_transform = np.linalg.inv(se3_to_matrix(gt_transform))
        scene_start.add_geometry(
            scene_gt.geometry[geom_name].copy(),
            geom_name=geom_name,
            node_name=node_name,
            transform=start_transform @ current_transform,
        )

    scene_start.export(output_path)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a GLB with ground-truth fragments placed in the pre-assembly "
            "input pose using the saved gt transforms."
        )
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help=(
            "Either a sample directory containing view_gt.glb and "
            "view_assembly_0.json, or an experiment directory containing "
            "assembly_results/."
        ),
    )
    parser.add_argument(
        "--gt-name",
        default="view_gt.glb",
        help="Name of the canonical ground-truth GLB file.",
    )
    parser.add_argument(
        "--json-name",
        default="view_assembly_0.json",
        help="Name of the JSON file containing gt_transform.",
    )
    parser.add_argument(
        "--output-name",
        default="view_start.glb",
        help="Filename for the exported pre-assembly GLB.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample_dirs = find_sample_dirs(args.input_path, args.gt_name, args.json_name)

    for sample_dir in sample_dirs:
        output_path = export_start_pose(
            sample_dir=sample_dir,
            gt_name=args.gt_name,
            json_name=args.json_name,
            output_name=args.output_name,
        )
        print(output_path)


if __name__ == "__main__":
    main()
