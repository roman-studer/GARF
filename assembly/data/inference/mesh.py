from typing import List, Literal

import trimesh
import numpy as np
from torch.utils.data import Dataset

from ..transform import recenter_pc, rotate_pc

COLORS = [
    [0, 204, 0],
    [204, 0, 0],
    [0, 0, 204],
    [127, 127, 0],
    [127, 0, 127],
    [0, 127, 127],
    [76, 153, 0],
    [153, 0, 76],
    [76, 0, 153],
    [153, 76, 0],
    [76, 0, 153],
    [153, 0, 76],
    [204, 51, 127],
    [204, 51, 127],
    [51, 204, 127],
    [51, 127, 204],
    [127, 51, 204],
    [127, 204, 51],
    [76, 76, 178],
    [76, 178, 76],
    [178, 76, 76],
]


class MeshInferenceDataset(Dataset):
    def __init__(
        self,
        name: str,
        meshes_paths: List[str],
        pad_to_parts: int = 20,
        num_points_to_sample: int = 1000,
        mesh_type: str = "obj",
        sample_method: Literal["uniform", "weighted"] = "uniform",
        min_points_per_part: int = 20,
        seed=1116,
        sample_strategy: Literal["uniform", "poisson"] = "uniform",
    ):
        self.name = name
        self.meshes_paths = meshes_paths
        self.pad_to_parts = max(pad_to_parts, len(meshes_paths))
        self.num_points_to_sample = num_points_to_sample
        self.meshes = [
            trimesh.load_mesh(mesh_path, file_type=mesh_type)
            for mesh_path in meshes_paths
        ]
        self.sample_method = sample_method
        self.min_points_per_part = min_points_per_part
        self.seed = seed
        self.numpy_rng = np.random.default_rng(seed=seed)
        self.sample_strategy = sample_strategy

        self.scale_max = 1.0
        for i in range(len(self.meshes)):
            self.scale_max = max(self.scale_max, max(self.meshes[i].extents))
        for i in range(len(self.meshes)):
            self.meshes[i].apply_scale(1 / self.scale_max)

    def __len__(self):
        return 1

    def get_meshes(self, name: str):
        return [
            {
                "vertices": mesh.vertices,
                "faces": mesh.faces,
                "color": COLORS[idx % len(COLORS)],
            }
            for idx, mesh in enumerate(self.meshes)
        ]

    def _pad_data(self, input_data: np.ndarray):
        """Pad data to shape [`self.max_parts`, data.shape[1], ...]."""
        d = np.array(input_data)
        pad_shape = (self.pad_to_parts,) + tuple(d.shape[1:])
        pad_data = np.zeros(pad_shape, dtype=np.float32)
        pad_data[: d.shape[0]] = d
        return pad_data

    def get_item_weighted(self):
        num_parts = len(self.meshes)
        areas = [mesh.area for mesh in self.meshes]
        total_area = sum(areas)
        points_per_part = [
            self.min_points_per_part
            + int(
                (
                    self.num_points_to_sample
                    - self.min_points_per_part * len(self.meshes)
                )
                * area
                / total_area
            )
            for area in areas
        ]
        anchor_idx = np.argmax(points_per_part)
        points_per_part[anchor_idx] += self.num_points_to_sample - sum(points_per_part)
        points_per_part[
            np.argmin(points_per_part)
        ] += self.num_points_to_sample - np.sum(points_per_part)
        sampled_pcds = []
        if self.sample_strategy == "poisson":
            for i in range(num_parts):
                pcd, idx = trimesh.sample.sample_surface_even(
                    self.meshes[i], count=points_per_part[i], seed=self.seed
                )
                if len(pcd) < points_per_part[i]:
                    concat_pcd, concat_idx = trimesh.sample.sample_surface(
                        self.meshes[i],
                        count=points_per_part[i] - len(pcd),
                        seed=self.seed,
                    )
                    pcd = np.concatenate([pcd, concat_pcd], axis=0)
                    idx = np.concatenate([idx, concat_idx], axis=0)
                sampled_pcds.append((pcd, idx))
        else:
            sampled_pcds = [
                trimesh.sample.sample_surface(
                    mesh=self.meshes[i],
                    count=points_per_part[i],
                )
                for i in range(len(self.meshes))
            ]

        pointclouds_gt = [pcd[0] for pcd in sampled_pcds]
        pointclouds_normals_gt = [
            self.meshes[i].face_normals[pcd[1]] for i, pcd in enumerate(sampled_pcds)
        ]

        offset = np.concatenate([[0], np.cumsum(points_per_part)])

        pointclouds_gt = np.concatenate(pointclouds_gt).astype(np.float32)  # (N, 3)
        pointclouds_normals_gt = np.concatenate(pointclouds_normals_gt).astype(
            np.float32
        )  # (N, 3)

        pointclouds, pointclouds_normals, quaternions, translations = [], [], [], []
        scale = []
        for part_idx in range(num_parts):
            start = offset[part_idx]
            end = offset[part_idx + 1]
            pointcloud, translation = recenter_pc(pointclouds_gt[start:end])
            pointcloud, pointcloud_normals, quaternion = rotate_pc(
                pointcloud,
                pointclouds_normals_gt[start:end],
                self.numpy_rng,
            )
            # Rescale
            current_scale = np.max(np.abs(pointcloud))
            scale.append(current_scale)
            pointcloud /= current_scale

            pointclouds.append(pointcloud)
            pointclouds_normals.append(pointcloud_normals)
            translations.append(translation)
            quaternions.append(quaternion)

        pointclouds = np.concatenate(pointclouds).astype(np.float32)  # [N, 3]
        pointclouds_normals = np.concatenate(pointclouds_normals).astype(np.float32)
        quaternions = np.stack(quaternions).astype(np.float32)  # [P, 4]
        translations = np.stack(translations).astype(np.float32)  # [P, 3]
        scale = np.array(scale).astype(np.float32)

        # Pad data
        points_per_part = self._pad_data(points_per_part).astype(np.int64)
        quaternions = self._pad_data(quaternions)
        translations = self._pad_data(translations)
        scale = self._pad_data(scale)

        # Ref-part
        ref_part = np.zeros((self.pad_to_parts), dtype=np.float32)
        ref_part_idx = np.argmax(points_per_part[:num_parts])
        ref_part[ref_part_idx] = 1
        ref_part = ref_part.astype(bool)

        return {
            "name": self.name,
            "num_parts": num_parts,
            "pointclouds": pointclouds,
            "pointclouds_gt": pointclouds_gt,
            "pointclouds_normals": pointclouds_normals,
            "pointclouds_normals_gt": pointclouds_normals_gt,
            "quaternions": quaternions,
            "translations": translations,
            "points_per_part": points_per_part,
            "graph": np.ones((self.pad_to_parts, self.pad_to_parts), dtype=np.float32),
            # Only for uniform sampling
            "scale": scale[:, np.newaxis],
            "ref_part": ref_part,
            # "init_rot": init_rot,
            "mesh_scale": self.scale_max,
        }

    def __getitem__(self, idx):
        if self.sample_method == "weighted":
            return self.get_item_weighted()
        else:
            raise ValueError(f"Unknown sample method: {self.sample_method}")
