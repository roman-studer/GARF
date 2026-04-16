import os
import sys


def _ensure_repo_python():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    venv_python = os.path.join(repo_root, ".venv", "bin", "python")
    if not os.path.exists(venv_python):
        return
    current_python = os.path.abspath(sys.executable)
    if current_python == os.path.abspath(venv_python):
        return
    os.execv(venv_python, [venv_python, os.path.abspath(__file__), *sys.argv[1:]])


_ensure_repo_python()

import numpy as np
import h5py
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from typing import List
from scipy.sparse.csgraph import connected_components
import trimesh

CATEGORIES = ["radii"]
SPLITS = ["train", "val"]
BASE_PATH = "./data/radii/radii"
SPLITS_PATH = "./data/radii/radii/data_split"
OUTPUT_PATH = "./data/radii/radii.hdf5"
MESH_EXTENSIONS = (".obj", ".ply")


def read_data_list():
    data_list = dict()
    for category in CATEGORIES:
        data_list[category] = dict()
        for split in SPLITS:
            print(f"Reading data list {category}/{split}")
            data_list[category][split] = []
            f = open(f"{SPLITS_PATH}/{category}.{split}.txt", "r", encoding="utf-8")
            lines = f.readlines()
            lines = [line.strip() for line in lines]
            f.close()
            for line in tqdm(lines):
                obj_path = os.path.join(BASE_PATH, line)
                if not os.path.exists(obj_path):
                    print(f"Missing {obj_path}")
                    continue
                fractures = os.listdir(obj_path)
                # If the directory directly contains mesh pieces, no need to dig deeper.
                # no need to dig deeper
                if fractures and all(x.endswith(MESH_EXTENSIONS) for x in fractures):
                    data_list[category][split].append(line)
                    continue
                fractures = filter(
                    lambda x: x.startswith("fractured") or x.startswith("mode"),
                    fractures,
                )
                for fracture in fractures:
                    name = os.path.join(line, fracture)
                    data_list[category][split].append(name)
    return data_list


def flatten_data_list(data_list):
    flattened_data_list = []
    for category in CATEGORIES:
        for split in SPLITS:
            for name in data_list[category][split]:
                flattened_data_list.append(name)
    return flattened_data_list


def are_meshes_connected(
    mesh_a: trimesh.Trimesh,
    mesh_b: trimesh.Trimesh,
    decimals: int = 5,
):
    """
    Check if two meshes are connected.

    Args:
        mesh_a (trimesh.Trimesh): The first mesh.
        mesh_b (trimesh.Trimesh): The second mesh.
        decimals (int, optional): The number of decimal places to round the vertices to. Defaults to 5.

    Returns:
        bool: True if the meshes are connected, False otherwise.
    """
    vertices_a = mesh_a.vertices
    vertices_b = mesh_b.vertices
    faces_a = mesh_a.faces
    faces_b = mesh_b.faces

    shared_faces_a = np.zeros(len(faces_a), dtype=bool)
    shared_faces_b = np.zeros(len(faces_b), dtype=bool)

    vertices_a = vertices_a.round(decimals=decimals)
    vertices_b = vertices_b.round(decimals=decimals)

    common_vertices = set(map(tuple, vertices_a)).intersection(map(tuple, vertices_b))

    # calculate common faces
    if len(common_vertices) > 0:
        for i, face_a in enumerate(faces_a):
            if all([tuple(vertices_a[vertex]) in common_vertices for vertex in face_a]):
                shared_faces_a[i] = True
        for i, face_b in enumerate(faces_b):
            if all([tuple(vertices_b[vertex]) in common_vertices for vertex in face_b]):
                shared_faces_b[i] = True

    return len(common_vertices) > 0, shared_faces_a, shared_faces_b


def get_graph(
    meshes: List[trimesh.Trimesh],
) -> np.ndarray:
    """
    Get the connectivity matrix of a list of meshes.

    Args:
        meshes (List[trimesh.Trimesh]): List of meshes

    Returns:
        np.ndarray: Graph matrix.
    """
    num_meshes = len(meshes)
    graph = np.zeros((num_meshes, num_meshes), dtype=bool)

    shared_faces = []
    for i in range(num_meshes):
        # negative values indicate that the face is not shared
        shared_faces.append(-np.ones(len(meshes[i].faces), dtype=np.int64))

    for i in range(num_meshes):
        for j in range(i + 1, num_meshes):  # Check each pair once
            connected, i_shared_faces_with_j, j_shared_faces_with_i = (
                are_meshes_connected(meshes[i], meshes[j])
            )
            if connected:
                graph[i, j] = True
                graph[j, i] = True
                shared_faces[i][i_shared_faces_with_j] = j
                shared_faces[j][j_shared_faces_with_i] = i

    return graph.astype(bool), shared_faces


def process_obj(name: str):
    obj_path = os.path.join(BASE_PATH, name)
    pieces = sorted(
        piece for piece in os.listdir(obj_path) if piece.endswith(MESH_EXTENSIONS)
    )
    if not pieces:
        print(f"Skipping {name}: no mesh files found in {obj_path}")
        return name, None

    pieces_names = [os.path.splitext(piece)[0] for piece in pieces]
    try:
        meshes = [trimesh.load(os.path.join(obj_path, piece)) for piece in pieces]
    except Exception as e:
        print(f"Error loading {name}: {e}")
        return name, None
    volumes = [mesh.volume for mesh in meshes]

    # largest to smallest
    volumes_order = np.argsort(volumes)[::-1]
    # keep the largest piece, move largest to the end
    volumes_order = np.concatenate([volumes_order[1:], volumes_order[:1]])

    result = dict()
    graph, shared_faces = get_graph(meshes=meshes)

    # removal start from the largest piece that does not split the graph
    removal_masks = []
    removal_order = []
    last_removal_mask = np.ones(len(meshes), dtype=bool)
    for _ in range(len(meshes)):
        for piece_idx in volumes_order:
            # this piece is already removed
            if not last_removal_mask[piece_idx]:
                continue
            removal_mask = last_removal_mask.copy()
            removal_mask[piece_idx] = False
            connected, _ = connected_components(graph[removal_mask][:, removal_mask])
            if connected == 1:
                last_removal_mask = removal_mask
                removal_masks.append(removal_mask)
                removal_order.append(piece_idx)
                break

    result["pieces_names"] = pieces_names
    result["removal_masks"] = np.array(removal_masks)
    result["removal_order"] = np.array(removal_order)
    result["pieces"] = dict()
    for piece_idx, mesh in enumerate(meshes):
        result["pieces"][piece_idx] = dict()
        result["pieces"][piece_idx]["vertices"] = mesh.vertices
        result["pieces"][piece_idx]["faces"] = mesh.faces
        result["pieces"][piece_idx]["shared_faces"] = shared_faces[piece_idx]

    return name, result


def main_process():
    data_list = read_data_list()
    flattened_data_list = flatten_data_list(data_list)
    h5_file = h5py.File(OUTPUT_PATH, "w")

    h5_file.create_group("data_split")
    for category in data_list:
        h5_file["data_split"].create_group(category)
        for split in data_list[category]:
            h5_file["data_split"][category]
            h5_file["data_split"][category].create_dataset(
                split, data=data_list[category][split]
            )

    pool = ProcessPoolExecutor(max_workers=32)
    for name, result in tqdm(
        pool.map(process_obj, flattened_data_list),
        total=len(flattened_data_list),
    ):
        if result is None:
            continue
        # skip if name already exists
        if name in h5_file:
            continue
        h5_file.create_group(name)
        h5_file[name].create_dataset("pieces_names", data=result["pieces_names"])
        h5_file[name].create_dataset(
            "removal_masks", data=result["removal_masks"], compression="gzip"
        )
        h5_file[name].create_dataset(
            "removal_order", data=result["removal_order"], compression="gzip"
        )
        h5_file[name].create_group("pieces")
        for piece_idx, data in result["pieces"].items():
            group = h5_file[name]["pieces"].create_group(str(piece_idx))
            group.create_dataset("vertices", data=data["vertices"], compression="gzip")
            group.create_dataset("faces", data=data["faces"], compression="gzip")
            group.create_dataset(
                "shared_faces", data=data["shared_faces"], compression="gzip"
            )

    h5_file.close()

if __name__ == "__main__":
    main_process()
