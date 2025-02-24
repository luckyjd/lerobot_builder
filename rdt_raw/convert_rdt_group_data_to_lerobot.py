import os
import time
import dataclasses
from pathlib import Path
import shutil
from typing import Literal

import h5py
from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.push_dataset_to_hub._download_raw import download_raw
import numpy as np
import torch
import tqdm
import tyro

SHAPE_14 = (14,)
FPS = 20


def create_empty_dataset(
        repo_id: str,
        robot_type: str,
        mode: Literal["video", "image"] = "image",
        has_velocity: bool = False,
        has_effort: bool = False,
        has_depth: bool = False,
) -> LeRobotDataset:
    base = [
        "linear",
        "angular"
    ]
    cameras = [
        "cam_high",
        "cam_left_wrist",
        "cam_right_wrist",
    ]

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": SHAPE_14,
            "names": [],
        },
        "action": {
            "dtype": "float32",
            "shape": SHAPE_14,
            "names": [],
        },
        "base_action": {
            "dtype": "float32",
            "shape": (len(base),),
            "names": [
                base,
            ],
        },
    }

    if has_velocity:
        features["observation.velocity"] = {
            "dtype": "float32",
            "shape": SHAPE_14,
            "names": [],
        }

    if has_effort:
        features["observation.effort"] = {
            "dtype": "float32",
            "shape": SHAPE_14,
            "names": [],
        }

    for cam in cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": mode,
            "shape": (3, 480, 640),
            "names": [
                "channels",
                "height",
                "width",
            ],
        }
        if has_depth:
            features[f"observation.images_depth.{cam}"] = {
                "dtype": mode,
                "shape": (3, 480, 640),
                "names": [
                    "channels",
                    "height",
                    "width",
                ],
            }

    if Path(LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(LEROBOT_HOME / repo_id)

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=FPS,
        robot_type=robot_type,
        features=features,
        image_writer_processes=10,
        image_writer_threads=5,
    )


def get_cameras(hdf5_files: list[Path]) -> list[str]:
    with h5py.File(hdf5_files[0], "r") as ep:
        # ignore depth channel, not currently handled
        return [key for key in ep["/observations/images"].keys()]

def get_cameras_depth(hdf5_files: list[Path]) -> list[str]:
    with h5py.File(hdf5_files[0], "r") as ep:
        # ignore depth channel, not currently handled
        return [key for key in ep["/observations/images_depth"].keys()]

def has_velocity(hdf5_folder: list[Path]) -> bool:
    hdf5_files = sorted(hdf5_folder[0].glob("episode_*.hdf5"))
    with h5py.File(hdf5_files[0], "r") as ep:
        return "/observations/qvel" in ep


def has_effort(hdf5_folder: list[Path]) -> bool:
    hdf5_files = sorted(hdf5_folder[0].glob("episode_*.hdf5"))
    with h5py.File(hdf5_files[0], "r") as ep:
        return "/observations/effort" in ep


def load_raw_images_per_camera(ep: h5py.File,
                               cameras: list[str],
                               has_depth: bool = False) -> dict[str, np.ndarray]:
    imgs_per_cam = {
        "color": {},
        "depth": {}
                    }
    for camera in cameras:
        # color
        uncompressed = ep[f"/observations/images/{camera}"].ndim == 4

        if uncompressed:
            # load all images in RAM
            imgs_array = ep[f"/observations/images/{camera}"][:]
        else:
            import cv2

            # load one compressed image after the other in RAM and uncompress
            imgs_array = []
            for data in ep[f"/observations/images/{camera}"]:
                buffer_array = np.frombuffer(data, dtype=np.uint8)
                img_bgr = cv2.imdecode(buffer_array, cv2.IMREAD_COLOR)
                imgs_array.append(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
            imgs_array = np.array(imgs_array)

        imgs_per_cam["color"][camera] = imgs_array

        # depth
        if has_depth:
            depth_ds = ep[f"/observations/images_depth/{camera}"]
            uncompressed_depth = (depth_ds.ndim == 3 or depth_ds.ndim == 4)

            if uncompressed_depth:
                depth_array = depth_ds[:]
            else:
                import cv2

                depth_array = []
                for data in depth_ds:
                    # Use IMREAD_ANYDEPTH or IMREAD_UNCHANGED for depth
                    buffer_array = np.frombuffer(data, dtype=np.uint8)
                    dimg = cv2.imdecode(buffer_array, cv2.IMREAD_UNCHANGED)
                    depth_array.append(dimg)
                depth_array = np.array(depth_array)

            imgs_per_cam["depth"][camera] = depth_array

    return imgs_per_cam


def load_raw_episode_data(
        ep_path: Path,
) -> tuple[dict[str, np.ndarray], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    with h5py.File(ep_path, "r") as ep:
        state = torch.from_numpy(ep["/observations/qpos"][:])
        action = torch.from_numpy(ep["/action"][:])
        base_action = torch.from_numpy(ep["/base_action"][:])

        velocity = None
        if "/observations/qvel" in ep:
            velocity = torch.from_numpy(ep["/observations/qvel"][:])

        effort = None
        if "/observations/effort" in ep:
            effort = torch.from_numpy(ep["/observations/effort"][:])

        cameras = [
            "cam_high",
            "cam_left_wrist",
            "cam_right_wrist",
        ]
        imgs_per_cam = load_raw_images_per_camera(ep, cameras)

    return imgs_per_cam, state, action, base_action, velocity, effort


def populate_dataset(
        dataset: LeRobotDataset,
        hdf5_files: list[Path],
        task: str,
        episodes: list[int] | None = None,
        has_depth: bool = False,
) -> LeRobotDataset:
    if episodes is None:
        episodes = range(len(hdf5_files))

    for ep_idx in tqdm.tqdm(episodes):
        ep_path = hdf5_files[ep_idx]

        imgs_per_cam, state, action, base_action, velocity, effort = load_raw_episode_data(ep_path)
        num_frames = state.shape[0]

        for i in range(num_frames):
            frame = {
                "observation.state": state[i],
                "action": action[i],
                "base_action": base_action[i],
            }

            for camera, img_array in imgs_per_cam["color"].items():
                frame[f"observation.images.{camera}"] = img_array[i]

            # if has_depth:
            #     for camera, img_depth_array in imgs_per_cam["depth"].items():
            #         frame[f"observation.images_depth.{camera}"] = img_depth_array[i]

            if velocity is not None:
                frame["observation.velocity"] = velocity[i]
            if effort is not None:
                frame["observation.effort"] = effort[i]

            dataset.add_frame(frame)
        dataset.save_episode(task=task)

    return dataset


def lerobot_builder(
        raw_dir: Path,
        repo_id: str,
        raw_repo_id: str | None = None,
        task: str = "DEBUG",
        episodes: list[int] | None = None,
        push_to_hub: bool = False,
        mode: Literal["video", "image"] = "image",
):
    if (LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(LEROBOT_HOME / repo_id)

    if not raw_dir.exists():
        if raw_repo_id is None:
            raise ValueError("raw_repo_id must be provided if raw_dir does not exist")
        download_raw(raw_dir, repo_id=raw_repo_id)

    all_paths = raw_dir.glob("*")
    hdf5_folders = [p for p in all_paths if os.path.isdir(p)]

    dataset = create_empty_dataset(
        repo_id,
        robot_type="rdt",
        mode=mode,
        has_effort=has_effort(hdf5_folders),
        has_velocity=has_velocity(hdf5_folders),
    )
    for task_dir in hdf5_folders:
        # try:
        task = task_dir.name
        task_start_time = time.time()
        hdf5_files = sorted(task_dir.glob("episode_*.hdf5"))

        dataset = populate_dataset(
            dataset,
            hdf5_files,
            task=task,
            episodes=episodes,
        )


        task_end_time = time.time()
        print(f"Finish task {task_dir.name} in {(task_end_time - task_start_time):.2f} seconds")
        # except Exception as ex:
        #     print(f"Error task {task_dir.name} : {ex}")
    dataset.consolidate()

    if push_to_hub:
        dataset.push_to_hub()


# def lerobot_test(
#     raw_dir: Path,
#     repo_id: str,
# ):
#     hdf5_files = sorted(raw_dir.glob("episode_*.hdf5"))
#     with h5py.File(hdf5_files[0], "r") as ep:
#         for key in ep.keys():
#             print(f"{key} : {ep[key]}")
#
#         for key in ep["observations"].keys():
#             try:
#                 print(f"List key of observations/{key} : {ep[f'observations/{key}'].keys()}")
#             except:
#                 print(f"Type of observations/{key} : {ep[f'observations/{key}']}")
#
#         print(f"DEPTH : {ep[f'observations/images_depth/cam_high']}")


if __name__ == "__main__":
    tyro.cli(lerobot_builder)
