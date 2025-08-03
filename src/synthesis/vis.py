import mitsuba as mi
import os
import argparse
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
import rerun as rr
from PIL import Image

from camera import get_camera_params, to_world_matrix_to_rerun_transform

# Set Mitsuba variant
mi.set_variant("llvm_ad_rgb")


def load_camera_data(json_path: str) -> Dict[str, Any]:
    """
    Load camera data from JSON file

    Args:
        json_path: Path to the camera_data.json file

    Returns:
        Dict containing camera data
    """
    with open(json_path, "r") as f:
        return json.load(f)


def load_scene_models(scene_dir: str):
    """
    Load 3D models from the scene directory

    Args:
        scene_dir: Path to the scene directory containing models
    """
    obj_dir = os.path.join(scene_dir, "models")
    if os.path.exists(obj_dir):
        for file in os.listdir(obj_dir):
            if file.endswith(".obj"):
                file_path = os.path.join(obj_dir, file)
                # Log the 3D model file
                rr.log_file_from_path(
                    file_path=file_path,
                    entity_path_prefix=f"/scene/models/{file[:-4]}",
                    static=True,
                )


def visualize_cameras(camera_data: Dict[str, Any], images_dir: str):
    """
    Visualize camera positions and orientations

    Args:
        camera_data: Camera data from JSON
        images_dir: Directory containing rendered images
    """
    frames = camera_data.get("frames", [])

    for i, frame in enumerate(frames):
        # Extract camera transform
        transform_matrix = np.array(frame["camera_to_world_transform"])

        # Extract camera position
        position = transform_matrix[:3, 3]

        # Extract camera orientation (rotation matrix)
        rotation = transform_matrix[:3, :3]

        # Create camera entity path
        camera_path = f"/cameras/camera_{i:04d}"

        rr.log("/cameras", rr.Transform3D(translation=[0, 0, 0], mat3x3=np.eye(3)))

        # Log camera transform
        rr.log(camera_path, rr.Transform3D(translation=position, mat3x3=rotation))

        # Log camera coordinate system
        rr.log(
            f"{camera_path}/axes",
            rr.Arrows3D(
                origins=np.array([[0, 0, 0]]),
                vectors=np.array([[0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]]),
                colors=np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]]),
            ),
        )

        # Log camera pinhole
        intrinsic_matrix = np.array(frame["camera_intrinsic_matrix"])
        film_size = frame["film_size"]

        rr.log(
            f"{camera_path}/pinhole",
            rr.Pinhole(
                focal_length=[intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]],
                width=film_size[0],
                height=film_size[1],
                camera_xyz=rr.ViewCoordinates.LUF,
            ),
        )

        # Log rendered image if it exists
        image_path = frame["file_path"]
        full_image_path = os.path.join(images_dir, os.path.basename(image_path))

        if os.path.exists(full_image_path):
            try:
                image = Image.open(full_image_path)
                image_array = np.array(image)
                rr.log(f"{camera_path}/pinhole", rr.Image(image_array))
                print(f"Loaded image for camera {i}: {os.path.basename(image_path)}")
            except Exception as e:
                print(f"Failed to load image for camera {i}: {e}")
        else:
            print(f"Image not found: {full_image_path}")
        


def visualize_coordinate_system():
    """
    Visualize the world coordinate system
    """
    rr.log(
        "/world/axes",
        rr.Arrows3D(
            origins=np.array([[0, 0, 0]]),
            vectors=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            colors=np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]]),
        ),
    )


def main():
    parser = argparse.ArgumentParser(description="Visualize synthesis output in Rerun")
    parser.add_argument(
        "--camera_data",
        type=str,
        default="output/camera_data.json",
        help="Path to camera_data.json file",
    )
    parser.add_argument(
        "--scene_dir",
        type=str,
        default="spaceship",
        help="Path to scene directory containing models",
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        default="output/images",
        help="Path to directory containing rendered images",
    )

    args = parser.parse_args()

    # Initialize Rerun
    rr.init("synthesis-visualization", spawn=True)

    # Get current directory
    current_dir = Path(__file__).parent

    # Load camera data
    camera_data_path = current_dir / args.camera_data
    if not camera_data_path.exists():
        print(f"Camera data file not found: {camera_data_path}")
        return

    print(f"Loading camera data from: {camera_data_path}")
    camera_data = load_camera_data(str(camera_data_path))

    # Load scene models
    scene_dir_path = current_dir / args.scene_dir
    if scene_dir_path.exists():
        print(f"Loading models from: {scene_dir_path}")
        load_scene_models(str(scene_dir_path))
    else:
        print(f"Scene directory not found: {scene_dir_path}")

    # Visualize coordinate system
    visualize_coordinate_system()

    # Visualize cameras
    images_dir_path = current_dir / args.images_dir
    if images_dir_path.exists():
        print(f"Loading images from: {images_dir_path}")
        visualize_cameras(camera_data, str(images_dir_path))
    else:
        print(f"Images directory not found: {images_dir_path}")

    print("Visualization started. Open Rerun viewer to see the results.")
    print("Press Ctrl+C to stop the visualization.")


if __name__ == "__main__":
    main()
