import mitsuba as mi
import os
import argparse
import json
import numpy as np
import math
from pathlib import Path
from typing import List, Dict, Any
import random

from camera import get_camera_params, Camera

# Set Mitsuba variant
mi.set_variant("llvm_ad_rgb")


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle numpy types"""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


def generate_hemisphere_camera_position(
    center: tuple[float, float, float] = (0, 0, 0),
    normal: tuple[float, float, float] = (0, 1, 0),
    radius: float = 4.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a random camera position on a hemisphere with given radius.
    Camera always points to the center of the hemisphere.

    Args:
        center: center of the hemisphere
        normal: normal vector of the hemisphere
        radius: radius of the hemisphere

    Returns:
        tuple: (position, to_world_matrix)
    """
    center = np.array(center)
    normal = np.array(normal)
    normal = normal / np.linalg.norm(normal)  # Normalize

    while True:
        position = np.random.uniform(-radius, radius, 3)
        if (
            np.dot(position, normal) > 0
            and abs(np.linalg.norm(position) - radius) < 0.3
        ):
            break
    position = center + position

    # direction to center
    forward = (center - position) / np.linalg.norm(center - position)

    # Build rotation matrix with proper normalization
    right = np.cross(forward, normal)
    right = right / np.linalg.norm(right)  # Normalize right vector

    up = np.cross(right, forward)  # This ensures up is perpendicular to both
    up = up / np.linalg.norm(up)  # Normalize up vector

    # Ensure forward is also normalized
    forward = forward / np.linalg.norm(forward)

    # Build rotation matrix - each column should be a unit vector
    rotation = np.column_stack([right, up, forward])

    # Verify the rotation matrix is orthogonal (no scale factors)
    # This ensures R^T * R = I
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
        # If not orthogonal, use Gram-Schmidt to orthogonalize
        right = right / np.linalg.norm(right)
        up = up - np.dot(up, right) * right
        up = up / np.linalg.norm(up)
        forward = np.cross(right, up)
        rotation = np.column_stack([right, up, forward])

    # Build transformation matrix
    to_world = np.eye(4)
    to_world[:3, :3] = rotation
    to_world[:3, 3] = position

    return position, to_world


def update_scene_camera(scene, to_world_matrix: np.ndarray):
    """
    Update the scene camera with new transformation matrix
    """
    params = mi.traverse(scene)

    if "PerspectiveCamera.to_world" in params:
        params["PerspectiveCamera.to_world"] = mi.Transform4f(to_world_matrix)
    elif "ThinLensCamera.to_world" in params:
        params["ThinLensCamera.to_world"] = mi.Transform4f(to_world_matrix)
    else:
        raise ValueError("Camera type not supported")

    params.update()


def render_image(scene, spp: int = 256) -> np.ndarray:
    """
    Render an image from the scene

    Args:
        scene: Mitsuba scene
        spp: samples per pixel

    Returns:
        np.ndarray: rendered image as RGB array
    """
    image = mi.render(scene, spp=spp)
    return image.numpy()


def camera_to_json_format(
    camera: Camera, image_path: str, frame_id: int
) -> Dict[str, Any]:
    """
    Convert camera data to JSON format similar to train.json

    Args:
        camera: Camera object
        image_path: path to the rendered image
        frame_id: frame identifier

    Returns:
        Dict: camera data in JSON format
    """
    # Convert intrinsic matrix to list format
    intrinsic_matrix = camera.intrinsic_matrix.tolist()

    # Convert to_world matrix to list format
    camera_to_world_transform = camera.to_world.tolist()

    # Convert film_size to list if it's a tuple
    film_size = (
        list(camera.film_size)
        if isinstance(camera.film_size, tuple)
        else camera.film_size
    )

    # Convert principal_point_offset to list if it's a tuple
    principal_point_offset = (
        list(camera.principal_point_offset)
        if isinstance(camera.principal_point_offset, tuple)
        else camera.principal_point_offset
    )

    # Convert x_fov to float if it's a numpy array
    x_fov = float(camera.x_fov) if hasattr(camera.x_fov, "item") else camera.x_fov

    return {
        "file_path": image_path,
        "frame_id": frame_id,
        "camera_to_world_transform": camera_to_world_transform,
        "camera_intrinsic_matrix": intrinsic_matrix,
        "film_size": film_size,
        "x_fov": x_fov,
        "principal_point_offset": principal_point_offset,
        "focus_distance": float(camera.focus_distance)
        if hasattr(camera, "focus_distance")
        else 1.0,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic images with camera data"
    )
    parser.add_argument(
        "--num_images", type=int, default=10, help="Number of images to generate"
    )
    parser.add_argument(
        "--output_dir", type=str, default="output", help="Output directory"
    )
    parser.add_argument(
        "--scene_file", type=str, default="spaceship/scene.xml", help="Scene file path"
    )
    parser.add_argument(
        "--spp", type=int, default=256, help="Samples per pixel for rendering"
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=3.0,
        help="Radius of hemisphere for camera positions",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)

    # Get current directory and load scene
    current_dir = Path(__file__).parent
    scene_path = current_dir / args.scene_file

    if not scene_path.exists():
        print(f"Scene file not found: {scene_path}")
        return

    print(f"Loading scene from: {scene_path}")
    scene = mi.load_file(str(scene_path))

    # Get original camera parameters for reference
    original_camera = get_camera_params(scene)
    print(f"Original camera film size: {original_camera.film_size}")
    print(f"Original camera FOV: {original_camera.x_fov}")

    # Prepare output data structure
    output_data = {
        "camera_angle_x": float(original_camera.x_fov)
        if hasattr(original_camera.x_fov, "item")
        else original_camera.x_fov,
        "film_size": list(original_camera.film_size)
        if isinstance(original_camera.film_size, tuple)
        else original_camera.film_size,
        "frames": [],
    }

    print(f"Generating {args.num_images} images...")

    for i in range(args.num_images):
        print(f"Generating image {i + 1}/{args.num_images}")

        # Generate random camera position
        position, to_world_matrix = generate_hemisphere_camera_position(
            center=(0, 0.7, 0), normal=(0, 1, 0), radius=args.radius
        )
        print(f"Camera position: {position}")

        # Update scene camera
        update_scene_camera(scene, to_world_matrix)

        # Render image
        image = render_image(scene, args.spp)

        # Save image
        image_filename = f"image_{i:04d}.png"
        image_path = images_dir / image_filename
        mi.util.write_bitmap(str(image_path), image)

        # Get updated camera parameters
        camera = get_camera_params(scene)

        # Convert to JSON format
        frame_data = camera_to_json_format(camera, f"./images/{image_filename}", i)

        output_data["frames"].append(frame_data)

    # Save JSON data
    json_path = output_dir / "camera_data.json"
    with open(json_path, "w") as f:
        json.dump(output_data, f, indent=2, cls=NumpyEncoder)

    print(f"Generated {args.num_images} images in {output_dir}")
    print(f"Camera data saved to: {json_path}")
    print(f"Images saved to: {images_dir}")


if __name__ == "__main__":
    main()
