import mitsuba as mi
from dataclasses import dataclass
import numpy as np
import math
import rerun as rr


def get_intrinsic_matrix(film_size: tuple[int, int], x_fov: float) -> np.ndarray:
    """
    Get 3x3 intrinsic matrix from film size and x_fov
    Args:
        film_size: tuple[int, int] - film size in pixels
        x_fov: float - field of view in degrees
    Returns:
        np.ndarray: 3x3 intrinsic matrix
    """
    # Convert Mitsuba types to Python types
    film_width = float(film_size[0])
    film_height = float(film_size[1])
    x_fov_float = float(x_fov)

    half_angle_rad = x_fov_float / 2 * math.pi / 180
    focal_length = film_width / (2 * math.tan(half_angle_rad))
    intrinsic_matrix = np.array(
        [
            [focal_length, 0, film_width / 2],
            [0, focal_length, film_height / 2],
            [0, 0, 1],
        ]
    )
    return intrinsic_matrix


@dataclass
class Camera:
    film_size: tuple[int, int]
    x_fov: float
    principal_point_offset: tuple[float, float]
    focus_distance: float

    # 4x4 transformation matrix from camera to world
    to_world: np.ndarray
    # 3x3 intrinsic matrix
    intrinsic_matrix: np.ndarray

    def get_pos(self) -> tuple[float, float, float]:
        """
        Get camera position
        """
        return self.to_world[:3, 3]

    def get_rpy(self) -> tuple[float, float, float]:
        """
        Get camera RPY (roll, pitch, yaw)
        """
        # get rotation matrix
        rotation_matrix = self.to_world[:3, :3]
        # get roll, pitch, yaw
        roll = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        pitch = math.atan2(
            -rotation_matrix[2, 0],
            math.sqrt(rotation_matrix[2, 1] ** 2 + rotation_matrix[2, 2] ** 2),
        )
        yaw = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        return roll, pitch, yaw

    def as_rerun_inhole(self) -> rr.Pinhole:
        """
        Convert camera to rerun pinhole
        """
        print("focus distance: ", self.focus_distance)
        print("focus distance type: ", type(self.focus_distance))
        print("shape of focus distance: ", self.focus_distance.shape)

        return rr.Pinhole(
            focal_length=[self.intrinsic_matrix[0, 0], self.intrinsic_matrix[1, 1]],
            width=self.film_size[0],
            height=self.film_size[1],
            camera_xyz=rr.ViewCoordinates.LUF,
        )


def to_world_matrix_to_rerun_transform(to_world: np.ndarray) -> rr.Transform3D:
    """
    Convert Mitsuba to_world matrix to rerun transform
    """
    print(to_world)
    print("to world type: ", type(to_world))
    print("to world shape: ", to_world.shape)
    print("to world type: ", type(to_world))
    return rr.Transform3D(translation=to_world[:3, 3], mat3x3=to_world[:3, :3])


def get_camera_params(scene) -> Camera:
    params = mi.traverse(scene)

    if "PerspectiveCamera.x_fov" in params:
        film_size = params["PerspectiveCamera.film.size"].numpy()
        x_fov = params["PerspectiveCamera.x_fov"].numpy()

        # convert to tuple
        film_size = (film_size[0], film_size[1])

        camera = Camera(
            film_size=film_size,
            x_fov=x_fov,
            focus_distance=1,
            principal_point_offset=(
                params["PerspectiveCamera.principal_point_offset_x"],
                params["PerspectiveCamera.principal_point_offset_y"],
            ),
            to_world=np.array(params["PerspectiveCamera.to_world"].matrix()),
            intrinsic_matrix=get_intrinsic_matrix(film_size, x_fov),
        )
        return camera

    elif "ThinLensCamera.x_fov" in params:
        file_size = params["ThinLensCamera.film.size"].numpy()
        x_fov = params["ThinLensCamera.x_fov"].numpy()

        # convert to tuple
        film_size = (file_size[0], file_size[1])
        camera = Camera(
            film_size=film_size,
            x_fov=x_fov,
            focus_distance=params["ThinLensCamera.focus_distance"].numpy().squeeze(),
            principal_point_offset=(0, 0),
            to_world=params["ThinLensCamera.to_world"].matrix.numpy().squeeze(),
            intrinsic_matrix=get_intrinsic_matrix(film_size, x_fov),
        )
        return camera
    else:
        raise ValueError("Camera type not supported")
