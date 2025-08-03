import mitsuba as mi
import os
import argparse
import rerun as rr
from PIL import Image
import numpy as np


from synthesis.camera import get_camera_params, to_world_matrix_to_rerun_transform

mi_variants = mi.variants()
print(mi_variants)

mi.set_variant("llvm_ad_rgb")

# get current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# scene directory
scene_dir = os.path.join(current_dir, "spaceship")

scene = mi.load_file(os.path.join(scene_dir, "scene.xml"))

# init rerun

rr.init("synthesis", spawn=True)

# log all the *.obj files in the scene directory
obj_dir = os.path.join(scene_dir, "models")
for file in os.listdir(obj_dir):
    if file.endswith(".obj"):
        # entity path prefix does not matter
        rr.log_file_from_path(file_path=os.path.join(obj_dir, file), entity_path_prefix=obj_dir, static=True)

# params = mi.traverse(scene)
# print(params)

# print("Camera file size: ", params["PerspectiveCamera.film.size"])
# print("Camera xFov: ", params["PerspectiveCamera.x_fov"])
# print("Camera principal point offset x: ", params["PerspectiveCamera.principal_point_offset_x"])
# print("Camera principal point offset y: ", params["PerspectiveCamera.principal_point_offset_y"])

# print("Camera to world: ", params["PerspectiveCamera.to_world"])

camera = get_camera_params(scene)
print("Camera: ", camera)



print("Camera position: ", camera.get_pos())
print("Camera RPY: ", camera.get_rpy())

# log xyz coordinates
rr.log(
    "/xyz",
    rr.Arrows3D(
        origins=np.array([[0, 0, 0]]),
        vectors=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        colors=np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]]),
    ),
)

# log camera
rr.log("/camera", to_world_matrix_to_rerun_transform(camera.to_world))


# log xyz coordinates
rr.log(
    "/camera/xyz",
    rr.Arrows3D(
        origins=np.array([[0, 0, 0]]),
        vectors=np.array([[0.3, 0, 0], [0, 0.3, 0], [0, 0, 0.3]]),
        colors=np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]]),
    ),
)

rr.log("/camera/pinhole", camera.as_rerun_inhole())

# log a image to numpy array
image = Image.open(os.path.join(scene_dir, "TungstenRender.png"))
image_array = np.array(image)
rr.log("/camera/pinhole/image", rr.Image(image_array))

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--render", action="store_true")
args = arg_parser.parse_args()

if args.render:
    image = mi.render(scene, spp=256)
    mi.util.write_bitmap("my_first_render.png", image)
    mi.util.write_bitmap("my_first_render.exr", image)
