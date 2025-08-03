# Synthesis CLI Tool

A command-line tool for generating synthetic images with camera data using Mitsuba renderer.

## Features

- Generate synthetic images from 3D scenes
- Position cameras randomly on a hemisphere (r=4m) pointing to origin
- Export camera data in JSON format compatible with NeRF training
- Support for both PerspectiveCamera and ThinLensCamera
- Reproducible results with random seed
- **Visualization tool** for viewing results in Rerun

## Installation

```bash
# Install dependencies using uv
uv sync
```

## Usage

### Basic Usage

```bash
# Generate 10 images with default settings
uv run python main.py

# Generate 50 images with custom settings
uv run python main.py --num_images 50 --output_dir my_dataset --spp 512 --radius 5.0

# Use a specific random seed for reproducibility
uv run python main.py --num_images 20 --seed 42

# Use a different scene file
# uv run python main.py --scene_file my_scene/scene.xml
```

### Visualization

```bash
# Visualize the synthesis output in Rerun
uv run python vis.py

# Visualize with custom parameters
uv run python vis.py --camera_data my_dataset/camera_data.json --scene_dir my_scene
```

The visualization tool will:

- Load and display the 3D models from the scene
- Show camera positions and orientations
- Display rendered images for each camera
- Show the world coordinate system

### Command Line Arguments

#### Main Tool (`main.py`)

- `--num_images`: Number of images to generate (default: 10)
- `--output_dir`: Output directory for images and JSON data (default: "output")
- `--scene_file`: Path to the scene XML file (default: "spaceship/scene.xml")
- `--spp`: Samples per pixel for rendering (default: 256)
- `--radius`: Radius of hemisphere for camera positions (default: 4.0)
- `--seed`: Random seed for reproducibility (default: None)

#### Visualization Tool (`vis.py`)

- `--camera_data`: Path to camera_data.json file (default: "output/camera_data.json")
- `--scene_dir`: Path to scene directory containing models (default: "spaceship")
- `--images_dir`: Path to directory containing rendered images (default: "output/images")

## Output Format

The tool generates:

1. **Images**: PNG files saved in `output_dir/images/`
2. **Camera Data**: JSON file `camera_data.json` with the following structure:

```json
{
  "camera_angle_x": 60.0,
  "film_size": [1280, 720],
  "frames": [
    {
      "file_path": "./images/image_0000.png",
      "frame_id": 0,
      "camera_to_world_transform": [
        [0.768, 0.64, 0.025, -0.101],
        [-0.64, 0.768, 0.03, -0.121],
        [0.0, 0.039, -0.999, 3.997],
        [0.0, 0.0, 0.0, 1.0]
      ],
      "camera_intrinsic_matrix": [
        [1108.51, 0.0, 640.0],
        [0.0, 1108.51, 360.0],
        [0.0, 0.0, 1.0]
      ],
      "film_size": [1280, 720],
      "x_fov": 60.0,
      "principal_point_offset": [0, 0],
      "focus_distance": 3.0
    }
  ]
}
```

## Camera Positioning

Cameras are positioned randomly on a hemisphere with:

- **Center**: Configurable center point (default: origin)
- **Normal**: Configurable normal vector defining the hemisphere orientation (default: Z-up)
- **Radius**: Configurable radius (default: 4 meters)
- **Direction**: Always pointing to the center of the hemisphere

### Examples

```bash
# Default: hemisphere centered at origin, opening upward
uv run python main.py --num_images 10

# Another settings fine quality
uv run python main.py --num_images 50 --spp 512

```
