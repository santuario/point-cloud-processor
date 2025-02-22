# Point Cloud Processor for 3D AI Twin Creation

This project provides an object-oriented Python script to process RGBD point cloud data from a depth camera, aiming to create a photorealistic 3D AI twin with a focus on accurate hair representation. The script cleans the raw data, segments the human subject, isolates the hair, and refines it for downstream modeling.

## Features
- **Data Cleaning**: Removes background and noise from the point cloud.
- **Human Segmentation**: Isolates the human subject using clustering.
- **Hair Segmentation**: Extracts hair based on color (HSV) properties.
- **Hair Refinement**: Enhances hair detail by filtering noise and regularizing points.
- **Visualization**: Displays intermediate and final results.
- **Modular Design**: Easy to extend or modify for specific needs.

## Requirements
- Python 3.8+
- Libraries:
  - `open3d` (point cloud processing)
  - `numpy` (array operations)
  - `opencv-python` (image processing)
  - `scikit-learn` (clustering)
- Install with:
  ```bash
  pip install open3d numpy opencv-python scikit-learn