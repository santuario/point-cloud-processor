import open3d as o3d
import numpy as np
import cv2
from sklearn.cluster import DBSCAN
import argparse

class PointCloudPortraitProcessor:
    """Class to process a 3D portrait point cloud into a photorealistic AI twin with enhanced hair."""

    def __init__(self, input_file, output_pcd="portrait_test.pcd"):
        """Initialize with input PLY file and optional output PCD path."""
        self.input_file = input_file
        self.output_pcd = output_pcd
        self.pcd = None
        self.pcd_clean = None
        self.pcd_human = None
        self.pcd_hair = None

    def load_point_cloud(self):
        """Load the PLY file as a point cloud."""
        self.pcd = o3d.io.read_point_cloud(self.input_file)
        if not self.pcd.has_points():
            raise ValueError(f"Failed to load point cloud from {self.input_file}")
        print(f"Loaded point cloud with {len(self.pcd.points)} points.")
        o3d.io.write_point_cloud(self.output_pcd, self.pcd)
        print(f"Saved as {self.output_pcd}")

    def clean_data(self, depth_threshold=2.0, nb_neighbors=20, std_ratio=2.0, voxel_size=0.01):
        """Clean the point cloud by removing background and noise, with optional downsampling."""
        if self.pcd is None:
            raise ValueError("Point cloud not loaded. Call load_point_cloud() first.")

        if voxel_size > 0:
            self.pcd = self.pcd.voxel_down_sample(voxel_size)
            print(f"Downsampled to {len(self.pcd.points)} points with voxel_size={voxel_size}")

        points = np.asarray(self.pcd.points)
        colors = np.asarray(self.pcd.colors) if self.pcd.has_colors() else None
        mask = points[:, 2] < depth_threshold
        self.pcd.points = o3d.utility.Vector3dVector(points[mask])
        if colors is not None:
            self.pcd.colors = o3d.utility.Vector3dVector(colors[mask])

        cl, ind = self.pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
        self.pcd_clean = self.pcd.select_by_index(ind)
        print(f"Cleaned point cloud to {len(self.pcd_clean.points)} points.")

    def segment_human(self, eps=0.05, min_samples=10):
        """Segment the human subject using DBSCAN with adaptive parameter tuning."""
        if self.pcd_clean is None:
            raise ValueError("Point cloud not cleaned. Call clean_data() first.")

        points = np.asarray(self.pcd_clean.points)
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
        labels = clustering.labels_

        valid_labels = labels[labels >= 0]
        if len(valid_labels) == 0:
            eps *= 1.5
            clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
            labels = clustering.labels_
            valid_labels = labels[labels >= 0]
            if len(valid_labels) == 0:
                raise ValueError("No clusters found even after tuning. Adjust parameters further.")

        largest_cluster = np.argmax(np.bincount(valid_labels))
        human_mask = labels == largest_cluster
        self.pcd_human = self.pcd_clean.select_by_index(np.where(human_mask)[0])
        print(f"Segmented human with {len(self.pcd_human.points)} points.")

    def segment_hair(self, lower_hair=(0, 20, 20), upper_hair=(30, 255, 255), eps=0.02, min_samples=5):
        """Segment hair using color or fallback to density/curvature if no colors."""
        if self.pcd_human is None:
            raise ValueError("Human not segmented. Call segment_human() first.")

        if self.pcd_human.has_colors():
            rgb = np.asarray(self.pcd_human.colors) * 255
            hsv = cv2.cvtColor(rgb.astype(np.uint8).reshape(-1, 1, 3), cv2.COLOR_RGB2HSV)
            hair_mask = cv2.inRange(hsv, np.array(lower_hair), np.array(upper_hair))
            hair_indices = np.where(hair_mask.flatten() > 0)[0]
        else:
            print("No color data. Falling back to density/curvature-based segmentation.")
            self.pcd_human.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
            points = np.asarray(self.pcd_human.points)
            normals = np.asarray(self.pcd_human.normals)
            curvature = np.linalg.norm(normals, axis=1)

            # Compute density by counting neighbors for each point
            kdtree = o3d.geometry.KDTreeFlann(self.pcd_human)
            density = np.zeros(len(points))
            for i, point in enumerate(points):
                [k, _, _] = kdtree.search_knn_vector_3d(point, 10)
                density[i] = k  # Number of neighbors within k=10

            # Define hair mask based on curvature and density thresholds
            hair_mask = (curvature > np.percentile(curvature, 70)) & (density > np.percentile(density, 60))
            hair_indices = np.where(hair_mask)[0]

        if len(hair_indices) == 0:
            raise ValueError("No hair points detected. Adjust color range or curvature thresholds.")
        
        self.pcd_hair = self.pcd_human.select_by_index(hair_indices)
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(np.asarray(self.pcd_hair.points))
        labels = clustering.labels_
        if len(labels[labels >= 0]) > 0:
            largest_cluster = np.argmax(np.bincount(labels[labels >= 0]))
            self.pcd_hair = self.pcd_hair.select_by_index(np.where(labels == largest_cluster)[0])
        print(f"Segmented hair with {len(self.pcd_hair.points)} points.")

    def refine_hair(self, nb_neighbors=15, std_ratio=1.5):
        """Refine hair by removing noise."""
        if self.pcd_hair is None:
            raise ValueError("Hair not segmented. Call segment_hair() first.")

        cl, ind = self.pcd_hair.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
        self.pcd_hair = self.pcd_hair.select_by_index(ind)
        print(f"Refined hair to {len(self.pcd_hair.points)} points.")

    def augment_hair(self, smooth_radius=0.05, densify_factor=2, smooth_iterations=3):
        """Augment hair for a more natural look by smoothing and densifying."""
        if self.pcd_hair is None:
            raise ValueError("Hair not refined. Call refine_hair() first.")

        points = np.asarray(self.pcd_hair.points)
        kdtree = o3d.geometry.KDTreeFlann(self.pcd_hair)
        smoothed_points = np.copy(points)
        
        for _ in range(smooth_iterations):
            for i, point in enumerate(points):
                [k, idx, _] = kdtree.search_radius_vector_3d(point, smooth_radius)
                if k > 1:
                    neighbor_points = points[idx]
                    smoothed_points[i] = np.mean(neighbor_points, axis=0)
            points = smoothed_points.copy()
        
        self.pcd_hair.points = o3d.utility.Vector3dVector(smoothed_points)

        colors = np.asarray(self.pcd_hair.colors) if self.pcd_hair.has_colors() else None
        new_points = []
        new_colors = []
        for i, point in enumerate(points):
            [k, idx, _] = kdtree.search_knn_vector_3d(point, 5)
            neighbors = points[idx]
            for _ in range(densify_factor):
                noise = np.random.normal(0, 0.005, 3)
                new_point = np.mean(neighbors, axis=0) + noise
                new_points.append(new_point)
                if colors is not None:
                    new_colors.append(colors[i])

        all_points = np.vstack((points, new_points))
        self.pcd_hair.points = o3d.utility.Vector3dVector(all_points)
        if colors is not None:
            all_colors = np.vstack((colors, new_colors))
            self.pcd_hair.colors = o3d.utility.Vector3dVector(all_colors)
        print(f"Augmented hair to {len(self.pcd_hair.points)} points.")

    def visualize(self, target="hair", window_name="Point Cloud Visualization"):
        """Render the point cloud at different stages."""
        targets = {"pcd": self.pcd, "clean": self.pcd_clean, "human": self.pcd_human, "hair": self.pcd_hair}
        if targets[target] is not None:
            o3d.visualization.draw_geometries([targets[target]], window_name=window_name)
        else:
            print(f"No {target} data available to visualize.")

    def visualize_clean_vs_enhanced(self, window_name="Cleaned vs Enhanced Hair"):
        """Visualize the cleaned point cloud and enhanced hair in the same window."""
        if self.pcd_clean is None or self.pcd_hair is None:
            raise ValueError("Cleaned or hair point cloud not available. Ensure clean_data() and augment_hair() are called.")

        pcd_clean_vis = o3d.geometry.PointCloud(self.pcd_clean)
        pcd_hair_vis = o3d.geometry.PointCloud(self.pcd_hair)

        clean_colors = np.zeros((len(pcd_clean_vis.points), 3))
        clean_colors[:, 2] = 1.0  # Blue
        pcd_clean_vis.colors = o3d.utility.Vector3dVector(clean_colors)

        if not pcd_hair_vis.has_colors():
            hair_colors = np.zeros((len(pcd_hair_vis.points), 3))
            hair_colors[:, 0] = 1.0  # Red
            pcd_hair_vis.colors = o3d.utility.Vector3dVector(hair_colors)

        o3d.visualization.draw_geometries([pcd_clean_vis, pcd_hair_vis], window_name=window_name)

    def save(self, filename, target="hair"):
        """Save the processed point cloud."""
        targets = {"pcd": self.pcd, "human": self.pcd_human, "hair": self.pcd_hair}
        if targets[target] is not None:
            o3d.io.write_point_cloud(filename, targets[target])
            print(f"Saved {target} point cloud to {filename}")
        else:
            print(f"No {target} data available to save.")

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Process a 3D point cloud portrait with enhanced hair.")
    parser.add_argument("input_file", help="Path to input PLY file")
    parser.add_argument("--output_file", default="processed_portrait_hair.pcd", help="Path to output PCD file")
    parser.add_argument("--depth_threshold", type=float, default=2.0, help="Depth threshold for background removal")
    parser.add_argument("--voxel_size", type=float, default=0.01, help="Voxel size for downsampling (0 to disable)")
    parser.add_argument("--eps_human", type=float, default=0.05, help="DBSCAN eps for human segmentation")
    parser.add_argument("--min_samples_human", type=int, default=10, help="DBSCAN min_samples for human segmentation")
    parser.add_argument("--eps_hair", type=float, default=0.02, help="DBSCAN eps for hair segmentation")
    parser.add_argument("--min_samples_hair", type=int, default=5, help="DBSCAN min_samples for hair segmentation")
    parser.add_argument("--lower_hair", nargs=3, type=int, default=[0, 20, 20], help="Lower HSV hair color range")
    parser.add_argument("--upper_hair", nargs=3, type=int, default=[30, 255, 255], help="Upper HSV hair color range")
    parser.add_argument("--smooth_radius", type=float, default=0.05, help="Radius for hair smoothing")
    parser.add_argument("--densify_factor", type=int, default=2, help="Factor for hair densification")
    parser.add_argument("--smooth_iterations", type=int, default=3, help="Number of smoothing iterations")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    processor = PointCloudPortraitProcessor(args.input_file, output_pcd="temp.pcd")

    processor.load_point_cloud()
    processor.clean_data(depth_threshold=args.depth_threshold, voxel_size=args.voxel_size)
    processor.segment_human(eps=args.eps_human, min_samples=args.min_samples_human)
    processor.segment_hair(
        lower_hair=args.lower_hair,
        upper_hair=args.upper_hair,
        eps=args.eps_hair,
        min_samples=args.min_samples_hair
    )
    processor.refine_hair()
    processor.augment_hair(
        smooth_radius=args.smooth_radius,
        densify_factor=args.densify_factor,
        smooth_iterations=args.smooth_iterations
    )

    #processor.visualize(target="human", window_name="Cleaned")

    processor.visualize_clean_vs_enhanced(window_name="Cleaned vs Enhanced Hair")
    processor.save(args.output_file, target="hair")