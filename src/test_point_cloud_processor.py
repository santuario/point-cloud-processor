import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import point_cloud_portrait_processor 

class TestPointCloudPortraitProcessor(unittest.TestCase):
    def setUp(self):
        """Set up a test instance of PointCloudPortraitProcessor."""
        self.processor = point_cloud_processor.PointCloudPortraitProcessor("test.ply", "test_output.pcd")

    @patch('open3d.io.read_point_cloud')
    def test_load_point_cloud(self, mock_read):
        """Test loading a point cloud."""
        # Mock a point cloud with sample points
        mock_pcd = MagicMock()
        mock_pcd.has_points.return_value = True
        mock_pcd.points = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float64)
        mock_read.return_value = mock_pcd

        self.processor.load_point_cloud()
        self.assertIsNotNone(self.processor.pcd)
        self.assertEqual(len(self.processor.pcd.points), 2)
        mock_read.assert_called_once_with("test.ply")

    def test_load_point_cloud_failure(self):
        """Test loading a point cloud that fails."""
        with patch('open3d.io.read_point_cloud') as mock_read:
            mock_pcd = MagicMock()
            mock_pcd.has_points.return_value = False
            mock_read.return_value = mock_pcd
            with self.assertRaises(ValueError):
                self.processor.load_point_cloud()

    @patch('open3d.geometry.PointCloud.remove_statistical_outlier')
    def test_clean_data(self, mock_outlier):
        """Test cleaning the point cloud."""
        # Mock a point cloud with points and colors
        mock_pcd = MagicMock()
        mock_pcd.points = np.array([[0, 0, 1], [0, 0, 3]], dtype=np.float64)
        mock_pcd.colors = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float64)
        mock_pcd.has_colors.return_value = True
        self.processor.pcd = mock_pcd

        mock_outlier.return_value = (True, [0])  # Mock outlier removal returning index 0
        self.processor.clean_data(depth_threshold=2.0)

        self.assertIsNotNone(self.processor.pcd_clean)
        points = np.asarray(self.processor.pcd_clean.points)
        self.assertEqual(len(points), 1)  # Should keep only points with Z < 2.0
        self.assertTrue(mock_pcd.remove_statistical_outlier.called)

    def test_clean_data_no_points(self):
        """Test cleaning with no point cloud loaded."""
        self.processor.pcd = None
        with self.assertRaises(ValueError):
            self.processor.clean_data()

    @patch('sklearn.cluster.DBSCAN')
    def test_segment_human(self, mock_dbscan):
        """Test segmenting the human subject."""
        # Mock a point cloud with points
        mock_pcd = MagicMock()
        mock_pcd.points = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=np.float64)
        self.processor.pcd_clean = mock_pcd

        # Mock DBSCAN to return labels with a largest cluster
        mock_cluster = MagicMock()
        mock_cluster.labels_ = np.array([0, 0, -1])  # Cluster 0 is largest, -1 is noise
        mock_dbscan.return_value = mock_cluster

        self.processor.segment_human(eps=0.05, min_samples=10)
        self.assertIsNotNone(self.processor.pcd_human)
        self.assertEqual(len(self.processor.pcd_human.points), 2)  # Should keep points in largest cluster

    def test_segment_human_no_clusters(self):
        """Test segmenting human with no valid clusters."""
        mock_pcd = MagicMock()
        mock_pcd.points = np.array([[0, 0, 0]], dtype=np.float64)
        self.processor.pcd_clean = mock_pcd

        with patch('sklearn.cluster.DBSCAN') as mock_dbscan:
            mock_cluster = MagicMock()
            mock_cluster.labels_ = np.array([-1])  # No valid clusters
            mock_dbscan.return_value = mock_cluster
            with self.assertRaises(ValueError):
                self.processor.segment_human(eps=0.05, min_samples=10)

    @patch('open3d.geometry.PointCloud.estimate_normals')
    @patch('open3d.geometry.KDTreeFlann')
    def test_segment_hair_no_color(self, mock_kdtree, mock_normals):
        """Test segmenting hair with no color data using density/curvature."""
        # Mock a point cloud without colors
        mock_pcd = MagicMock()
        mock_pcd.points = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 1]], dtype=np.float64)
        mock_pcd.has_colors.return_value = False
        self.processor.pcd_human = mock_pcd

        # Mock normals for curvature
        mock_pcd.normals = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]], dtype=np.float64)
        mock_normals.return_value = None  # No need to modify normals here

        # Mock KDTreeFlann for density
        mock_tree = MagicMock()
        mock_tree.search_knn_vector_3d.side_effect = lambda x, k: (3, [0, 1, 2], [0.0, 1.0, 2.0])  # 3 neighbors for each point
        mock_kdtree.return_value = mock_tree

        self.processor.segment_hair(eps=0.02, min_samples=5)
        self.assertIsNotNone(self.processor.pcd_hair)
        self.assertGreater(len(self.processor.pcd_hair.points), 0)

    def test_segment_hair_with_color(self):
        """Test segmenting hair with color data (simplified)."""
        mock_pcd = MagicMock()
        mock_pcd.points = np.array([[0, 0, 0]], dtype=np.float64)
        mock_pcd.colors = np.array([[0, 20, 20]], dtype=np.float64)  # Within hair range
        mock_pcd.has_colors.return_value = True
        self.processor.pcd_human = mock_pcd

        with patch('cv2.cvtColor') as mock_cvt, patch('cv2.inRange') as mock_range:
            mock_cvt.return_value = np.array([[0, 20, 20]], dtype=np.uint8)
            mock_range.return_value = np.array([[255]], dtype=np.uint8)  # Hair detected

            self.processor.segment_hair(lower_hair=(0, 0, 0), upper_hair=(30, 50, 50))
            self.assertIsNotNone(self.processor.pcd_hair)
            self.assertEqual(len(self.processor.pcd_hair.points), 1)

    @patch('open3d.geometry.PointCloud.remove_statistical_outlier')
    def test_refine_hair(self, mock_outlier):
        """Test refining the hair point cloud."""
        mock_pcd = MagicMock()
        mock_pcd.points = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float64)
        self.processor.pcd_hair = mock_pcd

        mock_outlier.return_value = (True, [0])  # Keep only first point
        self.processor.refine_hair()
        self.assertIsNotNone(self.processor.pcd_hair)
        self.assertEqual(len(self.processor.pcd_hair.points), 1)

    @patch('open3d.geometry.KDTreeFlann')
    def test_augment_hair(self, mock_kdtree):
        """Test augmenting the hair point cloud."""
        mock_pcd = MagicMock()
        mock_pcd.points = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float64)
        mock_pcd.has_colors.return_value = False
        self.processor.pcd_hair = mock_pcd

        mock_tree = MagicMock()
        mock_tree.search_radius_vector_3d.side_effect = lambda x, r: (2, [0, 1], [0.0, 1.0])  # 2 neighbors
        mock_tree.search_knn_vector_3d.side_effect = lambda x, k: (2, [0, 1], [0.0, 1.0])  # 2 neighbors
        mock_kdtree.return_value = mock_tree

        self.processor.augment_hair(smooth_radius=0.05, densify_factor=1, smooth_iterations=1)
        self.assertIsNotNone(self.processor.pcd_hair)
        self.assertGreater(len(self.processor.pcd_hair.points), 2)  # Should add points via densification

    @patch('open3d.visualization.draw_geometries')
    def test_visualize(self, mock_draw):
        """Test visualization of different targets."""
        mock_pcd = MagicMock()
        self.processor.pcd = mock_pcd
        self.processor.visualize(target="pcd")
        mock_draw.assert_called_once()

    @patch('open3d.io.write_point_cloud')
    def test_save(self, mock_write):
        """Test saving the point cloud."""
        mock_pcd = MagicMock()
        self.processor.pcd_hair = mock_pcd
        self.processor.save("test_output.pcd", target="hair")
        mock_write.assert_called_once_with("test_output.pcd", mock_pcd)

if __name__ == '__main__':
    unittest.main()