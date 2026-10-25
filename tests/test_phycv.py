"""
Unit tests for PhyCV physics-inspired edge detection.
"""

import unittest
import numpy as np
import cv2

from falkourier.image_path import _mask_phycv_pst, _mask_canny


class TestPhyCV(unittest.TestCase):
    def test_phycv_pst_mask_shape(self):
        # Create a synthetic test image with a circle
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.circle(img, (50, 50), 30, (255, 255, 255), 2)
        
        mask = _mask_phycv_pst(img)
        self.assertEqual(mask.shape, (100, 100))
        self.assertEqual(mask.dtype, np.uint8)
        # Should detect edges of the circle
        self.assertGreater(np.count_nonzero(mask), 0)

    def test_phycv_pst_grayscale_input(self):
        gray = np.zeros((80, 80), dtype=np.uint8)
        cv2.rectangle(gray, (20, 20), (60, 60), 255, 2)
        
        mask = _mask_phycv_pst(gray)
        self.assertEqual(mask.shape, (80, 80))
        self.assertGreater(np.count_nonzero(mask), 0)


if __name__ == "__main__":
    unittest.main()
