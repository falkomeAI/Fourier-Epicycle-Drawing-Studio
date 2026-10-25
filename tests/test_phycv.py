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
    def test_phycv_pst_cv_enhancements(self):
        img = np.zeros((120, 120, 3), dtype=np.uint8)
        cv2.circle(img, (60, 60), 35, (255, 255, 255), 3)
        # Add a tiny 2-pixel isolated speckle
        img[10:12, 10:12] = 255

        mask_clean = _mask_phycv_pst(img, clahe=True, bilateral=True, close_gaps=True, min_area=20, thin=False)
        self.assertEqual(mask_clean.shape, (120, 120))
        self.assertGreater(np.count_nonzero(mask_clean), 0)

        # Test skeleton thinning
        mask_thin = _mask_phycv_pst(img, clahe=True, bilateral=True, thin=True)
        self.assertEqual(mask_thin.shape, (120, 120))
        self.assertGreater(np.count_nonzero(mask_thin), 0)
        # Thinning should produce fewer or equal pixels than dilated mask
        self.assertLessEqual(np.count_nonzero(mask_thin), np.count_nonzero(mask_clean))

    def test_phycv_custom_pst_parameters(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.circle(img, (50, 50), 30, (255, 255, 255), 2)
        mask = _mask_phycv_pst(
            img, S=0.5, W=18.0, sigma_LPF=0.10, thresh_min=0.03, thresh_max=0.85, morph_flag=1
        )
        self.assertEqual(mask.shape, (100, 100))
        self.assertGreater(np.count_nonzero(mask), 0)


if __name__ == "__main__":
    unittest.main()
