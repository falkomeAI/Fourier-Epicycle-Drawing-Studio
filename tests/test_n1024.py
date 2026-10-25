"""
Tests for N=1024 harmonic scheduling and point scaling.
"""

import unittest
import numpy as np

import falkourier as fk


class TestN1024(unittest.TestCase):
    def test_schedule_progression(self):
        schedule = fk.get_n_schedule(1024)
        self.assertGreater(len(schedule), 1)
        self.assertEqual(schedule[-1], 1024)

    def test_sample_points_scaling(self):
        for n in [100, 500, 1024, 2000]:
            n_pts = fk.sample_points_for_n(n)
            self.assertGreaterEqual(n_pts, 2 * n)

    def test_hummingbird_fit(self):
        n_points = fk.sample_points_for_n(500)
        points = fk.SHAPES["hummingbird"](n_points)
        pts_norm = fk.normalize_path(points, target_span=2.5)

        fx = fk.fit_fourier(pts_norm[:, 0], 500)
        fy = fk.fit_fourier(pts_norm[:, 1], 500)

        self.assertEqual(len(fx.a), 500)
        self.assertEqual(len(fy.a), 500)


if __name__ == "__main__":
    unittest.main()
