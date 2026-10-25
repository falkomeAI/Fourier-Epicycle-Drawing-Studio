"""
Unit tests for falkourier core functionality.
"""

import unittest
import numpy as np

import falkourier as fk


class TestFalkourier(unittest.TestCase):
    def test_version_and_metadata(self):
        self.assertTrue(hasattr(fk, "__version__"))
        self.assertEqual(fk.__author__, "falkomeai")

    def test_shapes_available(self):
        expected = {"hummingbird", "heart", "star", "butterfly", "infinity", "rose"}
        for shape in expected:
            self.assertIn(shape, fk.SHAPES)
            pts = fk.SHAPES[shape](n_points=128)
            self.assertEqual(pts.ndim, 2)
            self.assertEqual(pts.shape[1], 2)
            self.assertEqual(len(pts), 128)

    def test_fit_fourier(self):
        pts = fk.SHAPES["heart"](n_points=256)
        fx = fk.fit_fourier(pts[:, 0], 20)
        fy = fk.fit_fourier(pts[:, 1], 20)
        self.assertEqual(len(fx.a), 20)
        self.assertEqual(len(fy.a), 20)
        self.assertEqual(len(fx.b), 20)
        self.assertEqual(len(fy.b), 20)

    def test_fit_complex_fourier(self):
        pts = fk.SHAPES["star"](n_points=256)
        z = pts[:, 0] + 1j * pts[:, 1]
        cfit = fk.fit_complex_fourier(z)
        self.assertIsInstance(cfit, dict)
        self.assertGreater(len(cfit), 0)

    def test_schedule(self):
        n_pts = fk.sample_points_for_n(100)
        self.assertGreaterEqual(n_pts, 200)
        sched = fk.get_n_schedule(500)
        self.assertEqual(sched[-1], 500)


if __name__ == "__main__":
    unittest.main()
