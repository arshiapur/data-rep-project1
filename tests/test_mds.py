import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('mds', ROOT / 'code/mds.py')
mds = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mds)


class MdsTests(unittest.TestCase):
    def setUp(self):
        self.square = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])

    def test_square_spectrum_and_distances(self):
        d = mds.distances(self.square, 2)
        expected = np.array([[0, 1, 2**.5, 1], [1, 0, 1, 2**.5],
                             [2**.5, 1, 0, 1], [1, 2**.5, 1, 0]])
        np.testing.assert_allclose(d, expected)
        y, w, tol, g = mds.classical_mds(d, 2)
        np.testing.assert_allclose(w, [1, 1, 0, 0], atol=1e-14)
        np.testing.assert_allclose(g, (self.square-.5) @ (self.square-.5).T, atol=1e-14)
        np.testing.assert_allclose(mds.distances(y, 2), d, atol=1e-14)
        np.testing.assert_allclose(y.mean(axis=0), 0, atol=1e-14)

    def test_handout_triangle(self):
        x = np.array([[0., 0.], [4., 0.], [0., 3.]])
        y, w, t, g = mds.classical_mds(mds.distances(x, 2), 2)
        np.testing.assert_allclose(g, np.array([[25,-23,-2],[-23,73,-50],[-2,-50,52]])/9, atol=1e-14)
        np.testing.assert_allclose(w, [12.96414799648327, 3.7025186701834, 0], atol=1e-12)

    def test_q_one_and_meaningful_negatives(self):
        d = mds.distances(self.square, 1)
        self.assertEqual(d[0, 2], 2)
        y, w, t, g = mds.classical_mds(d, 2)
        np.testing.assert_allclose(w, [2, 2, 0, -1], atol=1e-14)
        report = mds.diagnostics(d, y, w, t, g)
        self.assertEqual(report['negative_count'], 1)
        self.assertAlmostEqual(report['negative_mass'], 1)
        self.assertAlmostEqual(report['normalized_gram_error'], 1/3)

    def test_inf_and_fractional_q(self):
        x = np.array([[0., 0.], [3., 4.]])
        self.assertEqual(mds.distances(x, np.inf)[0, 1], 4)
        self.assertAlmostEqual(mds.distances(x, 1.5)[0, 1], (3**1.5+4**1.5)**(1/1.5))
        y, w, t, g = mds.classical_mds(mds.distances(self.square, np.inf), 2)
        np.testing.assert_allclose(w, [.5, .5, .5, 0], atol=1e-14)

    def test_large_q_and_duplicate_rows(self):
        x = np.array([[0., 0.], [1e100, 2e100], [0., 0.]])
        self.assertAlmostEqual(mds.distances(x, 1000000)[0, 1]/2e100, 1)
        self.assertEqual(mds.distances(x, 1000000)[0, 2], 0)

    def test_rank_deficient_and_zero_data(self):
        x = np.array([[0.], [1.], [2.]])
        y, w, t, g = mds.classical_mds(mds.distances(x, 2), 3)
        np.testing.assert_array_equal(y[:, 1:], 0)
        y, w, t, g = mds.classical_mds(np.zeros((3, 3)), 3)
        report = mds.diagnostics(np.zeros((3, 3)), y, w, t, g)
        self.assertEqual(report['normalized_distance_error'], 0)
        self.assertIsNone(report['positive_mass_fractions']['2D'])

    def test_tolerance_scales_quadratically(self):
        d = mds.distances(self.square, 1)
        _, w, t, _ = mds.classical_mds(d, 2)
        _, w2, t2, _ = mds.classical_mds(d * 1e-8, 2)
        np.testing.assert_allclose(w2, w * 1e-16, atol=1e-30)
        self.assertAlmostEqual(t2 / t, 1e-16, delta=1e-29)

    def test_normalization(self):
        for text in ('2', '2.0', '2.00', '02.000', '2e0', '+2.0'):
            self.assertEqual(mds.parse_q(text), (2., '2'))
        self.assertEqual(mds.parse_q('1.2500'), (1.25, '1.25'))
        self.assertEqual(mds.parse_q('inf'), (np.inf, 'inf'))

    def test_invalid_csv_and_arguments(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'bad.csv'
            for content in ('', 'a,b\n1,2\n3\n', 'a,b\n1,nan\n2,3\n',
                            'a,b\n1,inf\n2,3\n', 'a,b\n1,label\n2,label\n', '1,2\n'):
                path.write_text(content)
                with self.assertRaises(ValueError):
                    mds.read_features(path)
            for q in ('0', '-1', 'nan', 'Infinity', 'INF', '1e999', 'bad'):
                result = subprocess.run([sys.executable, str(ROOT/'code/mds.py'), '--input', str(path), '--q', q, '--dimensions', '2'], capture_output=True, text=True)
                self.assertEqual(result.returncode, 2, (q, result.stderr))
            for args in ([], ['--input', str(path), '--q', '2', '--dimensions', '4'],
                         ['--input', str(Path(tmp)/'missing.csv'), '--q', '2', '--dimensions', '2']):
                result = subprocess.run([sys.executable, str(ROOT/'code/mds.py'), *args], capture_output=True, text=True)
                self.assertEqual(result.returncode, 2)

    def test_cli_outputs_names_rows_and_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'My_Square.csv'
            np.savetxt(path, self.square, delimiter=',')
            for q in ('2.00', 'inf'):
                result = subprocess.run([sys.executable, str(ROOT/'code/mds.py'), '--input', str(path), '--q', q, '--dimensions', '3'], cwd=tmp, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                report = json.loads(result.stdout)
                token = '2' if q == '2.00' else 'inf'
                self.assertEqual(report['csv'], f'My_Square_mds_q{token}_3D.csv')
                self.assertTrue((Path(tmp)/report['png']).is_file())
                y = np.loadtxt(Path(tmp)/report['csv'], delimiter=',', skiprows=1)
                self.assertEqual(y.shape, (4, 3))
                if token == '2':
                    np.testing.assert_allclose(mds.distances(y, 2), mds.distances(self.square, 2), atol=1e-14)

    def test_iris_independent_centered_svd(self):
        x = mds.read_features(ROOT/'data/Iris.csv')
        centered = x - x.mean(axis=0)
        u, s, _ = np.linalg.svd(centered, full_matrices=False)
        for k in (2, 3):
            y, w, tol, g = mds.classical_mds(mds.distances(x, 2), k)
            reference = u[:, :k] * s[:k]
            left, _, right = np.linalg.svd(y.T @ reference)
            aligned = y @ (left @ right)
            np.testing.assert_allclose(aligned, reference, atol=1e-11, rtol=1e-11)
            np.testing.assert_allclose(w[:4], s**2, rtol=1e-12)
            np.testing.assert_allclose(g, centered @ centered.T, atol=1e-12)


if __name__ == '__main__':
    unittest.main(verbosity=2)
