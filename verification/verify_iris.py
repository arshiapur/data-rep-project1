"""Recheck saved CSVs against centered SVD; no PCA/MDS library routines."""
import hashlib
import json
from pathlib import Path
import platform
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
x = np.loadtxt(ROOT/'data/Iris.csv', delimiter=',', skiprows=1)
xc = x - x.mean(axis=0)
u, s, vt = np.linalg.svd(xc, full_matrices=False)
report = {'input_sha256': hashlib.sha256((ROOT/'data/Iris.csv').read_bytes()).hexdigest(),
          'python': platform.python_version(), 'numpy': np.__version__,
          'shape': list(x.shape), 'centered_svd_eigenvalues': (s*s).tolist(),
          'alignment': 'orthogonal only, reflections allowed; no fitted scale or translation', 'runs': {}}
for k in (2, 3):
    y = np.loadtxt(ROOT/f'results/Iris_mds_q2_{k}D.csv', delimiter=',', skiprows=1)
    expected = u[:, :k] * s[:k]
    left, _, right = np.linalg.svd(y.T @ expected)
    aligned = y @ (left @ right)
    error = float(np.linalg.norm(aligned-expected)/np.linalg.norm(expected))
    np.testing.assert_allclose(aligned, expected, atol=1e-11, rtol=1e-11)
    report['runs'][str(k)+'D'] = {'relative_frobenius_alignment_error': error,
        'maximum_absolute_alignment_error': float(np.abs(aligned-expected).max()), 'passed': True}
print(json.dumps(report, indent=2))
