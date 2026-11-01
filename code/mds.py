#!/usr/bin/env python3
"""Explicit STRAIN/classical MDS; see README.md for conventions and formulas."""
import argparse
import csv
from decimal import Decimal, InvalidOperation
import json
import math
import os
from pathlib import Path
import sys
import tempfile

import numpy as np


def parse_q(text):
    """Return (float exponent, canonical decimal filename token)."""
    if text == 'inf':
        return math.inf, 'inf'
    try:
        value = Decimal(text)
    except InvalidOperation as exc:
        raise argparse.ArgumentTypeError('Q must be a real >= 1 or literal inf') from exc
    if not value.is_finite() or value < 1:
        raise argparse.ArgumentTypeError('Q must be a real >= 1 or literal inf')
    q = float(value)
    if not math.isfinite(q):
        raise argparse.ArgumentTypeError('finite Q exceeds float64 range; use literal inf')
    # Fixed decimal notation, independent of Decimal context precision.
    token = format(value, 'f')
    if '.' in token:
        token = token.rstrip('0').rstrip('.')
    token = token.lstrip('+')
    token = token.lstrip('0') or '0'
    return q, token


def read_features(path):
    """Read all columns, optional all-text header; never select/drop label columns."""
    with Path(path).open(newline='', encoding='utf-8-sig') as handle:
        rows = list(csv.reader(handle))
    if not rows or not rows[0]:
        raise ValueError('CSV is empty')
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError('CSV must be rectangular with no blank rows')
    def numeric(cell):
        try:
            float(cell)
            return True
        except ValueError:
            return False
    flags = [numeric(cell) for cell in rows[0]]
    if not any(flags):
        rows = rows[1:]
    elif not all(flags):
        raise ValueError('first row must be numeric or an entirely nonnumeric header')
    try:
        x = np.array(rows, dtype=np.float64)
    except ValueError as exc:
        raise ValueError('all data cells must be numeric; remove labels upstream explicitly') from exc
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 1:
        raise ValueError('need at least two rows and one numeric feature')
    if not np.isfinite(x).all():
        raise ValueError('all features must be finite')
    return x


def distances(x, q):
    """Minkowski norm via max rescaling; no powers of unscaled differences."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or min(x.shape) < 1 or not np.isfinite(x).all():
        raise ValueError('expected a nonempty finite feature matrix')
    if not (q >= 1) or (not math.isfinite(q) and q != math.inf):
        raise ValueError('q must be >= 1 or positive infinity')
    n = len(x)
    d = np.zeros((n, n))
    with np.errstate(over='raise', invalid='raise', under='ignore'):
        for i in range(n - 1):
            delta = np.abs(x[i + 1:] - x[i])
            maximum = delta.max(axis=1)
            if q == math.inf:
                norm = maximum
            else:
                ratios = np.divide(delta, maximum[:, None], out=np.zeros_like(delta),
                                   where=maximum[:, None] != 0)
                norm = maximum * np.power(np.power(ratios, q).sum(axis=1), 1.0 / q)
            d[i, i + 1:] = norm
            d[i + 1:, i] = norm
    if not np.isfinite(d).all():
        raise ValueError('distances exceed float64 range')
    return d


def classical_mds(d, k):
    """Handout Eq. (6), symmetric EVD, descending PSD rank-k truncation."""
    d = np.asarray(d, dtype=np.float64)
    if d.ndim != 2 or d.shape[0] != d.shape[1] or len(d) < 2:
        raise ValueError('distance matrix must be square, n >= 2')
    if k not in (2, 3) or k > len(d):
        raise ValueError('dimensions must be 2 or 3 and no greater than row count')
    if not np.isfinite(d).all() or (d < 0).any():
        raise ValueError('distances must be finite and nonnegative')
    if not np.array_equal(d, d.T) or np.any(np.diag(d) != 0):
        raise ValueError('distance matrix must be symmetric with zero diagonal')
    with np.errstate(over='raise', invalid='raise'):
        a = d * d
        # Divide before summing to reduce overflow in the means.
        row_mean = (a / len(d)).sum(axis=1)
        grand_mean = (row_mean / len(d)).sum()
        gram = -0.5 * a + 0.5 * row_mean[:, None] + 0.5 * row_mean[None, :] - 0.5 * grand_mean
        gram = 0.5 * gram + 0.5 * gram.T
    values, vectors = np.linalg.eigh(gram)
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    tolerance = 100 * np.finfo(float).eps * len(d) * float(np.max(np.abs(values)))
    positive = values > tolerance
    y = np.zeros((len(d), k))
    count = min(k, int(positive.sum()))
    y[:, :count] = vectors[:, :count] * np.sqrt(values[:count])
    if not np.isfinite(y).all():
        raise ValueError('coordinates exceed float64 range')
    return y, values, tolerance, gram


def diagnostics(d, y, values, tolerance, gram):
    positive = values[values > tolerance]
    neg = values < -tolerance
    raw_neg = values < 0
    fitted = distances(y, 2)
    upper = np.triu_indices(len(d), 1)
    # Scaling cancels in the ratio and prevents squaring large raw distances.
    scale = float(d.max())
    if scale == 0:
        nrmse = sq_nrmse = 0.0
    else:
        target = d[upper] / scale
        estimate = fitted[upper] / scale
        nrmse = float(np.linalg.norm(target - estimate) / np.linalg.norm(target))
        sq_nrmse = float(np.linalg.norm(target**2 - estimate**2) / np.linalg.norm(target**2))
    spectral_scale = float(np.max(np.abs(values)))
    if spectral_scale == 0:
        strain = 0.0
    else:
        residual = values / spectral_scale
        residual = residual.copy()
        selected = min(y.shape[1], len(positive))
        residual[:selected] = 0
        strain = float(np.linalg.norm(residual) / np.linalg.norm(values / spectral_scale))
    total = float(positive.sum())
    return {
        'eigenvalues_descending': values.tolist(), 'eigenvalue_tolerance': tolerance,
        'positive_count': int(len(positive)),
        'negative_count': int(neg.sum()), 'negative_mass': float(-values[neg].sum()),
        'raw_negative_count': int(raw_neg.sum()), 'raw_negative_mass': float(-values[raw_neg].sum()),
        'negligible_negative_count': int(((values < 0) & ~neg).sum()),
        'positive_mass': total,
        'positive_mass_fractions': {str(k) + 'D': float(positive[:k].sum() / total) if total else None for k in (2, 3)},
        'normalized_distance_error': nrmse,
        'normalized_squared_distance_error': sq_nrmse,
        'normalized_gram_error': strain,
        'effective_dimensions': min(y.shape[1], len(positive)),
        'zero_padded_dimensions': max(0, y.shape[1] - len(positive)),
    }


def plot_coordinates(y, path, title):
    os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'mds-matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    k = y.shape[1]
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection='3d' if k == 3 else None)
    ax.scatter(*[y[:, j] for j in range(k)], s=22, color='#236da1', marker='o', alpha=0.8)
    ax.set_xlabel('MDS coordinate 1')
    ax.set_ylabel('MDS coordinate 2')
    midpoint = (y.min(axis=0) + y.max(axis=0)) / 2
    radius = float(np.ptp(y, axis=0).max()) / 2
    if radius == 0:
        radius = 0.5
    radius *= 1.05
    ax.set_xlim(midpoint[0] - radius, midpoint[0] + radius)
    ax.set_ylim(midpoint[1] - radius, midpoint[1] + radius)
    if k == 3:
        ax.set_zlabel('MDS coordinate 3')
        ax.set_zlim(midpoint[2] - radius, midpoint[2] + radius)
        ax.set_box_aspect((1, 1, 1))
    else:
        ax.set_aspect('equal', adjustable='box')
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--q', required=True, type=parse_q)
    parser.add_argument('--dimensions', required=True, type=int, choices=(2, 3))
    args = parser.parse_args(argv)
    try:
        if args.input.suffix.lower() != '.csv':
            raise ValueError('input must have a .csv extension')
        x = read_features(args.input)
        q, token = args.q
        d = distances(x, q)
        y, values, tolerance, gram = classical_mds(d, args.dimensions)
        report = diagnostics(d, y, values, tolerance, gram)
        stem = f'{args.input.stem}_mds_q{token}_{args.dimensions}D'
        csv_path, png_path = Path(stem + '.csv'), Path(stem + '.png')
        plot_coordinates(y, png_path, f'{args.input.stem}: classical MDS, q={token}, {args.dimensions}D')
        np.savetxt(csv_path, y, delimiter=',', header=','.join(f'MDS{i+1}' for i in range(args.dimensions)), comments='', fmt='%.17g')
        report.update(input=str(args.input), rows=x.shape[0], features=x.shape[1], q=token,
                      dimensions=args.dimensions, csv=str(csv_path), png=str(png_path))
        print(json.dumps(report, allow_nan=False))
    except (OSError, ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == '__main__':
    sys.exit(main())
