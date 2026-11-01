# Explicit STRAIN / classical multidimensional scaling

The supplied five-page **Multidimensional-Scaling.pdf** was read before implementation and is retained under `reference/`. It is the mathematical specification; its text is treated as reference material, not instructions overriding the request. Equation (6), the PSD rank-K truncation on pages 3–4, and the Euclidean/PCA observation on page 5 determine the implementation. No MDS, classical-scaling, kernel-PCA, or PCA routine is called.

## Run

Requires Python 3.10+, NumPy and Matplotlib (`python -m pip install -r requirements.txt`). From this submission's root:

```sh
cd results
python ../code/mds.py --input ../data/Iris.csv --q 2 --dimensions 2 > ../verification/Iris_q2_2D.stdout.json
python ../code/mds.py --input ../data/Iris.csv --q 2 --dimensions 3 > ../verification/Iris_q2_3D.stdout.json
```

The exact CLI is also available from the program directory:

```sh
cd code
python mds.py --input ../data/Iris.csv --q 2 --dimensions 2
```

Only `--input`, `--q`, and `--dimensions` are required. Outputs go to the **current working directory**, not the input directory. Each successful run writes a coordinate CSV and PNG. Existing files with the same names are replaced. The input basename loses only its final `.csv` extension; capitalization and underscores are preserved. For example, `My_Data.csv --q 2.00 --dimensions 3` produces `My_Data_mds_q2_3D.csv` and `.png`. Decimal exponents are rendered without exponent notation or trailing fractional zeros; literal `inf` becomes `qinf`. CSVs have `MDS1,...,MDSK` headers, no row index, and 17 significant digits.

## Algorithm and numerical policy

1. Keep the original numeric feature matrix and row order. Do not scale, standardize, normalize columns, reorder rows, or use class labels.
2. For each pair, let `m = max_l abs(x_il-x_jl)`. For finite real `q >= 1`, compute `m * (sum_l (abs(x_il-x_jl)/m)**q)**(1/q)`, taking zero for duplicate rows. For literal `inf`, use `m`. The ratios lie in [0,1], avoiding overflow from powers at large q; underflow of tiny powered ratios is harmless at float64 precision. This internal norm evaluation does not rescale the feature matrix or output embedding.
3. Square distances once: `A = D**2`. Use the handout's Equation (6): `G_ij = -A_ij/2 + rowmean_i/2 + rowmean_j/2 - grandmean/2`, equivalent to `-J A J / 2`, where `J = I - 11^T/n`. Symmetrize with `(G + G.T)/2` before the symmetric eigendecomposition.
4. Sort all eigenvalues and corresponding eigenvectors in descending order. Define `rho = max_i abs(lambda_i)` and `tau = 100 * n * eps_float64 * rho`. The factor 100 provides a conservative round-off margin for forming and diagonalizing the dense matrix; this is a numerical heuristic, not a proof that every small negative eigenvalue arises from round-off. Unlike a fixed absolute cutoff, tau has squared-distance units and changes quadratically with distance scale. If G is zero, tau is zero.
5. Values `< -tau` are meaningful negatives; values in `[-tau,0)` are negligible negatives. Values `> tau` count as numerically positive; values with absolute magnitude <= tau are treated as zero for coordinates. Retain at most the first K positive eigenpairs and form `Y[:,r] = sqrt(lambda_r) * v_r`. **No eigenvalue absolute values, coordinate normalization, or fitted scaling.** If fewer than K positives exist, append zero coordinate columns, still producing exactly K columns. All negative eigenvalues are excluded. Meaningful negatives indicate a non-Euclidean distance geometry; the embedding remains the PSD rank-K STRAIN approximation, rather than an exact distance reconstruction. Dropping tiny positive modes is the stated numerical interpretation of the handout's exact zero rule.

Float64 arithmetic is assumed. Finite q is rounded to float64 for distance computation, with the user-supplied decimal retained for the normalized filename. Finite decimal q outside float64 range is rejected (literal `inf` is supported). Inputs whose differences, squared distances, Gram computation or coordinates exceed float64 representability are rejected. Subnormal-scale quantities can lose precision or underflow; arbitrary precision is not claimed. Dense storage costs O(n²), eigendecomposition O(n³), and pairwise distances O(n²p). Degenerate eigenvectors can differ across numerical libraries; rotation, reflection and eigenvector signs do not change the embedding geometry.

## Input assumptions and validation

Input is a comma-separated rectangular feature matrix in UTF-8 (optional BOM), with either no header or one entirely nonnumeric header. Every subsequent cell must be numeric and finite; blank rows, missing cells, mixed text/data rows and label columns are rejected, not silently dropped. Numeric columns are all treated as features: numeric IDs or numeric class codes cannot be inferred automatically, so the caller must supply a feature-only file. A numeric-looking header is interpreted as data. At least two observations and one feature are required; K is exactly 2 or 3 and must not exceed the row count, consistent with K <= n in the handout. Infinite, NaN and q < 1 arguments are rejected; infinity is accepted only as the literal lowercase `inf`.

The supplied project `data/Iris.csv` is already feature-only: 150 rows, with `sepal_length,sepal_width,petal_length,petal_width` headers. It was copied byte-for-byte from `/workspace/data-rep-project1/data/Iris.csv`. Its hash is in `verification/Iris_alignment.json`. No external Iris replacement was downloaded. The reference handout was uploaded as `Multidimensional-Scaling.pdf`, although the request refers to `MDS.pdf`; these were assumed to mean the same attached handout.

Plots use a single uniform marker style, labeled coordinate axes and title, with equal units on all axes. In 3D both the numeric axis ranges and the display box aspect are equal. No class labels or class-specific colors are used. “Labeled visualization” is interpreted as axis/title labels, not point labels or class annotations.

## Machine-readable diagnostics

Successful stdout is a single JSON object, including **all n raw eigenvalues** in descending order without truncation, tau, effective/zero-padded dimensions, and:

- `negative_count`, `negative_mass`: number and sum of absolute eigenvalues strictly below `-tau`.
- `raw_negative_count`, `raw_negative_mass`: the same over all raw values below zero, including round-off; `negligible_negative_count` counts those in `[-tau,0)`.
- `positive_mass`: sum of eigenvalues above tau. `positive_mass_fractions["2D"]` and `["3D"]`: sum of the first 2 or 3 positive eigenvalues divided by all positive mass, irrespective of requested K. With zero positive mass these fractions are JSON null.
- `normalized_distance_error`: `sqrt(sum_(i<j) (D_ij - dY_ij)^2 / sum_(i<j) D_ij^2)`, where `dY` is **Euclidean** distance in the K-dimensional embedding, including when input q is not 2.
- `normalized_squared_distance_error`: `sqrt(sum_(i<j) (D_ij^2 - dY_ij^2)^2 / sum_(i<j) D_ij^4)`.
- `normalized_gram_error`: `||G - Y Y^T||_F / ||G||_F`, evaluated from the omitted raw eigenvalues. This is normalized STRAIN in Gram space, not a distance-minimizing fit.

The distance errors are diagnostics, not objectives minimized by this implementation; they are defined explicitly to avoid ambiguity in STRESS conventions. Common scale factors cancel before squared-error computation for numerical stability. For the all-zero distance matrix, all these errors are defined as zero. Invalid invocations exit with code 2 and argparse messages on stderr.

## Retained validation

```sh
OPENBLAS_NUM_THREADS=1 python -m unittest discover -s tests -v
OPENBLAS_NUM_THREADS=1 python verification/verify_iris.py
```

`verification/tests.txt` retains **11 passing tests**: unit square distances and spectrum (1,1,0,0), handout triangle's exact Gram matrix, q=1 square spectrum (2,2,0,-1), infinity and fractional q, large q, duplicate rows, rank deficiency and zero data, quadratic tolerance scaling, invalid CSV/CLI cases, filename normalization and successful CLI exports, and independent Iris validation. SVD is used only in independent verification, never in the MDS algorithm.

For independent verification, center the unchanged Iris features once, obtain `U,S,Vt` by SVD, and compare saved MDS coordinates with `U[:,:K]*S[:K]`. Fit an orthogonal alignment only (reflections allowed), without scaling or translation. `verification/Iris_alignment.json` retains relative Frobenius alignment errors of 1.52e-15 in 2D and 1.62e-15 in 3D, with maximum absolute errors below 1e-14.

| Iris q=2 | 2D | 3D |
|---|---:|---:|
| Positive mass fraction | 0.9776317750 | 0.9948169145 |
| Normalized distance error | 0.0418266315 | 0.0122210004 |
| Normalized squared-distance error | 0.0231858532 | 0.0065301574 |
| Normalized Gram error | 0.0193777135 | 0.0055954186 |

Both runs use tau = 2.0966604269e-9 and have zero meaningful negative eigenvalues. The full raw spectrum (including negligible negatives) is retained in each `verification/Iris_q2_*D.stdout.json`. Both coordinate CSVs and PNGs are in `results/`.
