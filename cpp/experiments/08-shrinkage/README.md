**Experiment 15 (Phase 10): Fixing High-Dimensional Covariance — Shrinkage and Random Matrix Clipping**

**Objective**  
Experiment 2 showed the sample covariance degrading as the number of assets N approaches the number of observations T: the condition number climbs by orders of magnitude, and the matrix becomes singular once N reaches T. This experiment re-runs that exact setup with two estimators designed for the regime, Ledoit-Wolf shrinkage and Marchenko-Pastur eigenvalue clipping. It measures what actually matters downstream: how far each estimate is from the truth, and how good a portfolio built from it is.

**Methodology**  
*Part 1.* Same ground truth as Experiment 2: unit variances, constant correlation 0.3, T = 50 observations. N is swept over 5, 10, 25, 40, 50, 60, 75 and 100 (past N = T this time), with 200 independent replications per N instead of one. Three estimators are compared:
- **Sample:** the Bessel-corrected sample covariance from Phase 2.
- **Ledoit-Wolf:** linear shrinkage towards a scaled identity, with the intensity δ estimated from the data.
- **RMT clipping:** eigenvalues of the sample correlation matrix below the Marchenko-Pastur noise edge (1 + √(N/T))² are replaced by their average, keeping only those above it as signal.

Metrics:
- The median condition number (infinite when the matrix is numerically singular).
- How often Cholesky succeeds.
- The relative Frobenius error ‖Σ̂ − Σ‖ / ‖Σ‖.
- Out-of-sample portfolio quality: the fully invested minimum-variance portfolio is built from each estimate, its variance is computed under the *true* covariance, and that is divided by the best achievable variance. 1.0 is perfect.

*Part 2.* PCA of the correlation matrix of the real 6-stock history (752 daily returns, hardened data), with the Marchenko-Pastur edge, the Ledoit-Wolf intensity and the number of eigenvalues RMT keeps as signal.

C++ only: the Python Ledoit-Wolf matches scikit-learn's implementation to 1e-10 for every configuration tested (including N > T), and the C++ matches the Python reference in both test suites.

**Results**

Minimum-variance portfolio variance relative to the best achievable (1.00 = perfect; "—" means the estimate is singular, so the portfolio cannot be computed):

| N | N/T | Sample | Ledoit-Wolf | RMT clipping |
| :---- | :---- | :---- | :---- | :---- |
| 5 | 0.1 | 1.09 | 1.04 | 1.06 |
| 10 | 0.2 | 1.22 | 1.09 | 1.09 |
| 25 | 0.5 | 2.04 | 1.26 | 1.19 |
| 40 | 0.8 | 5.23 | 1.37 | 1.23 |
| 50 | 1.0 | 4,954 (when it ran) | 1.44 | 1.24 |
| 60 | 1.2 | — | 1.47 | 1.25 |
| 75 | 1.5 | — | 1.45 | 1.25 |
| 100 | 2.0 | — | 1.43 | 1.26 |

Conditioning and accuracy (median condition number / Cholesky success rate / mean relative error):

| N | Sample | Ledoit-Wolf | RMT clipping |
| :---- | :---- | :---- | :---- |
| 5 | 5.4 / 100% / 0.30 | 3.3 / 100% / 0.28 | 4.1 / 100% / 0.24 |
| 25 | 112 / 100% / 0.42 | 25 / 100% / 0.39 | 18 / 100% / 0.28 |
| 40 | 1,268 / 100% / 0.45 | 45 / 100% / 0.42 | 28 / 100% / 0.30 |
| 50 | ∞ / 54.5% / 0.45 | 66 / 100% / 0.41 | 36 / 100% / 0.28 |
| 100 | ∞ / 0% / 0.47 | 126 / 100% / 0.43 | 76 / 100% / 0.30 |

Ledoit-Wolf's estimated shrinkage intensity was 0.30 at N = 5 and settled between 0.21 and 0.22 from N = 25 onwards. RMT clipping kept one signal eigenvalue (the market factor) in essentially every replication (average 0.97 at N = 5, 1.00 from N = 10).

Real data (6 stocks, 752 days; Marchenko-Pastur edge 1.19; Ledoit-Wolf δ = 0.020):

| Component | Eigenvalue | Variance explained | Above noise edge | Loadings: AAPL / CAT / JNJ / JPM / PG / XOM |
| :---- | :---- | :---- | :---- | :---- |
| PC1 | 2.47 | 41.1% | yes | 0.39 / 0.46 / 0.35 / 0.50 / 0.35 / 0.38 |
| PC2 | 1.33 | 22.2% | yes | 0.17 / -0.40 / 0.50 / -0.16 / 0.57 / -0.46 |
| PC3 | 0.77 | 12.8% | no | -0.81 / 0.10 / 0.45 / -0.11 / 0.13 / 0.31 |
| PC4-6 | 0.56, 0.48, 0.39 | 23.9% together | no | |

**Discussion**  
**The damage from noise shows up long before the matrix breaks.** The sample covariance stays invertible up to N = 40, so Experiment 2's Cholesky test passes. But the minimum-variance portfolio built from it already has twice the achievable variance at N/T = 0.5 and five times at 0.8. Inverting the matrix amplifies its smallest eigenvalues, exactly the ones that sampling noise biases downward the most. The optimizer then loads up on directions that only look riskless. Conditioning is the symptom, and portfolio quality is the cost.

**Both fixes remove most of that cost, and keep working past N = T.** From N = 25 upwards, Ledoit-Wolf holds the portfolio within 26% to 47% of optimal and RMT clipping within 19% to 26%, including at twice as many assets as observations, where the sample covariance cannot be used at all. Both stay positive definite in every replication, with condition numbers in the tens rather than the thousands or infinity.

**RMT clipping wins here, but the setup favours it.** The true matrix is a one-factor model: one large eigenvalue (the market) and N - 1 identical small ones. That is exactly the structure RMT clipping assumes, and it reliably finds the one factor. Ledoit-Wolf shrinks towards a scaled identity, which has no correlation at all, so with a true correlation of 0.3 it pulls in a partly wrong direction. Its Frobenius error is barely better than the sample's (0.41-0.43 vs 0.45-0.47), even though its portfolios are far better. A shrinkage target with constant correlation would suit this truth better. With several factors of different strengths, or eigenvalues near the noise edge, clipping's all-or-nothing choice is less clean. The ranking between the two fixes should not be generalized from this one structure, but the gap to the sample covariance should.

**A refinement of Experiment 2: Cholesky success is not a reliable singularity test at N = T.** With demeaning, a sample covariance from T observations has rank at most T - 1, so at N = T = 50 it is always singular in exact arithmetic. Experiment 2's single draw failed Cholesky there, as it should. Across 200 replications, though, Eigen's Cholesky "succeeded" 54.5% of the time on rounding noise. The minimum-variance portfolios built from those matrices had about 5,000 times the achievable variance. A matrix that passes Cholesky can still be useless. The eigenvalue spectrum (here, the smallest eigenvalue relative to the largest) is the trustworthy check.

**On real data the two fixes barely act, which is the correct behaviour.** With 752 observations for 6 stocks (N/T = 0.008), Ledoit-Wolf shrinks by only 2%, and the Marchenko-Pastur edge is 1.19. Two eigenvalues sit above it. PC1 (41% of the variance) has positive loadings on every stock: the market factor. PC2 (22%) contrasts the consumer-staples and healthcare names (PG 0.57, JNJ 0.50) with the cyclical and commodity names (XOM -0.46, CAT -0.40), a defensive-versus-cyclical split. The remaining four components fall inside the noise band. With this little dimensionality the sample covariance is already reliable, and these tools matter once the universe grows to dozens or hundreds of names.

**Limitations.** Synthetic truth with a single factor and equal correlations; Gaussian returns; one T. Ledoit-Wolf uses only the identity target; the constant-correlation target and nonlinear shrinkage are not tested. The real-data part has too few assets to show the high-dimensional benefit.

**Conclusion**  
The instability Experiment 2 found is now fixed. Both Ledoit-Wolf shrinkage and Marchenko-Pastur clipping keep the covariance well conditioned at any N/T, including N > T. More importantly, they cut the excess variance of a minimum-variance portfolio from 2 to 5 times optimal (and infinite once N ≥ T) down to roughly 1.2 to 1.5 times. Any part of the engine that inverts a covariance matrix of more than a handful of assets should use one of them.

**File Paths**  
/cpp/experiments/08-shrinkage/shrinkage_experiment.cpp and csv with results (estimator_comparison, real_data_pca) in /cpp/experiments/08-shrinkage/results/
