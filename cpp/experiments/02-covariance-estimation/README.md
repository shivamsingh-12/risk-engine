**Experiment 2: Sample Covariance Stability as N approaches T**

**Objective**  
Characterize the numerical stability of sample covariance estimation as N grows relative to a fixed number of observations (T), then determine at what N/T ratio the estimator’s conditioning degrades to the point of numerical failure.

**Motivation**  
Sample covariance estimation is foundational to almost every component of this project such as portfolio variance, VaR/ES, Monte Carlo. A covariance matrix that is not positive definite corrupts computation on top of it. That’s why this experiment establishes where that boundary lies.

**Methodology**  
A ground truth NxN covariance matrix sigma\_true with constant pairwise correlation and unit variance was made. Then its Cholesky factor L\_true was computed and used to transform NxT independent standard normal draws into synthetic returns with the true correlation structure.

The *sample* covariance sigma\_est was then estimated via the Bessel-corrected estimator, also independent of the ground truth used for generation. Two diagnostics were computed on est at each N:

1. Condition Number: Ratio fo largest to smallest eigenvalue using Eigen::SelfAdjointEigenSolver in C++ and np.linalg.eigvalsh in Python. A well conditioned matrix has a condition number near 1\.  
2. Cholesky decomposability: Whether LLT/np.linalg.cholesky succeeds on sigma\_est (not sigma\_true) whether sigma\_est remains strictly positive-definite from finite, noisy data.  
   

T was held fixed at 50 observations. N was swept across 5-60, spanning ratios from 0.1 to 1.2. The experiment was implemented independently in C++ 23 (Eigen/LLT/std::mt19937\_64) and Python (NumPy/LAPACK bindings, PCG64) as cross validation methodology.

**C++ 23 (Eigen)**

| N/T | Condition Number | Cholesky Succeeded |
| :---- | :---- | :---- |
| 0.1 | 3.11 | ✓ |
| 0.2 | 23.00 | ✓ |
| 0.5 | 121.61 | ✓ |
| 0.8 | 1,465.21 | ✓ |
| 1.0 | **1.44 × 10¹⁷** | ✗ |
| 1.2 | −2.48 × 10¹⁶ | ✗ |

**Python (NumPy/LAPACK)**

| N/T | Condition Number | Cholesky Succeeded |
| :---- | :---- | :---- |
| 0.1 | 4.90 | ✓ |
| 0.2 | 15.07 | ✓ |
| 0.5 | 193.91 | ✓ |
| 0.8 | 1,079.72 | ✓ |
| 1.0 | **6.14 × 10¹⁶** | ✗ |
| 1.2 | −2.21 × 10¹⁶ | ✗ |

**Discussion**  
**Conditioning degrades monotonically and predictably as N/T approaches 1\.** Both implementations show condition number climbing smoothly across N/T \= 0.1 through 0.8 from single digits to roughly three orders of magnitude, even while using unrelated PRNG engines and eigensolvers. This shows degradation is a property of the estimator itself under a shrinking effective sample size regime, not a bug in either codebase.

**The transition at N/T \= 1.0 is a genuine rank-deficiency event, not a mindless drift.** Condition number jumps 13-14 orders of magnitude in both languages which is consistent with theory: a sample covariance matrix estimated from T observations has rank at most T. Once N \>= T, sigma\_est is guaranteed to be rank-deficient, which is why Cholesky decomposition fails at and beyond this threshold, at the identical N/T ratio.

**The negative condition numbers observed at N/T \= 1.2 are a floating-point artifact.**   
A condition number is mathematically non negative. The negative values recorded here are from catastrophic cancellation in the eigenvalue solver: once N \> T, sigma\_est is rank-deficient and several eigenvalues are zero in arithmetic, but finite floating-point precision produces values landing marginally on either side of zero. So dividing by a near zero number gets a meaningless negative ratio, and provides more as a diagnostic signal.

**Cross-language agreement on the failure threshold, not just the trend, is the strongest evidence here.** Both Eigen::LLT and np.linalg.cholesky – independent LAPACK adjacent implementations fail at exact N/T \= 1.0. A shared bug across two unrelated third party linear algebra backends is improbable, strong evidence that the observed failure boundary reflects math behavior of the sample covariance estimator rather than an artifact of either codebase.

**Conclusion**  
Sample covariance estimation, as implemented in qre::linalg::sample\_covariance and its Python reference counterpart, behaves reliably while N remains below T and degrades predictably as N approaches T. This is consistent with the well known rank-deficient property of sample covariance matrices in the high dimensional or N \>= T regime. 

**File Paths**  
cpp/include/qre/linalg/cholesky.hpp and cpp/include/qre/linalg/covariance.hpp
