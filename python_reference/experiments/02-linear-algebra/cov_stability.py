#Python implementation of cov_stability.cpp for cross language referencing using numpy's cholesky
import csv
import numpy as np
RHO = 0.3
T = 50
NS = [5,10,25,40,50,60]
OUTPUT_PATH = "python_reference/experiments/02-linear-algebra/results/cov_stability_results.csv"
#NxN matrix of each entry being rho, and fill diagonal replaces with 1's for variance
#np.ndarray specifies that the expected output is a numpy array
def true_covariance(n: int, rho:float) -> np.ndarray:
    cov = np.full((n, n), rho)
    np.fill_diagonal(cov, 1.0)
    return cov

#accumulate results in a list, write csv once at end
def main() -> None:
    rows = []
    #independent seed generator N
    for i, n in enumerate(NS):
        rng = np.random.default_rng(seed=i)
        true_cov = true_covariance(n, RHO)
        L_true = np.linalg.cholesky(true_cov)
        #this makes it two independent numerical libraries, not copies but still similar -> more useful for comparision
        ind_draws = rng.standard_normal((n, T))
        #@ = np.matmul
        corr = L_true @ ind_draws
        transpose = corr.T
        est_cov = np.cov(transpose, rowvar=False, ddof=1)
        #eigenvalsh -> numpy's solver for symmetric matrices, direct analog of selfadjointsolver
        eigenval = np.linalg.eigvalsh(est_cov)
        cond_num = eigenval[-1]/eigenval[0]
        cholesky_success = True
        try:
            np.linalg.cholesky(est_cov)
        except np.linalg.LinAlgError:
            cholesky_success = False
        novert = n/T
        rows.append({
            "N": n,
            "T": T,
            "N over T": novert,
            "condition number": cond_num,
            "cholesky success": cholesky_success,
        })
        print(f"N={n} N/T={novert} condition number={cond_num:.4f} cholesky success ={cholesky_success}")
    with open(OUTPUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Results written to cov_stability_results.csv")

if __name__ == "__main__":
    main()


