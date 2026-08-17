#Reason to write this in Python: Wanted to add an actual plot for visual purposes, much easier in Python than C++
#Goal is just to see whether error shrinks as N grows as it is an earlier implementation

import csv
import numpy as np

#Fix parameters as language changes
MU = 0.0
SIGMA = 1.0
OUTPUT_PATH = "python_reference/experiments/01-probability/results/convergence_results.csv"

def ana_mean(mu: float, sigma: float) -> float:
    return mu
def ana_var(mu:float, sigma:float) -> float:
    return sigma ** 2

def main() -> None:
    ana_mean_val = ana_mean(MU, SIGMA)
    ana_var_val = ana_var(MU, SIGMA)

    #Geometric N increase
    ns = [10 ** k for k in range(1,7)]
    rows = []
    for i,n in enumerate(ns):
        #As discussed independent seeded gen for N to maintain congruence between py and cpp
        rng = np.random.default_rng(seed=i)
        samples = rng.normal(loc=MU, scale=SIGMA, size=n)
        emp_mean = np.mean(samples)
        #ddof = 1 for n -1 to maintain congruence
        emp_var = np.var(samples, ddof=1)
        mean_error = abs(emp_mean - ana_mean_val)
        var_error = abs(emp_var - ana_var_val)
        rows.append({
            "N": n,
            "empirical_mean": emp_mean,
            "analytical_mean": ana_mean_val,
            "mean_abs_error": mean_error,
            "empirical_var": emp_var,
            "analytical_var": ana_var_val,
            "var_abs_error": var_error,
        })
        #Fix error to floating point with 6 decimal places for accuracy
        print(f"N={n}  mean_error={mean_error:.6f}  var_error={var_error:.6f}")
        #Write CSV with same column order as C++ version -> difference between two files is meaningful
    with open(OUTPUT_PATH, "w", newline="") as f:
            #Use keys from first row for uniform column headers
        writer = csv.DictWriter(f,fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults written to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()