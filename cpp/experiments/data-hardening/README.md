**Experiment 8: Data Hardening — How Much Do Data Errors Distort VaR/ES, and What Does Hardening Recover?**

**Objective**  
Phase 7 computes risk from price history, so its numbers are only as good as that history. This experiment measures how much four realistic data errors distort the Phase 7 risk numbers, and how much of the damage the hardened data layer (`qre/data/csv_loader.hpp`, `corporate_actions.hpp`, `quality.hpp`, `point_in_time.hpp`) detects and repairs. It also quantifies survivorship bias from the old practice of dropping assets with partial histories.

**Methodology**  
Start from the real 6-stock history used in Phase 3 and Phase 7 (AAPL, CAT, JNJ, JPM, PG, XOM, 2021-01-04 to 2023-12-29, equal weights). That data passes the quality report with no flags, so it serves as the clean baseline. Inject one corruption at a time, then all four together:

| Scenario | Injected error |
| :---- | :---- |
| Unadjusted split | AAPL 4-for-1 split left unadjusted from 2022-08-05 (prices divided by 4 from then on) |
| Bad tick | XOM printed 35% too high for one day, 2021-10-19 |
| Stale run | JNJ frozen at the prior close for 10 trading days from 2022-03-14 |
| Gap | PG missing for 3 trading days from 2022-12-28 |

Each corrupted history goes through two treatments. **Naive** runs the Phase 7 pipeline directly; its only defence is a short forward fill. **Hardened** runs `harden()` first, which removes non-positive prices and spike-and-revert bad ticks and fills short gaps only inside each stock's listed life. It then back-adjusts the split candidates the quality report flags, standing in for an analyst confirming them against a corporate-actions feed, and then runs the same pipeline. Metrics are 1-day 99% historical VaR and ES, parametric VaR and the worst portfolio day, each compared with the clean baseline.

Survivorship check: CAT is made to "delist" after 2023-05-22. The risk of the portfolio actually held that day is estimated over the previous 500 days two ways. One uses the point-in-time universe (`universe_at`: every stock with complete data in the window). The other uses the Phase 3 practice of dropping any asset missing too much of the full history (`drop_missing_assets`, 5% threshold).

C++ only: every hardening function is cross-validated against the Python reference (`python_reference/qre/data/hardening.py`). The C++ and Python quality reports on the same corrupted fixture are identical, as asserted in both test suites.

**Results**

1-day 99% risk, with error against the clean baseline (historical VaR 2.525%, historical ES 2.891%, parametric VaR 2.306%, worst day -3.43%):

| Scenario | Treatment | Hist. VaR | Hist. ES | Param. VaR | Worst day | Flagged / repaired |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| Unadjusted split | Naive | +2.3% | **+39.5%** | +8.9% | -11.66% | — |
| | Hardened | 0.0% | 0.0% | 0.0% | -3.43% | 1 split flagged and confirmed |
| Bad tick | Naive | +2.3% | +4.9% | +4.1% | -3.66% | — |
| | Hardened | 0.0% | 0.0% | -0.01% | -3.43% | 1 bad tick removed and filled |
| Stale run | Naive | 0.0% | 0.0% | -0.08% | -3.43% | — |
| | Hardened | 0.0% | 0.0% | -0.08% | -3.43% | 1 stale run flagged, not changed |
| Gap | Naive | 0.0% | 0.0% | 0.00% | -3.43% | — |
| | Hardened | 0.0% | 0.0% | 0.00% | -3.43% | 3 days filled |
| All combined | Naive | +5.7% | **+43.9%** | +12.6% | -11.66% | — |
| | Hardened | 0.0% | 0.0% | -0.09% | -3.43% | all of the above |

Survivorship (portfolio held on 2023-05-23, 500-day window):

| Approach | Stocks used | Hist. VaR 99% | Param. VaR 99% |
| :---- | :---- | :---- | :---- |
| Point-in-time universe | 6 (what was held) | 2.703% | 2.506% |
| Drop assets with partial history | 5 (CAT removed) | 2.483% (-8.1%) | 2.408% (-3.9%) |

**Discussion**  
**An unadjusted split is the dangerous error, and it hides from VaR.** One fake -75% AAPL day turns into an -11.7% portfolio day, more than three times the real worst day. Historical VaR barely notices (+2.3%), because at 99% with 752 days it sits on roughly the eighth-worst loss and a single extra extreme day only shifts it one rank. ES averages the tail, so it jumps 39.5%. A risk report that monitors only VaR would miss this error, which is a practical argument for always reporting ES alongside it. The parametric figure rises 8.9% because one huge return inflates the sample variance.

**Detection is automatic; correction of the ambiguous case is not.** The quality report flagged the split with the exact 4:1 ratio, and applying it restored the baseline exactly. The hardening code deliberately does not apply split candidates itself: a genuine -50% crash is numerically identical to an unadjusted 2-for-1 split (there is a test for this). Here the "confirmation" step stood in for a corporate-actions feed. The exact recovery depends on the true ratio being known, which a real feed would supply.

**Bad ticks cost twice.** A spike that reverts creates two extreme returns, the jump and the fall back, and the fall back is a large loss. Removing the tick and filling it from the prior close recovered the baseline to within 0.01%.

**Stale runs and short gaps barely mattered here, and the report says why not to relax.** Ten frozen days in JNJ, the lowest-volatility stock, moved parametric VaR by 0.08%. Stale prices understate volatility and correlation, so the effect grows with longer runs, more volatile assets and illiquid instruments. Because no information exists to recover a stale price, the hardened layer flags it and leaves the decision to the user. The 3-day gap was already handled by Phase 7's forward fill, so the hardening adds nothing there. Its value for missing data is elsewhere: it never fabricates prices after a delisting, and it reports how much it filled.

**Dropping partial histories understates the risk of what you actually held.** Removing CAT from the whole history because it delisted later produces risk numbers for a 5-stock portfolio nobody held. Its historical 99% VaR is 8.1% lower than the 6-stock portfolio's. This is survivorship bias in miniature: the names that vanish tend to be the ones that caused losses, so dropping them flatters every backtest built on top. The point-in-time universe keeps each stock exactly as long as it was observable.

**Limitations.** Each corruption was injected once, at one date, with one magnitude, so the percentages are illustrative rather than general rates. The data was already adjusted by the vendor, so the corporate-actions code was exercised on injected errors, not on a real raw feed. The survivorship case is a single synthetic delisting. Point-in-time correctness covers price data only: fundamentals and index membership would need their own as-of history.

**Conclusion**  
On clean data the hardened layer changes nothing: the Phase 7 experiment re-run through it reproduced its results byte for byte. On corrupted data it recovers the clean risk numbers to within 0.1% in every scenario, and it flags rather than guesses on the two errors it cannot resolve alone (splits and stale prices). The unadjusted-split result is the headline: one bad corporate action raised ES by 40% while VaR moved 2%. The data layer is now trustworthy enough to serve as ground truth for the backtesting module, which comes next.

**File Paths**  
/cpp/experiments/data-hardening/corruption_injection.cpp and csv with results in /cpp/experiments/data-hardening/results/
