**Experiment 27 (Phase 22): Streaming Risk — Updating VaR on Every Tick**

**Objective**  
Everything so far computes risk once a day from closing prices. A trading desk wants its risk as prices move: the portfolio's value, P&L and VaR updated on every tick and pushed to a screen. This phase adds a streaming engine, `qre_stream`, built from four parts:
- **Incremental risk.** A tick in one asset updates the parametric and historical VaR in O(N + T) work, instead of recomputing them in O(N² + NT) (N assets, T historical scenarios).
- **A lock-free queue.** A single-producer, single-consumer ring buffer passes ticks from the feed thread to the risk thread with no locks.
- **A latency histogram.** Fixed-size and allocation-free, so it can time every tick.
- **A WebSocket and HTTP server.** Written from scratch over POSIX sockets, including the SHA-1 and base64 the handshake needs. It pushes snapshots to the browser.

The experiment measures what each design choice buys:
1. How much faster is an incremental update than recomputing, and how does that scale?
2. Do the incremental updates drift away from the true values?
3. What does a lock-free queue gain over a mutex?
4. What latency and CPU cost does the whole pipeline have?

**Methodology**  
*Part 1, cost per tick.* Synthetic books of N = 10 to 1,000 assets with T = 250 and 1,000 scenarios of daily returns, from a three-factor model with idiosyncratic noise; one asset in seven is held short. 20,000 random ticks, each moving one asset's price, are applied repeatedly. Three costs are timed:
- the incremental update alone
- the update plus a full snapshot (parametric VaR and ES, historical VaR and ES by selection, component VaR for every asset)
- a full recomputation that produces the same snapshot from scratch

*Part 2, drift.* A 100-asset book receives 10 million ticks with no resynchronization. At checkpoints, the running variance, exposures, scenario losses and historical VaR are compared with a recomputation from scratch.

*Part 3, queues.* Two threads, one producing and one consuming:
- **Maximum throughput:** 20 million messages (5 million for the mutex queue) as fast as possible; three runs each.
- **Latency under load:** the producer is paced at 10,000, 100,000 and 1,000,000 ticks per second. The consumer applies each tick to a 100-asset book and records the time from creation to update. Each setting runs three times and the run with the median p99 is reported.

The mutex queue is a `std::deque` behind a `std::mutex`.

*Part 4, the pipeline.* `qre_stream`'s pipeline on the six bundled stocks (equal weights, $1 million), fed by its synthetic feed at 1,000, 10,000 and 100,000 ticks per second and at full speed. Each setting runs for 4 seconds in each of two modes:
- **sleep-idle (the default):** the risk thread spins briefly when the queue is empty, then sleeps 50 microseconds at a time.
- **busy-poll:** it never sleeps.

Reported: tick-to-update latency percentiles and the CPU used, in cores.

All timings come from the 2-core build machine (shared, virtualized, GCC 13, Release build). On two cores the producer and consumer each have one core, and anything else the machine does interrupts one of them, which dominates the tails (p99 and beyond). Medians and throughput are reliable; tails would be much lower on a dedicated machine with isolated cores.

The C++ tests check:
- the incremental values against a full recomputation after every tick (to 10⁻¹²)
- the C++ against the Python reference (`python_reference/qre/stream/incremental.py`)
- the queues across threads (2 million messages, in order)
- the histogram against known percentiles
- SHA-1 against its published test vectors, the handshake against the example in RFC 6455, frames at every length encoding, the origin check, and a live server round trip on a real socket
- the whole pipeline fed from a stream of ticks against a direct computation

**Results**

Part 1: nanoseconds per tick:

| Assets | Scenarios | Incremental update | Update + snapshot | Full recomputation | Speed-up (update) | Speed-up (with snapshot) |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| 10 | 250 | 100 | 619 | 1,719 | 17× | 2.8× |
| 100 | 250 | 135 | 684 | 7,269 | 54× | 10.6× |
| 250 | 250 | 175 | 1,216 | 24,508 | 140× | 20.2× |
| 500 | 250 | 320 | 1,425 | 120,734 | 377× | 84.7× |
| 1,000 | 250 | 645 | 2,953 | 462,853 | 717× | 156.7× |
| 100 | 1,000 | 431 | 3,062 | 36,463 | 85× | 11.9× |
| 1,000 | 1,000 | 1,271 | 5,419 | 601,667 | 473× | 111.0× |

Part 2: relative error of the running values after N ticks, no resynchronization (prices moved by up to 3.7 times their starting level):

| Ticks | Variance | Exposures | Scenario losses | Historical VaR |
| :---- | :---- | :---- | :---- | :---- |
| 1,000 | 3.2 × 10⁻¹⁵ | 7.1 × 10⁻¹⁶ | 2.5 × 10⁻¹⁵ | 1.3 × 10⁻¹⁶ |
| 100,000 | 6.5 × 10⁻¹⁵ | 4.3 × 10⁻¹⁵ | 1.8 × 10⁻¹⁴ | 2.6 × 10⁻¹⁶ |
| 1,000,000 | 4.8 × 10⁻¹⁴ | 9.4 × 10⁻¹⁵ | 8.8 × 10⁻¹⁴ | 3.6 × 10⁻¹⁴ |
| 10,000,000 | 4.6 × 10⁻¹⁴ | 1.2 × 10⁻¹⁴ | 2.2 × 10⁻¹³ | 5.4 × 10⁻¹⁴ |

Part 3: queues. Maximum throughput over three runs: SPSC 119 to 188 million messages per second, mutex 3.6 to 4.3 million (27 to 53 times fewer).

Latency from tick to updated risk, consumer updating a 100-asset book (nanoseconds):

| Queue | Ticks per second | p50 | p99 | p99.9 | Mean |
| :---- | :---- | :---- | :---- | :---- | :---- |
| SPSC | 10,000 | 552 | 29,440 | 860,160 | 4,045 |
| Mutex | 10,000 | 1,360 | 32,512 | 663,552 | 5,013 |
| SPSC | 100,000 | 460 | 115,712 | 3,178,496 | 12,429 |
| Mutex | 100,000 | 1,584 | 45,568 | 4,030,464 | 15,935 |
| SPSC | 1,000,000 | 452 | 123,904 | 811,008 | 5,528 |
| Mutex | 1,000,000 | 18,176 | 3,244,032 | 3,768,320 | 150,269 |

Part 4: `qre_stream` pipeline, six holdings:

| Mode | Target ticks/s | Achieved ticks/s | CPU (cores) | p50 | p99 | p99.9 |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| sleep-idle | 1,000 | 1,000 | 0.21 | 59 μs | 130 μs | 0.9 ms |
| sleep-idle | 10,000 | 9,999 | 0.22 | 58 μs | 120 μs | 0.6 ms |
| sleep-idle | 100,000 | 99,989 | 0.26 | 60 μs | 219 μs | 10.6 ms |
| busy-poll | 1,000 | 1,000 | 1.06 | 0.8 μs | 12.7 μs | 1.2 ms |
| busy-poll | 10,000 | 9,999 | 1.06 | 0.6 μs | 62 μs | 1.1 ms |
| busy-poll | 100,000 | 99,990 | 1.09 | 1.1 μs | 9.9 μs | 0.3 ms |
| either | as fast as possible | 2.7 million | 1.98 | 23 ms | 33 ms | 34 ms |

**Discussion**  
**Incremental updating changes the cost from quadratic to linear, and the gap grows with the book.** A tick costs 100 to 650 nanoseconds with 250 scenarios, across a 100-fold range of book sizes: the update touches one row of the covariance matrix and one column of scenarios. Recomputing costs O(N² + NT): from 1.7 microseconds at 10 assets to 463 microseconds at 1,000, where the incremental update is 717 times faster. Producing a full snapshot narrows the gap, because the historical VaR needs a selection over all T scenarios and component VaR touches every asset. Even so, at 1,000 assets it is still 157 times faster than recomputing. A desk that publishes a few snapshots a second while applying every tick gets close to the update-only figure.

**The rounding drift is negligible.** After 10 million ticks, during which prices wandered by up to 3.7 times, every running value agreed with a recomputation to within 3 × 10⁻¹³ (relative). The errors grow only slowly, as accumulated rounding should, and stay about a billion times below anything that matters for risk. `qre_stream` still resynchronizes every 100,000 ticks (under half a millisecond of work at 1,000 assets), but the experiment shows that is insurance rather than a necessity.

**The lock-free queue is 27 to 53 times faster and keeps its median flat under load.** At maximum speed, the SPSC queue moved 119 to 188 million messages a second and the mutex queue about 4 million: every mutex operation is an atomic read-modify-write plus possible contention, while each SPSC operation is one plain write and one release store. Under load with real work per tick, the SPSC median stayed at 0.45 to 0.55 microseconds from 10,000 to 1,000,000 ticks a second. That is mostly the 100-asset update itself (about 0.15 μs) plus moving a cache line between cores. The mutex median grew from 1.4 μs to 18 μs. At a million ticks a second the mutex queue was saturated: its 99th percentile reached 3.2 milliseconds, because ticks waited in a growing backlog, while the SPSC queue kept up. At the lower rates the tails of both queues were similar and large (tens of microseconds at p99, up to a few milliseconds at p99.9). On a shared two-core machine those tails come from the operating system interrupting one of the two busy threads, not from the queue.

**Latency is a choice paid for in CPU.** The default sleep-idle mode used about a quarter of a core at any rate up to 100,000 ticks a second, with a median latency near 60 microseconds, set by the 50-microsecond sleep. Busy-polling cut the median to under 1.1 microseconds, a factor of about 60, at the cost of one full core, whatever the tick rate. A trading system dedicates the core; a laptop app should not, which is why sleep-idle is the default and `--busy-poll` an option. At full speed the feed produced 2.7 million ticks a second, faster than the risk thread could take them with its latency recording, so the queue filled and every tick waited behind 65,536 others for 23 to 34 milliseconds. That is what a saturated pipeline looks like. The cure is backpressure or conflation, keeping only the latest price per asset, which the queue does not do.

**Limitations.**
- **Machine:** one shared two-core virtual machine, so the tails are a property of the machine and the experiment should be rerun on dedicated hardware for tail numbers.
- **Feed:** synthetic. Real market data arrives in bursts, and a real feed handler would parse a binary protocol.
- **Risk model:** parametric and historical VaR are updated incrementally; the GARCH and EVT models of earlier phases are refitted daily, not per tick.
- **Server:** serves one machine. Origin checks and a localhost-only socket are its whole security model, and it has no TLS.

**Conclusion**  
- **Speed:** updating risk incrementally makes per-tick VaR cheap, under a microsecond for books of up to 1,000 assets, against up to 0.6 milliseconds to recompute.
- **Accuracy:** the updates stay accurate to 10⁻¹³ over 10 million ticks.
- **Queue:** a lock-free single-producer, single-consumer queue moves messages 27 to 53 times faster than a mutex and holds a sub-microsecond median at a million ticks a second.
- **Trade-off:** the remaining choice is between latency and CPU: 60 microseconds on a quarter core, or one microsecond on a full core.

**File Paths**  
/cpp/experiments/streaming/streaming_experiment.cpp and csv with results (tick_cost, drift, queues, pipeline) in /cpp/experiments/streaming/results/. The engine: /cpp/include/qre/stream/ (queue.hpp, latency.hpp, incremental.hpp, feed.hpp, websocket.hpp, pipeline.hpp) and /cpp/apps/qre_stream.cpp
