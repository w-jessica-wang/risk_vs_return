# Risk vs Return

**Is crypto carry just payment for crash risk?**

Perpetual futures never expire. To keep their price tied to spot, longs and shorts exchange a
*funding rate* every few hours. A market-neutral carry strategy — long the contracts with the
lowest funding, short the highest — collects that payment, and in naive backtests it looks
close to free money.

This project tests one idea:

> **If carry is compensation for crash risk, then managing the crash risk should also remove the return.**

| If carry is… | Then risk controls that cut crash exposure should… |
|---|---|
| a pure risk premium | take most of the return away along with the tail losses |
| partly mispricing (e.g. persistent retail demand for leverage) | leave a return that survives, net of costs |

Either outcome is a finding. The criteria that decide between them were fixed before any return was
computed ([PREREGISTRATION.md](PREREGISTRATION.md)), and every design choice is logged with its
reason in [DECISIONS.md](DECISIONS.md).

## Findings

Pre-registered design period 2020-05 → 2025-08; sealed holdout 2025-09 → 2026-08, read once after every
amendment was locked. Full results: [RESULTS.md](RESULTS.md).

1. **Carry is not crash compensation.** Positive skew, small losses in every crash (−5% against Bitcoin's
   −9% in the October 2025 cascade), and controlling for downside beta and coskewness barely moves its
   pricing (Fama-MacBeth t −3.26 → −2.90). Crash-risk controls cut returns, not tail risk.
2. **The cost model was the verdict, so it was measured, not assumed.** Fitted to 653,477 real taker orders
   at the strategy's execution time, market impact is a fifth of the pre-registered square-root assumption
   (Y = 0.21 vs 1.0), confirmed by walking 14,244 measured order books. Net Sharpe moves from −0.11 to 1.50.
3. **Carry is not known factors in disguise** — in the design period: alpha t = 4.5 after market, size,
   momentum, reversal and low-volatility factors (R² 0.03).
4. **It failed out of sample.** Holdout net Sharpe −2.5 under measured costs; no cross-sectional premium
   (t = 0.13). Funding income quadrupled and the price leg lost more.
5. **Why:** the premium was a 2020–21 squeeze regime. A funding pass-through regression (the perpetual-futures
   analogue of uncovered interest parity) shows prices moving *against* funding in 2020–21 and *with* it
   from late 2021: carry's total-return slope on funding was −2.7 and −1.9 in 2020–21 and is +0.05 (t = 0.3) in
   the holdout, i.e. prices absorb the funding actually paid. The design Sharpe averaged a Sharpe-3 regime with
   a Sharpe-0 one; the deflated Sharpe (0.20 over ten trials) already said not to trust it.

6. **Two other exchanges agree.** dYdX and Hyperliquid, whose histories are all post-2021, show no carry
   premium (gross Sharpe 0.70 and −0.91). On Hyperliquid funding income (+29%/yr) is more than cancelled by
   price (−54%/yr). A pre-registered test of the explanation there holds: Hyperliquid's funding
   pass-through is 1.50 (t = 3.8): prices absorb the funding there too, on an independent exchange.
7. **Maker execution, simulated on 63,910 limit orders replayed against real quotes and fills, triples
   capacity** (break-even ≈ $300M vs $100M) but doesn't help a small book: adverse selection on the orders
   that don't fill eats the spread saved.
8. **The regime change is dated formally:** one structural break in funding pass-through, on 1 September 2021
   (95% CI May 2021 – Mar 2022, bootstrap p < 0.0005), from −2.41 to +1.12. A natural explanation, newly
   listed tokens, was tested and not supported.
9. **A live forward test.** Two falsifiable predictions and a tail-robust strategy were frozen before any
   data after August 2026 was downloaded; each month's archive updates a dashboard, and the verdict is read
   once the window closes in February 2027.

![Cumulative returns](results/figures/cumulative.png)

## Design principles

- **Point-in-time or not at all.** Strategy code reads data only through `rvr/pit.py`, where every
  lookup is a strictly-before `merge_asof`. A funding rate is known only after its published
  timestamp (millisecond jitter included); a bar only after it closes; a contract's future
  delisting never removes it early. `tests/test_leakage.py` corrupts all future data and checks
  that no past lookup changes.
- **No survivorship bias.** The universe comes from Binance's archive bucket, which still holds
  contracts that died (LUNAUSDT's history ends in May 2022), not from today's symbol list.
- **Dead contracts stay dead.** Binance keeps archiving flat, zero-volume bars for months after a
  delisting — 8% of all bars. Only bars with volume count as trading; otherwise 122 dead contracts
  would have sat in the 2026 universe collecting fake funding.
- **Verified data.** Every archive is SHA-256-checked against Binance's published checksum
  before it touches disk. Every quirk in the raw files is counted in `quality_report.json` and
  documented in [DATA.md](DATA.md).
- **Costs before conclusions.** No return is reported without fees, spread and slippage.
- **A sealed holdout.** The final year stayed unread until the design and every amendment were frozen,
  then was read once ([HOLDOUT_LOG.md](HOLDOUT_LOG.md)).
- **Amendments are written before they are run.** Calibrating costs, the factor test, a cost-aware variant
  and the second venue were each specified in PREREGISTRATION.md first, with their motivation disclosed.

## Roadmap

| Phase | What | Status |
|---|---|---|
| 1 | Data pipeline: checksum-verified archives, tidy panels, point-in-time lookups, leakage tests, data audit | Done |
| 2 | Baseline carry strategy net of costs; pre-registration; sealed holdout | Done (design period) |
| 3 | The thesis test, two ways: crash-risk controls on the portfolio, and Fama-MacBeth pricing with crash exposure | Done (design period); holdout sealed |
| 3b | Measure trading costs from order-book archives (A1); factor spanning test (A2); cost-aware variant (A3); unseal the holdout | Done |
| 4 | Why it failed: funding pass-through by year, portfolio sorts (exploratory) | Done |
| 4b | Replication on dYdX (A5a) and Hyperliquid (A3), daily bars, design and holdout | Done |
| 4c | Maker execution: fill simulation on real quotes and fills (A5b) | Done |
| 5 | Forward test on unseen months, 2026-09 → 2027-02 (A4, A5c), monthly interim dashboard | Running; September done |
| 5b | Structural-break dating, mark-price funding, young-token test (A7) | Done |

## Quick start

```bash
pip install -r requirements.txt
pytest -q                          # synthetic tests, no network
python -m rvr.download --dry-run   # list archives and total size
python -m rvr.download             # mirror funding + 1h prices into data/raw (resumable)
python -m rvr.build                # tidy parquet panels into data/processed
python -m rvr.microstructure       # order-book sample at the execution time (design period only)
python -m rvr.costs                # fit the measured cost model (Amendment A1)
python -m rvr.run                  # every pre-registered analysis on the design period → results/
python -m rvr.diagnostics          # exploratory: pass-through and sorts (needs the holdout run)
python -m rvr.figures
```

## Layout

```
rvr/
  config.py     paths, archive endpoints, sample window
  listing.py    survivorship-free universe from the archive bucket
  download.py   checksum-verified, resumable mirror
  parse.py      raw-archive quirks, handled and counted
  build.py      parquet panels + quality report
  pit.py        point-in-time lookups (the only way strategy code reads data)
  universe.py   frozen list of non-crypto contracts
  spec.py       every pre-registered parameter
  holdout.py    the sealed holdout and its unseal log
  features.py   daily characteristics known by 00:05 UTC
  strategy.py   decision panel and strategy weights
  backtest.py   execution, halts, funding and costs
  microstructure.py  order-book sample: quoted spreads, depth, realised taker costs
  costs.py      cost models: pre-registered and measured (A1)
  factors.py    crypto factors and the spanning test (A2)
  diagnostics.py  why carry failed: pass-through, sorts, young-token test (exploratory)
  breaks.py     structural-break test for the regime change (A7a)
  figures.py    the charts in README and RESULTS
  execution.py  maker fill simulation and cost model (A5b)
  hyperliquid.py, dydx.py, replication.py   other venues (A3, A5a)
  dashboard.py  exports the forward-test dashboard's data
  stats.py      bootstrap CIs, Newey-West, deflated Sharpe, tails
  pricing.py    Fama-MacBeth
  verdict.py    the pre-registered verdict tables
  run.py        runs everything, writes results/
tests/          leakage, parsing, features, execution and P&L, costs, microstructure, factors, strategies,
                statistics, verdicts, holdout, replication — synthetic data, no network
PREREGISTRATION.md  the frozen design, written before any return was computed
RESULTS.md      results: design, holdout, costs, why carry failed
HOLDOUT_LOG.md  every time the holdout was read
DATA.md         sources, schemas, quirks, point-in-time rules
DECISIONS.md    decision log: what was chosen, and why
```

## License

Proprietary — see [LICENSE](LICENSE).
