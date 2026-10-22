# Decision log

Every design choice that could change a result: what was chosen and why.
The full frozen specification is [PREREGISTRATION.md](PREREGISTRATION.md).

## Research design

| Date | Decision | Choice | Why |
|---|---|---|---|
| 2026-09-15 | Research question | Does managing crash risk also remove carry's return? | Separates a risk premium from mispricing, and either answer is a finding |
| 2026-09-15 | Strategy family | Market-neutral funding carry, not price prediction | Price prediction is noisy and crowded; carry has an economic story to test |
| 2026-09-15 | Emphasis | Risk management and leakage control | Risk management *is* the experiment here, not an add-on |
| 2026-09-15 | Scope | Why the edge exists, factor spanning test, capacity curve, second-exchange replication |  |
| 2026-09-15 | Deferred | Live paper-trading forward test | Needs months; early-program applications are happening now |
| 2026-10-12 | Second test of the thesis | Fama-MacBeth: does funding still price returns after controlling for crash exposure? | Two independent tests (portfolio and pricing) that must agree are far harder to argue with than one |
| 2026-10-12 | Verdict criteria | Pre-registered tables in PREREGISTRATION.md §6–7, including a validity check that the controls really cut tail risk | Fixed before any return existed, so the goalposts can't move |

## Data pipeline

| Date | Decision | Choice | Why |
|---|---|---|---|
| 2026-09-15 | Data source | Binance bulk archives | The live API is blocked from US networks (HTTP 451) |
| 2026-09-15 | Universe base | USDT-margined perpetuals, delisted ones included | Largest set with one collateral currency; adding USDC/BUSD contracts would double-count the same coins |
| 2026-09-15 | Bar frequency | 1 hour | Funding settles as often as hourly; finer bars add size without changing carry |
| 2026-09-15 | Sample | 2020-01 → 2026-08 | First archived funding month through the last complete month |
| 2026-09-15 | When funding is known | Strictly after the published `calc_time`, jitter included | Conservative: a decision exactly on a settlement hour sees the previous rate |
| 2026-09-15 | Delistings | Never anticipated; a contract stays eligible until its trading stops | Dropping it earlier would use knowledge of the delisting |
| 2026-09-17 | What counts as trading | Only bars with volume; zero-volume bars ignored | Binance archives flat bars for months after a delisting; counting them put 122 dead contracts in the 2026-06-30 universe and ~11%/yr of fake funding on the short side |

## Phase 2–3 design (locked 2026-10-12, before any return was computed)

| Decision | Choice | Why |
|---|---|---|
| Non-crypto contracts | Excluded: 186 stock, ETF, commodity, pre-IPO, index and stablecoin contracts, classified by weekend-volume pattern plus names (`rvr/universe.py`) | The thesis is about crypto carry; stock contracts behave differently around market hours |
| Holdout | Final 12 months (2025-09 → 2026-08), sealed; keeps the October 2025 cascade out of sample | A real out-of-sample crash is the strongest possible test of a crash-risk thesis |
| Rebalance | Daily, decision 00:05 UTC, execution 01:00 UTC | Funding is persistent, so daily rebalancing saves costs; 00:05 follows the settlement all schedules share |
| Signal | Average hourly funding over the previous 72 h | Smooths single-settlement noise while staying current; 24 h and 168 h reported as logged trials |
| Minimum history | 30 days of trading | Excludes the volatile first weeks after listing |
| Restart gap | 7 days without a trade starts a new trading run | A contract back from dormancy is effectively a new listing |
| Liquidity filter | 7-day average daily volume ≥ $5M | Keeps costs realistic without emptying the early years |
| Fresh funding | Latest rate within 2 funding intervals | Second guard against funding records on contracts that aren't trading |
| Exit when trading stops | Last traded price with a 20% adverse haircut; settlements counted through the halt | Deliberately conservative; LUNA is the test case |
| Portfolio | Short top quintile, long bottom quintile, equal weight, gross 1, at least 20 eligible | Standard, transparent, no optimiser to overfit |
| Costs | 5 bp taker fee + half-spread by liquidity tier + square-root impact; break-even multiple reported | Conservative, and the break-even keeps the conclusion from hinging on one cost guess |
| Risk controls (thesis test) | Drop the most volatile quintile, hedge downside beta with BTC, target 10% volatility | Each targets a different source of crash exposure; each also run alone |
| Statistics | Block-bootstrap Sharpe CIs (primary), Newey-West t, deflated Sharpe over a trial log | Autocorrelation-robust, and corrected for every variant tried |

## Phase 3b–4 (2026-10-12)

| Decision | Choice | Why |
|---|---|---|
| How to use the holdout | Calibrate costs first, lock every amendment, then unseal once | Impact was the one assumption the verdict hung on; measuring it costs no holdout data |
| Cost calibration (A1) | Spread regression + square-root coefficient fitted to real taker orders around 01:00 UTC; depth-based book walk as an independent check; 2× stress | Keeps the pre-registered functional form and calibrates only its parameters; procedure written before any calibration number existed |
| Calibration sample | 5 bookTicker/aggTrades days (all the archive allows, spaced 2 months apart) and 64 bookDepth days (1st and 15th, 2023-01 → 2025-08), every eligible contract | Covers the whole cross-section at the actual execution time; design period only |
| Factor spanning test (A2) | MKT, SIZE, MOM, REV, LOWVOL built from the same universe and plumbing | Reversal is the obvious alternative explanation for carry; factors built in-house so nothing differs but the sort |
| Cost-aware variant (A3) | One no-trade buffer (exit at 40%), not searched | Costs are mostly turnover; a single first-principles rule keeps the trial count honest (9 trials) |
| Second venue (A3) | Hyperliquid, daily bars | Bybit and OKX block US networks or keep no history; Hyperliquid's API keeps only 5,000 candles per interval, so hourly bars don't reach 2023 |
| Holdout | Unsealed once, 2026-10-13, after A1–A3 were locked | Logged in HOLDOUT_LOG.md |
| Diagnosing the failure | Funding pass-through by year (the UIP test from FX carry), portfolio sorts, a real-time pass-through switch | Labelled exploratory: designed after the holdout was read |
| Testing the explanation (A4) | Two falsifiable predictions on 2026-09 → 2027-02, data not yet downloaded | The only clean test of an explanation built after seeing the holdout is data nobody has seen |
| Third venue (A5a) | dYdX v4, same rules as Hyperliquid, 5 bp fee | Reachable from the US, keeps wound-down markets; its thin late cross-section is reported, not worked around |
| Maker execution (A5b) | Post at the touch, cross after 30 min; fill simulated from real quotes and fills | The largest remaining cost is crossing the spread; a fill model fitted to replayed orders is testable, an assumed fill rate is not |
| Fill-rule fix (A5b) | Fill by traded quantity, not "complete on a trade-through" | The original rule made capacity rise with capital; caught by the capacity curve, logged, first fit kept |
| Tail-robust strategy (A5c) | Skip the extreme funding decile on each side; tested only forward | Built from the failure diagnosis, so only unseen data can test it |
| Interim monitoring | Monthly evaluation of the forward window to date, published on a dashboard, no decisions from it | Keeps the forward test public and falsifiable without moving the goalposts |
| Testing the explanation on other venues (A6) | Pass-through on Hyperliquid and dYdX, prediction b ≥ 0.5, written before running | The explanation came from Binance; independent exchanges test it without waiting for the forward window |
| Dating the regime change (A7a) | Least-squares mean break with bootstrap p-value and date interval; procedure fixed before running | Turns "it looks like 2022" into a tested claim with an interval |
| Mark-price funding (A7b) | Mirror Binance's mark-price bars and re-value every settlement; traded-close version stays primary | Removes the last known simplification in the P&L; result: no Sharpe moves by more than 0.002 |
| Young-token mechanism (A7c) | Interaction with a 180-day listing-age dummy | Tests the project's only untested interpretation; inconclusive, so the claim was removed |
