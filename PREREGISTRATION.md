# Pre-registration

**Written 2026-10-12, before any strategy return was computed.** At the time of writing, no carry
portfolio, signal-sorted return or Sharpe ratio had been calculated on any part of the data. The
only analyses run were data-quality checks (DATA.md) and the volume-pattern classification of
non-crypto contracts, neither of which touches returns.

Everything below is fixed. Bug fixes are allowed and are logged under **Amendments** with the
reason; any change to a parameter after results are seen is logged there too, and the original
result is still reported.

## 1. Samples

| | Period | Status |
|---|---|---|
| **Design** | decision days 2020-01-01 → 2025-08-30 (last exit 2025-08-31 01:00 UTC) | used for every result below |
| **Holdout** | decision days 2025-09-01 → 2026-08-30 | **sealed**; evaluated once, after Jessica approves unsealing |

The holdout contains the October 2025 liquidation cascade, kept there deliberately as a true
out-of-sample crash. `rvr/holdout.py` truncates every input at 2025-09-01 00:00 UTC unless run with
an explicit unseal, which is recorded in `HOLDOUT_LOG.md`.

## 2. Universe (re-evaluated at every decision)

A contract is eligible when all of the following hold, using only data available at the decision:

1. USDT-margined perpetual whose underlying is a crypto asset — not in `rvr/universe.py`'s frozen
   `NON_CRYPTO` list (stocks, ETFs, commodities, pre-IPO names, index contracts, USDC).
2. Trading: a bar with volume closed within the last **2 h**, and the current run of trading has
   lasted **30 days** (720 h). A run restarts after **7 days** (168 h) without a trade.
3. Liquidity: average daily USDT volume over the previous **7 days ≥ $5M**.
4. Fresh funding: the latest funding rate was published within **2 of the contract's funding intervals**.

No portfolio is held on a day with fewer than **20** eligible contracts.

## 3. Signal and baseline portfolio ("raw carry", **B**)

- **Decision** daily at **00:05 UTC**, after the 00:00 settlement that every funding schedule shares.
- **Signal** = sum of funding rates published in the previous **72 h**, divided by 72: the average
  hourly cost of being long. Reported annualised (× 8,760).
- **Portfolio:** with N eligible contracts and n = ⌊N/5⌋, short the n highest-signal contracts and long
  the n lowest. Each long +0.5/n, each short −0.5/n: gross 1, dollar-neutral.
- **Execution** at the close of the 00:00–01:00 UTC bar (01:00 UTC), 55 minutes after the decision.

## 4. Returns, execution and halts

A position of weight w held from t0 (day d, 01:00) to t1 (day d+1, 01:00) earns:

- **Price:** w × (exit / entry − 1).
- **Funding:** −w × Σ rateₛ × Pₛ / entry, over settlements with settle time in (t0, t1], where Pₛ is the
  last traded close at or before the settlement. (Longs pay positive funding; shorts receive it.)

**Execution price at time t:** the latest traded close in (t − 2 h, t]; if none, the first traded close
in (t, t + 7 days] (the order waits for trading to resume); if none, the contract has **halted**.

- A position whose entry has no execution price is **not established** (weight 0 that day; other
  weights are not rescaled).
- A held position whose exit has no execution price exits at the **last traded close with a 20%
  adverse haircut** (longs × 0.8, shorts × 1.2), and is credited or charged every settlement up to the
  first hour after its last trade. LUNA (last trade 15:00–16:00 UTC on 2022-05-12; −1.0% settlement
  published at 16:00:00.007) is the reference case.

## 5. Costs

Charged on every change in position, including drift back to target weights and halt exits, per
unit of traded notional:

| Component | Value |
|---|---|
| Fee | 5.0 bp (Binance regular-tier taker), every year |
| Half-spread, by the contract's 7-day average daily volume (ADV) | ≥ $1B: 0.5 bp · $100M–$1B: 2 bp · $20M–$100M: 5 bp · below $20M: 10 bp |
| Impact (square-root law) | 1.0 × σ_daily × √(traded notional / ADV), σ_daily = 30-day volatility of daily returns |

Fallbacks: σ_daily = 5% and ADV = $5M when unavailable. Main results use **$1M** of capital.
Capacity curve: $1M, $10M, $50M, $100M, $250M, $500M. Break-even cost multiple = gross mean ÷ cost mean.

## 6. The thesis test, portfolio form

**Managed strategy M** = B with all three crash-risk controls:

1. **Drop the most crash-prone names:** before ranking, remove the top quintile of eligible contracts
   by 30-day volatility of daily returns (contracts with fewer than 20 daily returns are removed too).
2. **Hedge crash exposure:** each contract's downside beta to BTC over the trailing 90 days, using only
   days BTC fell (at least 10 such days; otherwise ordinary 90-day beta with at least 20 days;
   otherwise 1.0). The portfolio's downside beta is offset with a BTCUSDT position.
3. **Volatility target:** scale the book to 10% annualised volatility using the trailing 60-day
   realised volatility of the unscaled managed strategy (at least 20 days; scale 1 before that).
   Only daily returns completed before the decision are used, so day d uses returns through day
   d − 2. Scale capped at 2.

Secondary variants M1, M2 and M3 apply each control alone.

**Validity check (pre-registered).** The controls must actually remove crash risk: M's 1% expected
shortfall per unit of volatility must be at least **25% lower** than B's. If not, the verdict is
*inconclusive: controls did not remove crash risk*.

**Verdict** (design-period net Sharpe; 95% CIs from §8):

| Condition | Verdict |
|---|---|
| B's CI includes 0 | No significant carry premium to explain |
| S_M ≤ 0.5 × S_B and M's CI includes 0 | **Risk premium** — managing crash risk removes the return |
| M's CI lower bound > 0 | **Edge survives** — part of carry is not crash compensation |
| Otherwise | Inconclusive |

## 7. The thesis test, pricing form (Fama-MacBeth)

Each design day, regress every eligible contract's next-day long return (price + funding, before
costs) on cross-sectionally standardised characteristics, then average the daily coefficients.

- **Model 1:** carry signal only.
- **Model 2:** carry + downside beta (90 d) + coskewness with BTC (90 d) + log ADV (7 d) + momentum
  (30 d) + volatility (30 d) + maximum daily return (30 d).

Both models use the **identical sample** (contract-days with every Model 2 characteristic), days with
at least 20 contracts. Characteristics and returns are winsorised at the 1st/99th percentile each day;
characteristics are z-scored. Inference: Newey-West standard errors, 20 lags.

| Condition | Verdict |
|---|---|
| Model 1 carry coefficient has \|t\| < 1.96 | No cross-sectional carry premium |
| Model 2 coefficient ≤ 50% of Model 1 in magnitude and \|t\| < 1.96 | **Risk premium** — crash exposure explains carry |
| Model 2 \|t\| ≥ 1.96 with Model 1's sign | **Edge survives** crash-risk controls |
| Otherwise | Inconclusive |

## 8. Statistics

- Daily net returns; Sharpe annualised with √365.
- **Primary CI:** stationary block bootstrap of the Sharpe ratio, mean block 20 days, 10,000
  resamples, seed 20261012, 2.5th–97.5th percentiles.
- **Secondary:** Newey-West t-statistic (20 lags) on the mean daily return.
- **Deflated Sharpe ratio** using every variant in the trial log (§9).
- **Tails:** skewness, excess kurtosis, worst day, 1% and 5% expected shortfall, maximum drawdown,
  downside beta to BTC.
- **Crash windows** (descriptive), from 2 days before to 5 after: 2020-03-12, 2021-05-19,
  2022-05-09, 2022-06-13, 2022-11-08, 2024-08-05; holdout: 2025-10-10.

## 9. Trial log

Every variant evaluated on the design period is appended to `results/trials.jsonl`. Pre-registered
variants: B, M1, M2, M3, M, and B with 24 h and 168 h signal windows (7 trials). Cost sensitivity and
capacity reuse the same weights and are not trials. Anything run beyond these is a new trial, logged
and counted.

## 10. Holdout

Run once on the sealed year with everything above frozen. **Confirms** if B and M keep their design-
period signs and M's 1% expected shortfall per unit of volatility is below B's. The October 2025
cascade is reported as a crash window.

## Amendments

- **2026-10-12, after the first design-period run:** no parameter changed. One diagnostic beyond §9
  (baseline executed at 02:00 instead of 01:00, to rule out a timing artifact) was logged as trial 8,
  so the deflated Sharpe ratio uses 8 trials.

### Amendment A1 — measured trading costs (written 2026-10-12, before any calibration result was seen)

**Motivation, disclosed.** The design-period portfolio verdict turns on the assumed market impact (§5):
fees and spreads alone leave the baseline a Sharpe of 1.35, the square-root term takes it to −0.11. The
impact coefficient (1.0) and spread tiers were assumptions. This amendment measures them from Binance's
order-book archives, using design-period days only, and fixes the procedure before any number exists.
Only the order-book sample was looked at when this was written (one contract, LINKUSDT on 2023-11-01,
used to check the parsers). Nothing in the calibration reads a strategy return.

**Data** (`rvr/microstructure.py`). Eligible contracts (§2) on sampled decision days:

- *Execution sample:* 2023-06-01, 2023-08-01, 2023-10-01, 2023-12-01, 2024-02-01 (the bookTicker archive
  covers only 2023-05-16 → 2024-03-30). Quoted half-spread and touch size, time-weighted over 00:50–01:10
  UTC; every taker order in 00:30–01:30 UTC (aggregate trades sharing a timestamp and side), costed
  against the mid quoted strictly before it.
- *Depth sample:* the 1st and 15th of each month, 2023-01 → 2025-08. Median cumulative notional within
  ±1 … ±5% of the mid over 00:50–01:10 UTC.

**Calibrated cost model** (per unit of traded notional q, for a contract with point-in-time ADV and σ_daily
as in §5):

    cost = fee (5 bp, unchanged) + ĥs + Ŷ × σ_daily × √(q / ADV)

1. **Half-spread ĥs:** OLS on the execution sample, log(half-spread) = a + b·log(ADV) + c·log(σ_daily);
   the prediction is applied to every contract-day and clipped to [0.1 bp, 50 bp].
2. **Impact coefficient Ŷ:** taker orders of at least $1,000, grouped by (contract, day, size bucket) with
   at least 5 orders. y = mean slippage − that contract-day's measured half-spread; x = σ_daily ×
   √(mean order notional / ADV). Ŷ = order-weighted least squares through the origin. 95% CI from a
   bootstrap over contracts.
3. Fallbacks as in §5.

**Reported, not used to set costs:** (i) the book-walk cost from measured depth at the baseline's actual
trade sizes, by year, against the calibrated model; (ii) whether depth at given ADV and volatility drifts
by year, which bounds the error of applying a 2023–25 calibration to 2020–22.

**Known bias, disclosed.** Realised slippage of orders that were sent understates the cost of a forced
order, because traders size orders to the liquidity they see. Every A1 result is also reported at 2× the
calibrated costs.

**Use.** No strategy parameter changes. Design-period results under A1 are cost sensitivity (same weights,
not trials). On the holdout, the §6 verdict table is applied twice: under the pre-registered §5 costs
(the primary verdict, unchanged) and under A1 costs.

### Amendment A2 — factor spanning test (written 2026-10-12, before it was run)

Is carry a new premium, or known crypto factors in disguise? Funding is high after prices rise, so carry
could be short-term reversal; the long leg is crowded shorts, so it could be momentum or size.

**Factors**, built from the same eligible universe, timing and long returns (price + funding, before
costs) as §3–4, equal weight within legs, each leg gross 0.5:

| Factor | Long | Short |
|---|---|---|
| MKT | every eligible contract (gross 1, long only) | — |
| SIZE | bottom tercile of 7-day ADV | top tercile |
| MOM | top tercile of 21-day return (Liu, Tsyvinski & Wu 2022 use three weeks) | bottom tercile |
| REV | bottom tercile of 7-day return | top tercile |
| LOWVOL | bottom tercile of 30-day volatility | top tercile |

Returns from daily closes known at the 00:05 decision. **Regressions** of the baseline's daily gross
return on (a) MKT; (b) MKT, SIZE, MOM; (c) MKT, SIZE, MOM, REV, LOWVOL. Newey-West, 20 lags. **Verdict:**
carry is *spanned* if the gross alpha in (c) has |t| < 1.96; otherwise *not spanned by known factors*.
Run on the design period now and on the holdout with everything else. Not a trial: no carry variant changes.

**A1/A2 outcome, recorded 2026-10-12 after the first run with them** (numbers in RESULTS.md): Ŷ = 0.21
(95% CI 0.18–0.23), and the gross alpha in A2's five-factor model has t = +4.52. Nothing above was changed
after these were seen.

### Amendment A3 — one cost-aware variant and a second venue (written 2026-10-12, before either was run)

**Motivation, disclosed.** A1 was run before this was written. Under calibrated costs the baseline's net
Sharpe is 1.50, and costs are still 38% of its gross return, almost all from daily turnover. The standard
remedy is a no-trade buffer, so one is tested here. It is a new trial (the 9th), designed from first
principles rather than searched; no other buffer width will be run.

**B_buffer.** As B (§3), except for exits: a contract held short stays short while it is eligible and in
the top **40%** of the signal; a long stays long while eligible and in the bottom **40%**. Each day the
book is (yesterday's holdings that pass that test) ∪ (today's top and bottom quintiles, as in B). Equal
weight within each side, each side gross 0.5. A day with fewer than 20 eligible contracts holds nothing,
and the book restarts from B's rule the next day.

**Second venue: Hyperliquid** (`rvr/hyperliquid.py`). Binance's other large competitors block US networks
or keep no history; Hyperliquid serves every perp's funding (hourly) and daily candles, delisted coins
included, from 2023-05. Its API keeps only the latest 5,000 candles per interval, so the replication runs on
**daily** bars, and what changes is forced by that:

- Decision at **00:00 UTC** using funding published strictly before it; entry at the open of day d's daily
  candle, exit at its close; a position earns the settlements in (d 00:00, d+1 00:00].
- Eligibility: the previous day's candle traded; 30 consecutive traded days (a gap of more than 7 days
  restarts the count); fresh funding (a settlement in the previous 2 hours); 7-day ADV above a floor. The floor
  is the largest of $5M, $2M and $1M that leaves at least 20 eligible contracts on 90% of decision days in
  2023-06 → 2025-08, chosen from volume data alone before any return is computed.
- Signal, portfolio, halts (20% haircut, exit at the last traded close), B_buffer: as on Binance.
- Costs: Hyperliquid's base taker fee (4.5 bp) plus A1's calibrated half-spread and impact (fitted on
  Binance, disclosed as such), with σ and ADV from Hyperliquid's own candles. Pre-registered §5 costs are also reported.
- Periods: design 2023-06-01 → 2025-08-30, holdout 2025-09-01 → 2026-08-30, as on Binance.

**Replication verdict** (each period): *replicates* if B's gross Sharpe has a 95% CI above zero and the
Fama-MacBeth Model 1 carry slope (§7) has |t| ≥ 1.96 and Binance's sign; *partly replicates* if exactly one
holds; otherwise *does not replicate*. Model 2 is reported with the same seven characteristics as §7, all of
which come from daily closes.

### Amendment A4 — a forward test of the explanation (written 2026-10-13, before any data after 2026-08-31 was downloaded)

**Motivation, disclosed.** The holdout failed, and the explanation in RESULTS.md §5 — carry pays only when
prices do not absorb funding, and since 2022 they mostly have — was built after the holdout was read. Data
already seen cannot test it. Months that nobody has downloaded can.

**Window:** decision days **2026-09-01 → 2027-02-27** (six months), from Binance's monthly archives, run
once when the 2027-02 archive is published: `python -m rvr.run --forward --unseal "<reason>"`.
Everything is frozen as of this amendment: §2–§5, A1's fitted costs (`results/design/costs.json`), B and
B_buffer.

**Predictions** (each can fail):

1. **Carry stays dead.** B's net Sharpe under A1 costs is **≤ 0**. Falsified if it is positive with a 95%
   CI above zero.
2. **Parity persists.** The window's mean funding pass-through b (`rvr/diagnostics.py`) is **≥ 0.5**.
   Falsified if it is below 0.5 with |t vs 1| ≥ 1.96.
3. Reported, not predicted: B gated on a trailing 180-day pass-through below 0.5 (the real-time switch
   in RESULTS.md §5), and B_buffer.

If prediction 1 fails while 2 holds, the explanation is wrong about what drives carry; if both fail, the
squeeze regime is back.

### Amendment A5 — a third venue, maker execution, and a tail-robust strategy (written 2026-10-13, before any of them was estimated or run)

**A5a. Third venue: dYdX v4** (`rvr/dydx.py`). Public indexer, reachable from US networks, hourly funding
and daily candles from late 2023, wound-down markets included (223 of 296). Everything as A3's Hyperliquid
replication — daily bars, decision at 00:00 UTC, the same floor rule, verdict and periods — except the fee:
dYdX's base taker fee, **5 bp**. Seen before writing this: the market list and current volumes (only ETH,
BTC and SOL trade above $1M a day today), and 2024 volumes for six alts (LINK, AVAX ≈ $12M a day).

**A5b. Maker execution** (`rvr/execution.py`). The baseline crosses the spread on every trade. The
alternative: at 01:00 UTC post a limit order at the touch for the whole trade; after **τ = 30 minutes**
send any unfilled remainder as a taker order.

*Fill simulation*, on A1's execution sample (5 days, every eligible contract), posting at every 5 minutes
from 00:30 to 01:00 UTC, both sides, sizes $1K, $3K, $10K, $30K, $100K:
- the order joins the back of the queue: queue ahead = the touch size quoted when it is posted;
- it fills as taker volume trades at its price (after the queue ahead), and completely if a trade prints
  through its price; it is never repriced;
- implementation shortfall against the mid at posting, per unit: filled part −half-spread at posting +
  maker fee; unfilled part signed mid move over τ + half-spread at τ + A1 impact + taker fee.

*Model for the backtest*, fitted on that simulation and nothing else:
- fill fraction f = logistic(α + β·log(q / touch size) + γ·log σ_daily), with touch size predicted from
  log ADV and log σ (OLS on A1's quoted touch sizes);
- adverse move of unfilled orders, in units of daily volatility: κ = mean(signed mid move | not fully
  filled) / σ_daily;
- expected cost per unit = f·(maker fee − ĥs) + (1 − f)·(taker fee + ĥs + κ·σ_daily + Ŷ·σ_daily·√(q/ADV)).

Fees: Binance regular tier, maker **2 bp**, taker 5 bp. Reported as a cost model on unchanged weights (B,
B_buffer), like A1: not a trial. Its design and holdout numbers are descriptive (the holdout has been read);
it is applied, frozen, in the A4 forward window.

**A5c. Tail-robust carry, "B_core"** — built after the holdout showed the funding tails breaking (RESULTS.md
§5), so it is tested **only** on the A4 forward window. As B, but short contracts between the 70th and 90th
percentile of the signal and long those between the 10th and 30th, skipping the extreme decile on each side;
equal weight, gross 1. Logged as trial 10 when its (contaminated) design and holdout numbers are computed.
Forward prediction, written now: **none** — it is a test, and either sign is informative.

**A4/A5 addendum — interim monitoring (written 2026-10-13, before any forward data was downloaded).** As each
monthly archive is published, the forward window is evaluated to date and posted to a public dashboard
(`python -m rvr.run --forward --interim --unseal "<reason>"`, logged in HOLDOUT_LOG.md like every read).
Interim results are monitoring only: no parameter, strategy or prediction changes because of them, and the
verdicts are those of the full window when 2027-02 is archived.

**A5b bug fix (2026-10-13, after the first design run with it).** The fill rule "completely if a trade prints
through its price" is wrong for large orders: no trade can print through a price while a large order still
rests there. It made fills *rise* with order size and the maker capacity curve rise with capital (net Sharpe
1.91 at $1M, 2.04 at $500M), which is how it was caught. Fixed: an order fills by the quantity traded at or
through its price after the queue ahead. The fix is more conservative; the first fit is kept in
`results/design/maker_v0.json` for the record.

### Amendment A6 — funding parity on the other venues (written 2026-10-13, before it was run)

**Motivation.** The parity explanation (RESULTS.md §5) was built on Binance. Hyperliquid and dYdX are
independent order books with different traders, and their pass-through has never been computed: only their
carry portfolios (A3, A5a) were. If the explanation is right, it should hold there too.

**Test.** The §5 pass-through regression (`rvr.diagnostics.daily_pass_through`: daily cross-sectional OLS of a
long's next-day price return on its expected daily funding cost, both winsorised at 1%), on each venue's
eligible contracts (A3 rules) for every day with at least 20, from each venue's first day to 2026-08-30.
Newey-West, 20 lags.

**Prediction:** on each venue the mean pass-through b is **≥ 0.5**. Falsified on a venue if b < 0.5 with
|t vs 1| ≥ 1.96. Reported by year as well.

### Amendment A7 — dating the break, mark-price funding, and the young-token mechanism (written 2026-10-13, before any of them was run or downloaded)

**Status, disclosed.** A7a and A7c analyse data already seen (design and holdout) to sharpen claims made after
the holdout; they are exploratory, and fixing the procedure here only stops it being tuned to the answer. A7b
corrects a known simplification. None changes a strategy or adds a trial.

**A7a. A formal test for the regime change** (`rvr/breaks.py`). On the daily Binance pass-through slopes b_t
(RESULTS.md §5; 2020-05 → 2026-08), fit a single shift in mean at an unknown date by least squares (Bai 1997),
with 15% trimming at each end.
- *Test:* sup-F over all candidate dates; p-value from 2,000 stationary-bootstrap resamples of the demeaned
  series (mean block 20 days, seed 20261012), which keeps its autocorrelation under the null of no break.
- *Break date interval:* 95% interval from re-estimating the date on 2,000 block-bootstrap series built from
  the fitted two-mean model plus resampled residuals.
- *A second break:* the same test inside each segment, reported, never chosen between.
- *Prediction:* a break with p < 0.05, dated between 2021-06 and 2022-12, with a pre-break mean below 0.5 and a
  post-break mean of at least 0.5.

**A7b. Funding on the mark price** (`rvr/download.py`, `rvr/backtest.py`). §4 values each settlement with the
last traded close; Binance charges funding on the mark price. Binance's 1h mark-price archives are mirrored
and verified like the rest, and each settlement is valued at the close of the mark-price bar ending at its
settlement hour, falling back to the last traded close where no mark bar exists (that fallback count is
reported). Design and holdout are re-run on unchanged weights under §5, A1 and A5b costs; the §4 version stays
the primary result and both are reported. *Prediction:* no net Sharpe changes by more than 0.10.

**A7c. Are newly listed tokens behind parity?** Hypothesis from RESULTS.md §5: extreme funding increasingly
marks young tokens whose supply is unlocking, and their prices absorb funding. Each day, across eligible
contracts, regress next-day price return on expected funding cost f, a young dummy Y (first traded bar fewer
than **180 days** before the decision, from past bars only), and f × Y; average the daily coefficients with
Newey-West errors (20 lags). Same winsorising and minimum of 20 contracts as §7.
- *Samples:* 2022-01 → 2026-08 (the parity period) and, separately, 2020-05 → 2021-12.
- *Prediction:* in 2022-01 → 2026-08 the interaction coefficient is positive with t ≥ 1.96 (young contracts pass
  more funding into price). Falsified if it is negative with |t| ≥ 1.96; inconclusive otherwise.

**Bug fixes from an outside review (2026-10-13).** Logged here as §4/§8 require; no parameter changed.
1. *Maximum drawdown* omitted the starting wealth of 1, so a loss on the first day did not count
   (`max_drawdown([-0.2])` returned 0). Fixed; the test now checks hand-computed values.
2. *Funding during a delayed exit.* §4 lets an exit wait for trading to resume, but settlements after the
   scheduled exit and before the actual one were dropped. Each position now earns every settlement between its
   actual entry and actual exit. This arises only around one archive gap (2022-02-26 → 03-01, 30 positions).
   A delayed position's whole P&L is still booked on the day it was opened, a convention now documented.
3. *Interpretation.* RESULTS.md called price pass-through b = 1 "funding parity". Because funding mean-reverts
   (realised next-day funding ≈ 0.7 of the signal), the no-carry benchmark is b = ρ; the total-return slope
   (b − ρ, zero for no carry) is now the headline measure. The A4 and A6 predictions are unchanged, as written.
No verdict changed; net Sharpe ratios moved by at most 0.001.
