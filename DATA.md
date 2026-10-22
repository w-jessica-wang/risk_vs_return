# Data

## Source

Binance USDⓈ-M futures bulk archives at <https://data.binance.vision>, one file per symbol per month.

| Dataset | Archive path | Coverage |
|---|---|---|
| Funding rates | `data/futures/um/monthly/fundingRate/{SYMBOL}/` | 2020-01 onward |
| 1h klines | `data/futures/um/monthly/klines/{SYMBOL}/1h/` | 2020-01 onward |

- Every archive has a SHA-256 `.CHECKSUM` sidecar. `rvr/download.py` verifies it before writing.
- The live REST API (`fapi.binance.com`) returns HTTP 451 from US networks. Nothing here uses it.
- Sample: **2020-01 → 2026-08**, the first archived funding month through the last complete month
  when the pipeline was built.

**Universe.** Every symbol with a funding-rate archive quoted in USDT. On 2026-09-15 the bucket
listed 952 symbols with funding archives: 865 USDT, 40 USDC and 47 legacy BUSD-margined.
Delisted contracts keep their archives, so the universe includes contracts that died.

**Mirror (2026-09-17).** 41,734 archives (20,351 funding, 21,383 price; 723 MB), all checksums
verified, no failures. Built into 2,634,145 funding settlements and 15,295,659 hourly bars
across 865 contracts (864 with price archives; `GAIBUSDT` has funding records only).

## Processed outputs (`data/processed/`)

All timestamps are int64 epoch milliseconds, UTC.

**`funding.parquet`** — one row per settlement

| Column | Meaning |
|---|---|
| `symbol` | e.g. `BTCUSDT` |
| `calc_time` | exact publish time from Binance, jitter included. **Decides when the rate was known.** |
| `settle_hour` | `calc_time` rounded to the nearest hour. For accruing P&L only. |
| `interval_h` | hours between settlements for this contract at this time (1, 2, 4 or 8) |
| `rate` | funding rate for the interval |
| `rate_per_hour` | `rate / interval_h`, comparable across schedules |

Sign convention: `rate > 0` means longs pay shorts. A short of notional N receives `rate × N` at settlement.

**`prices_1h/{SYMBOL}.parquet`** — hourly bars: `open_time`, `close_time`, `open`, `high`, `low`,
`close`, `volume`, `quote_volume` (USDT), `count`, `taker_buy_quote_volume`. A bar is known only after
`close_time`. Zero-volume bars are kept as archived; `rvr/pit.py` ignores them.

**`listings.parquet`** — first and last bar, last traded bar and settlements per symbol.
**Descriptive only.** It contains each contract's future, so it must never be used to choose a universe.

**`quality_report.json`** — counts of every anomaly below, in total and per symbol.

## Quirks found in the raw archives

| Quirk | Evidence | Handling |
|---|---|---|
| **Zombie price bars after delisting** | Binance keeps archiving flat, zero-volume hourly bars for months after a contract stops trading: 1,267,588 bars (8.3% of all bars); 129 dead contracts (e.g. `FTMUSDT`, `KLAYUSDT`, `UNFIUSDT`) print them through 2026-08 | Kept in the panel, ignored by every lookup. Counting them as trading would have put 122 dead contracts in the 2026-06-30 universe (761 instead of 639) |
| **Zombie funding records** | 76,000 settlements (2.9%) on contracts with no trade in the prior 24h, mostly the flat default rate (0.01% per 8h); `FTMUSDT` printed 0.01% for two weeks after it died | A contract must be trading to be eligible. Left in, a short would "earn" ~11% a year on a contract nobody could trade |
| **Dormant then revived contracts** | `LITUSDT` (4,500h), `BTCSTUSDT`, `AIAUSDT`, `CVXUSDT`, `SLPUSDT`, `PUMPUSDT`: zombie bars and no funding, then trading resumes; `BNTUSDT` relisted after 2.3 years | A new trading run starts after a long gap and must rebuild minimum history |
| Funding records before trading starts | Most new listings show funding 1–5 days before their first bar; `BNTUSDT`'s 2021 listing has funding but no price archive | No traded bars → not eligible |
| **Non-crypto contracts from 2026-03** | Stocks, ETFs, commodities and pre-IPO names (e.g. `AAPLUSDT`, `SPYUSDT`, `EWYUSDT`, `CLUSDT`, `NATGASUSDT`, `OPENAIUSDT`); 122 of 639 live contracts on 2026-06-30 first traded after 2026-03-01. All 45 conflicting duplicate settlements are on stock contracts | Excluded: 186 contracts on the frozen `NON_CRYPTO` list (`rvr/universe.py`, PREREGISTRATION.md §2) |
| Early price archives have no header row | `BTCUSDT-1h-2020-01.csv` starts with a data row; `2024-01` has a header | Header detected from the first field |
| Funding `calc_time` has millisecond publish jitter | 1,186,493 settlements (45%); e.g. `LUNAUSDT` 2022-05: `1652371200007`, `1652400000014`; none beyond one minute | Exact time kept for visibility; `settle_hour` for accrual |
| Funding interval varies by contract and over time | 1h: 79,107 · 2h: 3,492 · 4h: 1,498,911 · 8h: 1,052,635 settlements; 289 contracts changed schedule | `rate_per_hour` |
| Extreme funding in stress | `LUNAUSDT` −0.65% to −0.90% per 8h through its collapse on 2022-05-11/12; `ALPACAUSDT` ±4% per hour in 2025-04 | Kept. These are the crash observations the thesis is about |
| Trading halts mid-crash | `LUNAUSDT`'s last traded bar is 15:00–16:00 UTC on 2022-05-12 (close $0.008, down from $0.030 at 12:00; $54M traded). A −1.0% settlement was published at 16:00:00.007, as trading stopped; zombie bars and the default 0.01% rate follow | Contract leaves the universe after `max_staleness_hours`; a held position exits at the last traded close with a 20% adverse haircut (PREREGISTRATION.md §4) |
| Missing price hours | 7,883 hours across 60 contracts (maintenance and halts) | Counted, never filled |
| Timestamp units | Futures archives are milliseconds through 2026-08; Binance's spot archives moved to microseconds in 2025 | Normalised defensively |

## Mark prices (Amendment A7b)

`data/futures/um/monthly/markPriceKlines/{SYMBOL}/1h/`, mirrored and checksum-verified by `rvr/download.py`
(dataset `mark_1h`): 22,519 archives, 2020-01 → 2026-09, same 12-column layout as trade bars with zero volume.
`rvr/build.py` keeps full on-the-hour bars with a positive close in `mark_1h.parquet` (symbol, close_time, mark).
A settlement is valued on the mark bar ending at its settlement hour; 2,058 design-period and 3,526
holdout-year settlements on held positions have none and keep the traded close.

## Order-book sample (Amendment A1)

Daily archives from the same bucket, `data/futures/um/daily/{dataset}/{SYMBOL}/`, checksum-verified,
reduced to the execution window by `rvr/microstructure.py` and discarded; only the summaries are kept.
Design-period days only; the sampler refuses any day whose window reaches the holdout.

| Dataset | Coverage in the archive | Sampled | Kept (`data/processed/`) |
|---|---|---|---|
| `bookTicker` (every best bid/ask change) | 2023-05-16 → 2024-03-30 only | 2023-06-01, 08-01, 10-01, 12-01, 2024-02-01; every eligible contract (913 contract-days) | `micro_execution.parquet`: time-weighted half-spread, touch notional, tick size over 00:50–01:10 UTC |
| `aggTrades` (every fill) | full sample | same days | same file: taker orders over 00:30–01:30 UTC, bucketed by size (count, mean notional, mean slippage against the prior mid) |
| `bookDepth` (snapshot every ~30 s of notional within ±1…±5% of the mid) | 2023-01 onward | 1st and 15th of each month, 2023-01 → 2025-08; every eligible contract (14,244 of 14,575 contract-days available) | `micro_bookDepth.parquet`: median depth per band over 00:50–01:10 UTC |

Quirks: a **taker order** is reconstructed as the aggregate trades sharing a timestamp and side (two takers
hitting the book in the same millisecond would merge; rare at these sizes). Quotes are matched to orders
**strictly before** the fill, because the archive logs the quote update caused by the fill at the same
millisecond. Large contracts' bookTicker files run to 130 MB a day; reading stops once the window has passed.

## Hyperliquid (Amendment A3)

Public API `https://api.hyperliquid.xyz/info`, reachable from US networks, cached under
`data/raw/hyperliquid` by `rvr/hyperliquid.py`.

| Data | Endpoint | Coverage | Note |
|---|---|---|---|
| Universe | `meta` | 234 perps, 56 flagged delisted | delisted coins keep their history: no survivorship bias |
| Funding | `fundingHistory` | from 2023-05; hourly (8-hourly in the first weeks) | 500 rows per request, rate-limited (~45 weight a page of 1,200 a minute) |
| Prices | `candleSnapshot`, `1d` | full history | only the latest 5,000 candles per interval are served, so hourly bars do not reach 2023 |

Mirror (2026-10-13): 234 coins' daily candles; funding for the 228 that ever reached $1M ADV (no failures).
Quirks: delisted coins keep printing **zero funding** after they stop trading (`FTM`), the same
zombie pattern as Binance; candle `v` is base-asset volume, so quote volume is approximated as volume × the
average of open and close.

## dYdX v4 (Amendment A5a)

Public indexer `https://indexer.dydx.trade/v4`, cached under `data/raw/dydx` by `rvr/dydx.py`: 296 markets
(223 wound down, `FINAL_SETTLEMENT`), hourly `historicalFunding` (1,000 rows a request) and `1DAY` candles
with USD volume, 2023-10 → 2026-08; 4.9M funding records, 36% of them exactly zero (markets before real
trading and after wind-down). Breadth is the binding limit: ~20–30 contracts above $1M a day in 2024, under
10 from mid-2025.

## Point-in-time rules

Strategy code reads data only through `rvr/pit.py`. Every lookup is
`merge_asof(direction="backward", allow_exact_matches=False)`.

1. **Funding** is known strictly after `calc_time`. A decision exactly on a settlement hour sees the
   *previous* rate, because the new one was published milliseconds later.
2. **Prices** come only from bars with volume, known strictly after `close_time`, and go stale after
   `max_staleness_hours`.
3. **Trailing volume** covers bars that closed in `[t − window, t)`.
4. **Universe:** a contract is eligible while a traded bar has closed within `max_staleness_hours` and
   its current run of trading has lasted `min_history_hours`. A run restarts after more than
   `restart_gap_hours` without a trade. Nothing about a run is visible before its first bar closes,
   and a future delisting never removes a contract early.

`tests/test_leakage.py` checks every rule, including zombie and revived contracts, then corrupts all
data at or after a cutoff and asserts no lookup at or before the cutoff changes.
