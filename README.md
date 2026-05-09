# deribit-polymarket-scanner

Compares Polymarket crypto binary markets against Deribit options-implied probabilities to find mispricings.

## What it does

Scans Polymarket for BTC/ETH binary markets, fetches the Deribit vol surface (calls + puts, per-expiry forward prices), and prices each market using the appropriate model:

| Market type | Example | Model |
|---|---|---|
| Digital option | "Will BTC be above $100K on Dec 31?" | N(d₂) via Black-Scholes |
| Range market | "Will BTC be between $80K–$82K on May 10?" | Digital call spread |
| Barrier / one-touch | "Will BTC reach $150K in 2026?" | GBM first-passage (Reiner-Rubinstein) |

For each market the scanner reports:

- **PM price** — current Polymarket mid
- **Deribit-implied probability** — from the vol surface
- **Edge** — PM price minus Deribit-implied (%)
- **Action** — BUY\_YES or BUY\_NO
- **Cost / monthly return / APY**
- **Liquidity** — Polymarket order book depth

## Usage

```bash
# Install dependencies
python3 -m pip install -r requirements.txt
python3 -m pip install dspy-ai   # optional — enables LLM-based market parsing

# Run with defaults (min edge 3%, min liquidity $10K)
python3 scripts/deribit_scanner.py

# Custom thresholds
python3 scripts/deribit_scanner.py --min-edge 5 --min-liquidity 10000

# Disable LLM fallback, run continuously every 10 minutes
python3 scripts/deribit_scanner.py --no-llm --loop --interval 600

# Run tests
python3 -m pytest
```

### Flags

| Flag | Default | Description |
|---|---|---|
| `--min-edge` | 3.0 | Minimum edge % to display |
| `--min-liquidity` | 10000 | Minimum Polymarket liquidity ($) |
| `--no-llm` | off | Use regex-only market matching (no dspy) |
| `--loop` | off | Run continuously |
| `--interval` | 600 | Seconds between scans (with `--loop`) |
| `--risk-free-rate` | 0.05 | Risk-free rate for pricing |

## Data sources

Both APIs are public and require no authentication.

- **Polymarket** — `/events` endpoint, filtered to crypto markets
- **Deribit** — `/public/get_instruments` + `/public/get_order_book` for the full options chain

## Caveats

**This is a pricing tool, not an arbitrage bot.** Edge is statistical, not risk-free.

- **Hedging cost**: Polymarket trades binary outcomes; Deribit trades vanilla options. Replicating a digital with a call/put spread introduces pin risk near expiry.
- **Vol surface extrapolation**: Deribit has finite strike coverage. Probabilities for extreme strikes are extrapolated and marked with `~` — treat these as rough estimates.
- **Barrier pricing**: The Reiner-Rubinstein formula assumes continuous GBM monitoring. Polymarket barrier markets resolve on 1-minute candles, so the modelled probability will be slightly higher than realised.
- **Settlement basis risk**: Polymarket uses its own oracle; Deribit settles on its BTC/ETH index. These can diverge at expiry, especially around exchange outages or low-liquidity windows.
- **Vol surface staleness**: Deribit vol quotes are fetched at scan time. Fast-moving markets can make the snapshot stale within minutes.

No positions are opened, modified, or closed. The scanner is read-only. Authenticated trading helpers respect `DRY_RUN=true` by returning simulated order/cancel results without touching the CLOB client.
