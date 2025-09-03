import argparse
from pathlib import Path
import datetime as dt

import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
from trading_script import download_price_data  # reuse your robust accessor



def load_portfolio_totals(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    totals = df[df["Ticker"] == "TOTAL"].copy()
    if totals.empty:
        raise RuntimeError(f"No 'TOTAL' rows found in {csv_path}. Run trading_script.py at least once.")
    totals["Date"] = pd.to_datetime(totals["Date"])
    totals = totals.sort_values("Date")
    # Normalize to $100 baseline from the first TOTAL row
    first_equity = float(totals["Total Equity"].iloc[0])
    totals["Equity ($100)"] = totals["Total Equity"] / first_equity * 100.0
    return totals[["Date", "Equity ($100)"]]


def download_sp500(start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    """
    Pull S&P 500 levels using the same robust accessor as trading_script:
    - Yahoo via yfinance
    - Stooq via pandas-datareader
    - Stooq CSV
    - Proxy: ^GSPC -> SPY (already handled inside download_price_data)
    Normalize to $100 at the first available date.
    """
    fr = download_price_data("^GSPC", start=start_date, end=end_date + pd.Timedelta(days=1), progress=False)
    df = fr.df
    if df.empty:
        # final fallback: try SPY directly (in case proxy path isn’t hit)
        fr = download_price_data("SPY", start=start_date, end=end_date + pd.Timedelta(days=1), progress=False)
        df = fr.df

    if df.empty:
        raise RuntimeError("Failed to fetch S&P 500 (both ^GSPC and SPY) via resilient accessor.")

    # Ensure datetime index and sort
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")
    df = df.sort_index()

    # Build normalized series
    close = df["Close"].astype(float).dropna()
    if close.empty:
        raise RuntimeError("S&P data has no Close values after fetch.")
    base = float(close.iloc[0])
    spx_norm = (close / base) * 100.0
    out = spx_norm.reset_index()
    out.columns = ["Date", "SPX ($100)"]
    return out



def make_plot(chatgpt: pd.DataFrame, spx: pd.DataFrame, out_path: Path, show: bool) -> None:
    # Join by date for clean annotation points (not required for plotting)
    # We’ll plot separately so different calendars don’t block.
    plt.figure(figsize=(10, 6))
    # Minimal styling that works headless
    plt.plot(chatgpt["Date"], chatgpt["Equity ($100)"], label="ChatGPT ($100)", linewidth=2)
    plt.plot(spx["Date"], spx["SPX ($100)"], label="S&P 500 ($100)", linestyle="--", linewidth=2)

    # Annotations
    cg_last_date = chatgpt["Date"].iloc[-1]
    cg_last_val = float(chatgpt["Equity ($100)"].iloc[-1])
    spx_last_date = spx["Date"].iloc[-1]
    spx_last_val = float(spx["SPX ($100)"].iloc[-1])

    plt.text(cg_last_date, cg_last_val * 1.01, f"{cg_last_val-100:+.1f}%", fontsize=9)
    plt.text(spx_last_date, spx_last_val * 1.01, f"{spx_last_val-100:+.1f}%", fontsize=9)

    plt.title("ChatGPT Micro-Cap vs S&P 500 (Normalized to $100)")
    plt.xlabel("Date")
    plt.ylabel("Value of $100")
    plt.xticks(rotation=15)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    if show:
        plt.show()
    plt.close()
    print(f"SAVED:{out_path}")  # <-- helpful for orchestrator logs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="chatgpt_portfolio_update.csv",
                    help="Path to chatgpt_portfolio_update.csv")
    ap.add_argument("--outdir", default="Graphs", help="Directory to save the PNG")
    ap.add_argument("--no-show", action="store_true", help="Don’t call plt.show() (CI/headless)")
    args = ap.parse_args()

    csv_path = Path(args.file)
    if not csv_path.exists():
        # fallback to old location
        alt = Path("Scripts and CSV Files") / "chatgpt_portfolio_update.csv"
        if alt.exists():
            csv_path = alt
        else:
            raise FileNotFoundError(f"Could not find portfolio CSV at {args.file} or {alt}")

    chatgpt = load_portfolio_totals(csv_path)
    start_date = pd.to_datetime(chatgpt["Date"].min())
    end_date = pd.to_datetime(chatgpt["Date"].max())
    spx = download_sp500(start_date, end_date)

    stamp = dt.datetime.now().strftime("%Y_%m_%d")
    out_path = Path(args.outdir) / f"chatgpt_vs_spx_{stamp}.png"
    make_plot(chatgpt, spx, out_path, show=not args.no_show)


if __name__ == "__main__":
    print("generating graph...")
    main()
