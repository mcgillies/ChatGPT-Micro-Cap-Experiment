import argparse
from pathlib import Path
import datetime as dt

import matplotlib.pyplot as plt
import pandas as pd
from trading_script import download_price_data  # reuse robust accessor


def load_portfolio_totals(csv_path: Path) -> pd.DataFrame:
    """Return Date and Total Equity (no normalization)."""
    df = pd.read_csv(csv_path)
    totals = df[df["Ticker"] == "TOTAL"].copy()
    if totals.empty:
        raise RuntimeError(f"No 'TOTAL' rows found in {csv_path}. Run trading_script.py at least once.")
    totals["Date"] = pd.to_datetime(totals["Date"])
    totals = totals.sort_values("Date")
    out = totals[["Date", "Total Equity"]].rename(columns={"Total Equity": "Portfolio Equity ($)"})
    return out


def download_sp500(start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    """
    Pull S&P 500 using resilient accessor via trading_script.download_price_data.
    Tries ^GSPC, then SPY if needed. Returns Date and SPX Level.
    """
    fr = download_price_data("^GSPC", start=start_date, end=end_date + pd.Timedelta(days=1), progress=False)
    df = fr.df.copy()
    if df.empty:
        fr = download_price_data("SPY", start=start_date, end=end_date + pd.Timedelta(days=1), progress=False)
        df = fr.df.copy()

    if df.empty:
        raise RuntimeError("Failed to fetch S&P 500 (^GSPC) or proxy (SPY).")

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")
    df = df.sort_index()

    close = df["Close"].astype(float).dropna()
    if close.empty:
        raise RuntimeError("S&P data has no Close values after fetch.")

    out = close.reset_index()
    out.columns = ["Date", "S&P 500 Level"]
    return out


def make_plot(chatgpt: pd.DataFrame, spx: pd.DataFrame, out_path: Path, show: bool) -> None:
    """
    Dual-axis plot:
      - Left Y: Portfolio Equity ($)
      - Right Y: S&P 500 Level
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Left axis: portfolio equity in dollars
    l1, = ax1.plot(chatgpt["Date"], chatgpt["Portfolio Equity ($)"], linewidth=2, label="Portfolio Equity ($)")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Portfolio Equity ($)")
    ax1.grid(True, which="both", alpha=0.3)

    # Right axis: SPX level
    ax2 = ax1.twinx()
    l2, = ax2.plot(spx["Date"], spx["S&P 500 Level"], linestyle="--", linewidth=2, label="S&P 500 Level")
    ax2.set_ylabel("S&P 500 Level")

    # Title & legend
    plt.title("Portfolio vs. S&P 500 (Absolute, Not Normalized)")
    # Build a combined legend
    lines = [l1, l2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper left")

    fig.autofmt_xdate()
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)
    print(f"SAVED:{out_path}")  # picked up by orchestrator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="chatgpt_portfolio_update.csv",
                    help="Path to chatgpt_portfolio_update.csv")
    ap.add_argument("--outdir", default="Graphs", help="Directory to save the PNG")
    ap.add_argument("--no-show", action="store_true", help="Don’t call plt.show() (CI/headless)")
    args = ap.parse_args()

    csv_path = Path(args.file)
    if not csv_path.exists():
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
    out_path = Path(args.outdir) / f"portfolio_vs_spx_{stamp}.png"
    make_plot(chatgpt, spx, out_path, show=not args.no_show)


if __name__ == "__main__":
    print("generating graph...")
    main()

