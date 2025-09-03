# orchestrate.py
import os, json, subprocess, sys, datetime as dt
from pathlib import Path
import pandas as pd

from trading_script import download_price_data, last_trading_date
import pandas as pd


# --- env & LLM client ---
from dotenv import load_dotenv
load_dotenv()

from pathlib import Path
import shutil
import datetime as dt

from openai import OpenAI
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------- Config you can tweak ----------
REPO_ROOT = Path(__file__).resolve().parent
REPORTS = REPO_ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

# If your repo has specific CSVs/paths, wire them here:
LIVE_POSITIONS_CANDIDATES = [
    "Scripts and CSV Files/Live Portfolio.csv",
    "data/positions.csv",
    "positions.csv",
]

# Guardrails (paper mode): keep sizing sane even if LLM goes wild
MAX_NAMES = 10
MAX_WEIGHT_PER_NAME = 0.2     # 20% per position
MAX_TURNOVER = 0.5            # don't trade more than 50% of portfolio in one pass




# ---------- Helpers ----------

def _ordinal(n: int) -> str:
    # 1 -> 1st, 2 -> 2nd, 3 -> 3rd, 4 -> 4th, etc.
    return f"{n}{'th' if 11<=n%100<=13 else {1:'st',2:'nd',3:'rd'}.get(n%10,'th')}"

def pretty_date(d: dt.datetime | dt.date) -> str:
    if isinstance(d, dt.datetime):
        d = d.date()
    return f"{d.strftime('%B')} {_ordinal(d.day)}, {d.year}"

def build_user_prompt(positions_df: pd.DataFrame | None, cash: float | None, equity: float | None,
                      last_thesis: str = "") -> str:
    # compact holdings table (25 rows max)
    holdings_block = "None"
    if positions_df is not None and not positions_df.empty:
        show_cols = [c for c in ["ticker","shares","buy_price","cost_basis","stop_loss"] if c in positions_df.columns]
        small = positions_df[show_cols].head(25).copy()
        holdings_block = small.to_string(index=False)

    snapshot_lines = []
    if cash is not None:   snapshot_lines.append(f"Cash: ${cash:,.2f}")
    if equity is not None: snapshot_lines.append(f"Total Equity: ${equity:,.2f}")
    snapshot_block = " | ".join(snapshot_lines) if snapshot_lines else "N/A"

    return f"""
Context:
- Current holdings (truncated): 
{holdings_block}

Snapshot:
{snapshot_block}

Last Analyst Thesis (optional):
{last_thesis}

Instructions:
Propose actionable **buy/sell/hold** updates for next session within rules. 
Bias to liquid micro-caps; respect stops and budget. 
Return ONLY the JSON per schema; no prose.
"""

def _last_close(tkr: str) -> float | None:
    try:
        end = last_trading_date() + pd.Timedelta(days=1)
        start = end - pd.Timedelta(days=7)
        fr = download_price_data(tkr.upper().strip(), start=start, end=end, progress=False)
        if fr.df.empty:
            return None
        close = float(fr.df["Close"].iloc[-1])
        return close if close > 0 else None
    except Exception:
        return None
    

def build_system_prompt(max_names: int, max_w: float, cash: float | None) -> str:
    max_price = int(max(1, min(50, (cash or 100))))  # if cash small, keep picks cheap; cap at $50
    return f"""
You are a professional-grade, *conservative* portfolio analyst whose only goal is alpha **within strict rules**.

Follow these exactly:f
- **Budget discipline:** Use only available cash. Track cash implicitly via target *trade* weights (fraction of total equity). No new capital.
- **Execution limits:** Full shares only (engine enforces). No options, shorting, leverage, derivatives. Long-only.
- **Universe bias:** Prefer easily tradable U.S. micro-caps (<$300M). Existing positions may exceed $300M; you may only sell/hold those.
- **Liquidity:** Favor liquid names; avoid tickers unlikely to fill at open. Beware very low ADV/spreads.
- **Risk control:** Respect provided stop-losses; propose new stops only if defensible.
- **Cadence:** Daily EOD updates. Deep research Fri/Sat only (if needed, say so in `notes`).
- **Ticker validity:** Use ONLY real US-listed symbols (NYSE/NASDAQ/AMEX). Never invent placeholders (e.g., MCC/XYZ/ABC). If unsure, omit.
- **Affordability:** Prefer tickers priced **≤ ${max_price}** so at least 1 share is realistically buyable with current cash.

Output **ONLY valid JSON** matching this exact schema (no markdown, no backticks):
{{
  "universe": ["TICKER", "..."],
  "actions": [
    {{
      "ticker": "TICKER",
      "side": "buy" | "sell" | "hold",
      "weight": 0.00,
      "stop_loss": 0.00,
      "horizon_days": 5,
      "confidence": 0.0,
      "rationale": "≤30 words"
    }}
  ],
  "notes": "≤50 words on session-level risks or gaps"
}}

Hard constraints for `actions`:
- 1–{max_names} tickers max; ≤{int(max_w*100)}% per single name.
- Sum of buy notionals must fit within available cash.
- Only propose **buy** if ≥1 share is plausibly affordable with current cash.
- If **no** action is warranted, return "actions": [] with a terse "notes".

Be concise; rationale lines are short. JSON must parse.
"""


def is_real_ticker(tkr: str) -> bool:
    return _last_close(tkr) is not None

def filter_to_real_and_affordable(proposed: dict, cash: float | None) -> dict:
    acts = proposed.get("actions", [])
    keep = []
    for a in acts:
        t = str(a.get("ticker","")).upper().strip()
        if not t:
            continue
        px = _last_close(t)
        if px is None:           # not priceable -> drop
            continue
        if cash is not None and px > cash:  # can’t afford even 1 share -> drop
            continue
        a["ticker"] = t
        keep.append(a)
    proposed["actions"] = keep
    # optional: also clean the 'universe'
    uni = proposed.get("universe")
    if isinstance(uni, list):
        proposed["universe"] = [t.upper() for t in uni if isinstance(t, str) and is_real_ticker(t)]
    return proposed


# --- add near the top (after imports) ---
def read_state_for_prompt(csv_path: Path) -> tuple[pd.DataFrame|None, float|None, float|None]:
    import pandas as pd
    if not csv_path.exists():
        return None, None, None
    df = pd.read_csv(csv_path)
    # latest holdings (non TOTAL, non SELL)
    non_total = df[df["Ticker"] != "TOTAL"].copy()
    if not non_total.empty:
        non_total["Date"] = pd.to_datetime(non_total["Date"])
        latest_date = non_total["Date"].max()
        holdings = non_total[non_total["Date"] == latest_date]
        holdings = holdings[~holdings["Action"].astype(str).str.startswith("SELL")]
        holdings = holdings.rename(columns={"Ticker":"ticker","Shares":"shares",
                                            "Buy Price":"buy_price","Cost Basis":"cost_basis",
                                            "Stop Loss":"stop_loss"})
        holdings = holdings[["ticker","shares","buy_price","cost_basis","stop_loss"]]
    else:
        holdings = None
    # cash + equity from TOTAL
    totals = df[df["Ticker"] == "TOTAL"].copy()
    cash = equity = None
    if not totals.empty:
        totals["Date"] = pd.to_datetime(totals["Date"])
        latest = totals.sort_values("Date").iloc[-1]
        cash = float(latest["Cash Balance"])
        equity = float(latest["Total Equity"])
    return holdings, cash, equity

import re

def parse_execution_stdout(stdout: str) -> dict:
    """Grab the last JSON block printed by trading_script.py (the summary)."""
    # find last {...} block
    m = list(re.finditer(r"\{[\s\S]*\}", stdout))
    if not m:
        return {}
    last = m[-1].group(0)
    try:
        return json.loads(last)
    except Exception:
        return {}

def summarize_proposals(proposed: dict) -> tuple[list[str], list[str], list[str]]:
    """Return (buys, sells, holds) as nice bullet strings."""
    buys, sells, holds = [], [], []
    for a in proposed.get("actions", []):
        t = a.get("ticker","").upper()
        side = (a.get("side") or "").lower()
        w = a.get("weight", 0.0)
        stop = a.get("stop_loss", None)
        conf = a.get("confidence", None)
        why = a.get("rationale","")
        line = f"- **{t}** — weight ~{int(round(w*100))}%"
        if stop not in (None, "", 0):
            line += f", stop {stop}"
        if conf is not None:
            line += f", conf {conf:.2f}"
        if why:
            # keep it short
            line += f" — {why[:140]}"
        if side == "buy":
            buys.append(line)
        elif side == "sell":
            sells.append(line)
        else:
            holds.append(f"- **{t}** — hold ({why[:140]})" if why else f"- **{t}** — hold")
    return buys, sells, holds

def summarize_applied(applied_actions: list[dict]) -> list[str]:
    """Pretty-print what the engine actually did in paper mode."""
    lines = []
    for a in applied_actions:
        t = str(a.get("ticker","")).upper()
        s = (a.get("side") or "").upper()
        status = a.get("status","")
        sh = a.get("shares", None)
        px = a.get("price", None)
        if status == "filled":
            part = f"- {s}: **{t}**"
            if sh is not None: part += f" x {int(sh)}"
            if px is not None: part += f" @ ${px:.2f}"
            lines.append(part)
        else:
            lines.append(f"- {s}: **{t}** — {status}")
    return lines

def read_latest_cash_equity(csv_path: Path) -> tuple[float|None, float|None]:
    try:
        df = pd.read_csv(csv_path)
        tot = df[df["Ticker"] == "TOTAL"].copy()
        if tot.empty: return None, None
        tot["Date"] = pd.to_datetime(tot["Date"])
        latest = tot.sort_values("Date").iloc[-1]
        return float(latest["Cash Balance"]), float(latest["Total Equity"])
    except Exception:
        return None, None



def read_positions_csv():
    import pandas as pd
    for p in LIVE_POSITIONS_CANDIDATES:
        fp = REPO_ROOT / p
        if fp.exists():
            return pd.read_csv(fp)
    # fallback: empty df
    return None

def is_real_ticker(tkr: str) -> bool:
    """
    Treat a ticker as 'real' if we can fetch at least one recent price bar
    using the same resilient accessor as trading_script.py.
    """
    try:
        end = last_trading_date() + pd.Timedelta(days=1)
        start = end - pd.Timedelta(days=7)
        fr = download_price_data(tkr.upper().strip(), start=start, end=end, progress=False)
        return not fr.df.empty
    except Exception:
        return False

def filter_to_real_tickers(proposed: dict) -> dict:
    acts = proposed.get("actions", [])
    keep = []
    for a in acts:
        t = str(a.get("ticker","")).upper().strip()
        if not t:
            continue
        if is_real_ticker(t):
            a["ticker"] = t
            keep.append(a)
        # else drop silently
    proposed["actions"] = keep
    # optional: also purge any fake symbols from "universe"
    if "universe" in proposed and isinstance(proposed["universe"], list):
        proposed["universe"] = [t.upper() for t in proposed["universe"] if isinstance(t, str) and is_real_ticker(t)]
    return proposed


def render_md(proposed, executed, plot_path, llm_notes, csv_path: Path):
    now = dt.datetime.now()
    buys, sells, holds = summarize_proposals(proposed)

    # parse applied actions from trading_script stdout
    exec_obj = {}
    if isinstance(executed, dict):
        exec_obj = parse_execution_stdout(executed.get("stdout","") or "")
    applied = exec_obj.get("applied_actions", []) or []

    applied_lines = summarize_applied(applied)
    cash, equity = read_latest_cash_equity(csv_path)

    lines = []
    lines.append(f"# Daily Portfolio Decisions — {now:%Y-%m-%d}")
    if equity is not None or cash is not None:
        lines.append("")
        lines.append("**Snapshot**")
        if equity is not None: lines.append(f"- Total Equity: ${equity:,.2f}")
        if cash is not None:   lines.append(f"- Cash: ${cash:,.2f}")

    lines.append("\n## AI Proposal (Paper Mode)")
    if buys:
        lines.append("**Buys**")
        lines.extend(buys)
    if sells:
        lines.append("\n**Sells**")
        lines.extend(sells)
    if holds:
        lines.append("\n**Holds**")
        lines.extend(holds)
    if not (buys or sells or holds):
        lines.append("_No actions proposed today._")

    lines.append("\n## Executed (Simulated Fills)")
    if applied_lines:
        lines.extend(applied_lines)
    else:
        # fallback: show return code if nothing parsed
        rc = executed.get("returncode") if isinstance(executed, dict) else None
        lines.append(f"_No fills parsed (rc={rc})._")

    if llm_notes:
        lines.append("\n## Notes")
        lines.append(llm_notes)

    if plot_path:
        # embed image (works in GitHub/email clients that support markdown images)
        lines.append("\n## Performance")
        # both embed and path (some clients ignore images)
        lines.append(f"![Performance Chart]({plot_path})")
        lines.append(f"\n_Plot saved:_ `{plot_path}`")

    # Always include the raw JSON at the end (for audit/debug), but collapsed under a heading
    lines.append("\n---\n### Debug / Raw\n**Proposed JSON**")
    lines.append("```json")
    lines.append(json.dumps(proposed, indent=2))
    lines.append("```")
    lines.append("\n**Engine stdout (tail)**")
    tail = (executed.get("stdout","") or "")
    lines.append("```\n" + tail[-2000:] + "\n```")  # last ~2k chars

    return "\n".join(lines)


def send_email(markdown_text, subject):
    # Reuse your daily_email.py email method if you want. Minimal inline here:
    from email.message import EmailMessage
    import ssl, smtplib
    from_addr = os.getenv("FROM_EMAIL"); to_addr = os.getenv("TO_EMAIL")
    if not (from_addr and to_addr):
        print("No FROM_EMAIL/TO_EMAIL set; skipping email.")
        return
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = from_addr
    msg["To"] = to_addr
    msg.set_content(markdown_text)
    if os.getenv("GMAIL_SMTP","0") == "1":
        app_pw = os.getenv("GMAIL_APP_PASSWORD")
        ctx = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=ctx) as srv:
            srv.login(from_addr, app_pw)
            srv.send_message(msg)
    else:
        import requests
        sg = os.getenv("SENDGRID_API_KEY")
        if not sg:
            print("No email method configured; skipping email.")
            return
        r = requests.post(
            "https://api.sendgrid.com/v3/mail/send",
            headers={"Authorization": f"Bearer {sg}", "Content-Type": "application/json"},
            json={"personalizations":[{"to":[{"email":to_addr}]}],
                  "from":{"email":from_addr},
                  "subject":subject,
                  "content":[{"type":"text/plain","value":markdown_text}]},
            timeout=30)
        r.raise_for_status()

# ---------- LLM prompting ----------
def ask_llm_for_actions(prompt_text, context_text="", system_prompt=None):
    system = system_prompt or (
        "You are a portfolio assistant. Output ONLY valid JSON matching the schema. "
        f"Weights are fractions of total portfolio (0.0–1.0). Use <= {MAX_NAMES} names, "
        f"no single name > {MAX_WEIGHT_PER_NAME} weight. If no change, return actions:[]."
    )
    user = f"""CONTEXT:
{context_text}

PROMPT:
{prompt_text}

RESPONSE SCHEMA (JSON ONLY):
{{
  "universe": ["TICKER", "..."],
  "actions": [
    {{
      "ticker": "TICKER",
      "side": "buy" | "sell" | "hold",
      "weight": 0.0,
      "rationale": "one-paragraph why",
      "horizon_days": 1,
      "confidence": 0.0
    }}
  ],
  "notes": "any caveats or market risks"
}}
Return ONLY JSON, no backticks."""
    r = client.chat.completions.create(
        model=MODEL, temperature=0.2,
        messages=[{"role":"system","content":system},
                  {"role":"user","content":user}],
    )
    txt = r.choices[0].message.content.strip()
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        start, end = txt.find("{"), txt.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(txt[start:end+1])
        raise


# ---------- Guardrails ----------
def sanitize_actions(proposed):
    """
    - Limit number of names
    - Clamp weights
    - Ensure sides valid
    - Optional: cap aggregate turnover
    """
    actions = proposed.get("actions", [])
    cleaned = []
    for a in actions[:MAX_NAMES]:
        t = a.get("ticker","").upper().strip()
        side = a.get("side","hold").lower()
        if side not in {"buy","sell","hold"}: side = "hold"
        w = float(a.get("weight", 0.0))
        # clamp
        if w < 0: w = 0.0
        if w > MAX_WEIGHT_PER_NAME: w = MAX_WEIGHT_PER_NAME
        cleaned.append({
            "ticker": t,
            "side": side,
            "weight": round(w, 4),
            "rationale": a.get("rationale",""),
            "horizon_days": int(a.get("horizon_days", 5)),
            "confidence": float(a.get("confidence", 0.5)),
        })
    # (optional) turnover cap: normalize total |Δw| <= MAX_TURNOVER if you track current weights
    return {
        "universe": proposed.get("universe", []),
        "actions": cleaned,
        "notes": proposed.get("notes","")
    }

# ---------- Execution (paper) ----------
def run_trading_script(order_file: Path) -> dict:
    """Always call trading_script.py via subprocess (simpler, avoids import issues)."""
    cmd = [
        sys.executable,
        str(REPO_ROOT / "trading_script.py"),
        "--paper",
        "--order-file",
        str(order_file),
        "--file",
        str(REPO_ROOT / "chatgpt_portfolio_update.csv"),
    ]
    try:
        out = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=600)
        return {
            "status": "ok",
            "stdout": out.stdout,
            "stderr": out.stderr,
            "returncode": out.returncode,
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

    try:
        out = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=600)
        return {"status":"ok","stdout":out.stdout, "stderr":out.stderr, "returncode":out.returncode}
    except Exception as e:
        return {"status":"error","error":str(e)}

def run_generate_graph() -> Path | None:
    gp = REPO_ROOT / "generate_graph.py"
    if not gp.exists():
        return None
    try:
        out = subprocess.run(
            [sys.executable, str(gp), "--file", str(REPO_ROOT/"chatgpt_portfolio_update.csv"), "--no-show"],
            cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=600
        )
        # prefer explicit "SAVED:" lines if your script prints them
        saved = None
        for line in out.stdout.splitlines():
            if line.startswith("SAVED:"):
                saved = Path(line.split("SAVED:",1)[1].strip())
                break
        # fallback: newest image in Graphs/
        if not saved:
            plots_dir = REPO_ROOT / "Graphs"
            imgs = list(plots_dir.glob("*.png")) if plots_dir.exists() else []
            saved = max(imgs, key=lambda p: p.stat().st_mtime) if imgs else None
        if saved and saved.exists():
            # write a stable alias at repo root
            stable = REPO_ROOT / "Results.png"
            try:
                shutil.copyfile(saved, stable)
            except Exception:
                stable = saved  # fall back to original if copy fails
            return stable
        return None
    except Exception:
        return None


def main():
    # 1) Build the LLM context and prompt
    # You said you already have prompts—paste them here:
    YOUR_LONG_PROMPT = """
    Using recent data and my micro-cap selection heuristics, propose portfolio actions.
    Constraints: max 10 names, ≤20% per name, be mindful of liquidity.
    Prioritize catalysts, survivability, and risk control. Paper mode only.
    """

    # (Optional) Add a tiny context string with current positions / cash; you can expand later.
    positions_df = read_positions_csv()
    context = ""
    if positions_df is not None and not positions_df.empty:
        sample = positions_df.head(25)
        context = "Current positions (truncated 25 rows):\n" + sample.to_string(index=False)

    csv_path = REPO_ROOT / "chatgpt_portfolio_update.csv"
    positions_df, cash_val, equity_val = read_state_for_prompt(csv_path)

    sys_prompt = build_system_prompt(MAX_NAMES, MAX_WEIGHT_PER_NAME, cash_val)
    user_prompt = build_user_prompt(positions_df, cash_val, equity_val, last_thesis="")

    raw = ask_llm_for_actions(prompt_text=user_prompt, context_text="", system_prompt=sys_prompt)
    proposed = sanitize_actions(raw)
    proposed = filter_to_real_and_affordable(proposed, cash=cash_val)


    # 3) Save proposed orders JSON
    order_file = REPORTS / f"proposed_actions_{dt.datetime.now():%Y_%m_%d}.json"
    order_file.write_text(json.dumps(proposed, indent=2), encoding="utf-8")

    # 4) Paper “execute” via trading_script.py
    executed = run_trading_script(order_file)

    # 5) Generate plot
    plot_path = run_generate_graph()

    # 6) Make a report + email
    notes = proposed.get("notes","")
    csv_path = REPO_ROOT / "chatgpt_portfolio_update.csv"
    md = render_md(proposed, executed, str(plot_path) if plot_path else None, notes, csv_path)
    report_path = REPORTS / f"paper_run_{dt.datetime.now():%Y_%m_%d}.md"
    report_path.write_text(md, encoding="utf-8")

    send_email(md, subject="Micro-Cap — Paper Run (Auto)")

    print(f"Done. Wrote:\n - {order_file}\n - {report_path}")
    if plot_path: print(f" - Plot: {plot_path}")

if __name__ == "__main__":
    main()
