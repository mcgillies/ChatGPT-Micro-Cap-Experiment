import os, ssl, datetime as dt, pandas as pd
from email.message import EmailMessage
from pathlib import Path
# --- add near the top ---
from dotenv import load_dotenv
load_dotenv()  # make sure .env variables are loaded

from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # set OPENAI_MODEL in .env if you want

def llm_notes(positions_df, pnl_today):
    sys_prompt = (
        "You are a cautious portfolio assistant. "
        "Give 3 bullets: (1) notable moves/risk (2) watchlist idea (3) action to consider. "
        "No trade instructions; paper-mode only."
    )

    # keep prompt small to avoid token bloat
    show_cols = [c for c in ["Ticker","Qty","AvgCost","LastPrice","MktValue"] if c in positions_df.columns]
    small = positions_df[show_cols].copy() if not positions_df.empty else positions_df
    if not small.empty and len(small) > 25:
        small = small.head(25)  # cap rows
    pos_txt = small.to_string(index=False) if not small.empty else "No positions."

    user_prompt = f"Positions (truncated):\n{pos_txt}\n\nApprox PnL today: {pnl_today:+.2f}%"

    try:
        r = client.chat.completions.create(
            model=MODEL,
            temperature=0.2,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return r.choices[0].message.content.strip()
    except Exception as e:
        # helpful error message in the email if model/keys are wrong
        return f"(LLM error: {e})"


def load_positions():
    """
    Try to read a positions CSV from the repo; else use a placeholder.
    Adjust this to match your repo’s actual file (e.g., 'Scripts and CSV Files/Live Portfolio.csv').
    """
    candidates = [
        "Scripts and CSV Files/Live Portfolio.csv",
        "data/positions.csv",
        "positions.csv",
    ]
    for p in candidates:
        if Path(p).exists():
            df = pd.read_csv(p)
            # Expect columns like: Ticker, Qty, AvgCost, LastPrice, MktValue
            return df
    # Fallback placeholder
    return pd.DataFrame([
        {"Ticker":"ABC", "Qty":100, "AvgCost":2.10, "LastPrice":2.34, "MktValue":234.0},
        {"Ticker":"XYZ", "Qty":50,  "AvgCost":5.80, "LastPrice":5.10, "MktValue":255.0},
    ])

def compute_pnl_today(df):
    # If you have a “PrevClose” column, replace this with ((Last-PrevClose)/PrevClose)*100
    if df.empty or "AvgCost" not in df or "LastPrice" not in df:
        return 0.0
    # naive proxy: value vs. cost
    val = (df["LastPrice"] * df.get("Qty", 0)).sum()
    cost = (df["AvgCost"] * df.get("Qty", 0)).sum()
    if cost == 0:
        return 0.0
    return (val / cost - 1.0) * 100.0

def make_markdown(df, notes, pnl_today):
    now = dt.datetime.now()
    title = f"# Daily Snapshot — {now:%Y-%m-%d}\n"
    # Compact table
    show_cols = [c for c in ["Ticker","Qty","AvgCost","LastPrice","MktValue"] if c in df.columns]
    table_md = (
        "## Positions\n\n" +
        (df[show_cols].to_markdown(index=False) if not df.empty else "_No positions_\n")
    )
    meta_md = f"\n\n**Paper-mode:** no orders executed.\n\n**PnL (rough):** {pnl_today:+.2f}%\n"
    notes_md = "\n\n## LLM Notes (3 bullets)\n" + notes + "\n"
    return title + table_md + meta_md + notes_md

def send_email(markdown_text, subject):
    use_gmail = os.getenv("GMAIL_SMTP","0") == "1"
    if use_gmail:
        from_addr = os.getenv("FROM_EMAIL"); to_addr = os.getenv("TO_EMAIL")
        app_pw = os.getenv("GMAIL_APP_PASSWORD")
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = from_addr
        msg["To"] = to_addr
        msg.set_content(markdown_text)
        ctx = ssl.create_default_context()
        import smtplib
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=ctx) as srv:
            srv.login(from_addr, app_pw)
            srv.send_message(msg)
    else:
        # SendGrid option (set SENDGRID_API_KEY)
        import requests
        sg = os.getenv("SENDGRID_API_KEY")
        if not sg:
            raise RuntimeError("No email method configured. Set GMAIL_SMTP=1 + GMAIL_APP_PASSWORD or SENDGRID_API_KEY.")
        r = requests.post(
            "https://api.sendgrid.com/v3/mail/send",
            headers={"Authorization": f"Bearer {sg}", "Content-Type": "application/json"},
            json={
                "personalizations":[{"to":[{"email":os.getenv("TO_EMAIL")}]}],
                "from":{"email":os.getenv("FROM_EMAIL")},
                "subject":subject,
                "content":[{"type":"text/plain","value":markdown_text}]
            }, timeout=30
        )
        r.raise_for_status()

def main():
    df = load_positions()
    pnl = compute_pnl_today(df)
    notes = llm_notes(df, pnl)
    md = make_markdown(df, notes, pnl)

    # Save artifact (nice to keep a record locally)
    reports = Path("reports"); reports.mkdir(exist_ok=True)
    path = reports / f"snapshot_{dt.datetime.now():%Y_%m_%d}.md"
    path.write_text(md, encoding="utf-8")

    send_email(md, subject="Micro-Cap — Daily Snapshot (paper mode)")
    print(f"Sent. Also saved: {path}")

if __name__ == "__main__":
    main()
