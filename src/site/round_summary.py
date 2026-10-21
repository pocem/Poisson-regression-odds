"""
An AI-written summary of the round (Google's Gemini, free API tier), shown at
the top of the website: the upcoming predictions next to Bet365's odds, the
outcomes with an expected value of at least MIN_EV, and how the model and
Bet365 did in each league's latest round.

The summary is saved to data/summary/round_summary.json together with a hash of
the data it was written from, and only regenerated when that data changes -- so
the daily runs don't use up the free quota when nothing moved. Without a
GEMINI_API_KEY (or on an API error) the last saved summary is kept.

Called by build_site.py; to (re)generate it on its own after a site build:

    python src/site/round_summary.py
"""

import hashlib
import json
import os
import time
from datetime import datetime, timezone

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SUMMARY_JSON = os.path.join(ROOT, "data", "summary", "round_summary.json")
# Free-tier models, tried in order: the free tier is the first to be turned away
# when a model is overloaded, so a busy model falls through to the next one.
MODELS = ["gemini-3.8-flash", "gemini-3.6-flash", "gemini-3.5-flash-lite"]
API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
RETRY_WAITS = [10, 30]  # seconds before retrying an overloaded / rate-limited model
RETRYABLE = {429, 500, 503}

SYSTEM = """You write the weekly round summary for a football prediction website. A bivariate \
Poisson regression model predicts home win / draw / away win chances for Premier League, \
Bundesliga, La Liga, Serie A and Ligue 1 matches, and the site compares them with Bet365's odds.

You get the data as JSON. Write for football fans who are not statisticians:
- Open with the round's big picture in one or two sentences.
- Then the outcomes with high expected value: what the model sees that Bet365 doesn't, \
strongest cases first.
- Then, for leagues with played results, how the model and Bet365 did in the latest round \
(log loss: lower is better), picking out the few most notable hits and misses rather than \
listing them all.
- Keep it to roughly 300 words, in short paragraphs of plain text. No markdown, headings, \
bullet points or tables. Quote only the numbers that make a point, not every figure available.

Rules: chances are percentages and odds are decimal odds. Don't discuss bookmaker margins or \
overrounds, and don't add figures that aren't in the data. Over its test seasons the model has \
been slightly less accurate than Bet365, so present disagreements as worth a closer look, \
never as tips or sure things, and don't encourage betting."""


def _pct(p):
    return None if p is None else round(100 * p, 1)


def summary_input(data):
    """The compact, stable view of the site's data that the summary is written from."""
    leagues = []
    for lg in data["leagues"]:
        nx = lg["next"]
        upcoming = []
        for g in (nx["games"] if nx else []):
            upcoming.append({
                "kickoff": g["kickoff"][:10] if g["kickoff_tbc"] else g["kickoff"][:16],
                "match": f"{g['home']} v {g['away']}",
                "model_chance_pct_hda": [_pct(x) for x in g["p"]],
                "model_odds_hda": g["odds"],
                "bet365_odds_hda": g["b365"],
                "bet365_chance_pct_hda": [_pct(x) for x in g["b365_prob"]] if g["b365_prob"] else None,
                "elo": g["elo"],
                "expected_goals": g["xg"],
            })
        latest = max((m["round"] for m in lg["matches"]), default=None)
        last_round = [{
            "match": f"{m['home']} {m['score'][0]}-{m['score'][1]} {m['away']}",
            "result": m["result"],
            "model_pick": m["pick"], "bet365_pick": m["book_pick"],
            "model_chance_pct_hda": [_pct(x) for x in m["p"]],
            "bet365_odds_hda": m["b365"],
            "log_loss": {"model": round(m["ll"]["model"], 3),
                         "bet365": None if m["ll"]["b365"] is None else round(m["ll"]["b365"], 3)},
        } for m in lg["matches"] if m["round"] == latest]
        s = lg["summary"]
        leagues.append({
            "league": lg["name"],
            "next_round": nx["round"] if nx else None,
            "upcoming": upcoming,
            "latest_played_round": latest,
            "latest_round_results": last_round,
            "season_so_far": None if not s else {
                "matches": s["n"], "log_loss_model": round(s["ll_model"], 3),
                "log_loss_bet365": None if s["ll_b365"] is None else round(s["ll_b365"], 3),
                "favourite_won_model_pct": _pct(s["acc_model"]), "favourite_won_bet365_pct": _pct(s["acc_b365"]),
            },
        })
    value = [{
        "league": r["league"], "kickoff": r["kickoff"][:16], "match": f"{r['home']} v {r['away']}",
        "outcome": r["label"], "model_chance_pct": _pct(r["p_model"]), "bet365_chance_pct": _pct(r["p_b365"]),
        "bet365_odds": r["odds_b365"], "expected_value_pct": _pct(r["ev"]),
    } for r in data["differences"]]
    return {"value_outcomes_ev_10pct_or_more": value, "leagues": leagues}


def load_saved():
    if os.path.exists(SUMMARY_JSON):
        with open(SUMMARY_JSON, encoding="utf-8") as f:
            return json.load(f)
    return None


class Unavailable(RuntimeError):
    """The model is overloaded, rate-limited or missing -- another model may work."""


def write_summary(payload):
    """(text, model) from the first model in MODELS that answers."""
    errors = []
    for model in MODELS:
        for wait in RETRY_WAITS + [None]:
            try:
                return _generate(model, payload)
            except Unavailable as e:
                errors.append(f"{model}: {e}")
                if wait is None or "HTTP 404" in str(e):
                    break
                print(f"  {model} unavailable ({str(e)[:60]}), retrying in {wait}s")
                time.sleep(wait)
    raise RuntimeError("no model available -- " + " | ".join(e[:160] for e in errors))


def _generate(model, payload):
    resp = requests.post(
        API_URL.format(model=model),
        headers={"x-goog-api-key": os.environ["GEMINI_API_KEY"], "Content-Type": "application/json"},
        json={
            "systemInstruction": {"parts": [{"text": SYSTEM}]},
            "contents": [{"role": "user", "parts": [{"text":
                "Here is this round's data:\n\n" + json.dumps(payload, ensure_ascii=False, indent=1)
                + "\n\nWrite the round summary."}]}],
            # Generous: on thinking models the reasoning counts towards this limit too.
            "generationConfig": {"maxOutputTokens": 8192},
        },
        timeout=120,
    )
    if resp.status_code in RETRYABLE or resp.status_code == 404:
        raise Unavailable(f"HTTP {resp.status_code}: {resp.text[:300]}")
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
    body = resp.json()
    blocked = body.get("promptFeedback", {}).get("blockReason")
    if blocked:
        raise RuntimeError(f"prompt blocked ({blocked})")
    candidates = body.get("candidates") or []
    if not candidates:
        raise RuntimeError("no candidates in the response")
    finish = candidates[0].get("finishReason")
    if finish not in (None, "STOP"):
        raise RuntimeError(f"generation stopped early ({finish})")
    parts = candidates[0].get("content", {}).get("parts", [])
    text = "".join(p.get("text", "") for p in parts if not p.get("thought")).strip()
    if not text:
        raise RuntimeError("empty response")
    return text, body.get("modelVersion", model)


def round_summary(data):
    """{"text", "generated_at", "model"} for the page, or None if there's none yet."""
    payload = summary_input(data)
    if not any(lg["upcoming"] or lg["latest_round_results"] for lg in payload["leagues"]):
        return None
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]
    saved = load_saved()
    if saved and saved.get("input_hash") == digest:
        print("Round summary: data unchanged, keeping the saved summary")
        return saved
    if not os.environ.get("GEMINI_API_KEY"):
        print("Round summary: GEMINI_API_KEY not set, keeping the saved summary")
        return saved

    try:
        text, model = write_summary(payload)
    except (requests.RequestException, ValueError, RuntimeError) as e:
        print(f"  WARNING: couldn't write the round summary ({e}); keeping the saved one")
        return saved
    summary = {"text": text, "model": model, "input_hash": digest,
               "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    os.makedirs(os.path.dirname(SUMMARY_JSON), exist_ok=True)
    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Round summary: written by {model}, saved to {os.path.relpath(SUMMARY_JSON, ROOT)}")
    return summary


if __name__ == "__main__":
    import re
    html = open(os.path.join(ROOT, "generated_site", "index.html"), encoding="utf-8").read()
    raw = re.search(r'<script id="data" type="application/json">(.*?)</script>', html, re.S).group(1)
    result = round_summary(json.loads(raw.replace("<\\/", "</")))
    print(result["text"] if result else "No summary.")
