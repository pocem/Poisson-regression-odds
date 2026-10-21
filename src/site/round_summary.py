"""
Claude's written summary of the round, shown at the top of the website:
the upcoming predictions next to Bet365's odds, the outcomes with an expected
value of at least MIN_EV, and how the model and Bet365 did in each league's
latest round.

The summary is saved to data/summary/round_summary.json together with a hash of
the data it was written from, and only regenerated when that data changes -- so
the daily runs don't pay for a new summary when nothing moved. Without an
ANTHROPIC_API_KEY (or on an API error) the last saved summary is kept.

Called by build_site.py; to (re)generate it on its own after a site build:

    python src/site/round_summary.py
"""

import hashlib
import json
import os
from datetime import datetime, timezone

import anthropic

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SUMMARY_JSON = os.path.join(ROOT, "data", "summary", "round_summary.json")
MODEL = "claude-opus-5-5"

SYSTEM = """You write the weekly round summary for a football prediction website. A bivariate \
Poisson regression model predicts home win / draw / away win chances for Premier League, \
Bundesliga, La Liga, Serie A and Ligue 1 matches, and the site compares them with Bet365's odds.

You get the data as JSON. Write for football fans who are not statisticians:
- Open with the round's big picture in one or two sentences.
- Then the outcomes with high expected value: what the model sees that Bet365 doesn't, \
grouped or ordered so the strongest cases come first. Only use the numbers you are given.
- Then, for leagues with played results, how the model and Bet365 did in the latest round \
(log loss: lower is better), naming notable hits and misses.
- Keep it to roughly 250-400 words, in short paragraphs of plain text. No markdown, headings, \
bullet points or tables.

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
                "matches": s["n"], "log_loss_model": s["ll_model"], "log_loss_bet365": s["ll_b365"],
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


def write_summary(payload):
    client = anthropic.Anthropic()
    response = client.beta.messages.create(
        model=MODEL,
        max_tokens=16000,
        system=SYSTEM,
        output_config={"effort": "medium"},
        # On a safety decline, the API re-runs the request on a suitable fallback model.
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        messages=[{"role": "user", "content":
                   "Here is this round's data:\n\n" + json.dumps(payload, ensure_ascii=False, indent=1)
                   + "\n\nWrite the round summary."}],
    )
    if response.stop_reason == "refusal":
        raise RuntimeError(f"request declined ({response.stop_details})")
    if response.stop_reason == "max_tokens":
        raise RuntimeError("summary was cut off at max_tokens")
    text = "".join(b.text for b in response.content if b.type == "text").strip()
    if not text:
        raise RuntimeError("empty response")
    return text, response.model


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
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Round summary: ANTHROPIC_API_KEY not set, keeping the saved summary")
        return saved

    try:
        text, model = write_summary(payload)
    except (anthropic.APIError, RuntimeError) as e:
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
