# AI Conversion Co-Pilot — reference pipeline

A runnable Python implementation of the proposal's architecture: User Persona
Engine, Car Persona Engine, Matching Engine, Adaptive Engagement (LinUCB),
and the Agent Layer (Persona Card Generation + Pre-Call Briefing), wired
together end to end against synthetic data.

## Files

| File | Maps to |
|---|---|
| `data_gen.py` | Data Layer — synthetic `user_events`, `crm_leads`, `engagement_log`, `car_catalog`, `call_log`, `order_data`, unified into `user_features_daily` / `car_features_current` (Step 0) |
| `user_persona_engine.py` | HDBSCAN clustering + XGBoost propensity scoring + SHAP drivers |
| `car_persona_engine.py` | Condition scoring + plain-language condition summary |
| `matching_engine.py` | Cosine-similarity matching (the proposal's own day-one/cold-start approach), with `recall_at_k`/`precision_at_k` ready to wire up once real booking outcomes exist |
| `adaptive_engagement.py` | Real LinUCB contextual bandit (ridge regression per arm, UCB arm selection), simulated against a hidden reward function |
| `agent_layer.py` | Persona Card Generation Agent + Pre-Call Briefing Agent. `call_llm()` is the one seam where a real Claude API call belongs — swap it in and delete the template fallback |
| `main.py` | Orchestrates all of the above end to end |

## Running it

```bash
pip install -r requirements.txt
python main.py
```

This writes three files: `kpis.json` (funnel + guardrail metrics, same shape
as the proposal's KPI table), `scored_leads.csv` (every synthetic user with
cluster, propensity score, and outcome), and `sample_briefings.txt` (pre-call
briefings for 5 example leads, exactly as an RM would see them).

Every module also runs standalone for quick inspection, e.g. `python
user_persona_engine.py` or `python adaptive_engagement.py`.

## What's real vs. simplified here

- **Real**: HDBSCAN clustering, XGBoost + SHAP, LinUCB with ridge regression
  per arm and proper context vectors, cosine-similarity matching (this is
  literally the proposal's specified cold-start path, not a simplification
  of it).
- **Simplified for a reference build with no warehouse/vendor access**:
  the six source tables are synthetic; car condition/damage detection is a
  synthetic stand-in for the Ravin.ai-style vendor call; the two-tower neural
  matcher isn't trained (there's no historical booking log yet to train it
  on — that's precisely why the proposal specifies cosine similarity as the
  correct thing to ship on day one); persona card/briefing text generation
  uses deterministic templates instead of a live Claude call (see
  `agent_layer.call_llm`, which is the exact integration point).
- Every module degrades gracefully if `hdbscan`, `xgboost`, or `shap` aren't
  installed (falls back to KMeans / logistic regression / linear
  contributions respectively), so this still runs with a minimal environment.

## One honest caveat

This codebase was written and reviewed carefully, but the sandbox this was
built in couldn't execute Python this session (a temporary environment
outage, not a project issue), so it has not been run end to end here. Run
`python main.py` on your end — if anything errors, share the traceback and
it'll get fixed immediately. The companion file, `prototype.html`, *has*
been exercised as thoroughly as static review allows and is meant to be the
one you open first to see the system work.
