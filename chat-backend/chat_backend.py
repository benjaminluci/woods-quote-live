# chat_backend.py — Woods Quoting Assistant
# - Uses GET /quote?model=...&dealer_number=... (+ options)
# - Detects models like bb60.30, bw12.40, ds8.30 → normalized to uppercase
# - Stores dealer after dealer-discount and auto-injects dealer_number into quote calls
# - Full knowledge prompt included (verbatim)
# - Tool schema includes a comprehensive set of option fields (+ additionalProperties=True)

from __future__ import annotations
import os, re, json, time, logging, traceback
from typing import Any, Dict, List, Tuple

import requests
from flask import Flask, request, jsonify
try:
    from flask_cors import CORS
except Exception:
    CORS = None

# ---------------- Env & logging ----------------
QUOTE_API_BASE = os.environ.get("QUOTE_API_BASE", "https://woods-quote-api.onrender.com").rstrip("/")
OPENAI_MODEL   = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
HTTP_TIMEOUT   = float(os.environ.get("HTTP_TIMEOUT", "60"))
RETRY_ATTEMPTS = int(os.environ.get("RETRY_ATTEMPTS", "2"))
LOG_LEVEL      = os.environ.get("LOG_LEVEL", "INFO").upper()

logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
log = logging.getLogger("woods-qa")

# ---------------- OpenAI (optional; planner) ----------------
client = None
try:
    from openai import OpenAI
    if os.environ.get("OPENAI_API_KEY"):
        client = OpenAI()
except Exception:
    client = None

# ---------------- Flask & CORS ----------------
app = Flask(__name__)
if CORS:
    allow = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "").split(",") if o.strip()]
    if allow:
        CORS(app, resources={r"/chat": {"origins": allow}}, supports_credentials=False)
    else:
        CORS(app, supports_credentials=False)

# ---------------- In-memory sessions ----------------
SESS: Dict[str, Dict[str, Any]] = {}
SESSION_TTL = 60 * 30  # 30 minutes

# ---------------- FULL KNOWLEDGE (verbatim) ----------------
KNOWLEDGE = r"""
You are a quoting assistant for Woods Equipment dealership staff. Your primary job is to retrieve part numbers, list prices, dealer discounts, and configuration requirements exclusively from the Woods Pricing API. You never fabricate data, never infer pricing, and only ask configuration questions when required.

---
Core Rules
- A dealer number is required before quoting. Use the Pricing API to look up the dealer’s discount. Do not begin quotes or give pricing without it.
- Dealer numbers may be remembered within a session and across multiple quotes for the same user unless the dealer provides a new number.
- All model, accessory, and pricing data must be pulled directly from the API. Never invent, infer, reuse, or cache data.
- Every quote must pull fresh pricing from the API for all items — including list prices and accessories.
- If a valid part number returns no price, quoting must stop and inform the dealer to escalate the issue.

---
API Error Handling
- Retry any connector error once automatically before showing an error.
- If retry succeeds, proceed normally.
- If retry fails, show: “There was a system error while retrieving data. Please try again shortly. If the issue persists, escalate to Benjamin Luci at 615-516-8802.”

---
Pricing Logic
1. Retrieve list price for each part number from API.
2. Apply dealer discount from lookup.
3. Apply 12% cash discount on top of dealer discount, unless dealer discount is 5% — then the cash discount is 5%.
4. These calculations are enforced on the backend and provided as `_enforced_totals` inside the API response.

Formula:
Final Net = list_price_total − dealer_discount_total − cash_discount_total
This is pre-calculated in `_enforced_totals.final_net`. Use it directly.

---
_enforced_totals format (example):
- list_price_total → full list price of all items combined
- dealer_discount_total → total discount from list price based on dealer discount
- cash_discount_total → additional discount based on payment terms
- final_net → amount after dealer + cash discounts ✅

---
Quote Output Format
- Begin with "**Woods Equipment Quote**"
- Include the dealer name and dealer number below the title
- Show:
  - Show:
  - List Price → `_enforced_totals.list_price_total`
  - Dealer Discount → `_enforced_totals.dealer_discount_total`
  - Cash Discount → `_enforced_totals.cash_discount_total`
  - Final Net ✅ → `_enforced_totals.final_net`
- Include: “Cash discount included only if paid within terms.”
- Omit the "Subtotal" section
- If `_enforced_totals` is missing or pricing cannot be determined, stop and say: “Unable to find pricing, please contact Benjamin Luci at 615-516-8802.”

---
Session Handling
- Remember dealer number across quotes in the same session
- Remember selected model/config only within a single quote
- Always re-pull prices between quotes
- Never say “API says…” — present info as system output

---
Access Control
- Never disclose pricing from one dealer to another
- If dealer needs help finding dealer number, direct them to the Woods dealer portal

---
Accessory Handling
- If a dealer requests a specific accessory (e.g., tires, chains, dual hub):
  - Attempt API lookup
  - If priced, add as separate line item
  - If not priced, stop and show the escalation message
- Never treat a dealer-requested accessory as included by default

---
Interaction Style
- Ask one config question at a time
- Never combine multiple questions into a single message
- Format multiple options as lettered vertical lists:

Example:
Which Turf Batwing size do you need?

A. 12 ft  
B. 15 ft  
C. 17 ft

- Wait for user response before proceeding

---
Box Scraper Correction
- Valid widths: 48 in (4 ft), 60 in (5 ft), 72 in (6 ft), 84 in (7 ft)

---
Disc Harrow Fix
- If API returns the same required spacing prompt repeatedly:
  - Detect the loop
  - Stop quoting
  - Say: “The system is stuck on a required disc spacing selection. Please escalate to Benjamin Luci at 615-516-8802.”
  - Do not retry endlessly

---
Correction Enforcement
- Do not calculate Final Net by subtracting dealer discount only.
- Final Net must subtract both dealer discount AND cash discount.
- Use the exact value from `_enforced_totals.final_net`. Never recalculate.
"""

# (Optional helper synopsis for the model; harmless to include)
FAMILY_GUIDE = r"""
Guide (only if /quote does not already ask):
- BrushFighter: bf_choice, bf_choice_id, drive.
- BrushBull: bb_shielding, bb_tailwheel (sometimes).
- Batwing: bw_duty, bw_driveline (540/1000), shielding_rows, deck_rings, bw_tire_qty.
- Dual Spindle (DS/MDS): ds_mount, ds_shielding, ds_driveline (540/1000), tire_id, tire_qty.
- Turf Batwing (TBW): width_ft (12/15/17), tbw_duty, front_rollers, chains.
- Finish Mowers: finish_choice (tk/tkp/rd990x), rollers, chains.
- Box Scraper: bs_width_in (48/60/72/84), bs_duty, bs_choice_id.
- Grading Scraper: gs_width_in, gs_choice_id.
- Landscape Rake: lrs_width_in, lrs_grade, lrs_choice_id.
- Rear Blade: rb_width_in, rb_duty, rb_choice_id.
- Post Hole Digger: pd_model, auger_id.
- Disc Harrow: dh_width_in, dh_duty (DHS/DHM), dh_spacing_id.
- Tillers (DB/RT/RTR): tiller_series, tiller_width_in, tiller_rotation, tiller_choice_id.
- Bale Spear / Pallet Fork / Quick Hitch: use specific choice_id/part_id.
- Stump Grinder: hydraulics_id.
"""

SYSTEM_PROMPT = KNOWLEDGE + "\n\n--- CONFIG GUIDE ---\n" + FAMILY_GUIDE

# ---------------- HTTP helper ----------------
def http_get(path: str, params: Dict[str, Any]) -> Tuple[Dict[str, Any], int, str]:
    """GET with one retry, JSON-first parse."""
    url = f"{QUOTE_API_BASE}{path}"
    attempts, last_exc = 0, None
    while attempts < max(RETRY_ATTEMPTS, 1):
        try:
            r = requests.get(url, params=params, timeout=HTTP_TIMEOUT)
            ctype = (r.headers.get("content-type") or "").lower()
            body = r.json() if "application/json" in ctype else {"raw_text": r.text}
            return body, r.status_code, r.url
        except Exception as e:
            last_exc = e
            attempts += 1
            if attempts >= max(RETRY_ATTEMPTS, 1):
                break
            time.sleep(0.35)
    log.error("http_get failed %s params=%s error=%s", url, params, last_exc)
    return {"error": str(last_exc)}, 599, url

# ---------------- Model detection ----------------
# Matches: bb60.30, bw12.40, ds8.30, mds12.40, etc.
MODEL_RE = re.compile(r"\b([A-Za-z]{2,3}\d{1,2}\.\d{2})\b")

# Canonical families (from your list), adjusted for valid Python
FAMILY_CATALOG = [
    {
        "family": "disc_harrow",
        "aliases": ["disc", "disc harrow", "harrow", "dh", "dhs", "dhm"],
        "model_prefixes": ["DH", "DHS", "DHM"],
    },
    {
        "family": "batwing",
        "aliases": ["batwing", "bw", "flex wing"],
        "model_prefixes": ["BW"],
    },
    {
        "family": "turf_batwing",
        "aliases": ["turf batwing", "tbw", "batwing finish mower"],
        "model_prefixes": ["TBW"],
    },
    {
        "family": "brushbull",
        "aliases": ["brush bull", "brushbull", "bb", "bushhog", "bush hog"],
        "model_prefixes": ["BB"],
    },
    {
        "family": "brushfighter",
        "aliases": ["brushfighter", "brush fighter", "bf"],
        "model_prefixes": ["BF"],
    },
    {
        "family": "box_scraper",
        "aliases": ["box scraper", "box blade", "bs"],
        "model_prefixes": ["BS"],
    },
    {
        "family": "grading_scraper",
        "aliases": ["grading scraper", "gs", "land plane"],
        "model_prefixes": ["GS"],
    },
    {
        "family": "landscape_rake",
        "aliases": ["landscape rake", "lrs", "rake"],
        "model_prefixes": ["LR"],
    },
    {
        "family": "rear_blade",
        "aliases": ["rear blade", "rb", "blade"],
        "model_prefixes": ["RB"],
    },
    {
        "family": "post_hole_digger",
        "aliases": ["post hole digger", "phd", "posthole digger"],
        "model_prefixes": ["PD"],
    },
    {
        "family": "pallet_fork",
        "aliases": ["pallet fork", "forks", "fork"],
        "model_prefixes": ["PF"],
    },
    {
        "family": "quick_hitch",
        "aliases": ["quick hitch", "hitch"],
        "model_prefixes": ["TQH"],
    },
    {
        "family": "stump_grinder",
        "aliases": ["stump grinder"],
        "model_prefixes": ["TSG"],
    },
    {
        "family": "bale_spear",
        "aliases": ["bale spear", "spear"],
        "model_prefixes": ["BS"],
    },
    {
        "family": "tiller",
        "aliases": ["tiller", "db", "rt", "rtr"],
        "model_prefixes": ["DB", "RT", "RTR"],
    },
    {
        "family": "rear_finish",
        "aliases": ["finish mower", "rd990x", "tk", "tkp", "grooming mower"],
        "model_prefixes": ["RD", "TK", "TKP"],
    },
    {
        "family": "dual_spindle",
        "aliases": ["dual spindle", "ds", "mds"],
        "model_prefixes": ["DS", "MDS"],
    },
]


# Model detectors
MODEL_DOT_RE = re.compile(r"\b([A-Za-z]{2,4}\d{1,2}\.\d{2})\b", re.IGNORECASE)   # e.g., BB72.30
MODEL_SERIES_RE = re.compile(r"\b([A-Za-z]{2,4}\d{2,3})\b", re.IGNORECASE)       # e.g., DHS64, BW12, DS8

# Some APIs label the spacing question inconsistently.
# No matter which one comes back, we must send `dh_spacing_id` with the ID value.

def detect_model(text: str) -> str | None:
    t = text or ""
    m = MODEL_DOT_RE.search(t)
    if m:
        return m.group(1).upper()
    m = MODEL_SERIES_RE.search(t)
    if m:
        return m.group(1).upper()
    return None


# Build alias/prefix indexes from FAMILY_CATALOG
_alias_patterns: List[Tuple[re.Pattern, str]] = []
_prefix_to_families: Dict[str, List[str]] = {}

for entry in FAMILY_CATALOG:
    fam = entry["family"]
    # compile aliases to whole-word-ish regexes
    for a in entry.get("aliases", []):
        # allow matches anywhere but prefer word-ish boundaries
        rx = re.compile(rf"(?:^|\b){re.escape(a)}(?:\b|$)", re.IGNORECASE)
        _alias_patterns.append((rx, fam))
    # collect model prefixes
    for p in entry.get("model_prefixes", []):
        _prefix_to_families.setdefault(p.upper(), []).append(fam)

# order alias patterns by longest alias first (helps disambiguate like "batwing finish mower" vs "batwing")
_alias_patterns.sort(key=lambda ap: len(ap[0].pattern), reverse=True)


def _disambiguate_bs(text: str) -> str | None:
    """Special disambiguation for the 'BS' prefix collision (box_scraper vs bale_spear)."""
    t = (text or "").lower()
    if "spear" in t or "bale spear" in t:
        return "bale_spear"
    if "box" in t or "blade" in t:
        return "box_scraper"
    return None


def detect_family_slug(text: str) -> str | None:
    t = text or ""

    # 1) Alias wins (most reliable)
    for rx, fam in _alias_patterns:
        if rx.search(t):
            return fam

    # 2) If we see a model-like token, try its prefix (e.g., DHS64 → DHS)
    m = MODEL_SERIES_RE.search(t)
    if m:
        token = m.group(1).upper()  # e.g., DHS64
        # try 4, 3, then 2 letter prefixes
        for L in (4, 3, 2):
            if len(token) >= L:
                prefix = token[:L]
                fams = _prefix_to_families.get(prefix)
                if fams:
                    if len(fams) == 1:
                        return fams[0]
                    # ambiguous: handle BS special-case
                    if prefix == "BS":
                        resolved = _disambiguate_bs(t)
                        if resolved:
                            return resolved
                    # more-than-one family for this prefix: bail (let model path proceed)
                    break

    # 3) No alias/prefix found
    return None

# ---------------- Conversation Helpers -----------------




# ---------------- Tools (function schemas) ----------------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "woods_dealer_discount",
            "description": "Look up dealer info by dealer_number (name + discount).",
            "parameters": {
                "type": "object",
                "properties": {
                    "dealer_number": {"type": "string", "description": "Dealer number (e.g., 178055)"},
                },
                "required": ["dealer_number"],
                "additionalProperties": False
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "woods_quote",
            "description": "GET /quote with model= and dealer_number= plus any required options.",
            "parameters": {
                "type": "object",
                "properties": {
                    # REQUIRED CORE
                    "model": {"type": "string", "description": "Exact model, e.g., BB60.30, BW12.40, DS8.30"},
                    "dealer_number": {"type": "string"},

                    # -------- GENERIC / SHARED --------
                    "quantity": {"type": "integer"},
                    "part_id": {"type": "string"},
                    "accessory_id": {"type": "string"},
                    "family": {"type": "string"},
                    "width": {"type": "string"},
                    "width_ft": {"type": "string"},
                    "width_in": {"type": "string"},

                    # -------- BRUSHBULL (BB) --------
                    "bb_shielding": {"type": "string", "description": "Belt or Chain"},
                    "bb_tailwheel": {"type": "string", "description": "Single or Dual, if applicable"},

                    # -------- BATWING (BW/TBW) --------
                    "bw_duty": {"type": "string"},
                    "bw_driveline": {"type": "string", "description": "540 or 1000"},
                    "bw_tire_qty": {"type": "integer"},
                    "shielding_rows": {"type": "string"},
                    "deck_rings": {"type": "string"},
                    "tbw_duty": {"type": "string"},
                    "front_rollers": {"type": "string"},
                    "chains": {"type": "string"},

                    # -------- DUAL SPINDLE (DS/MDS) --------
                    "ds_mount": {"type": "string"},
                    "ds_shielding": {"type": "string"},
                    "ds_driveline": {"type": "string", "description": "540 or 1000"},
                    "tire_id": {"type": "string"},
                    "tire_qty": {"type": "integer"},

                    # -------- DISC HARROW (DHS/DHM) --------
                    "dh_width_in": {"type": "string"},
                    "dh_duty": {"type": "string", "description": "DHS or DHM"},
                    "dh_spacing_id": {"type": "string"},

                    # -------- BOX / GRADING SCRAPERS --------
                    "bs_width_in": {"type": "string"},
                    "bs_duty": {"type": "string"},
                    "bs_choice_id": {"type": "string"},
                    "gs_width_in": {"type": "string"},
                    "gs_choice_id": {"type": "string"},

                    # -------- LANDSCAPE RAKE --------
                    "lrs_width_in": {"type": "string"},
                    "lrs_grade": {"type": "string"},
                    "lrs_choice_id": {"type": "string"},

                    # -------- REAR BLADE --------
                    "rb_width_in": {"type": "string"},
                    "rb_duty": {"type": "string"},
                    "rb_choice_id": {"type": "string"},

                    # -------- POST HOLE DIGGER --------
                    "pd_model": {"type": "string"},
                    "auger_id": {"type": "string"},

                    # -------- TILLERS (DB/RT/RTR) --------
                    "tiller_series": {"type": "string"},
                    "tiller_width_in": {"type": "string"},
                    "tiller_rotation": {"type": "string"},
                    "tiller_choice_id": {"type": "string"},

                    # -------- FINISH MOWERS --------
                    "finish_choice": {"type": "string"},
                    "rollers": {"type": "string"},

                    # -------- STUMP GRINDER / MISC --------
                    "hydraulics_id": {"type": "string"},
                    "choice_id": {"type": "string"},
                },
                "required": ["dealer_number"],
                "additionalProperties": True  # safety net for any new fields
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "woods_health",
            "description": "Pricing API health.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

# ---------------- Tool implementations ----------------
def tool_woods_dealer_discount(args: Dict[str, Any]) -> Dict[str, Any]:
    dealer_number = str(args.get("dealer_number") or "").strip()
    body, status, used = http_get("/dealer-discount", {"dealer_number": dealer_number})
    return {"ok": status == 200, "status": status, "url": used, "body": body}

def tool_woods_quote(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calls /quote with sanitized params, logs request/response, and injects `_enforced_totals`
    computed from API summary (dealer net already applied by API).
    """
    # -------- Build params (strip empty) --------
    params = {k: v for k, v in (args or {}).items() if v not in (None, "")}

    # -------- Inject last-known family/model if missing --------
    dealer_number = str(params.get("dealer_number") or "")
    sess = None
    for s in SESS.values():
        d = s.get("dealer") or {}
        if str(d.get("dealer_number") or "") == dealer_number:
            sess = s
            break

    if sess:
        if "family" not in params and sess.get("last_family"):
            params["family"] = sess["last_family"]
        if "model" not in params and sess.get("last_model"):
            params["model"] = sess["last_model"]

    # -------- Family-specific param normalization --------
    try:
        import re

        # ===================== Disc Harrow =====================
        has_dh = any(k.startswith("dh_") for k in params.keys())
        if has_dh:
            # Mid-flow: the API expects param questions, not a model guess
            params.pop("model", None)
            params.setdefault("family", "disc_harrow")

            # Normalize width alias -> dh_width_in
            if "width" in params and "dh_width_in" not in params:
                params["dh_width_in"] = str(params.pop("width")).strip()

            # Spacing: move ID from alt keys to dh_spacing_id
            if "dh_spacing_id" not in params:
                for alt in ("dh_choice", "dh_spacing", "dh_spacing_choice"):
                    val = params.get(alt)
                    if isinstance(val, str) and re.match(r"^\d{6,}[A-Z]?$", val.strip()):
                        params["dh_spacing_id"] = val.strip()
                        break

            # Blade shorthand -> API label
            if "dh_blade" in params:
                t = str(params["dh_blade"]).strip().lower()
                if t in {"n", "notched"}:
                    params["dh_blade"] = "Notched (N)"
                elif t in {"c", "combo"}:
                    params["dh_blade"] = "Combo (C)"

        # ===================== Bale Spear =====================
        has_bs_flow = (
            "balespear_choice" in params
            or ("model" in params and isinstance(params["model"], str) and params["model"].upper().startswith("BS"))
            or "part_id" in params
            or params.get("family") == "bale_spear"
        )
        if has_bs_flow:
            params.setdefault("family", "bale_spear")
            # Do not let a model short-circuit the question flow
            mdl = params.pop("model", None)

            # Ensure the answer is under the expected question field
            if "balespear_choice" not in params:
                # Prefer a part ID if present (e.g., 1037171 / 1037798)
                pid = params.get("part_id")
                if isinstance(pid, str) and re.match(r"^\d{6,}[A-Z]?$", pid.strip()):
                    params["balespear_choice"] = pid.strip()
                elif isinstance(mdl, str) and mdl.strip():
                    # Some backends accept the model token as the choice value
                    params["balespear_choice"] = mdl.strip().upper()
                # (Optionally) drop part_id if your API is strict:
                # params.pop("part_id", None)

        # ===================== Pallet Fork =====================
        has_pf_flow = (
            "pf_choice" in params
            or ("model" in params and isinstance(params["model"], str) and params["model"].upper().startswith(("PF", "PFW")))
            or "part_id" in params
            or params.get("family") == "pallet_fork"
        )
        if has_pf_flow:
            params.setdefault("family", "pallet_fork")
            mdl = params.pop("model", None)  # avoid model short-circuit mid-flow

            if "pf_choice" not in params:
                # Prefer a part ID if present (e.g., 1037171)
                pid = params.get("part_id")
                if isinstance(pid, str) and re.match(r"^\d{6,}[A-Z]?$", pid.strip()):
                    params["pf_choice"] = pid.strip()
                elif isinstance(mdl, str) and mdl.strip():
                    # Fall back to the model token if given (common from LLM)
                    params["pf_choice"] = mdl.strip().upper()
                # (Optionally) drop part_id if your API is strict:
                # params.pop("part_id", None)

    except Exception as _e:
        log.warning("param normalization skipped: %s", _e)

    log.info("QUOTE CALL params=%s", json.dumps(params, ensure_ascii=False))

    # -------- HTTP call --------
    body, status, used = http_get("/quote", params)

    # -------- Log response shape --------
    try:
        if status == 200 and isinstance(body, dict):
            rq = body.get("required_questions") or []
            rq_names = [q.get("name") for q in rq]
            log.info(
                "QUOTE RESP status=%s mode=%s model=%s rq=%s",
                status, body.get("mode"), body.get("model"), rq_names
            )
        else:
            log.info("QUOTE RESP status=%s (non-json or error) body=%s", status, str(body)[:500])
    except Exception as e:
        log.warning("QUOTE RESP log error: %s", e)

    # -------- Enforce cash discount --------
    try:
        if status == 200 and isinstance(body, dict):
            summary = body.get("summary", {}) or {}
            list_total = float(summary.get("subtotal_list", 0) or 0)
            dealer_net = float(summary.get("total", 0) or 0)

            dealer_discount = None
            if sess:
                try:
                    dealer_discount = float(sess.get("dealer", {}).get("discount"))
                except Exception:
                    pass

            if list_total > 0 and dealer_net > 0 and dealer_discount is not None:
                dealer_discount_amt = list_total - dealer_net
                cash_discount_pct = 0.05 if abs(dealer_discount - 0.05) < 1e-9 else 0.12
                cash_discount_amt = dealer_net * cash_discount_pct
                final_net = dealer_net - cash_discount_amt

                body["_enforced_totals"] = {
                    "list_price_total": round(list_total, 2),
                    "dealer_discount_total": round(dealer_discount_amt, 2),
                    "cash_discount_total": round(cash_discount_amt, 2),
                    "final_net": round(final_net, 2),
                }
                log.info("ENFORCED TOTALS %s: %s", dealer_number, json.dumps(body["_enforced_totals"]))
    except Exception as e:
        log.warning("Failed to inject _enforced_totals: %s", e)

    return {"ok": status == 200, "status": status, "url": used, "body": body}


def tool_woods_health(args: Dict[str, Any]) -> Dict[str, Any]:
    body, status, used = http_get("/health", {})
    return {"ok": status == 200, "status": status, "url": used, "body": body}

# ---------------- Planner ----------------
def run_ai(messages: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
    if not client:
        return "Planner unavailable (missing OPENAI_API_KEY).", messages
    completion = client.chat.completions.create(
        model=OPENAI_MODEL, temperature=0.2, messages=messages, tools=TOOLS, tool_choice="auto"
    )
    history = messages[:]
    rounds = 0
    while True:
        rounds += 1
        if rounds > 8:
            history.append({"role": "assistant", "content": "Tool loop exceeded. Please try again."})
            return "Tool loop exceeded. Please try again.", history

        msg = completion.choices[0].message
        tool_calls = getattr(msg, "tool_calls", None)
        if not tool_calls:
            history.append({"role": "assistant", "content": msg.content or ""})
            return msg.content or "", history

        # record the assistant step
        history.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments or "{}"},
                } for tc in tool_calls
            ]
        })

        # execute each tool
        for tc in tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except Exception:
                args = {}

            # ALWAYS inject dealer_number for woods_quote if the model forgot it
            if name == "woods_quote" and not args.get("dealer_number"):
                # scan history for the last dealer lookup
                dn = None
                for m in reversed(history):
                    if m.get("role") == "tool" and m.get("name") == "woods_dealer_discount":
                        try:
                            payload = json.loads(m.get("content") or "{}")
                            b = payload.get("body") or {}
                            dn = b.get("dealer_number") or b.get("number")
                            if dn:
                                break
                        except Exception:
                            pass
                if dn:
                    args["dealer_number"] = str(dn)

            if name == "woods_dealer_discount":
                result = tool_woods_dealer_discount(args)
            elif name == "woods_quote":
                result = tool_woods_quote(args)
            elif name == "woods_health":
                result = tool_woods_health(args)
            else:
                result = {"ok": False, "status": 0, "url": "", "body": {"error": "Unknown tool"}}

            history.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "name": name,
                "content": json.dumps(result),
            })

        completion = client.chat.completions.create(
            model=OPENAI_MODEL, temperature=0.2, messages=history, tools=TOOLS, tool_choice="auto"
        )

# ---------------- Utilities ----------------
DEALER_NUM_RE = re.compile(r"\b(\d{5,9})\b")

def extract_dealer(text: str) -> str | None:
    m = DEALER_NUM_RE.search(text or "")
    return m.group(1) if m else None

def add_tool_exchange(convo: List[Dict[str, Any]], name: str, args: Dict[str, Any], result: Dict[str, Any]) -> None:
    call_id = f"{name}-{int(time.time()*1000)}"
    convo.append({
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": call_id,
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)},
        }]
    })
    convo.append({
        "role": "tool",
        "tool_call_id": call_id,
        "name": name,
        "content": json.dumps(result),
    })

# ---------------- Routes ----------------
@app.post("/chat")
def chat():
    try:
        payload = request.get_json(silent=True) or {}
        user_message = str(payload.get("message") or "").strip()
        if not user_message:
            return jsonify({"error": "Missing message"}), 400

        # ---------- Session resolve + GC ----------
        now = time.time()
        sid = request.headers.get("X-Session-Id") or payload.get("session_id") or f"anon-{int(now*1000)}"
        # GC old sessions
        for k in list(SESS.keys()):
            if now - SESS[k].get("updated_at", now) > SESSION_TTL:
                SESS.pop(k, None)

        sess = SESS.setdefault(sid, {"messages": [], "dealer": None, "updated_at": now})
        sess["updated_at"] = now

        # ---------- Build conversation (system → history) ----------
        convo: List[Dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        convo.extend(sess["messages"])

        # ---------- Auto: Dealer detection from this message ----------
        dn_from_msg = extract_dealer(user_message)
        if dn_from_msg:
            res = tool_woods_dealer_discount({"dealer_number": dn_from_msg})
            add_tool_exchange(convo, "woods_dealer_discount", {"dealer_number": dn_from_msg}, res)
            if res.get("ok"):
                body = res.get("body") or {}
                sess["dealer"] = {
                    "dealer_number": body.get("dealer_number") or dn_from_msg,
                    "dealer_name": body.get("dealer_name") or "",
                    "discount": body.get("discount"),
                }

        # ---------- Resolve dealer number ----------
        dealer_num = None
        if sess.get("dealer", {}).get("dealer_number"):
            dealer_num = str(sess["dealer"]["dealer_number"])
        else:
            for m in reversed(convo):
                if m.get("role") == "tool" and m.get("name") == "woods_dealer_discount":
                    try:
                        payload = json.loads(m.get("content") or "{}")
                        b = payload.get("body") or {}
                        dn = b.get("dealer_number") or b.get("number")
                        if dn:
                            dealer_num = str(dn)
                            break
                    except Exception:
                        pass

        # ---------- Auto-trigger: model OR family ----------
        model = detect_model(user_message)
        family_slug = detect_family_slug(user_message) if not model else None

        if dealer_num and (model or family_slug):
            args = {"dealer_number": dealer_num}
            if model:
                args["model"] = model
                sess["last_model"] = model
            if family_slug:
                args["family"] = family_slug
                sess["last_family"] = family_slug

            res = tool_woods_quote(args)
            add_tool_exchange(convo, "woods_quote", args, res)

        # ---------- Handle follow-up (e.g., "15", "A", etc.) ----------
        elif dealer_num:
            args = {"dealer_number": dealer_num}

            # Restore model/family if they existed from earlier
            if sess.get("last_model"):
                args["model"] = sess["last_model"]
            if sess.get("last_family"):
                args["family"] = sess["last_family"]

            # Let OpenAI decide rest via planner; but restore context here
            res = tool_woods_quote(args)
            add_tool_exchange(convo, "woods_quote", args, res)

        # ---------- Append user and run planner ----------
        convo.append({"role": "user", "content": user_message})
        reply, hist = run_ai(convo)

        # ---------- Trim + persist history ----------
        MAX_KEEP = 80
        trimmed = [m for m in hist if m.get("role") != "system"]
        if len(trimmed) > MAX_KEEP:
            trimmed = trimmed[-MAX_KEEP:]
        sess["messages"] = trimmed

        dealer_badge = sess.get("dealer") or {}
        return jsonify({
            "reply": reply,
            "dealer": {
                "dealer_number": dealer_badge.get("dealer_number"),
                "dealer_name": dealer_badge.get("dealer_name"),
            }
        })
    except Exception as e:
        logging.exception("chat error")
        return jsonify({
            "reply": ("There was a system error while retrieving data. Please try again shortly. "
                      "If the issue persists, escalate to Benjamin Luci at 615-516-8802."),
            "error": str(e),
            "trace": traceback.format_exc(limit=1),
        }), 200

@app.get("/health")
def health():
    try:
        api = tool_woods_health({})
    except Exception as e:
        api = {"ok": False, "body": {"error": str(e)}}
    return jsonify({
        "ok": True,
        "planner": bool(client),
        "quote_api_base": QUOTE_API_BASE,
        "api_health": api.get("body") or {},
        "retry_attempts": RETRY_ATTEMPTS,
        "http_timeout": HTTP_TIMEOUT,
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    app.run(host="0.0.0.0", port=port, debug=False)
