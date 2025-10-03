# chat_backend.py â€” Woods Quoting Assistant
# - Uses GET /quote?model=...&dealer_number=... (+ options)
# - Detects models like bb60.30, bw12.40, ds8.30 â†’ normalized to uppercase
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
- A dealer number is required before quoting. Use the Pricing API to look up the dealerâ€™s discount. Do not begin quotes or give pricing without it.
- Dealer numbers may be remembered within a session and across multiple quotes for the same user unless the dealer provides a new number.
- All model, accessory, and pricing data must be pulled directly from the API. Never invent, infer, reuse, or cache data.
- Every quote must pull fresh pricing from the API for all items â€” including list prices and accessories.
- If a valid part number returns no price, quoting must stop and inform the dealer to escalate the issue.

---
API Error Handling
- Retry any connector error once automatically before showing an error.
- If retry succeeds, proceed normally.
- If retry fails, show: â€œThere was a system error while retrieving data. Please try again shortly. If the issue persists, escalate to Benjamin Luci at 615-516-8802.â€

---
Pricing Logic
1. Retrieve list price for each part number from API.
2. Apply dealer discount from lookup.
3. Apply 12% cash discount on top of dealer discount, unless dealer discount is 5% â€” then the cash discount is 5%.
4. These calculations are enforced on the backend and provided as `_enforced_totals` inside the API response.

Formula:
Final Net = list_price_total âˆ’ dealer_discount_total âˆ’ cash_discount_total
This is pre-calculated in `_enforced_totals.final_net`. Use it directly.

---
_enforced_totals format (example):
- list_price_total â†’ full list price of all items combined
- dealer_discount_total â†’ total discount from list price based on dealer discount
- cash_discount_total â†’ additional discount based on payment terms
- final_net â†’ amount after dealer + cash discounts âœ…

---
Quote Output Format
- Begin with "**Woods Equipment Quote**"
- Include the dealer name and dealer number below the title
- Show:
  - Show:
  - List Price â†’ `_enforced_totals.list_price_total`

  - Dealer Discount â†’ `_enforced_totals.dealer_discount_total`

  - Cash Discount â†’ `_enforced_totals.cash_discount_total`


  - Final Net âœ… â†’ `_enforced_totals.final_net`
- Include: â€œCash discount included only if paid within terms.â€
- Please make Final Net in bold and bigger text than the other items
- Please include $ sign in front of money values and add proper commas and decimals
- Please show each line item that is included in the quote on the final output
- Omit the "Subtotal" section
- If `_enforced_totals` is missing or pricing cannot be determined, stop and say: â€œUnable to find pricing, please contact Benjamin Luci at 615-516-8802.â€

---
Session Handling
- Remember dealer number across quotes in the same session
- Remember selected model/config only within a single quote
- Always re-pull prices between quotes
- Never say â€œAPI saysâ€¦â€ â€” present info as system output

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
  - Say: â€œThe system is stuck on a required disc spacing selection. Please escalate to Benjamin Luci at 615-516-8802.â€
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

    # 2) If we see a model-like token, try its prefix (e.g., DHS64 â†’ DHS)
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

                    # -------- BRUSHFIGHTER (BF) ------
                    "bf_choice": {"type": "string"},
                    "bf_choice_id": {"type": "string"},
                    "drive": {"type": "string", "description": "hydraulic/gear/pto if applicable"},
                    "bf_drive": {"type": "string"},
  
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
    Call /quote with a merged view of ALL known params for the in-progress quote.
    On every turn:
      - Merge previously remembered params for this session+family.
      - Overlay any newly provided params (newest wins).
      - Send the full merged param set to the API.

    After response:
      - If mode == "questions": remember the merged param set for this family.
      - If mode == "quote": clear per-quote memory (keep dealer info).
    """
    import re
    from flask import request as _flask_req

    # --- helpers --------------------------------------------------------------

    def _clean_params(d: Dict[str, Any]) -> Dict[str, Any]:
        """Drop empty values; coerce simple strings."""
        out = {}
        for k, v in (d or {}).items():
            if v in (None, "", []):  # treat [] as empty too
                continue
            if isinstance(v, str):
                v = v.strip()
                if v == "":
                    continue
            out[k] = v
        return out

    def _infer_family_from_keys(p: Dict[str, Any]) -> Optional[str]:
        """Best-effort family inference from key prefixes or known fields."""
        keys = {k.lower() for k in p.keys()}
        for fam, pfx in {
            "pallet_fork": "pf_",
            "bale_spear": "balespear_",
            "quick_hitch": "qh_",
            "rear_finish": "finish_",
            "disc_harrow": "dh_",
            "batwing": "bw_",
            "rear_blade": "rb_",
            "landscape_rake": "lrs_",
            "grading_scraper": "gs_",
            "post_hole_digger": "pd_",
            "tiller": "tiller_",
        }.items():
            if any(k.startswith(pfx) for k in keys):
                return fam
        # some families use unprefixed selector fields
        if "width_ft" in keys:
            return "batwing"
        return None

    # --- build params for this turn ------------------------------------------

    # 1) start with only the non-empty incoming args
    turn_params = _clean_params(args or {})

    # 2) get session + per-quote context
    sid = (
        _flask_req.headers.get("X-Session-Id")
        or (_flask_req.get_json(silent=True) or {}).get("session_id")
    )
    sess_entry = SESS.setdefault(sid, {}) if sid else {}
    qc = sess_entry.setdefault("quote_ctx", {"family": None, "answers": {}})

    # 3) determine active family: prefer incoming, else remembered, else infer
    fam = (turn_params.get("family") or qc.get("family") or _infer_family_from_keys(turn_params))
    if fam:
        turn_params["family"] = fam

    # 4) merge previously remembered params for this family
    #    (only if it's the same family; never mix families)
    merged = {}
    # always include dealer_number if present (from this turn or remembered)
    dealer_number = turn_params.get("dealer_number") or (qc.get("answers") or {}).get("dealer_number")
    if dealer_number:
        merged["dealer_number"] = dealer_number

    if fam and qc.get("family") == fam and isinstance(qc.get("answers"), dict):
        # start from remembered answers
        for k, v in qc["answers"].items():
            if k == "dealer_number":  # already handled
                continue
            merged[k] = v

    # overlay current turn params (newest answers win)
    for k, v in turn_params.items():
        merged[k] = v

    # 5) safety: avoid sending 'model' while we are still answering menus
    #    (some families derive model later; sending early can cause 404/drift)
    fam_lower = (fam or "").lower()
    if fam_lower in {"pallet_fork", "bale_spear", "quick_hitch", "rear_finish", "disc_harrow", "batwing"}:
        merged.pop("model", None)

    # 6) log and call API
    log.info("QUOTE CALL params=%s", json.dumps(merged, ensure_ascii=False))
    body, status, used = http_get("/quote", merged)

    # --- log response shape ---------------------------------------------------
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

    # --- persist/clear per-quote memory --------------------------------------
    try:
        if sid and isinstance(body, dict):
            mode = body.get("mode")

            if mode == "questions":
                # remember family and ALL current merged params for this family
                if fam:
                    qc["family"] = fam
                # Store everything except transient fields we never want to carry forever.
                # We *do* keep choice labels/ids, quantities, etc. Dealer number is kept outside.
                answers = qc.setdefault("answers", {})
                for k, v in merged.items():
                    if k == "dealer_number":
                        continue
                    answers[k] = v

            elif mode == "quote":
                # quote finished: clear per-quote memory, keep dealer info
                dealer = sess_entry.get("dealer")  # preserved elsewhere
                sess_entry["quote_ctx"] = {"family": None, "answers": {}}
                sess_entry["err_count"] = 0
                if dealer:
                    sess_entry["dealer"] = dealer  # explicit (likely redundant, but safe)
    except Exception as e:
        log.warning("quote memory persist/clear failed: %s", e)

    # --- compute client-side enforced totals (unchanged) ----------------------
    try:
        if status == 200 and isinstance(body, dict):
            summary = body.get("summary", {}) or {}
            list_total = float(summary.get("subtotal_list", 0) or 0)
            dealer_net = float(summary.get("total", 0) or 0)

            # Find dealer discount from session (stored as decimal, e.g., 0.24 or 0.05)
            dealer_number = str(merged.get("dealer_number") or "")
            dealer_discount = None
            for s in SESS.values():
                d = s.get("dealer") or {}
                if str(d.get("dealer_number") or "") == dealer_number:
                    try:
                        dealer_discount = float(d.get("discount"))
                    except Exception:
                        dealer_discount = None
                    break

            if list_total > 0 and dealer_net > 0 and dealer_discount is not None:
                dealer_discount_amt = list_total - dealer_net
                # Cash discount rule: 5% if dealer discount == 5%, else 12%
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
        # Ensure quote_ctx exists (per-quote "notebook")
        sess.setdefault("quote_ctx", {"family": None, "answers": {}})
        sess.setdefault("err_count", 0)

        # ---------- Build conversation (system â†’ history) ----------
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
                        payload2 = json.loads(m.get("content") or "{}")
                        b = payload2.get("body") or {}
                        dn = b.get("dealer_number") or b.get("number")
                        if dn:
                            dealer_num = str(dn)
                            break
                    except Exception:
                        pass

        # ---------- Auto-trigger: model OR family in user message ----------
        model = detect_model(user_message)
        family_slug = detect_family_slug(user_message) if not model else None

        if dealer_num and (model or family_slug):
            args = {"dealer_number": dealer_num}
            if model:
                args["model"] = model   # allow first-call model; quote tool will drop it mid-flow if needed
            if family_slug:
                args["family"] = family_slug

            # If only dealer_number somehow, skip preflight
            if set(args.keys()) == {"dealer_number"}:
                log.info("Skip /quote: only dealer_number present (auto-trigger)")
            else:
                res = tool_woods_quote(args)

                # ---- EARLY RETURN if tool requested an auto-reset ----
                if isinstance(res, dict) and isinstance(res.get("body"), dict) and res["body"].get("_auto_reset"):
                    try:
                        sess["messages"] = []
                        sess["quote_ctx"] = {"family": None, "answers": {}}
                        sess["err_count"] = 0
                    except Exception:
                        pass
                    dealer_badge = sess.get("dealer") or {}
                    return jsonify({
                        "reply": res["body"]["message"],
                        "dealer": {
                            "dealer_number": dealer_badge.get("dealer_number"),
                            "dealer_name": dealer_badge.get("dealer_name"),
                        }
                    })

                add_tool_exchange(convo, "woods_quote", args, res)

        # ---------- Handle follow-up (e.g., "15", "A", "pin", etc.) ----------
        elif dealer_num:
            args = {"dealer_number": dealer_num}

            # Prefer active quote_ctx (family set there)
            qc = sess.get("quote_ctx") or {}
            if qc.get("family"):
                args["family"] = qc["family"]

            # If we only have dealer_number, don't ping /quote yet â€” let the planner ask next
            if set(args.keys()) == {"dealer_number"}:
                log.info("Skip /quote: only dealer_number present (follow-up)")
            else:
                res = tool_woods_quote(args)

                # ---- EARLY RETURN if tool requested an auto-reset ----
                if isinstance(res, dict) and isinstance(res.get("body"), dict) and res["body"].get("_auto_reset"):
                    try:
                        sess["messages"] = []
                        sess["quote_ctx"] = {"family": None, "answers": {}}
                        sess["err_count"] = 0
                    except Exception:
                        pass
                    dealer_badge = sess.get("dealer") or {}
                    return jsonify({
                        "reply": res["body"]["message"],
                        "dealer": {
                            "dealer_number": dealer_badge.get("dealer_number"),
                            "dealer_name": dealer_badge.get("dealer_name"),
                        }
                    })

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