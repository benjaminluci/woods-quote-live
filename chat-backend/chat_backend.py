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
- Title line: Woods Equipment Quote
- Dealer line: <Dealer Name> – Dealer #<dealer_number>

- Then list each quoted line item back-to-back (NO blank lines between items), numbered with a bold index:
  **1.** <Primary description> (<qty>) – List: $<list_price>
       — <secondary details, if any>
  **2.** <Next item> (<qty>) – List: $<list_price>
       — <secondary details, if any>
  (Continue numbering 3., 4., …)

  Rules:
  - Use **bold** only for the item numbers (e.g., **1.**, **2.**, …).
  - Keep each item to 1–2 lines total:
    • Line 1: primary description, quantity, and list price
    • Optional Line 2: start with an em dash (—) and add concise details (model options, notes, requirements)
  - Do not insert blank lines between items.

- Separator line (shorter):
  ------------------------------

- Discount block (values MUST come from `_enforced_totals`):
  <dealer_discount_percent>% Dealer Discount: $<_enforced_totals.dealer_discount_total>
  <cash_discount_percent>% Cash Discount: $<_enforced_totals.cash_discount_total>

- Final line (BOLD the entire line):
  **Final Dealer Net: $<_enforced_totals.final_net>**

- Footer:
  Cash discount included only if paid within terms.

- Money formatting:
  - Always include $ sign, commas, and two decimals.

- Secondary details guidance (only when applicable; keep concise):
  - Call out configuration details (e.g., duty, driveline, spacing, blade style)
  - Clarify accessory inclusion/separation (e.g., tires quoted separately; required kits)
  - Use part numbers in parentheses when useful (e.g., (639996))

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
    Quote helper that:
      - Remembers all params for the active quote (per-session).
      - Sends the full merged parameter set on every turn.
      - Normalizes Disc Harrow fields.
      - Normalizes menu families (pallet_fork, bale_spear, quick_hitch) and promotes choice → *_choice_id,
        including when the user provides the raw ID itself.
      - On family change, drops keys from prior families (no cross-contamination).
      - Clears per-quote memory on completion (keeps dealer info).
      - Enforces cash discount totals for UI.
    """
    import re
    from flask import request as _flask_req

    # ---------------- helpers ----------------
    def _clean_params(d: Dict[str, Any]) -> Dict[str, Any]:
        out = {}
        for k, v in (d or {}).items():
            if v in (None, "", []):
                continue
            if isinstance(v, str):
                v = v.strip()
                if v == "":
                    continue
            out[k] = v
        return out

    def _infer_family_from_keys(p: Dict[str, Any]) -> Optional[str]:
        keys = {k.lower() for k in p.keys()}
        pref = {
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
        }
        for fam, pfx in pref.items():
            if any(k.startswith(pfx) for k in keys):
                return fam
        if "width_ft" in keys:
            return "batwing"
        return None

    def _family_prefix(fam: str) -> Optional[str]:
        return {
            "pallet_fork": "pf",
            "bale_spear": "balespear",
            "quick_hitch": "qh",
            "rear_finish": "finish",
            "disc_harrow": "dh",
            "batwing": "bw",
            "rear_blade": "rb",
            "landscape_rake": "lrs",
            "grading_scraper": "gs",
            "post_hole_digger": "pd",
            "tiller": "tiller",
        }.get((fam or "").lower())

    # Choice key overrides (API expects these specific id field names)
    CHOICE_ID_KEY_OVERRIDES = {
        "dh_choice": "dh_spacing_id",  # Disc Harrow
    }

    MENU_FAMILIES = {"pallet_fork", "bale_spear", "quick_hitch"}

    # ---------------- build params ----------------
    turn_params = _clean_params(args or {})
    sid = (
        _flask_req.headers.get("X-Session-Id")
        or (_flask_req.get_json(silent=True) or {}).get("session_id")
    )
    sess_entry = SESS.setdefault(sid, {}) if sid else {}
    # quote_ctx: per-quote state for this session
    qc = sess_entry.setdefault("quote_ctx", {"family": None, "answers": {}, "choice_map": {}})

    # Decide family
    fam = (turn_params.get("family") or qc.get("family") or _infer_family_from_keys(turn_params))
    if fam:
        turn_params["family"] = fam
    fam_lower = (fam or "").lower()

    old_fam = qc.get("family")
    old_answers = qc.get("answers") or {}
    choice_map = qc.get("choice_map") or {}

    # Start merged payload with dealer_number (from now or remembered)
    merged: Dict[str, Any] = {}
    dealer_number = turn_params.get("dealer_number") or old_answers.get("dealer_number")
    if dealer_number:
        merged["dealer_number"] = dealer_number

    # If we're still on the same family, include remembered answers first
    if fam and old_fam == fam and isinstance(old_answers, dict):
        for k, v in old_answers.items():
            if k == "dealer_number":
                continue
            merged[k] = v

    # Overlay current turn params (newest wins)
    for k, v in turn_params.items():
        merged[k] = v

    # ------------- if family changed, drop keys from the previous family -------------
    if fam and old_fam and old_fam != fam:
        old_prefix = _family_prefix(old_fam)
        if old_prefix:
            # remove any keys starting with "<oldprefix>_" or "dh_" etc.
            keys_to_drop = [k for k in list(merged.keys()) if k.startswith(old_prefix + "_")]
            for k in keys_to_drop:
                if k not in ("dealer_number", "family"):
                    merged.pop(k, None)
        # also drop the choice_map of the old family (avoid mismapping)
        choice_map = {}

    # ---------------- family-specific normalization ----------------

    # Disc Harrow normalization
    try:
        has_dh = any(k.startswith("dh_") for k in merged.keys())
        if fam_lower == "disc_harrow" or has_dh:
            merged.setdefault("family", "disc_harrow")
            # Normalize width alias
            if "width" in merged and "dh_width_in" not in merged:
                merged["dh_width_in"] = str(merged.pop("width")).strip()
            # Blade shorthand → compact code (or change to long label if your API needs it)
            if "dh_blade" in merged:
                t = str(merged["dh_blade"]).strip().lower()
                if t in {"n", "notched"}:
                    merged["dh_blade"] = "N"  # or "Notched (N)"
                elif t in {"c", "combo", "combination"}:
                    merged["dh_blade"] = "C"  # or "Combo (C)"
            # If dh_choice already looks like an ID, promote to dh_spacing_id
            if "dh_choice" in merged and "dh_spacing_id" not in merged:
                val = str(merged.get("dh_choice") or "").strip()
                if re.match(r"^\d{6,}[A-Z]?$", val):
                    merged["dh_spacing_id"] = val
                    # merged.pop("dh_choice", None)  # optional
    except Exception as _e:
        log.warning("disc harrow normalization skipped: %s", _e)

    # Menu family normalization (PF / Bale Spear / Quick Hitch)
    try:
        if fam_lower in MENU_FAMILIES:
            prefix = _family_prefix(fam_lower)  # 'pf', 'balespear', 'qh'
            choice_field = f"{prefix}_choice"          # e.g., pf_choice
            choice_id_field = f"{prefix}_choice_id"    # e.g., pf_choice_id

            # Promote generic ids into the exact *_choice_id field
            if choice_id_field not in merged:
                for alt in ("choice_id", "part_id", "part_no", f"{prefix}_id"):
                    if merged.get(alt):
                        merged[choice_id_field] = str(merged[alt]).strip()
                        break

            # If the user gave the choice value under the question name,
            # promote it to *_choice_id when it's clearly an ID
            if choice_id_field not in merged and merged.get(choice_field):
                raw = str(merged[choice_field]).strip()
                # Treat pure ids as ids (digits or digits+suffix like 1039976F)
                if re.match(r"^\d{6,}[A-Z]?$", raw):
                    merged[choice_id_field] = raw

            # If we now have either id or label, suppress model mid-flow
            if merged.get(choice_id_field) or merged.get(choice_field):
                merged.pop("model", None)
    except Exception as _e:
        log.warning("menu-family normalization skipped: %s", _e)

    # ---------------- Generic choice→ID promotion using cached maps ----------------
    try:
        cmap = choice_map or {}
        for qname, maps in list(cmap.items()):
            # Which ID key should this map to?
            expected_id_key = CHOICE_ID_KEY_OVERRIDES.get(
                qname,
                (qname if qname.endswith("_id") else f"{qname}_id")
            )
            if merged.get(expected_id_key):
                continue
            val = merged.get(qname)
            if not val:
                continue
            sval = str(val).strip()
            vlow = sval.lower()

            # 1) If the user already gave an exact ID-looking token, accept it directly
            if re.match(r"^\d{6,}[A-Z]?$", sval):
                merged[expected_id_key] = sval
                if expected_id_key != qname:
                    merged.pop(qname, None)
                continue

            # 2) Map letters/labels/model tokens via cached choices_with_ids
            _id = (
                (maps.get("by_letter") or {}).get(vlow)
                or (maps.get("by_label") or {}).get(vlow)
                or (maps.get("by_model") or {}).get(vlow)
            )
            if _id:
                merged[expected_id_key] = _id
                if expected_id_key != qname:
                    merged.pop(qname, None)
    except Exception as _e:
        log.warning("generic choice-id promotion skipped: %s", _e)

    # ---------------- safety: don’t send model mid-questions ----------------
    if fam_lower in {"pallet_fork", "bale_spear", "quick_hitch", "rear_finish", "disc_harrow", "batwing"}:
        merged.pop("model", None)

    # ---------------- call API ----------------
    log.info("QUOTE CALL params=%s", json.dumps(merged, ensure_ascii=False))
    body, status, used = http_get("/quote", merged)

    # ---------------- log response ----------------
    try:
        if status == 200 and isinstance(body, dict):
            rq = body.get("required_questions") or []
            rq_names = [q.get("name") for q in rq]
            log.info("QUOTE RESP status=%s mode=%s model=%s rq=%s",
                     status, body.get("mode"), body.get("model"), rq_names)
        else:
            log.info("QUOTE RESP status=%s (non-json or error) body=%s", status, str(body)[:500])
    except Exception as e:
        log.warning("QUOTE RESP log error: %s", e)

    # ---------------- persist/clear memory ----------------
    try:
        if sid and isinstance(body, dict):
            mode = body.get("mode")
            if mode == "questions":
                # Remember family and ALL current merged params for this family
                if fam:
                    qc["family"] = fam
                answers = qc.setdefault("answers", {})
                for k, v in merged.items():
                    if k == "dealer_number":
                        continue
                    answers[k] = v

                # Cache choice maps for ALL questions with choices_with_ids
                rqs = body.get("required_questions") or []
                choice_map = qc.setdefault("choice_map", {})
                for q in rqs:
                    qname = (q.get("name") or "").strip()
                    cwids = q.get("choices_with_ids") or []
                    if not qname or not cwids:
                        continue
                    by_letter, by_label, by_model = {}, {}, {}
                    for idx, item in enumerate(cwids):
                        _id = str(item.get("id") or "").strip()
                        _label = str(item.get("label") or "").strip()
                        if not _id or not _label:
                            continue
                        by_letter[chr(ord("a") + idx)] = _id
                        by_label[_label.lower()] = _id
                        # first token before " —" is often the model
                        model_tok = _label.split(" —", 1)[0].split()[0].strip().lower()
                        if model_tok:
                            by_model[model_tok] = _id
                    choice_map[qname] = {
                        "by_letter": by_letter,
                        "by_label": by_label,
                        "by_model": by_model,
                    }

            elif mode == "quote":
                # finished — clear per-quote state; keep dealer
                dealer = sess_entry.get("dealer")
                sess_entry["quote_ctx"] = {"family": None, "answers": {}, "choice_map": {}}
                sess_entry["err_count"] = 0
                if dealer:
                    sess_entry["dealer"] = dealer
    except Exception as e:
        log.warning("quote memory persist/clear failed: %s", e)

    # ---------------- enforce client-side totals (unchanged) ----------------
    try:
        if status == 200 and isinstance(body, dict):
            summary = body.get("summary", {}) or {}
            list_total = float(summary.get("subtotal_list", 0) or 0)
            dealer_net = float(summary.get("total", 0) or 0)

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

# ---------------- Tool-message sanitizer ----------------
def _sanitize_messages_for_tools(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Keep only tool messages that directly respond to the most recent assistant tool_calls.
    Prevents: "messages with role 'tool' must be a response..."
    """
    out: List[Dict[str, Any]] = []
    valid_tool_ids: set = set()
    for m in messages:
        role = m.get("role")
        if role == "assistant":
            # reset valid set to only the latest assistant tool calls
            valid_tool_ids = set()
            for tc in (m.get("tool_calls") or []):
                tc_id = (tc.get("id") or tc.get("tool_call_id") or "").strip()
                if tc_id:
                    valid_tool_ids.add(tc_id)
            out.append(m)
        elif role == "tool":
            tcid = (m.get("tool_call_id") or "").strip()
            if tcid and tcid in valid_tool_ids:
                out.append(m)
            else:
                # drop orphan tool message
                continue
        else:
            out.append(m)
    return out


# ---------------- Safe helper shims (use your originals if present) ----------------
def _safe_extract_dealer(text: str) -> Optional[str]:
    # Prefer user's extract_dealer if defined
    fn = globals().get("extract_dealer")
    if callable(fn):
        try:
            return fn(text)
        except Exception:
            pass
    # Fallback: grab a 5-7 digit sequence
    m = re.search(r"\b(\d{5,7})\b", text or "")
    return m.group(1) if m else None

def _safe_detect_model(text: str) -> Optional[str]:
    fn = globals().get("detect_model")
    if callable(fn):
        try:
            return fn(text)
        except Exception:
            pass
    # Very light fallback: model-ish tokens with letters+digits (allow dot)
    m = re.search(r"\b([A-Z]{1,4}\d{2,}[A-Z0-9\.]*)\b", (text or "").upper())
    return m.group(1) if m else None

def _safe_detect_family_slug(text: str) -> Optional[str]:
    fn = globals().get("detect_family_slug")
    if callable(fn):
        try:
            return fn(text)
        except Exception:
            pass
    t = (text or "").lower()
    # Minimal fallback mapping; your original may be richer
    if "pallet" in t and "fork" in t: return "pallet_fork"
    if "bale" in t and "spear" in t: return "bale_spear"
    if "quick" in t and "hitch" in t: return "quick_hitch"
    if "batwing" in t: return "batwing"
    if "brush bull" in t or "brushbull" in t: return "brushbull"
    if "brushfighter" in t or "brush fighter" in t: return "brushfighter"
    if "rear finish" in t or "finish mower" in t: return "rear_finish"
    if "landscape rake" in t or "lrs" in t: return "landscape_rake"
    if "disc harrow" in t or "dh" in t: return "disc_harrow"
    return None


# ---------------- Quote context helpers ----------------
def _qc_get(sess: Dict[str, Any]) -> Dict[str, Any]:
    qc = sess.setdefault("quote_ctx", {"family": None, "answers": {}})
    if "answers" not in qc or not isinstance(qc["answers"], dict):
        qc["answers"] = {}
    return qc

def _qc_clear(sess: Dict[str, Any]) -> None:
    sess["quote_ctx"] = {"family": None, "answers": {}}

def _qc_merge_and_build_args(sess: Dict[str, Any], base_args: Dict[str, Any]) -> Dict[str, Any]:
    qc = _qc_get(sess)

    new_family = (base_args.get("family") or "").strip().lower() if base_args else ""
    if new_family and new_family != (qc.get("family") or ""):
        # new quote; reset per-quote answers but keep dealer badge
        qc["family"] = new_family
        qc["answers"] = {}

    merged = {}
    merged.update(qc["answers"])
    for k, v in (base_args or {}).items():
        if v not in (None, ""):
            merged[k] = v

    if "family" not in merged and qc.get("family"):
        merged["family"] = qc["family"]

    dealer_num = (sess.get("dealer") or {}).get("dealer_number")
    if dealer_num and "dealer_number" not in merged:
        merged["dealer_number"] = str(dealer_num)

    # Persist (but not dealer_number)
    qc["answers"].update({k: v for k, v in merged.items() if k != "dealer_number"})
    if merged.get("family"):
        qc["family"] = merged["family"]
    sess["quote_ctx"] = qc

    # Helpful debug line (keep or remove):
    log.info("QUOTE MERGED args=%s", json.dumps(merged, ensure_ascii=False))
    return merged


# ---------------- Planner ----------------
def run_ai(messages: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
    if not client:
        return "Planner unavailable (missing OPENAI_API_KEY).", messages

    msgs = _sanitize_messages_for_tools(messages)
    completion = client.chat.completions.create(
        model=OPENAI_MODEL, temperature=0.2, messages=msgs, tools=TOOLS, tool_choice="auto"
    )
    history = msgs[:]
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
            "tool_calls": [tc.model_dump() for tc in tool_calls],
        })

        # execute tool calls and append tool results
        for tc in tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except Exception:
                args = {}

            # Ensure dealer_number is present when we can infer it
            if "dealer_number" not in args:
                dn = None
                # try dealer badge in session tool messages (present in history)
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

        # continue the tool loop
        history = _sanitize_messages_for_tools(history)
        completion = client.chat.completions.create(
            model=OPENAI_MODEL, temperature=0.2, messages=history, tools=TOOLS, tool_choice="auto"
        )


# ---------------- HTTP Routes ----------------
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
        for k in list(SESS.keys()):
            if now - SESS[k].get("updated_at", now) > SESSION_TTL:
                SESS.pop(k, None)

        sess = SESS.setdefault(sid, {"messages": [], "dealer": None, "updated_at": now})
        sess["updated_at"] = now
        _qc_get(sess)  # ensure quote_ctx exists
        sess.setdefault("err_count", 0)

        # ---------- Build conversation ----------
        convo: List[Dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        convo.extend(sess["messages"])

        # ---------- Dealer detection from this message (safe shim) ----------
        dn_from_msg = _safe_extract_dealer(user_message)
        if dn_from_msg:
            res = tool_woods_dealer_discount({"dealer_number": dn_from_msg})
            # helper existed in your file; keep using if present, else inline append
            if "add_tool_exchange" in globals() and callable(globals()["add_tool_exchange"]):
                add_tool_exchange(convo, "woods_dealer_discount", {"dealer_number": dn_from_msg}, res)
            else:
                convo.append({
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "id": f"auto_dealer_{int(time.time()*1000)}",
                        "type": "function",
                        "function": {"name": "woods_dealer_discount", "arguments": json.dumps({"dealer_number": dn_from_msg})}
                    }]
                })
                convo.append({"role": "tool", "tool_call_id": convo[-1]["tool_calls"][0]["id"], "name": "woods_dealer_discount", "content": json.dumps(res)})

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
        model = _safe_detect_model(user_message)
        family_slug = _safe_detect_family_slug(user_message) if not model else None

        if dealer_num and (model or family_slug):
            base_args = {"dealer_number": dealer_num}
            if model:
                base_args["model"] = model
            if family_slug:
                base_args["family"] = family_slug

            if set(base_args.keys()) == {"dealer_number"}:
                log.info("Skip /quote: only dealer_number present (auto-trigger)")
            else:
                args = _qc_merge_and_build_args(sess, base_args)
                res = tool_woods_quote(args)

                # auto-reset support
                if isinstance(res, dict) and isinstance(res.get("body"), dict) and res["body"].get("_auto_reset"):
                    sess["messages"] = []
                    _qc_clear(sess)
                    sess["err_count"] = 0
                    dealer_badge = sess.get("dealer") or {}
                    return jsonify({
                        "reply": res["body"]["message"],
                        "dealer": {
                            "dealer_number": dealer_badge.get("dealer_number"),
                            "dealer_name": dealer_badge.get("dealer_name"),
                        }
                    })

                if "add_tool_exchange" in globals() and callable(globals()["add_tool_exchange"]):
                    add_tool_exchange(convo, "woods_quote", args, res)
                else:
                    convo.append({
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{
                            "id": f"auto_quote_{int(time.time()*1000)}",
                            "type": "function",
                            "function": {"name": "woods_quote", "arguments": json.dumps(args)}
                        }]
                    })
                    convo.append({"role": "tool", "tool_call_id": convo[-1]["tool_calls"][0]["id"], "name": "woods_quote", "content": json.dumps(res)})

        # ---------- Follow-up (no new model/family in text) ----------
        elif dealer_num:
            base_args = {"dealer_number": dealer_num}
            qc = _qc_get(sess)
            if qc.get("family"):
                base_args["family"] = qc["family"]

            if set(base_args.keys()) == {"dealer_number"}:
                log.info("Skip /quote: only dealer_number present (follow-up)")
            else:
                args = _qc_merge_and_build_args(sess, base_args)
                res = tool_woods_quote(args)

                if isinstance(res, dict) and isinstance(res.get("body"), dict) and res["body"].get("_auto_reset"):
                    sess["messages"] = []
                    _qc_clear(sess)
                    sess["err_count"] = 0
                    dealer_badge = sess.get("dealer") or {}
                    return jsonify({
                        "reply": res["body"]["message"],
                        "dealer": {
                            "dealer_number": dealer_badge.get("dealer_number"),
                            "dealer_name": dealer_badge.get("dealer_name"),
                        }
                    })

                if "add_tool_exchange" in globals() and callable(globals()["add_tool_exchange"]):
                    add_tool_exchange(convo, "woods_quote", args, res)
                else:
                    convo.append({
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{
                            "id": f"auto_quote_{int(time.time()*1000)}",
                            "type": "function",
                            "function": {"name": "woods_quote", "arguments": json.dumps(args)}
                        }]
                    })
                    convo.append({"role": "tool", "tool_call_id": convo[-1]["tool_calls"][0]["id"], "name": "woods_quote", "content": json.dumps(res)})

        # ---------- Append user and run planner ----------
        convo.append({"role": "user", "content": user_message})
        reply, hist = run_ai(convo)

        # ---------- Trim + persist history ----------
        MAX_KEEP = 80
        trimmed = [m for m in hist if m.get("role") != "system"]
        if len(trimmed) > MAX_KEEP:
            trimmed = trimmed[-MAX_KEEP:]
        sess["messages"] = trimmed

        # Clear per-quote memory after a final quote
        try:
            last_tool = next((m for m in reversed(trimmed) if m.get("role") == "tool" and m.get("name") == "woods_quote"), None)
            if last_tool:
                payload3 = json.loads(last_tool.get("content") or "{}")
                body3 = payload3.get("body") or {}
                if isinstance(body3, dict) and body3.get("mode") == "quote":
                    _qc_clear(sess)
        except Exception:
            pass

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