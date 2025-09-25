import os
import re
import json
import time
import logging
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional, Tuple

import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("woods-quote-backend")

# =========================
# Config
# =========================
QUOTE_API_BASE = os.getenv("QUOTE_API_BASE", "https://woods-quote-api.onrender.com")
ALLOWED_ORIGINS = [o.strip() for o in os.getenv(
    "ALLOWED_ORIGINS",
    "https://woodsequipment-quote.onrender.com"
).split(",") if o.strip()]

# CORS/Flask
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ALLOWED_ORIGINS}}, supports_credentials=False)

# =========================
# Session store (in-memory)
# =========================
SESSIONS: Dict[str, Dict[str, Any]] = {}

# =========================
# FULL Knowledge Block (your rules)
# =========================
KNOWLEDGE = r"""
This GPT is a quoting assistant for Woods dealership staff. It retrieves part numbers, list prices, dealer discounts, and configuration requirements exclusively from the Woods Pricing API. It never fabricates data and only asks configuration questions when required.

---
Core Rules
- Dealer number is required before quoting. Use the API to look up the dealer’s discount. Do not begin quotes or give pricing without it.
- Dealer numbers may be remembered within a session and across multiple quotes for the same user. This eliminates the need for the dealer to repeatedly enter their number unless they explicitly want to change it.
- All model, accessory, and pricing data must come directly from the Woods Pricing API. Never invent, infer, reuse, or cache pricing or configurations.
- Do not reuse prices across quotes. Every quote must pull fresh data — including list prices — from the API for all base units, tires, hub kits, and accessories.
- If the API returns no price for a valid part number, quoting must stop and inform the dealer that escalation is needed.

API Error Handling
- If an API call fails with a connector error on the first attempt, automatically retry the request once before showing an error.
- If the retry succeeds, proceed as normal without showing the error.
- If the retry also fails, show: “There was a system error while retrieving data. Please try again shortly. If the issue persists, escalate to Benjamin Luci at 615-516-8802.”

Pricing Logic
1) Retrieve List Price from API for each part number — never reuse from past quotes
2) Apply dealer discount from lookup
3) Always apply 12% cash discount after dealer discount (unless dealer discount is exactly 5%)
4) Format quote as plain text, customer-ready
⚠️ Always apply the 12% cash discount unless the dealer discount is exactly 5%. This step is not optional.

Quote Output
- Begin with "Woods Equipment Quote" in bold
- Include the dealer name and dealer number below the title
- Final Dealer Net shown boldly with ✅
- Omit the "Subtotal" section
- Include: List Price → Discount → Cash Discount → Final Net
- Include: “Cash discount included only if paid within terms.”
- If a model or part cannot be priced, say: “Unable to find pricing... contact Benjamin Luci at 615-516-8802.”

Session Handling
- Store and reuse dealer number for all quotes within the same session and across multiple quotes for the same user, unless they provide a different dealer number.
- Store and reuse selected model and config within a single quote only.
- Between quotes, always start fresh and re-pull prices for all items.
- Never say “API says…” — always present information as system output.

Access Control
- Never disclose one dealer’s pricing to another.
- Direct dealers to portal to find their dealer number if needed.

Accessory Handling
- If the dealer explicitly requests an accessory or option (e.g., dual hub kit, chains, tires):
  - Always attempt an API lookup, regardless of whether the API marks the item as standard.
  - If pricing is returned, add it as a separate line item.
  - If no price is returned, stop quoting that item and show the escalation message.
- Never silently skip or treat a dealer-requested accessory as standard equipment.

Interaction Style
- Ask configuration questions one at a time only. Never combine multiple decisions into a single message.
- Always format multi-choice questions vertically for clarity.
- Present multiple options as lettered lists (A, B, C...) for easy dealer selection and to avoid confusion with numbered pricing formats.

Example:
Which Turf Batwing size do you need?

A. 12 ft
B. 15 ft
C. 17 ft

- Wait for the user’s response before asking the next configuration question.

Box Scraper Correction
- Valid Box Scraper widths include: 48 in (4 ft), 60 in (5 ft), 72 in (6 ft), 84 in (7 ft)

Disc Harrow Fix
- If the API repeatedly returns the same required disc spacing prompt even after the user has answered:
  - Detect this loop and stop quoting.
  - Say: “The system is stuck on a required disc spacing selection. Please escalate to Benjamin Luci at 615-516-8802.”
  - Do not keep retrying the same input. Do not allow an infinite loop.

Correction Enforcement
- Quotes must not stop at the dealer discount stage.
- If dealer discount ≠ 5%, the 12% cash discount must be calculated and clearly shown.
- Never omit or forget the cash discount application.
- If the final output does not include the cash discount deduction, that quote is invalid and must be corrected immediately.
"""

PARAM_HINTS = r"""
Valid families: brushfighter, brushbull, dual_spindle, batwing, turf_batwing, rear_finish,
box_scraper, grading_scraper, landscape_rake, rear_blade, disc_harrow, post_hole_digger,
tiller, bale_spear, pallet_fork, quick_hitch, stump_grinder.

Common params (subset):
- dealer_number (string)
- model (e.g., BB60.30, BW12.40)
- family (see list above)
- width_ft (e.g., "12", "15", "20")
- bw_driveline: "540" or "1000"
- bb_shielding: "Belt" | "Chain" | "Single Row" | "Double Row"
- tire_id / tire_qty when API requires tires
- accessory/accessory_id/accessory_ids for add-ons
- q (free-text if needed, but prefer structured params first)
"""

SYSTEM_PROMPT = KNOWLEDGE + PARAM_HINTS + r"""
You MUST return strict JSON in this exact shape:
{
  "action": "dealer_lookup" | "quote" | "ask" | "smalltalk",
  "reply": "string (may be empty if not needed)",
  "params": { ... Woods API params ... }
}

Routing rules:
- If the user message contains ONLY a dealer number (e.g., “dealer #178200” or “178200”), use:
  { "action":"dealer_lookup", "params": {"dealer_number":"178200"} }
- If the message contains both a dealer number and a quote request, do NOT return dealer_lookup.
  Instead, return { "action":"quote", "params": { "dealer_number": "...", ...other params... } }.
- If the message is a quote request without a dealer number, return:
  { "action":"ask", "reply":"Please provide your dealer number to begin." }.
- Map obvious natural language to params (e.g., “12 foot batwing, 540 RPM, laminated tires” → family=batwing, width_ft=12, bw_driveline=540).
- Only ask for missing fields via { "action":"ask", "reply":"<one question at a time>" }.
"""

# =========================
# Regex / parsing helpers
# =========================
DEALER_REGEX = re.compile(r"dealer\s*#?\s*(\d{3,})", re.I)
MODEL_TOKEN = re.compile(r"\b([a-z]{2,3}\d{2,}\.\d{2})\b", re.I)  # e.g., bb72.30, bw12.40
FAMILY_TOKENS = {
    "brushbull": {"aliases": ["brushbull", "bb"], "family": "BrushBull"},
    "dual spindle": {"aliases": ["dual spindle"], "family": "Dual Spindle"},
    "batwing": {"aliases": ["batwing"], "family": "Batwing"},
    "turf batwing": {"aliases": ["turf batwing"], "family": "Turf Batwing"},
    "rear finish": {"aliases": ["rear finish", "rmm"], "family": "Rear Finish"},
    "box scraper": {"aliases": ["box scraper", "box blade"], "family": "Box Scraper"},
    "grading scraper": {"aliases": ["grading scraper"], "family": "Grading Scraper"},
    "landscape rake": {"aliases": ["landscape rake"], "family": "Landscape Rake"},
    "rear blade": {"aliases": ["rear blade"], "family": "Rear Blade"},
    "disc harrow": {"aliases": ["disc harrow", "disc"], "family": "Disc Harrow"},
    "post hole digger": {"aliases": ["post hole digger", "phd"], "family": "Post Hole Digger"},
    "tiller": {"aliases": ["tiller"], "family": "Tiller"},
    "quick hitch": {"aliases": ["quick hitch", "qh"], "family": "Quick Hitch"},
    "stump grinder": {"aliases": ["stump grinder"], "family": "Stump Grinder"},
    "bale spear": {"aliases": ["bale spear"], "family": "Bale Spear"},
    "pallet fork": {"aliases": ["pallet fork"], "family": "Pallet Fork"},
}

def parse_dealer_in_text(text: str) -> Optional[str]:
    m = DEALER_REGEX.search(text or "")
    return m.group(1) if m else None

def parse_model(text: str) -> Optional[str]:
    m = MODEL_TOKEN.search(text or "")
    return m.group(1).lower() if m else None

def parse_family(text: str) -> Optional[str]:
    t = (text or "").lower()
    for key, spec in FAMILY_TOKENS.items():
        if any(alias in t for alias in spec["aliases"]):
            return spec["family"]
    return None

def infer_family_from_model_or_text(model: Optional[str], text: str) -> Optional[str]:
    if model:
        m = model.lower()
        if m.startswith("bw"):  # batwing / turf batwing distinction left to API
            # If "turf" appears, treat as Turf Batwing
            if "turf" in text.lower():
                return "Turf Batwing"
            return "Batwing"
        if m.startswith("bb"):
            return "BrushBull"
    fam = parse_family(text)
    return fam

def parse_driveline(text: str) -> Optional[str]:
    t = (text or "").lower().strip()
    if "540" in t:
        return "540"
    if "1000" in t or "1,000" in t:
        return "1000"
    return None

def normalize_choice_input(user_text: str, choices: List[Dict[str, str]]) -> Optional[str]:
    """
    Map A/B/C, 1/2/3, id match, label match, normalized tokens.
    Returns the chosen choice id (string) or None.
    """
    if not user_text:
        return None
    t = user_text.strip().lower()

    # A/B/C mapping
    alpha = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4}
    if t in alpha and alpha[t] < len(choices):
        return str(choices[alpha[t]].get("id"))

    # 1/2/3 mapping
    if t.isdigit():
        idx = int(t) - 1
        if 0 <= idx < len(choices):
            return str(choices[idx].get("id"))

    # direct id or label match
    for ch in choices:
        cid = str(ch.get("id", "")).lower()
        lab = str(ch.get("label", "")).lower()
        if t == cid or t == lab:
            return str(ch.get("id"))

    # tokens for driveline
    if "540" in t:
        for ch in choices:
            if "540" in str(ch.get("id", "")) or "540" in str(ch.get("label", "")):
                return str(ch.get("id"))
    if "1000" in t:
        for ch in choices:
            if "1000" in str(ch.get("id", "")) or "1000" in str(ch.get("label", "")):
                return str(ch.get("id"))

    return None

# =========================
# Family Trees (guided fallback)
# =========================
FAMILY_TREES: Dict[str, List[Dict[str, Any]]] = {
    "BrushBull": [
        {
            "name": "bb_duty",
            "question": "Which duty class for BrushBull?",
            "choices_with_ids": [
                {"id": ".30", "label": "Standard Duty (.30)"},
                {"id": ".40", "label": "Standard Plus (.40)"},
                {"id": ".50", "label": "Heavy Duty (.50)"},
                {"id": ".60", "label": "Extreme Duty (.60)"},
            ],
        },
        {
            "name": "bb_shielding",
            "question": "Belt or Chain shielding for BrushBull?",
            "choices_with_ids": [
                {"id": "Belt", "label": "Belt"},
                {"id": "Chain", "label": "Chain"},
            ],
        },
    ],
    "Batwing": [
        {
            "name": "driveline",
            "question": "Which driveline do you want?",
            "choices_with_ids": [
                {"id": "540", "label": "540 RPM"},
                {"id": "1000", "label": "1000 RPM"},
            ],
        },
        {
            "name": "tires",
            "question": "Which tire package?",
            "choices_with_ids": [
                {"id": "laminated", "label": "Laminated"},
                {"id": "air-filled", "label": "Air-filled"},
            ],
        },
    ],
    "Turf Batwing": [
        {
            "name": "driveline",
            "question": "Which driveline do you want?",
            "choices_with_ids": [
                {"id": "540", "label": "540 RPM"},
                {"id": "1000", "label": "1000 RPM"},
            ],
        },
        {
            "name": "tires",
            "question": "Which tire package?",
            "choices_with_ids": [
                {"id": "laminated", "label": "Laminated"},
                {"id": "air-filled", "label": "Air-filled"},
            ],
        },
    ],
    "Dual Spindle": [
        {
            "name": "driveline",
            "question": "Which driveline do you want?",
            "choices_with_ids": [
                {"id": "540", "label": "540 RPM"},
                {"id": "1000", "label": "1000 RPM"},
            ],
        }
    ],
}

# Families where driveline is a valid question
FAMILIES_NEED_DRIVELINE = {"Batwing", "Turf Batwing", "Dual Spindle"}

DEFAULT_CHOICES = {
    "driveline": [
        {"id": "540", "label": "540 RPM"},
        {"id": "1000", "label": "1000 RPM"},
    ]
}

def needs_driveline(family: Optional[str]) -> bool:
    return family in FAMILIES_NEED_DRIVELINE

# =========================
# Quote API Calls
# =========================
def call_quote_api(path: str, params: Dict[str, Any], timeout=6.0, retries=1) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    GET to Quote API with small retry for cold starts.
    Returns (json, error)
    """
    url = f"{QUOTE_API_BASE.rstrip('/')}/{path.lstrip('/')}"
    last_err = None
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code >= 500:
                last_err = f"{r.status_code} from Quote API"
                time.sleep(0.25 + 0.25 * attempt)
                continue
            r.raise_for_status()
            return r.json(), None
        except requests.RequestException as e:
            last_err = str(e)
            time.sleep(0.25 + 0.25 * attempt)
    return None, last_err

def fetch_dealer_discount(dealer_number: str) -> Optional[float]:
    data, err = call_quote_api("/dealer-discount", {"dealer_number": dealer_number}, retries=1)
    if err or not data:
        return None
    # Expect {"dealer_number":"...","dealer_discount":24,"dealer_name":"..."}
    return float(data.get("dealer_discount")) if data.get("dealer_discount") is not None else None

# =========================
# Discount math
# =========================
def money(x: Any) -> Decimal:
    return Decimal(str(x)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

def apply_discounts(msrp: float, dealer_pct: float) -> Tuple[Decimal, Decimal, Decimal]:
    msrp_d = Decimal(str(msrp))
    d = Decimal(str(dealer_pct)) / Decimal("100")
    dealer_price = msrp_d * (Decimal("1") - d)
    # Only apply 12% cash discount if dealer discount != 5%
    cash_rate = Decimal("0.12") if Decimal(str(round(dealer_pct, 2))) != Decimal("5") else Decimal("0")
    cash_price = dealer_price * (Decimal("1") - cash_rate)
    return money(dealer_price), money(cash_price), cash_rate

# =========================
# Core quoting orchestration
# =========================
def set_pending_question(sess: Dict[str, Any], base_params: Dict[str, Any], q: Dict[str, Any]) -> Dict[str, Any]:
    """Normalizes and stores a pending question in session. Returns a UI payload."""
    name = q.get("name") or ""
    question = q.get("question") or "Please choose:"
    choices = q.get("choices") or []
    choices_with_ids = q.get("choices_with_ids") or []

    # Inject defaults for driveline if upstream omitted choices
    if (not choices_with_ids and not choices) and name == "driveline":
        choices_with_ids = DEFAULT_CHOICES["driveline"]
        if not question or question == "Please choose:":
            question = "Which driveline do you want?"

    sess["in_progress_params"] = {k: v for k, v in (base_params or {}).items() if k != "dealer_number"}
    sess["pending_question"] = {
        "name": name,
        "question": question,
        "choices_with_ids": choices_with_ids,
        "choices": choices,
    }

    ui_choices = choices_with_ids or [{"id": str(c), "label": str(c)} for c in choices]
    if (not ui_choices) and name == "driveline":
        ui_choices = DEFAULT_CHOICES["driveline"]

    return {
        "type": "choices",
        "name": name,
        "question": question,
        "choices": ui_choices,
    }

def do_quote(sess: Dict[str, Any], params: Dict[str, Any]):
    """Merge context + new params; call Quote API; format question UI or final quote."""
    # Always include dealer_number
    dealer_number = params.get("dealer_number") or sess.get("dealer_number")
    if not dealer_number:
        return jsonify({"reply": "Please provide your dealer number to begin (e.g., dealer #178200)."})
    params["dealer_number"] = dealer_number

    # Track family for gating
    family = params.get("family") or sess.get("family") or infer_family_from_model_or_text(params.get("model"), params.get("_user_message", ""))
    if family:
        sess["family"] = family

    # Call Quote API
    data, err = call_quote_api("/quote", params, retries=1)
    state = {"dealer_number": dealer_number, "dealer_name": sess.get("dealer_name")}
    debug = {"params_sent": params}

    if err or not data:
        return jsonify({
            "reply": f"Quote API error: {err or 'no data'}. Please try again.",
            "state": state, "debug": debug
        })

    mode = data.get("mode")

    # If API needs questions
    if mode == "questions":
        # Expect a structure; we unify into one next question
        nxt = None
        # Prefer API-provided 'required_questions' or 'question'
        if isinstance(data.get("required_questions"), list) and data["required_questions"]:
            nxt = data["required_questions"][0]
        elif isinstance(data.get("question"), dict):
            nxt = data["question"]

        if not nxt:
            # Fall back to family tree step if available
            step = next_family_step(sess)
            if step:
                ui = set_pending_question(sess, params, step)
                return jsonify({
                    "reply": step.get("question", "Please choose:"),
                    "ui": ui, "state": state, "debug": debug
                })
            # Otherwise ask the user to clarify
            return jsonify({
                "reply": "I need a bit more configuration to continue. What option would you like?",
                "state": state, "debug": debug
            })

        # Optional gating: suppress driveline where not applicable
        if (nxt.get("name") in {"driveline", "bw_driveline", "ds_driveline"}) and not needs_driveline(family):
            # Skip this, try next family step or re-call API without setting pending
            step = next_family_step(sess)
            if step:
                ui = set_pending_question(sess, params, step)
                return jsonify({
                    "reply": ui["question"], "ui": ui, "state": state, "debug": debug
                })
            # If no step, just re-ask API hoping it returns a different field
            data2, err2 = call_quote_api("/quote", params, retries=1)
            if data2 and data2.get("mode") != "questions":
                # process as quote
                return format_quote_response(sess, data2, state, debug)
            # If still questions, set it but inject defaults for driveline so user can answer
            # fall through to set_pending_question

        ui = set_pending_question(sess, params, nxt)
        return jsonify({
            "reply": ui["question"],
            "ui": ui,
            "state": state, "debug": debug
        })

    # Final quote
    return format_quote_response(sess, data, state, debug)

def format_quote_response(sess: Dict[str, Any], data: Dict[str, Any], state: Dict[str, Any], debug: Dict[str, Any]):
    # Persist dealer info if provided by API
    if data.get("dealer_name"):
        sess["dealer_name"] = data["dealer_name"]
    if data.get("dealer_discount") is not None:
        sess["dealer_discount"] = float(data["dealer_discount"])

    # Prefer API summary, but ensure discount policy is applied
    msrp = data.get("msrp")
    dealer_pct = (
        float(data["dealer_discount"]) if data.get("dealer_discount") is not None
        else float(sess.get("dealer_discount") or 0.0)
    )

    price_lines = []
    if msrp is not None and dealer_pct is not None:
        dp, cp, cr = apply_discounts(float(msrp), dealer_pct)
        price_lines.append(f"MSRP: ${money(msrp)}")
        price_lines.append(f"Dealer price ({dealer_pct:.0f}% off): ${dp}")
        if cr > 0:
            price_lines.append(f"Cash price (additional 12%): ${cp}")
    summary = data.get("summary") or ""
    if price_lines:
        summary = (summary + "\n\n" if summary else "") + "\n".join(price_lines)

    reply = summary or "Here is your quote."
    resp = {"reply": reply, "state": state, "debug": debug, "mode": "quote"}
    # Clear pending once we have a quote
    sess.pop("pending_question", None)
    sess.pop("in_progress_params", None)
    sess["params"] = {}
    return jsonify(resp)

def next_family_step(sess: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    fam = sess.get("family")
    if not fam or fam not in FAMILY_TREES:
        return None
    asked = set()
    if "in_progress_params" in sess:
        asked |= set(sess["in_progress_params"].keys())
    if "params" in sess:
        asked |= set(sess["params"].keys())
    for step in FAMILY_TREES[fam]:
        if step["name"] not in asked:
            return step
    return None

# =========================
# Flask endpoints
# =========================
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "ok": True,
        "quote_api_base": QUOTE_API_BASE,
        "allowed_origins": ALLOWED_ORIGINS,
    })

@app.route("/chat", methods=["POST"])
def chat():
    payload = request.get_json(force=True, silent=True) or {}
    user_message = (payload.get("message") or "").strip()
    body_session_id = payload.get("session_id")
    header_session_id = request.headers.get("X-Session-Id")
    session_id = header_session_id or body_session_id or f"anon-{int(time.time()*1000)}"

    # Get or init session
    sess = SESSIONS.setdefault(session_id, {"params": {}})

    # Dealer: precedence Body JSON → X-Dealer-Number → message text → session
    dealer_number = (
        payload.get("dealer_number")
        or request.headers.get("X-Dealer-Number")
        or parse_dealer_in_text(user_message)
        or sess.get("dealer_number")
    )
    if dealer_number and dealer_number != sess.get("dealer_number"):
        sess["dealer_number"] = dealer_number
        # try fetching name/discount to echo
        info, err = call_quote_api("/dealer-discount", {"dealer_number": dealer_number}, retries=1)
        if not err and info:
            sess["dealer_name"] = info.get("dealer_name") or sess.get("dealer_name")
            if info.get("dealer_discount") is not None:
                sess["dealer_discount"] = float(info["dealer_discount"])

    # Detect if the user clearly started a new quote (model or family mentioned)
    model_in_text = parse_model(user_message)
    fam_in_text = parse_family(user_message)
    started_new_quote = bool(model_in_text or fam_in_text)

    # === Stuck driveline fix: clear pending if user starts a new quote ===
    if sess.get("pending_question") and started_new_quote:
        sess.pop("pending_question", None)
        sess.pop("in_progress_params", None)
        sess["params"] = {}

    # If no dealer yet, guide to set it
    if not sess.get("dealer_number"):
        maybe = parse_dealer_in_text(user_message)
        if maybe:
            sess["dealer_number"] = maybe
            # Echo confirmation
            dn = sess["dealer_number"]
            # Optionally fetch discount to confirm
            info, _ = call_quote_api("/dealer-discount", {"dealer_number": dn}, retries=1)
            if info:
                sess["dealer_name"] = info.get("dealer_name") or sess.get("dealer_name")
                if info.get("dealer_discount") is not None:
                    sess["dealer_discount"] = float(info["dealer_discount"])
            name = sess.get("dealer_name") or ""
            disc = sess.get("dealer_discount")
            msg = f"Using {name or 'this dealer'} (Dealer #{dn})"
            if disc is not None:
                msg += f" with a {int(disc)}% dealer discount"
            msg += " for this session. ✅\n\nWhat would you like me to quote?"
            return jsonify({"reply": msg, "state": {"dealer_number": dn, "dealer_name": name}})
        return jsonify({"reply": "Please provide your dealer number to begin (e.g., dealer #178200).",
                        "state": {"dealer_number": None}})

    # === Pending question resolution ===
    pending = sess.get("pending_question")
    if pending and not started_new_quote:
        name = (pending.get("name") or "").lower()
        cwids = pending.get("choices_with_ids") or []
        ui_choices = cwids or [{"id": str(c), "label": str(c)} for c in (pending.get("choices") or [])]

        # Accept free-text for driveline even if no choices
        if ("driveline" in name) and not ui_choices:
            drv = parse_driveline(user_message)
            if drv:
                # merge and advance
                merged = {}
                merged.update(sess.get("in_progress_params") or {})
                merged.update(sess.get("params") or {})
                merged["driveline"] = drv
                merged["dealer_number"] = sess["dealer_number"]
                # clear pending
                sess.pop("pending_question", None)
                sess.pop("in_progress_params", None)
                sess["params"] = {}
                # annotate for debug
                merged["_user_message"] = user_message
                return do_quote(sess, merged)
            # No parse -> re-ask with defaults
            return jsonify({
                "reply": (pending.get("question") or "Which driveline do you want?") + "\n\nA. 540 RPM\nB. 1000 RPM",
                "ui": {"type": "choices", "name": pending.get("name") or "driveline",
                       "question": pending.get("question") or "Which driveline do you want?",
                       "choices": DEFAULT_CHOICES["driveline"]},
                "state": {"dealer_number": sess["dealer_number"], "dealer_name": sess.get("dealer_name")},
            })

        # Normal mapping if choices exist
        chosen_id = normalize_choice_input(user_message, ui_choices)
        if chosen_id is None and ui_choices:
            # If user clicked button, they might send the id directly; try exact match
            if user_message:
                for ch in ui_choices:
                    if user_message.strip().lower() == str(ch.get("id", "")).lower():
                        chosen_id = str(ch.get("id"))
                        break

        if chosen_id is None and ui_choices:
            # Re-ask
            return jsonify({
                "reply": pending.get("question") or "Please choose:",
                "ui": {"type": "choices", "name": pending.get("name"), "question": pending.get("question"),
                       "choices": ui_choices},
                "state": {"dealer_number": sess["dealer_number"], "dealer_name": sess.get("dealer_name")},
            })

        # Apply chosen value and continue
        merged = {}
        merged.update(sess.get("in_progress_params") or {})
        merged.update(sess.get("params") or {})
        merged[pending.get("name")] = chosen_id if chosen_id is not None else user_message
        merged["dealer_number"] = sess["dealer_number"]

        # Clear pending and proceed
        sess.pop("pending_question", None)
        sess.pop("in_progress_params", None)
        sess["params"] = {}
        merged["_user_message"] = user_message
        return do_quote(sess, merged)

    # === No pending question: interpret user message ===
    dn = sess["dealer_number"]

    # Simple intents
    if user_message.lower() in {"help", "knowledge", "rules"}:
        return jsonify({"reply": KNOWLEDGE.strip(), "state": {"dealer_number": dn, "dealer_name": sess.get("dealer_name")}})

    model = model_in_text
    family = fam_in_text or infer_family_from_model_or_text(model, user_message)
    base_params: Dict[str, Any] = {"dealer_number": dn, "_user_message": user_message}

    if model:
        base_params["model"] = model
    if family:
        base_params["family"] = family
        sess["family"] = family

    if not (model or family):
        # Non-quote chatter
        return jsonify({"reply": "What would you like me to quote? (e.g., “bb60.30”, “bw12.40”, or “disc harrow”)",
                        "state": {"dealer_number": dn, "dealer_name": sess.get("dealer_name")}})

    return do_quote(sess, base_params)

# =========================
# Entrypoint
# =========================
if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    app.run(host="0.0.0.0", port=port, debug=True)

