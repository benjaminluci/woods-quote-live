import os
import json
import re
import time
import logging
from typing import Any, Dict, Tuple, Optional, List

import requests
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
from openai import OpenAI

logging.basicConfig(level=logging.INFO)

# =========================
# Config
# =========================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QUOTE_API_BASE = (os.getenv("QUOTE_API_BASE") or "").rstrip("/")
ALLOWED_ORIGINS = [o.strip() for o in (os.getenv("ALLOWED_ORIGINS") or "*").split(",")]

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

app = Flask(__name__)
CORS(
    app,
    resources={r"/*": {"origins": ALLOWED_ORIGINS}},
    allow_headers=["Content-Type", "X-Session-Id", "X-Dealer-Number"],
    methods=["GET", "POST", "OPTIONS"],
    supports_credentials=False,
    max_age=86400,
)

# In-memory sessions keyed by client-provided ID
# { dealer_number, dealer_discount, dealer_name, pending_question, in_progress_params, params }
SESSIONS: Dict[str, Dict[str, Any]] = {}

DEALER_REGEX = re.compile(r"dealer\s*#?\s*(\d{3,})", re.I)

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
# Parsers
# =========================
MODEL_RE = re.compile(r"\b((?:BB|BW|MDS|DS|TBW|RB|BS|GS|LRS|DHS|DHM|PD|DB|RT|RTR|TSG|PF|TQH)\s*\d+(?:\.\d+)?)\b", re.I)
WIDTH_FT_RE = re.compile(r"\b(\d{1,2})\s*(?:ft|foot|feet)\b", re.I)
WIDTH_IN_RE = re.compile(r"\b(\d{2,3})\s*(?:in|inch|inches|\"|”)\b", re.I)
DRIVELINE_RE = re.compile(r"\b(540|1000|1k)\b", re.I)
DRIVELINE_NAMES = {"driveline", "bw_driveline", "ds_driveline"}  # lowercase matches below
DEFAULT_DRIVELINE_CHOICES = [
    {"id": "540", "label": "540 RPM"},
    {"id": "1000", "label": "1000 RPM"},
]


def norm_model(s: str) -> str:
    return s.upper().replace(" ", "")

def parse_shielding(text: str) -> Optional[str]:
    t = text.lower()
    if "chain" in t or "chains" in t:
        return "Chain"
    if "belt" in t:
        return "Belt"
    if "single row" in t:
        return "Single Row"
    if "double row" in t:
        return "Double Row"
    return None

def parse_family(text: str) -> Optional[str]:
    t = text.lower()
    mapping = {
        "brushfighter": "brushfighter",
        "brush bull": "brushbull",
        "brushbull": "brushbull",
        "dual spindle": "dual_spindle",
        "batwing": "batwing",
        "turf batwing": "turf_batwing",
        "rear discharge": "rear_finish",
        "finish mower": "rear_finish",
        "box scraper": "box_scraper",
        "grading scraper": "grading_scraper",
        "landscape rake": "landscape_rake",
        "rear blade": "rear_blade",
        "disc harrow": "disc_harrow",
        "post hole digger": "post_hole_digger",
        "tiller": "tiller",
        "bale spear": "bale_spear",
        "pallet fork": "pallet_fork",
        "quick hitch": "quick_hitch",
        "stump grinder": "stump_grinder",
    }
    for k, v in mapping.items():
        if k in t:
            return v
    return None

def parse_width_ft(text: str) -> Optional[str]:
    m = WIDTH_FT_RE.search(text)
    if m:
        return m.group(1)
    m2 = re.search(r"\b(10|12|13|15|20)\b", text)
    return m2.group(1) if m2 else None

def parse_driveline(text: str) -> Optional[str]:
    m = DRIVELINE_RE.search(text)
    if not m:
        return None
    val = m.group(1)
    return "1000" if val.lower() == "1k" else val

def intent_from_text(text: str) -> Optional[str]:
    t = text.lower()
    if any(w in t for w in ["quote", "price", "how much", "cost", "need a", "get me a"]):
        return "quote"
    if MODEL_RE.search(text):
        return "quote"
    if parse_family(text):
        return "quote"
    return None

def extract_params_from_text(text: str) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    mm = MODEL_RE.search(text)
    if mm:
        params["model"] = norm_model(mm.group(1))
    fam = parse_family(text)
    if fam:
        params["family"] = fam
    wft = parse_width_ft(text)
    if wft:
        params["width_ft"] = wft
    drv = parse_driveline(text)
    if drv:
        params["bw_driveline"] = drv
    sh = parse_shielding(text)
    if sh:
        params["bb_shielding"] = sh

    t = text.lower()
    tire_terms: List[str] = []
    if "laminated" in t:
        tire_terms.append("laminated tires")
    if "foam" in t:
        tire_terms.append("foam-filled tires")
    if "ag" in t and "tire" in t:
        tire_terms.append("ag tires")
    if tire_terms:
        params["q"] = ", ".join(tire_terms)
    return params

# =========================
# Small helpers
# =========================
def extract_dealer_number(text: str) -> Optional[str]:
    if not text:
        return None
    m = DEALER_REGEX.search(text)
    return m.group(1) if m else None

def letter_or_number_choice_to_index(msg: str, count: int) -> Optional[int]:
    s = (msg or "").strip().lower()
    if not s or count <= 0:
        return None
    if len(s) == 1 and s.isalpha():
        idx = ord(s) - ord("a")
        return idx if 0 <= idx < count else None
    if s.isdigit():
        idx = int(s) - 1
        return idx if 0 <= idx < count else None
    return None

def format_question(q: Dict[str, Any]) -> str:
    title = q.get("question") or "Please choose:"
    labels = [str(c.get("label", "")).strip() for c in q.get("choices_with_ids", [])]
    if not labels and q.get("choices"):
        labels = [str(c).strip() for c in q.get("choices", [])]
    if not labels:
        return title
    abc = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    opts = [f"{abc[i]}. {labels[i]}" for i in range(len(labels))]
    return f"{title}\n\n" + "\n".join(opts)

# =========================
# OpenAI + API helpers
# =========================
def call_openai_for_plan(user_message: str) -> dict:
    if not client:
        return {"action": "smalltalk", "reply": "Server missing OPENAI_API_KEY.", "params": {}}
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.2,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        response_format={"type": "json_object"},
    )
    try:
        return json.loads(completion.choices[0].message.content)
    except Exception as e:
        return {"action": "ask", "reply": f"I couldn’t parse that. Please rephrase. ({e})", "params": {}}

def call_quote_api(path: str, params: dict, retries: int = 1, timeout: int = 60) -> Tuple[dict, int, str]:
    if not QUOTE_API_BASE:
        return ({"error": "QUOTE_API_BASE not configured"}, 500, "")
    base = QUOTE_API_BASE.rstrip("/")
    url = f"{base}/{path.lstrip('/')}"
    last_exc = None
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            try:
                data = r.json()
            except Exception:
                data = {"raw_text": r.text}
            return (data, r.status_code, r.url)
        except Exception as e:
            last_exc = e
            if attempt < retries:
                time.sleep(1.2)
                continue
            return ({"error": f"Upstream error: {last_exc}"}, 502, url)

def call_dealer_discount(dealer_number: str) -> dict:
    url = f"{QUOTE_API_BASE}/dealer-discount"
    r = requests.get(url, params={"dealer_number": dealer_number}, timeout=30)
    try:
        return r.json()
    except Exception:
        return {"error": r.text}

def apply_discounts_from_summary(summary: dict, dealer_discount_rate: float, items: list):
    subtotal = float(summary.get("subtotal_list", 0.0)) or sum(float(i.get("subtotal", 0)) for i in items)
    dealer_amt = round(subtotal * dealer_discount_rate, 2)
    after_dealer = round(subtotal - dealer_amt, 2)
    if abs(dealer_discount_rate - 0.05) < 1e-9:
        cash_amt = 0.0
        final_net = after_dealer
        cash_line = "No cash discount applied (dealer discount is exactly 5%)."
    else:
        cash_amt = round(after_dealer * 0.12, 2)
        final_net = round(after_dealer - cash_amt, 2)
        cash_line = f"Cash Discount (12%): -${cash_amt:,.2f}"
    return subtotal, dealer_amt, after_dealer, cash_amt, final_net, cash_line

def format_quote_output(dealer_name: str, dealer_number: str, resp: dict, forced_rate: Optional[float] = None) -> str:
    rd = resp.get("response_data", resp)
    items = rd.get("items", [])
    summary = rd.get("summary", {})
    dealer_rate = float(forced_rate if forced_rate is not None else summary.get("dealer_discount_rate", 0.0))

    subtotal, dealer_amt, after_dealer, cash_amt, final_net, cash_line = apply_discounts_from_summary(
        summary, dealer_rate, items
    )

    lines: List[str] = []
    lines.append("**Woods Equipment Quote**")
    lines.append(f"Dealer: {dealer_name} (#{dealer_number})\n")

    for it in items:
        desc = it.get("desc") or it.get("model") or ""
        part = it.get("part_id") or it.get("id") or ""
        unit = float(it.get("unit_price", 0.0))
        lines.append(f"Model: {desc}")
        if part:
            lines.append(f"Woods Part No.: {part}")
        lines.append(f"List Price: ${unit:,.2f}\n")

    lines.append(f"Dealer Discount ({int(dealer_rate*100)}%): -${dealer_amt:,.2f}")
    lines.append(f"After Dealer Discount: ${after_dealer:,.2f}")
    lines.append(cash_line)
    lines.append(f"✅ Final Dealer Net: ${final_net:,.2f}")
    lines.append("Cash discount included only if paid within terms.")
    return "\n".join(lines)

# =========================
# Family trees (mimic GPT action logic)
# =========================
FAMILY_TREES: Dict[str, list] = {
    "brushbull": [
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
        {
            "name": "bb_tailwheel",
            "question": "Single or Dual tailwheel?",
            "choices_with_ids": [
                {"id": "Single", "label": "Single"},
                {"id": "Dual", "label": "Dual"},
            ],
        },
    ],
    "dual_spindle": [
        {
            "name": "ds_mount",
            "question": "Mounted (MDS) or Pull-type (DS)?",
            "choices_with_ids": [
                {"id": "MDS", "label": "Mounted (MDS)"},
                {"id": "DS", "label": "Pull-type (DS)"},
            ],
        },
        {
            "name": "ds_shielding",
            "question": "Belt or Chain shielding?",
            "choices_with_ids": [
                {"id": "Belt", "label": "Belt"},
                {"id": "Chain", "label": "Chain"},
            ],
        },
        {
            "name": "ds_driveline",
            "question": "What driveline speed do you need?",
            "choices_with_ids": [
                {"id": "540", "label": "540 RPM"},
                {"id": "1000", "label": "1000 RPM"},
            ],
        },
    ],
    "batwing": [
        {
            "name": "bw_driveline",
            "question": "Which driveline do you want?",
            "choices_with_ids": [
                {"id": "540", "label": "540"},
                {"id": "1000", "label": "1000"},
            ],
        },
        # tires and qty via API
    ],
    "turf_batwing": [
        {
            "name": "tbw_duty",
            "question": "Which Turf Batwing series/duty?",
            "choices_with_ids": [
                {"id": "TBW12", "label": "12 ft Turf Batwing"},
                {"id": "TBW15", "label": "15 ft Turf Batwing"},
                {"id": "TBW17", "label": "17 ft Turf Batwing"},
            ],
        },
    ],
    "rear_finish": [
        {
            "name": "finish_choice",
            "question": "Which rear-finish configuration do you need?",
            "choices_with_ids": [
                {"id": "rear_discharge", "label": "Rear discharge"},
                {"id": "side_discharge", "label": "Side discharge"},
            ],
        },
    ],
    "box_scraper": [
        {
            "name": "bs_width_in",
            "question": "Choose a box scraper width:",
            "choices_with_ids": [
                {"id": "48", "label": "48 in (4 ft)"},
                {"id": "60", "label": "60 in (5 ft)"},
                {"id": "72", "label": "72 in (6 ft)"},
                {"id": "84", "label": "84 in (7 ft)"},
            ],
        },
        {
            "name": "bs_duty",
            "question": "Which duty class for the box scraper?",
            "choices_with_ids": [
                {"id": "standard", "label": "Standard"},
                {"id": "heavy", "label": "Heavy Duty"},
            ],
        },
    ],
    "grading_scraper": [
        {
            "name": "gs_width_in",
            "question": "Choose a grading scraper width:",
            "choices_with_ids": [
                {"id": "60", "label": "60 in (5 ft)"},
                {"id": "72", "label": "72 in (6 ft)"},
                {"id": "84", "label": "84 in (7 ft)"},
            ],
        },
    ],
    "landscape_rake": [
        {
            "name": "lrs_width_in",
            "question": "Choose a rake width:",
            "choices_with_ids": [
                {"id": "60", "label": "60 in (5 ft)"},
                {"id": "72", "label": "72 in (6 ft)"},
                {"id": "84", "label": "84 in (7 ft)"},
            ],
        },
        {
            "name": "lrs_grade",
            "question": "Standard or Premium grade?",
            "choices_with_ids": [
                {"id": "standard", "label": "Standard"},
                {"id": "P", "label": "Premium (P)"},
            ],
        },
    ],
    "rear_blade": [
        {
            "name": "rb_width_in",
            "question": "Choose a rear blade width:",
            "choices_with_ids": [
                {"id": "72", "label": "72 in (6 ft)"},
                {"id": "84", "label": "84 in (7 ft)"},
                {"id": "96", "label": "96 in (8 ft)"},
            ],
        },
        {
            "name": "rb_duty",
            "question": "Which duty class for the rear blade?",
            "choices_with_ids": [
                {"id": "standard", "label": "Standard"},
                {"id": "medium", "label": "Medium"},
                {"id": "heavy", "label": "Heavy Duty"},
            ],
        },
    ],
    "disc_harrow": [
        {
            "name": "dh_width_in",
            "question": "What working width do you want?",
            "choices_with_ids": [
                {"id": "48", "label": "48 in (4 ft)"},
                {"id": "64", "label": "64 in (5 ft)"},
                {"id": "80", "label": "80 in (6 ft)"},
                {"id": "96", "label": "96 in (8 ft)"},
            ],
        },
        {
            "name": "dh_duty",
            "question": "Which duty class?",
            "choices_with_ids": [
                {"id": "standard", "label": "Standard"},
                {"id": "heavy", "label": "Heavy Duty"},
            ],
        },
        {
            "name": "dh_blade",
            "question": "Blade style?",
            "choices_with_ids": [
                {"id": "notched", "label": "Notched"},
                {"id": "smooth", "label": "Smooth"},
                {"id": "combo", "label": "Notched/Smooth Combo"},
            ],
        },
    ],
    "post_hole_digger": [
        {
            "name": "pd_model",
            "question": "Which Post Hole Digger model?",
            "choices_with_ids": [
                {"id": "PD25.21", "label": "PD25.21"},
                {"id": "PD35.31", "label": "PD35.31"},
                {"id": "PD95.51", "label": "PD95.51"},
            ],
        },
    ],
    "tiller": [
        {
            "name": "tiller_series",
            "question": "Which tiller series?",
            "choices_with_ids": [
                {"id": "DB", "label": "DB (Light Duty Dirt Breaker)"},
                {"id": "RT", "label": "RT (Commercial Duty)"},
            ],
        },
        {
            "name": "tiller_width_in",
            "question": "Tiller width?",
            "choices_with_ids": [
                {"id": "60", "label": "60 in (5 ft)"},
                {"id": "72", "label": "72 in (6 ft)"},
                {"id": "84", "label": "84 in (7 ft)"},
            ],
        },
        {
            "name": "tiller_rotation",
            "question": "Rotation (RT only)?",
            "choices_with_ids": [
                {"id": "forward", "label": "Forward (RT)"},
                {"id": "reverse", "label": "Reverse (RTR)"},
            ],
        },
    ],
    "bale_spear": [],
    "pallet_fork": [],
    "quick_hitch": [
        {
            "name": "qh_choice_id",
            "question": "Which quick hitch category?",
            "choices_with_ids": [
                {"id": "TQH1", "label": "TQH1 (Category 1)"},
                {"id": "TQH2", "label": "TQH2 (Category 2)"},
            ],
        }
    ],
    "stump_grinder": [
        {
            "name": "hydraulics_id",
            "question": "Hydraulic option?",
            "choices_with_ids": [
                {"id": "standard", "label": "Standard flow"},
                {"id": "high_flow", "label": "High flow"},
            ],
        },
    ],
}

def infer_family_from_model_or_text(params: Dict[str, Any], user_text: str = "") -> Optional[str]:
    fam = params.get("family")
    if fam:
        return fam
    m = (params.get("model") or "").upper()
    if m.startswith("BB"):
        return "brushbull"
    if m.startswith("BW"):
        return "batwing"
    if m.startswith("DS") or m.startswith("MDS"):
        return "dual_spindle"
    pf = parse_family(user_text)
    return pf

def next_family_tree_question(params: Dict[str, Any], user_text: str = "") -> Optional[Dict[str, Any]]:
    fam = infer_family_from_model_or_text(params, user_text)
    if not fam:
        return None
    tree = FAMILY_TREES.get(fam)
    if not tree:
        return None
    answered = set(k for k, v in (params or {}).items() if v not in (None, "", []))
    for q in tree:
        if q["name"] not in answered:
            return {
                "name": q["name"],
                "question": q["question"],
                "choices_with_ids": list(q.get("choices_with_ids", [])),
            }
    return None

# =========================
# Pending-question state helpers
# =========================

def set_pending_question(sess: Dict[str, Any], base_params: Dict[str, Any], q: Dict[str, Any]) -> None:
    name = (q.get("name") or "").strip()
    question = q.get("question", "Please choose:")
    choices = q.get("choices", [])
    choices_with_ids = q.get("choices_with_ids", [])

    # If the API asked for driveline but sent no choices, inject 540/1000 defaults
    if name.lower() in DRIVELINE_NAMES and not choices_with_ids and not choices:
        choices_with_ids = list(DEFAULT_DRIVELINE_CHOICES)
        if not question or question == "Please choose:":
            question = "Which driveline do you want?"

    ctx = {k: v for k, v in (base_params or {}).items() if k != "dealer_number"}
    sess["in_progress_params"] = ctx
    sess["pending_question"] = {
        "name": name,
        "choices": choices,
        "choices_with_ids": choices_with_ids,
        "question": question,
    }

# =========================
# Core quoting helper (API + family-tree fallback)
# =========================

def do_quote(sess: Dict[str, Any], params: Dict[str, Any]):
    """
    Merge prior context + new params → call GET /quote.
    If API returns mode: "questions", store pending_question and return UI choices.
    If API returns mode: "quote", format and return the quote.
    """
    # Ensure dealer number
    dealer_number = params.get("dealer_number") or sess.get("dealer_number")
    if not dealer_number:
        return jsonify({
            "reply": "Please provide your dealer number to begin (e.g., dealer #178200).",
            "state": {"dealer_number": None, "dealer_name": sess.get("dealer_name")},
        })
    params["dealer_number"] = dealer_number
    dn = dealer_number

    # Track/remember family if known (helps gating and family-tree fallback)
    if "family" in params and params["family"]:
        sess["family"] = params["family"]

    # Call Quote API
    quote_json, err = call_quote_api("/quote", params, retries=1)
    state = {"dealer_number": dn, "dealer_name": sess.get("dealer_name")}
    debug = {"params_sent": params}

    if err or not quote_json:
        return jsonify({
            "reply": f"Quote API error: {err or 'no data'}. Please try again.",
            "state": state,
            "debug": debug,
        })

    mode = quote_json.get("mode")

    # 1) Upstream asks for more info (questions)
    if mode == "questions":
        rd = quote_json.get("response_data", quote_json)
        rq = rd.get("required_questions", [])
        if rq:
            q = rq[0]

            # Driveline safety: if API omitted choices, inject 540/1000 defaults
            name_l = (q.get("name") or "").strip().lower()
            if name_l in {"driveline", "bw_driveline", "ds_driveline"} and not q.get("choices_with_ids") and not q.get("choices"):
                q = {
                    **q,
                    "choices_with_ids": [
                        {"id": "540", "label": "540 RPM"},
                        {"id": "1000", "label": "1000 RPM"},
                    ],
                    "question": q.get("question") or "Which driveline do you want?",
                }

            # Store as pending and build UI block
            set_pending_question(sess, params, q)

            choices_struct = (
                q.get("choices_with_ids")
                or [{"id": str(c), "label": str(c)} for c in (q.get("choices") or [])]
            )

            # Belt-and-suspenders: if still empty for driveline, provide defaults
            if not choices_struct and name_l in {"driveline", "bw_driveline", "ds_driveline"}:
                choices_struct = [
                    {"id": "540", "label": "540 RPM"},
                    {"id": "1000", "label": "1000 RPM"},
                ]

            qtext = q.get("question") or "Please choose:"
            return jsonify({
                "reply": qtext,
                "ui": {
                    "type": "choices",
                    "name": q.get("name"),
                    "question": qtext,
                    "choices": choices_struct,
                },
                "state": state,
                "debug": debug,
            })

        # No concrete next question → try family-tree fallback if present
        step = next_family_step(sess)
        if step:
            set_pending_question(sess, params, step)
            return jsonify({
                "reply": step.get("question", "Please choose:"),
                "ui": {
                    "type": "choices",
                    "name": step.get("name"),
                    "question": step.get("question"),
                    "choices": step.get("choices_with_ids") or [
                        {"id": str(c), "label": str(c)} for c in (step.get("choices") or [])
                    ],
                },
                "state": state,
                "debug": debug,
            })

        # Otherwise ask for clarification
        return jsonify({
            "reply": "I need a bit more configuration to continue. What option would you like?",
            "state": state,
            "debug": debug,
        })

    # 2) Final quote
    # Persist dealer info if present
    if quote_json.get("dealer_name"):
        sess["dealer_name"] = quote_json["dealer_name"]
    if quote_json.get("dealer_discount") is not None:
        sess["dealer_discount"] = float(quote_json["dealer_discount"])

    # Prefer existing formatter if present
    if "summary" in quote_json or "msrp" in quote_json:
        # Use your existing formatter if you have one
        try:
            return format_quote_response(sess, quote_json, state, debug)
        except NameError:
            pass  # fall through to minimal formatting below

    # Fallback minimal formatting if no formatter available
    reply = quote_json.get("summary") or "Here is your quote."
    # Clear pending on success
    sess.pop("pending_question", None)
    sess.pop("in_progress_params", None)
    sess["params"] = {}
    return jsonify({
        "reply": reply,
        "state": state,
        "debug": debug,
        "mode": "quote",
    })

# =========================
# Routes
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

    # Session
    sess = SESSIONS.setdefault(session_id, {"params": {}})

    # Dealer precedence: JSON body → X-Dealer-Number header → message text → session
    dealer_number = (
        payload.get("dealer_number")
        or request.headers.get("X-Dealer-Number")
        or parse_dealer_in_text(user_message)
        or sess.get("dealer_number")
    )
    if dealer_number and dealer_number != sess.get("dealer_number"):
        sess["dealer_number"] = dealer_number
        # Optional: fetch dealer name/discount to echo
        info, _err = call_quote_api("/dealer-discount", {"dealer_number": dealer_number}, retries=1)
        if info:
            if info.get("dealer_name"):
                sess["dealer_name"] = info["dealer_name"]
            if info.get("dealer_discount") is not None:
                sess["dealer_discount"] = float(info["dealer_discount"])

    dn = sess.get("dealer_number")

    # If no dealer yet, try to guide to set it
    if not dn:
        maybe = parse_dealer_in_text(user_message)
        if maybe:
            sess["dealer_number"] = maybe
            info, _ = call_quote_api("/dealer-discount", {"dealer_number": maybe}, retries=1)
            if info:
                if info.get("dealer_name"):
                    sess["dealer_name"] = info["dealer_name"]
                if info.get("dealer_discount") is not None:
                    sess["dealer_discount"] = float(info["dealer_discount"])
            name = sess.get("dealer_name") or ""
            disc = sess.get("dealer_discount")
            msg = f"Using {name or 'this dealer'} (Dealer #{maybe})"
            if disc is not None:
                msg += f" with a {int(disc)}% dealer discount"
            msg += " for this session. ✅\n\nWhat would you like me to quote?"
            return jsonify({
                "reply": msg,
                "state": {"dealer_number": maybe, "dealer_name": name},
            })
        return jsonify({
            "reply": "Please provide your dealer number to begin (e.g., dealer #178200).",
            "state": {"dealer_number": None, "dealer_name": sess.get("dealer_name")},
        })

    # Detect if the user started a new quote (model token or known family)
    model_in_text = parse_model(user_message)
    fam_in_text = parse_family(user_message)
    started_new_quote = bool(model_in_text or fam_in_text)

    # Optional: clear stale pending if starting a brand-new request
    if sess.get("pending_question") and started_new_quote:
        sess.pop("pending_question", None)
        sess.pop("in_progress_params", None)
        sess["params"] = {}

    # 2) Multiple choice resolution (only if we are not starting a new quote)
    pending = sess.get("pending_question")
    if pending and not started_new_quote:
        cwids = pending.get("choices_with_ids") or []
        labels = (
            [str(c.get("label", "")).strip() for c in cwids]
            if cwids else
            [str(c).strip() for c in (pending.get("choices") or [])]
        )

        # If upstream provided no labels (problematic for UI), handle driveline specially
        if not labels:
            pname = (pending.get("name") or "").strip().lower()
            if pname in {"driveline", "bw_driveline", "ds_driveline"}:
                # Accept free-text 540/1000
                drv = parse_driveline(user_message)
                if drv:
                    merged: Dict[str, Any] = {}
                    merged.update(sess.get("in_progress_params") or {})
                    merged.update(sess.get("params") or {})
                    merged[pending.get("name")] = drv
                    merged["dealer_number"] = dn
                    # clear pending/context and continue
                    sess.pop("pending_question", None)
                    sess.pop("in_progress_params", None)
                    sess["params"] = {}
                    return do_quote(sess, merged)

                # No parse → present default buttons so the UI isn't blank
                return jsonify({
                    "reply": pending.get("question", "Which driveline do you want?"),
                    "ui": {
                        "type": "choices",
                        "name": pending.get("name") or "driveline",
                        "question": pending.get("question") or "Which driveline do you want?",
                        "choices": [
                            {"id": "540", "label": "540 RPM"},
                            {"id": "1000", "label": "1000 RPM"},
                        ],
                    },
                    "state": {"dealer_number": dn, "dealer_name": sess.get("dealer_name")},
                })

            # Non-driveline: original behavior (re-ask text only)
            return jsonify({
                "reply": pending.get("question", "Please choose:"),
                "state": {"dealer_number": dn, "dealer_name": sess.get("dealer_name")},
            })

        # We have choices; map A/B/1/2/id/label to a choice id
        ui_choices = cwids or [{"id": str(c), "label": str(c)} for c in (pending.get("choices") or [])]
        chosen_id = normalize_choice_input(user_message, ui_choices)

        # Also allow exact id match (user clicked button text as id)
        if chosen_id is None and user_message:
            for ch in ui_choices:
                if user_message.strip().lower() == str(ch.get("id", "")).lower():
                    chosen_id = str(ch.get("id"))
                    break

        if chosen_id is None:
            # Re-ask with choices
            return jsonify({
                "reply": pending.get("question", "Please choose:"),
                "ui": {
                    "type": "choices",
                    "name": pending.get("name"),
                    "question": pending.get("question"),
                    "choices": ui_choices,
                },
                "state": {"dealer_number": dn, "dealer_name": sess.get("dealer_name")},
            })

        # Apply chosen value and continue
        merged = {}
        merged.update(sess.get("in_progress_params") or {})
        merged.update(sess.get("params") or {})
        merged[pending.get("name")] = chosen_id
        merged["dealer_number"] = dn

        # Clear pending and proceed
        sess.pop("pending_question", None)
        sess.pop("in_progress_params", None)
        sess["params"] = {}
        return do_quote(sess, merged)

    # 3) No pending question: interpret user message and start/continue a quote
    base_params: Dict[str, Any] = {"dealer_number": dn, "_user_message": user_message}
    if model_in_text:
        base_params["model"] = model_in_text
    if fam_in_text:
        base_params["family"] = fam_in_text
        sess["family"] = fam_in_text

    if not (model_in_text or fam_in_text):
        return jsonify({
            "reply": "What would you like me to quote? (e.g., “bb60.30”, “bw12.40”, or “disc harrow”)",
            "state": {"dealer_number": dn, "dealer_name": sess.get("dealer_name")},
        })

    return do_quote(sess, base_params)

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    app.run(host="0.0.0.0", port=port, debug=True)

