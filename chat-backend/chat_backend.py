import os, json, re, requests, logging, time
from typing import Any, Dict, Tuple
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
    allow_headers=["Content-Type", "X-Session-Id"],
    methods=["GET", "POST", "OPTIONS"],
    supports_credentials=False,
    max_age=86400,
)

# In-memory sessions keyed by client-provided ID
SESSIONS: Dict[str, Dict[str, Any]] = {}  # { dealer_number, dealer_discount, dealer_name, pending_question, in_progress_params, params }

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
  { "action":"ask", "reply":"Please provide your dealer number to begin." }
- Map obvious natural language to params (e.g., “12 foot batwing, 540 RPM, laminated tires” → family=batwing, width_ft=12, bw_driveline=540, possibly tire keywords via q if you can’t map a specific tire_id).
- Only ask for missing fields via { "action":"ask", "reply":"<one question at a time>" }.

Examples:

User: "dealer #178200"
Return:
{"action":"dealer_lookup","reply":"","params":{"dealer_number":"178200"}}

User: "178647"
Return:
{"action":"dealer_lookup","reply":"","params":{"dealer_number":"178647"}}

User: "dealer #178200 quote me a 12 foot batwing with 540 rpm"
Return:
{"action":"quote","reply":"","params":{"dealer_number":"178200","family":"batwing","width_ft":"12","bw_driveline":"540"}}

User: "quote me a BB60.30 with chain shielding"
Return:
{"action":"quote","reply":"","params":{"model":"BB60.30","bb_shielding":"Chain"}}
"""

# =========================
# Parsers
# =========================
MODEL_RE = re.compile(r"\b((?:BB|BW|MDS|DS|TBW|RB|BS|GS|LRS|DHS|DHM|PD|DB|RT|RTR|TSG|PF|TQH)\s*\d+(?:\.\d+)?)\b", re.I)
WIDTH_FT_RE = re.compile(r"\b(\d{1,2})\s*(?:ft|foot|feet)\b", re.I)
WIDTH_IN_RE = re.compile(r"\b(\d{2,3})\s*(?:in|inch|inches|\"|”)\b", re.I)
DRIVELINE_RE = re.compile(r"\b(540|1000|1k)\b", re.I)


def norm_model(s: str) -> str:
    return s.upper().replace(" ", "")

def parse_shielding(text: str) -> str | None:
    t = text.lower()
    if "chain" in t or "chains" in t: return "Chain"
    if "belt" in t: return "Belt"
    if "single row" in t: return "Single Row"
    if "double row" in t: return "Double Row"
    return None

def parse_family(text: str) -> str | None:
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

def parse_width_ft(text: str) -> str | None:
    m = WIDTH_FT_RE.search(text)
    if m: return m.group(1)
    m2 = re.search(r"\b(10|12|13|15|20)\b", text)  # common batwing sizes
    return m2.group(1) if m2 else None

def parse_driveline(text: str) -> str | None:
    m = DRIVELINE_RE.search(text)
    if not m: return None
    val = m.group(1)
    return "1000" if val.lower() == "1k" else val

def intent_from_text(text: str) -> str | None:
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
    if mm: params["model"] = norm_model(mm.group(1))
    fam = parse_family(text)
    if fam: params["family"] = fam
    wft = parse_width_ft(text)
    if wft: params["width_ft"] = wft
    drv = parse_driveline(text)
    if drv: params["bw_driveline"] = drv
    sh = parse_shielding(text)
    if sh: params["bb_shielding"] = sh

    # Tire hints as free text for API matching when we can't map exact part id
    t = text.lower()
    tire_terms = []
    if "laminated" in t: tire_terms.append("laminated tires")
    if "foam" in t: tire_terms.append("foam-filled tires")
    if "ag" in t and "tire" in t: tire_terms.append("ag tires")
    if tire_terms: params["q"] = ", ".join(tire_terms)
    return params

# =========================
# Other helpers (NEW for robust Q&A)
# =========================

def extract_dealer_number(text: str) -> str | None:
    if not text:
        return None
    m = DEALER_REGEX.search(text)
    return m.group(1) if m else None


def letter_or_number_choice_to_index(msg: str, count: int) -> int | None:
    s = (msg or "").strip().lower()
    if not s or count <= 0:
        return None
    if len(s) == 1 and s.isalpha():
        idx = ord(s) - ord('a')
        return idx if 0 <= idx < count else None
    if s.isdigit():
        idx = int(s) - 1
        return idx if 0 <= idx < count else None
    return None


def format_question(q: Dict[str, Any]) -> str:
    title = q.get("question") or "Please choose:"
    # Prefer labels from choices_with_ids; fall back to choices
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
    """
    Call the upstream Quote API with optional retry (helps on cold starts).
    Returns: (json_or_fallback_dict, status_code, final_url)
    """
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


def format_quote_output(dealer_name, dealer_number, resp, forced_rate=None):
    rd = resp.get("response_data", resp)
    items = rd.get("items", [])
    summary = rd.get("summary", {})
    dealer_rate = float(forced_rate if forced_rate is not None else summary.get("dealer_discount_rate", 0.0))

    subtotal, dealer_amt, after_dealer, cash_amt, final_net, cash_line = apply_discounts_from_summary(summary, dealer_rate, items)

    lines = []
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
# NEW: save context with each question
# =========================

def set_pending_question(sess: Dict[str, Any], base_params: Dict[str, Any], q: Dict[str, Any]):
    """
    Save the current context and the next question so we can merge the user's answer
    with the previously provided params (e.g., keep model/family/width).
    """
    ctx = {k: v for k, v in (base_params or {}).items() if k != "dealer_number"}
    sess["in_progress_params"] = ctx
    sess["pending_question"] = {
        "name": q.get("name", ""),
        "choices": q.get("choices", []),
        "choices_with_ids": q.get("choices_with_ids", []),
        "question": q.get("question", "Please choose:")
    }


# =========================
# Local family trees (mimic GPT action logic)
# =========================
# Each family defines an ordered list of questions. Each question has:
# - name: field name to send to Quote API
# - question: human text
# - choices_with_ids: list of {id,label} the backend will accept
#
# You can extend or edit these safely without touching the rest of the code.

FAMILY_TREES: Dict[str, list] = {
    # BrushFighter (BF): API usually lists exact configs (bf_choice_id); no fixed local choices here.

    # BrushBull rotary cutters (BBxx.xx)
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
        # API will finalize (bb_choice_id) if needed
    ],

    # Dual-spindle cutters (DS / MDS)
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
        # API completes with ds choice ids if applicable
    ],

    # Batwing (BWxx.xx) – tires/qty are API-driven after driveline
    "batwing": [
        {
            "name": "bw_driveline",
            "question": "Which driveline do you want?",
            "choices_with_ids": [
                {"id": "540", "label": "540"},
                {"id": "1000", "label": "1000"},
            ],
        },
        # API will ask tire_id and bw_tire_qty
    ],

    # Turf Batwing (TBW): mirror of Batwing, but API will drive series/options
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
        # API will refine and pick exact model/options thereafter
    ],

    # Rear finish mowers
    "rear_finish": [
        {
            "name": "finish_choice",
            "question": "Which rear-finish configuration do you need?",
            "choices_with_ids": [
                {"id": "rear_discharge", "label": "Rear discharge"},
                {"id": "side_discharge", "label": "Side discharge"},
            ],
        },
        # API will ask width/model specifics
    ],

    # Box Scraper (BS) – widths are inches
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
        # API returns bs_choice_id to finalize
    ],

    # Grading Scraper (GS)
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
        # API returns gs_choice_id
    ],

    # Landscape Rake (LRS)
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
        # API returns lrs_choice_id
    ],

    # Rear Blade (RB)
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
        # API returns rb_choice_id
    ],

    # Disc Harrow (DH)
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
        # API will ask dh_spacing_id as needed (we will forward that)
    ],

    # Post Hole Digger (PD)
    "post_hole_digger": [
        {
            "name": "pd_model",
            "question": "Which PD model?",
            "choices_with_ids": [
                {"id": "PD25.21", "label": "PD25.21"},
                {"id": "PD35.31", "label": "PD35.31"},
                {"id": "PD95.51", "label": "PD95.51"},
            ],
        },
        # API will return auger options; we’ll answer with auger_id
    ],

    # Tiller (DB/RT/RTR)
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
        # API returns tiller_choice_id for exact unit
    ],

    # Bale Spear
    "bale_spear": [
        # API will list bspear_choice_id; no stable, local choices to hard-code.
    ],

    # Pallet Fork
    "pallet_fork": [
        # API will list pf_choice_id
    ],

    # Quick Hitch
    "quick_hitch": [
        # API will ask to pick TQH1 (Cat 1) or TQH2 (Cat 2) via qh_choice_id; we can optionally seed:
        {
            "name": "qh_choice_id",
            "question": "Which quick hitch category?",
            "choices_with_ids": [
                {"id": "TQH1", "label": "TQH1 (Category 1)"},
                {"id": "TQH2", "label": "TQH2 (Category 2)"},
            ],
        }
    ],

    # Stump Grinder
    "stump_grinder": [
        {
            "name": "hydraulics_id",
            "question": "Hydraulic option?",
            "choices_with_ids": [
                {"id": "standard", "label": "Standard flow"},
                {"id": "high_flow", "label": "High flow"},
            ],
        },
        # API finalizes exact model/part
    ],
}


def infer_family_from_model_or_text(params: Dict[str, Any], user_text: str = "") -> str | None:
    # explicit family wins
    fam = params.get("family")
    if fam:
        return fam
    # infer from model code
    m = (params.get("model") or "").upper()
    if m.startswith("BB"): return "brushbull"
    if m.startswith("BW"): return "batwing"
    if m.startswith("DS") or m.startswith("MDS"): return "dual_spindle"
    # infer from free text via parse_family
    pf = parse_family(user_text)
    return pf


def next_family_tree_question(params: Dict[str, Any], user_text: str = "") -> Dict[str, Any] | None:
    fam = infer_family_from_model_or_text(params, user_text)
    if not fam: return None
    tree = FAMILY_TREES.get(fam)
    if not tree: return None
    answered = set(k for k, v in (params or {}).items() if v not in (None, "", []))
    for q in tree:
        if q["name"] not in answered:
            # Return a copy so callers can mutate safely
            return {
                "name": q["name"],
                "question": q["question"],
                "choices_with_ids": list(q.get("choices_with_ids", []))
            }
    return None


# =========================
# Core quoting helper (drive from API required_questions) + family-tree fallback
# =========================

def do_quote(sess: Dict[str, Any], extra_params: Dict[str, Any]):
    dn = sess.get("dealer_number")
    if not dn:
        return jsonify({"reply": "Please provide your dealer number to begin (e.g., dealer #178200)."})

    # Merge context → params
    params: Dict[str, Any] = {}
    params.update(sess.get("in_progress_params", {}) or {})
    params.update(sess.get("params", {}) or {})
    params.update(extra_params or {})
    params["dealer_number"] = dn

    quote_json, status, used_url = call_quote_api("/quote", params)

    # 1) Upstream asks for more info
    if status == 200 and quote_json.get("mode") == "questions":
        rd = quote_json.get("response_data", quote_json)
        rq = rd.get("required_questions", [])
        if rq:
            q = rq[0]
            set_pending_question(sess, params, q)
            qtext = format_question(q)
            # attach UI block so frontend can render real buttons
            choices_struct = q.get("choices_with_ids") or [
                {"id": str(c), "label": str(c)} for c in (q.get("choices") or [])
            ]
            resp = make_response(jsonify({
                "reply": qtext,
                "ui": {
                    "type": "choices",
                    "name": q.get("name"),
                    "question": q.get("question"),
                    "choices": choices_struct,
                },
                "state": {"dealer_number": dn, "dealer_name": sess.get("dealer_name")},
                "debug": {"quote_api_url": used_url, "status": status, "params_sent": params}
            }))
            resp.set_cookie("dealer_number", dn, max_age=60*60*24*30, httponly=False, samesite="Lax")
            return resp

    # 2) Upstream returned a quote
    if status == 200 and quote_json.get("mode") == "quote":
        dealer_name = sess.get("dealer_name", "")
        dealer_rate = float(sess.get("dealer_discount",
                          quote_json.get("response_data", {}).get("summary", {}).get("dealer_discount_rate", 0.0)))
        out = format_quote_output(dealer_name, dn, quote_json, forced_rate=dealer_rate)
        resp = make_response(jsonify({
            "reply": out,
            "debug": {"quote_api_url": used_url, "status": status, "params_sent": params}
        }))
        resp.set_cookie("dealer_number", dn, max_age=60*60*24*30, httponly=False, samesite="Lax")
        return resp

    # 3) Family-tree fallback: if API didn’t guide us, ask the next local question
    lf_q = next_family_tree_question(params, user_text=(extra_params.get("q") or ""))
    if lf_q:
        set_pending_question(sess, params, lf_q)
        qtext = format_question(lf_q)
        resp = make_response(jsonify({
            "reply": qtext,
            "ui": {
                "type": "choices",
                "name": lf_q.get("name"),
                "question": lf_q.get("question"),
                "choices": lf_q.get("choices_with_ids"),
            },
            "state": {"dealer_number": dn, "dealer_name": sess.get("dealer_name")},
            "debug": {"note": "family_tree_fallback", "params_sent": params}
        }))
        resp.set_cookie("dealer_number", dn, max_age=60*60*24*30, httponly=False, samesite="Lax")
        return resp

    return jsonify({"reply": "There was a system error while retrieving data. Please try again shortly. If the issue persists, escalate to Benjamin Luci at 615-516-8802."})


# =========================
# Routes
# =========================

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "ok": True,
        "quote_api_base": QUOTE_API_BASE,
        "allowed_origins": ALLOWED_ORIGINS
    })


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True, silent=True) or {}
    user_message = (data.get("message") or "").strip()
    session_id = request.headers.get("X-Session-Id") or (data.get("session_id") or "default")
    if not user_message:
        return jsonify({"error": "Missing message"}), 400

    # Session & dealer number (payload → header → regex → session → cookie)
    sess = SESSIONS.setdefault(session_id, {})
    dn = None
    if data.get("dealer_number"):
        dn = str(data["dealer_number"]).strip()
    if not dn:
        hdr_dn = request.headers.get("X-Dealer-Number")
        if hdr_dn:
            dn = str(hdr_dn).strip()
    if not dn:
        dn = extract_dealer_number(user_message)
    if not dn:
        dn = sess.get("dealer_number")
    if not dn:
        dn = request.cookies.get("dealer_number")
    if dn:
        sess["dealer_number"] = dn

    logging.info("sid=%s payload_dn=%s header_dn=%s cookie_dn=%s sess_dn=%s",
                 session_id, data.get("dealer_number"),
                 request.headers.get("X-Dealer-Number"),
                 request.cookies.get("dealer_number"),
                 sess.get("dealer_number"))

    # 0) Shortcuts: help/knowledge
    if user_message.lower() in {"help", "knowledge", "instructions", "how do i quote", "tips"}:
        kb = KNOWLEDGE.strip() if KNOWLEDGE else "No internal knowledge is configured yet."
        return jsonify({"reply": kb})

    # 1) Dealer-only message → always update dealer (even if one is already set)
    only_dn = extract_dealer_number(user_message)
    if only_dn and not intent_from_text(user_message):
        info = call_dealer_discount(only_dn)
        rd = info if isinstance(info, dict) else {}
        if "discount" in rd:
            sess["dealer_discount"] = float(rd["discount"])
            sess["dealer_name"] = rd.get("dealer_name", "")
            sess["dealer_number"] = only_dn
            resp = make_response(jsonify({"reply": f"Using {sess['dealer_name']} (Dealer #{only_dn}) with a {int(sess['dealer_discount']*100)}% dealer discount for this session. ✅

What would you like me to quote?"}))
            resp.set_cookie("dealer_number", only_dn, max_age=60*60*24*30, httponly=False, samesite="Lax")
            return resp

    # 2) If user is answering last multiple-choice, resolve using IDs/labels and keep context
    pending = sess.get("pending_question")
    if pending and dn:
        cwids = pending.get("choices_with_ids") or []
        labels = [str(c.get("label", "")).strip() for c in cwids] if cwids else [str(c).strip() for c in (pending.get("choices") or [])]

        if not labels:
            return jsonify({"reply": pending.get("question", "Please choose:")})

        idx = letter_or_number_choice_to_index(user_message, len(labels))
        if idx is None:
            low = user_message.strip().lower()
            # exact label
            for i, lab in enumerate(labels):
                if lab.lower() == low:
                    idx = i; break
            # substring label
            if idx is None:
                for i, lab in enumerate(labels):
                    if low in lab.lower():
                        idx = i; break
            # exact ID
            if idx is None and cwids:
                ids = [str(c.get("id", "")).strip().lower() for c in cwids]
                for i, _id in enumerate(ids):
                    if _id and _id == low:
                        idx = i; break

        if idx is None:
            letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            opts = "
".join(f"{letters[i]}. {labels[i]}" for i in range(len(labels)))
            return jsonify({
                "reply": f"{pending.get('question','Please choose:')}

{opts}",
                "ui": {
                    "type": "choices",
                    "name": pending.get("name"),
                    "question": pending.get("question"),
                    "choices": cwids or [{"id": str(c), "label": str(c)} for c in (pending.get("choices") or [])]
                },
                "state": {"dealer_number": dn, "dealer_name": sess.get("dealer_name")}
            })

        # Prefer IDs when available
        value = cwids[idx].get("id") if cwids else (pending.get("choices") or [None])[idx]
        name = pending.get("name", "")
        if isinstance(value, str) and value.isdigit() and name.endswith("_qty"):
            value = int(value)

        merged = {}
        merged.update(sess.get("in_progress_params", {}) or {})
        merged.update(sess.get("params", {}) or {})
        merged[name] = value

        # clear pending/context and stale params
        sess.pop("pending_question", None)
        sess.pop("in_progress_params", None)
        sess["params"] = {}

        return do_quote(sess, merged)

    # 3) With a dealer number: always drive from the Quote API using merged context (or family tree)
    if dn:
        parsed = extract_params_from_text(user_message)
        extra = parsed if parsed else {"q": user_message}
        return do_quote(sess, extra)

    # 4) Planner fallback — only when we still don't have a dealer
    plan = call_openai_for_plan(user_message)
    action, reply, params = plan.get("action"), plan.get("reply") or "", plan.get("params", {})

    if action == "dealer_lookup":
        if not params.get("dealer_number"):
            return jsonify({"reply": "Please provide your dealer number (e.g., dealer #178200)."})
        info = call_dealer_discount(params["dealer_number"])
        rd = info if isinstance(info, dict) else {}
        if "discount" in rd:
            sess["dealer_discount"] = float(rd["discount"])
            sess["dealer_name"] = rd.get("dealer_name", "")
            sess["dealer_number"] = params["dealer_number"]
            resp = make_response(jsonify({"reply": f"Using {sess['dealer_name']} (Dealer #{sess['dealer_number']}) with a {int(sess['dealer_discount']*100)}% dealer discount for this session. ✅

What would you like me to quote?"}))
            resp.set_cookie("dealer_number", sess["dealer_number"], max_age=60*60*24*30, httponly=False, samesite="Lax")
            return resp
        return jsonify({"reply": "Could not retrieve dealer discount. Please try again shortly."})

    if action == "quote":
        if not params.get("dealer_number"):
            return jsonify({"reply": "Please provide your dealer number to begin (e.g., dealer #178200)."})
        # Keep behavior consistent: route through do_quote (lets family tree help too)
        sess["dealer_number"] = params["dealer_number"]
        return do_quote(sess, params)

    if action == "ask":
        return jsonify({"reply": reply or "What would you like to quote?"})

    # default smalltalk
    return jsonify({"reply": reply or "How can I help with your Woods quote today?"})


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    app.run(host="0.0.0.0", port=port, debug=True)
