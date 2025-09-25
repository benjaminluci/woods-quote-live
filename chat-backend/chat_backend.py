import os
import json
import re
import time
import logging
from typing import Any, Dict, Tuple, Optional, List

import requests
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS

# Optional OpenAI planner (used for AI-like chatter and non-quote intents)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore

logging.basicConfig(level=logging.INFO)

# =========================
# Config
# =========================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QUOTE_API_BASE = (os.getenv("QUOTE_API_BASE") or "").rstrip("/")
ALLOWED_ORIGINS = [o.strip() for o in (os.getenv("ALLOWED_ORIGINS") or "*").split(",")]

client = OpenAI(api_key=OPENAI_API_KEY) if (OPENAI_API_KEY and OpenAI) else None

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
# { dealer_number, dealer_discount, dealer_name, pending_question, in_progress_params, params, family, last_quote }
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
You are an orchestrator. On every turn you MUST return strict JSON:

{
  "action": "dealer_lookup" | "quote" | "ask" | "adjust" | "smalltalk",
  "reply": "string (optional for quote/adjust)",
  "params": { /* Woods API params or local adjust fields */ }
}

Semantics:
- dealer_lookup: user gave a dealer number; return {"dealer_number":"..."}.
- quote: we are ready to call the Quote API (family/model/params as far as known).
- ask: ask ONE crisp question when a key param is missing (no pricing).
- adjust: user asked for setup/markup/total math on the last or explicit base amount.
- smalltalk: chitchat or general Q&A NOT related to quoting.

Tips:
- Normalize families from natural phrases/typos: “box blade”→box_scraper, “brush bull”→brushbull, “bat wimg”→batwing, etc.
- When user says “add dual hub kit”, this is a quote action with accessory intent (NOT adjust).
- For “setup $200 / markup 12% / margin 10” on a recent quote, return {"action":"adjust"} with fields:
  { "base": "last" | 20000.00, "setup_fee": 200, "markup_pct": 12 }
- If user provides a new model/family mid-session, it means a new quote; do NOT ask about old pending question.

Family-tree guidance (only when API doesn’t specify next step):
- Batwing → bw_driveline → tires (tire_id) → qty (bw_tire_qty)
- BrushBull → duty → shielding → tailwheel
- Disc Harrow → width_in → duty → blade → spacing_id
(If API returns required_questions, prefer those names.)

Routing rules:
- If message ONLY contains a dealer number (e.g., “178647” or “dealer #178647”), return dealer_lookup.
- If the message clearly requests a quote (mentions model like BW12.40 or a family), return quote with normalized params.
- If it’s a post-quote math request (setup/markup/total), return adjust.
- Otherwise smalltalk or ask.
"""

# =========================
# Parsers & matchers
# =========================
DEALER_REGEX = re.compile(r"\b(?:dealer\s*#?\s*)?(\d{5,7})\b", re.I)

MODEL_RE = re.compile(
    r"\b((?:BB|BW|MDS|DS|TBW|RB|BS|GS|LRS|DHS|DHM|PD|DB|RT|RTR|TSG|PF|TQH)\s*\d+(?:[.\-]\d+)?|\bBF\d{1,2}\.\d{2})\b",
    re.I,
)
WIDTH_FT_RE = re.compile(r"\b(\d{1,2})\s*(?:ft|foot|feet)\b", re.I)
WIDTH_IN_RE = re.compile(r"\b(\d{2,3})\s*(?:in|inch|inches|\"|”)\b", re.I)
DRIVELINE_RE = re.compile(r"\b(540|1000|1k)\b", re.I)

# Driveline defaults when API omits choices
DRIVELINE_NAMES = {"driveline", "bw_driveline", "ds_driveline"}  # lowercase
DEFAULT_DRIVELINE_CHOICES = [
    {"id": "540", "label": "540 RPM"},
    {"id": "1000", "label": "1000 RPM"},
]

import difflib
FAMILY_SYNONYMS = {
    "box scraper": ["box blade", "boxscraper", "box-blade", "bbx", "boxscrpr"],
    "brushbull": ["brush bull", "brushbul", "brush-bull", "bb", "brshbull"],
    "batwing": ["bat wing", "batwimg", "batwong", "bat wimg"],
    "dual spindle": ["dual-spindle", "dualspindle", "ds"],
    "disc harrow": ["disc-harrow", "disk harrow", "disk-harrow", "disc", "disk"],
    "rear blade": ["rearblade", "rear-blade", "rb"],
    "grading scraper": ["grading-scraper", "gradingscraper", "gs"],
    "landscape rake": ["landscape-rake", "lscape rake", "lr"],
    "rear finish": ["rmm", "rear mower", "finish mower", "rear-finish"],
    "turf batwing": ["turf-batwing", "turf bat wing"],
    "post hole digger": ["post-hole digger", "phd", "post hole", "auger"],
    "tiller": ["rototiller", "roto tiller"],
    "quick hitch": ["quick-hitch", "qh"],
    "stump grinder": ["stump-grinder", "sg"],
    "bale spear": ["bale-spear", "bale fork"],
    "pallet fork": ["pallet-fork", "forks"],
}
FAMILY_CANON = {
    "box scraper": "box_scraper",
    "brushbull": "brushbull",
    "batwing": "batwing",
    "dual spindle": "dual_spindle",
    "disc harrow": "disc_harrow",
    "rear blade": "rear_blade",
    "grading scraper": "grading_scraper",
    "landscape rake": "landscape_rake",
    "rear finish": "rear_finish",
    "turf batwing": "turf_batwing",
    "post hole digger": "post_hole_digger",
    "tiller": "tiller",
    "quick hitch": "quick_hitch",
    "stump grinder": "stump_grinder",
    "bale spear": "bale_spear",
    "pallet fork": "pallet_fork",
}

def extract_dealer_number(text: str) -> Optional[str]:
    if not text:
        return None
    m = DEALER_REGEX.search(text)
    return m.group(1) if m else None

def parse_model(text: str) -> Optional[str]:
    m = MODEL_RE.search(text or "")
    if not m:
        return None
    tok = m.group(1).upper().replace(" ", "")
    tok = tok.replace("-", ".")
    if "." not in tok and len(tok) > 2 and tok[-2:].isdigit():
        tok = tok[:-2] + "." + tok[-2:]
    return tok

def parse_family(text: str) -> Optional[str]:
    t = (text or "").lower()
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

def fuzzy_family(text: str) -> Optional[str]:
    t = (text or "").lower()
    # direct alias hit
    for base, alts in FAMILY_SYNONYMS.items():
        if base in t or any(a in t for a in alts):
            return FAMILY_CANON[base]
    # fuzzy on tokens
    tokens = re.findall(r"[a-z]+(?:\s[a-z]+)?", t)
    candidates = list(FAMILY_CANON.keys())
    for tok in tokens:
        best = difflib.get_close_matches(tok, candidates, n=1, cutoff=0.86)
        if best:
            return FAMILY_CANON[best[0]]
    return None

def parse_width_ft(text: str) -> Optional[str]:
    m = WIDTH_FT_RE.search(text or "")
    if m:
        return m.group(1)
    m2 = re.search(r"\b(10|12|13|15|20)\b", text or "")
    return m2.group(1) if m2 else None

def parse_driveline(text: str) -> Optional[str]:
    m = DRIVELINE_RE.search(text or "")
    if not m:
        return None
    val = m.group(1)
    return "1000" if val.lower() == "1k" else val

def parse_shielding(text: str) -> Optional[str]:
    t = (text or "").lower()
    if "chain" in t or "chains" in t:
        return "Chain"
    if "belt" in t:
        return "Belt"
    if "single row" in t:
        return "Single Row"
    if "double row" in t:
        return "Double Row"
    return None

def intent_from_text(text: str) -> Optional[str]:
    t = (text or "").lower()
    if any(w in t for w in ["quote", "price", "how much", "cost", "get me a", "need a"]):
        return "quote"
    if MODEL_RE.search(text or ""):
        return "quote"
    if parse_family(text) or fuzzy_family(text):
        return "quote"
    return None

def extract_params_from_text(text: str) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    m = parse_model(text)
    if m:
        params["model"] = m
    fam = parse_family(text) or fuzzy_family(text)
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

    # lightweight tire hints to prime API
    t = (text or "").lower()
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
# Helpers
# =========================
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
        labels = [str(c).strip() for c in (q.get("choices") or [])]
    if not labels:
        return title
    abc = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    opts = [f"{abc[i]}. {labels[i]}" for i in range(len(labels))]
    return f"{title}\n\n" + "\n".join(opts)

def infer_field_from_question(question: str, family: Optional[str]) -> Optional[str]:
    qt = (question or "").lower()
    fam = (family or "").lower()
    # Batwing
    if "what size" in qt and "batwing" in qt:
        return "bw_size_ft"
    if "duty" in qt and "batwing" in qt:
        return "bw_duty"
    if "driveline" in qt:
        return "bw_driveline" if "batwing" in fam else ("ds_driveline" if "dual" in fam else "driveline")
    if "how many" in qt and "tire" in qt:
        return "bw_tire_qty" if "batwing" in fam else "tire_qty"
    if "tire option" in qt:
        return "tire_id"
    # Disc Harrow
    if ("working width" in qt or "what width" in qt) and "disc" in qt:
        return "dh_width_in"
    if "duty" in qt and "disc" in qt:
        return "dh_duty"
    if "blade style" in qt:
        return "dh_blade"
    if "disc spacing" in qt:
        return "dh_spacing_id"
    # BrushBull
    if "what size" in qt and "brushbull" in qt:
        return "bb_size_ft"
    if "shield" in qt and "brushbull" in qt:
        return "bb_shielding"
    return None

def _try_local_adjustment_reply(message: str, sess: Dict[str, Any]) -> Optional[str]:
    """
    Post-quote (or ad-hoc) adjustments with robust parsing.

    Triggers when the user mentions:
      • setup / set up / setup fee  (with a number), OR
      • markup / mark it up / margin (with a number), OR
      • "add $X to the price/total/net/quote" (treat as setup)

    Base amount selection:
      • Default = last quote's Final Dealer Net (or summary total).
      • Only treat a $-amount in the message as the BASE if paired with
        "take/use/base/starting from/on that/that (total/net/price)" keywords.
      • Never treat the setup dollar as the base.

    Won’t hijack accessory requests (e.g., “add dual hub kit”).
    """
    if not message:
        return None
    msg = message.lower()

    # Gates to avoid hijacking accessories
    mentions_setup = bool(re.search(r"\b(set\s*up|setup|setup\s*fee)\b", msg))
    mentions_markup = bool(re.search(r"\b(mark(?:\s*it)?\s*up|markup|margin)\b", msg))
    # "add $300 to the price/total/net/quote"
    add_price_setup = bool(re.search(
        r"\badd\b.*\$\s*[0-9][0-9,]*(?:\.[0-9]{1,2})?.*\b(price|total|net|quote|final)\b", msg
    ))

    # Allow quick follow-up: "what's the total" after an adjustment
    if not (mentions_setup or mentions_markup or add_price_setup):
        if re.search(r"\b(total|out\s*the\s*door|all\s*in)\b", msg) and sess.get("last_adjustment"):
            last_adj = sess["last_adjustment"]
            return f"✅ Adjusted Total: ${float(last_adj.get('total', 0.0)):,.2f}"
        return None

    NUM = r"([0-9][0-9,]*(?:\.[0-9]{1,2})?)"
    def _num(s: str) -> float:
        return float(s.replace(",", ""))

    # Parse setup amount (if any). Accept number before/after keyword OR "add $X to the price..."
    setup_re = (
        re.search(rf"(?:set\s*up|setup|setup\s*fee)\D{{0,12}}{NUM}", msg)
        or re.search(rf"{NUM}\D{{0,6}}(?:set\s*up|setup|setup\s*fee)", msg)
    )
    if not setup_re and add_price_setup:
        setup_re = re.search(rf"\badd\b.*\$\s*{NUM}", msg)

    # Parse markup/margin % (number before/after; % optional)
    markup_re = (
        re.search(rf"(?:mark(?:\s*it)?\s*up|markup|margin)\D{{0,12}}{NUM}\s*%?", msg)
        or re.search(rf"{NUM}\s*%?\s*(?:mark(?:\s*it)?\s*up|markup|margin)", msg)
    )

    if not (setup_re or markup_re):
        return None

    # -------- Determine BASE amount safely --------
    # Start with last quote
    base_amt = 0.0
    last = sess.get("last_quote") or {}
    if last:
        base_amt = float(last.get("final_net") or last.get("summary", {}).get("total") or 0.0)

    # Candidate explicit base (only when paired with base words)
    BASE_WORDS = r"(?:take|use|base|starting\s*from|on\s*that|that(?:\s+(?:total|price|net|quote))?)"
    base_kw_before = re.search(rf"{BASE_WORDS}\D{{0,12}}\$\s*{NUM}", msg)
    base_kw_after  = re.search(rf"\$\s*{NUM}\D{{0,12}}(?:as\s*base|to\s*use|to\s*start)", msg)
    base_match = base_kw_before or base_kw_after

    # Extract setup amount (to avoid mistaking it for base)
    setup_amt = _num(setup_re.group(1)) if setup_re else 0.0

    if base_match:
        cand = _num(base_match.group(1))
        # If the candidate equals the setup value and we have a last quote, keep last-quote base
        if not (abs(cand - setup_amt) < 1e-6 and base_amt > 0):
            base_amt = cand

    # If still no base, we can't compute
    if base_amt <= 0:
        return None

    # -------- Compute --------
    markup_pct = float(markup_re.group(1)) if markup_re else 0.0
    setup_fee = setup_amt

    marked = base_amt * (1.0 + markup_pct / 100.0) if markup_pct else base_amt
    total = marked + setup_fee

    # Save for "what's the total?" follow-ups
    sess["last_adjustment"] = {"base": base_amt, "setup_fee": setup_fee, "markup_pct": markup_pct, "total": total}

    lines = [f"Starting from ${base_amt:,.2f}"]
    if markup_pct:
        lines.append(f"Markup ({markup_pct:.0f}%): +${(marked - base_amt):,.2f} → ${marked:,.2f}")
    if setup_fee:
        lines.append(f"Setup Fee: +${setup_fee:,.2f} → ${total:,.2f}")
    lines.append(f"✅ Adjusted Total: ${total:,.2f}")
    return "\n".join(lines)

# =========================
# OpenAI + Quote API
# =========================
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

# =========================
# Discount math & formatting
# =========================
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
    pf = parse_family(user_text) or fuzzy_family(user_text)
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
# Pending-question state
# =========================
def set_pending_question(sess: Dict[str, Any], base_params: Dict[str, Any], q: Dict[str, Any]) -> None:
    # Infer a field name if the API omitted it (prevents empty-key params later)
    def infer_field_name(question_text: str, family: Optional[str]) -> Optional[str]:
        qt = (question_text or "").lower()
        fam = (family or (sess.get("family") or "")).lower()
        if "what size" in qt and "batwing" in qt:
            return "bw_size_ft"
        if "duty" in qt and "batwing" in qt:
            return "bw_duty"
        if "driveline" in qt:
            if "dual" in fam:
                return "ds_driveline"
            if "batwing" in fam:
                return "bw_driveline"
            return "driveline"
        if "how many" in qt and "tire" in qt:
            return "bw_tire_qty" if "batwing" in fam else "tire_qty"
        if "tire" in qt and "option" in qt:
            return "tire_id"
        if "shield" in qt and "brushbull" in fam:
            return "bb_shielding"
        return None

    name = (q.get("name") or "").strip()
    if not name:
        name = infer_field_name(q.get("question"), base_params.get("family"))

    ctx = {k: v for k, v in (base_params or {}).items() if k != "dealer_number"}
    sess["in_progress_params"] = ctx
    sess["pending_question"] = {
        "name": name or "",
        "choices": q.get("choices", []),
        "choices_with_ids": q.get("choices_with_ids", []),
        "question": q.get("question", "Please choose:"),
    }

# =========================
# Core quoting helper
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

    # 1) API needs more info
    if status == 200 and quote_json.get("mode") == "questions":
        rd = quote_json.get("response_data", quote_json)
        rq = rd.get("required_questions", [])
        if rq:
            q = rq[0]

            # Driveline safety: ensure buttons appear even if API omits choices
            name_l = (q.get("name") or "").strip().lower()
            if name_l in {"driveline", "bw_driveline", "ds_driveline"} and not q.get("choices_with_ids") and not q.get("choices"):
                q = dict(q)
                q["choices_with_ids"] = [
                    {"id": "540", "label": "540 RPM"},
                    {"id": "1000", "label": "1000 RPM"},
                ]
                if not q.get("question"):
                    q["question"] = "Which driveline do you want?"

            set_pending_question(sess, params, q)
            qtext = format_question(q)

            # Build UI choices (fallback to defaults for driveline if still empty)
            choices_struct = (
                q.get("choices_with_ids")
                or [{"id": str(c), "label": str(c)} for c in (q.get("choices") or [])]
            )
            if not choices_struct and name_l in {"driveline", "bw_driveline", "ds_driveline"}:
                choices_struct = [
                    {"id": "540", "label": "540 RPM"},
                    {"id": "1000", "label": "1000 RPM"},
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
                "debug": {"quote_api_url": used_url, "status": status, "params_sent": params},
            }))
            resp.set_cookie("dealer_number", dn, max_age=60*60*24*30, httponly=False, samesite="Lax")
            return resp

    # 2) API returned a quote
    if status == 200 and quote_json.get("mode") == "quote":
        dealer_name = sess.get("dealer_name", "")
        dealer_rate = float(
            sess.get(
                "dealer_discount",
                quote_json.get("response_data", {}).get("summary", {}).get("dealer_discount_rate", 0.0),
            )
        )

        # Persist last-quote figures for AI math
        rd = quote_json.get("response_data", quote_json)
        items = rd.get("items", [])
        summary = rd.get("summary", {})
        subtotal, dealer_amt, after_dealer, cash_amt, final_net, _cash_line = apply_discounts_from_summary(
            summary, dealer_rate, items
        )
        sess["last_quote"] = {
            "subtotal": subtotal,
            "dealer_discount_amt": dealer_amt,
            "after_dealer": after_dealer,
            "cash_discount_amt": cash_amt,
            "final_net": final_net,
            "summary": summary,
        }
        sess["post_quote"] = True  # favor AI/planner for non-quote follow-ups

        out = format_quote_output(dealer_name, dn, quote_json, forced_rate=dealer_rate)
        resp = make_response(jsonify({
            "reply": out,
            "state": {"dealer_number": dn, "dealer_name": dealer_name},
            "debug": {"quote_api_url": used_url, "status": status, "params_sent": params},
        }))
        resp.set_cookie("dealer_number", dn, max_age=60*60*24*30, httponly=False, samesite="Lax")
        return resp

    # 3) Family-tree fallback (when API gives no required_questions)
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
                "choices": lf_q.get("choices_with_ids") or [
                    {"id": str(c), "label": str(c)} for c in (lf_q.get("choices") or [])
                ],
            },
            "state": {"dealer_number": dn, "dealer_name": sess.get("dealer_name")},
            "debug": {"note": "family_tree_fallback", "params_sent": params},
        }))
        resp.set_cookie("dealer_number", dn, max_age=60*60*24*30, httponly=False, samesite="Lax")
        return resp

    # 4) Generic failure (also covers upstream HTML/error bodies)
    raw = quote_json.get("raw_text", "")
    if isinstance(raw, str) and "<!doctype" in raw.lower():
        logging.error("Upstream returned non-JSON HTML at %s", used_url)
    return jsonify({
        "reply": "There was a system error while retrieving your quote. Please try again. If the issue persists, contact support.",
        "state": {"dealer_number": dn, "dealer_name": sess.get("dealer_name")},
        "debug": {"quote_api_url": used_url, "status": status, "params_sent": params},
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
    try:
        data = request.get_json(force=True, silent=True) or {}
        user_message = (data.get("message") or "").strip()
        if not user_message:
            return jsonify({"error": "Missing message"}), 400

        # Session id
        body_sid = data.get("session_id")
        header_sid = request.headers.get("X-Session-Id")
        session_id = header_sid or body_sid or f"anon-{int(time.time()*1000)}"
        sess = SESSIONS.setdefault(session_id, {})

        # Dealer: JSON body → header → text (dealer #123456 OR bare 123456) → session → cookie
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

        # Dealer-only message (no quote intent)
        only_dn = extract_dealer_number(user_message)
        if only_dn and not intent_from_text(user_message):
            info = call_dealer_discount(only_dn)
            rd = info if isinstance(info, dict) else {}
            if "discount" in rd:
                sess["dealer_discount"] = float(rd["discount"])
                sess["dealer_name"] = rd.get("dealer_name", "")
                sess["dealer_number"] = only_dn
                reply_msg = (
                    f"Using {sess['dealer_name']} (Dealer #{only_dn}) with a "
                    f"{int(sess['dealer_discount'] * 100)}% dealer discount for this session. ✅\n\n"
                    "What would you like me to quote?"
                )
                resp = make_response(jsonify({
                    "reply": reply_msg,
                    "state": {"dealer_number": only_dn, "dealer_name": sess["dealer_name"]},
                }))
                resp.set_cookie("dealer_number", only_dn, max_age=60*60*24*30, httponly=False, samesite="Lax")
                return resp

        # Allow multiple quotes in one session (reset quote-scoped state but keep dealer)
        def _reset_quote_state(s):
            s.pop("pending_question", None)
            s.pop("in_progress_params", None)
            s["params"] = {}
            s.pop("family", None)
            s.pop("post_quote", None)

        model_in_text = bool(MODEL_RE.search(user_message or ""))
        fam_in_text = parse_family(user_message) or fuzzy_family(user_message)
        started_new_quote = bool(model_in_text or fam_in_text)
        if sess.get("pending_question") and started_new_quote:
            _reset_quote_state(sess)

        # ---- Local AI-like adjustments FIRST (forgiving; won’t hijack accessories) ----
        def _try_local_adjustment_reply(message: str, sess: Dict[str, Any]) -> Optional[str]:
            """
            Triggers ONLY when the message mentions setup/set up OR markup/mark it up/margin
            along with a number. Uses explicit $-based base if present; otherwise the last quote.
            Avoids misreading the '12' from '12% margin' as the base.
            """
            if not message:
                return None
            msg = message.lower()

            # Keyword gates (avoid intercepting "add dual hub kit")
            mentions_setup = bool(re.search(r"\b(set\s*up|setup|setup\s*fee)\b", msg))
            mentions_markup = bool(re.search(r"\b(mark(?:\s*it)?\s*up|markup|margin)\b", msg))
            if not (mentions_setup or mentions_markup):
                # Follow-up: "what's the total?" after a prior adjustment
                if re.search(r"\b(total|out\s*the\s*door|all\s*in)\b", msg) and sess.get("last_adjustment"):
                    last_adj = sess["last_adjustment"]
                    return f"✅ Adjusted Total: ${float(last_adj.get('total', 0.0)):,.2f}"
                return None

            # tolerant numbers
            NUM = r"([0-9][0-9,]*(?:\.[0-9]{1,2})?)"

            setup_re = (
                re.search(rf"(?:set\s*up|setup|setup\s*fee)\D{{0,12}}{NUM}", msg)
                or re.search(rf"{NUM}\D{{0,6}}(?:set\s*up|setup|setup\s*fee)", msg)
            )
            markup_re = (
                re.search(rf"(?:mark(?:\s*it)?\s*up|markup|margin)\D{{0,12}}{NUM}\s*%?", msg)
                or re.search(rf"{NUM}\s*%?\s*(?:mark(?:\s*it)?\s*up|markup|margin)", msg)
            )

            if not (setup_re or markup_re):
                return None

            def _num(s: str) -> float:
                return float(s.replace(",", ""))

            # Base detection:
            # 1) Prefer last quote if available.
            base_amt = 0.0
            last = sess.get("last_quote") or {}
            if last:
                base_amt = float(last.get("final_net") or last.get("summary", {}).get("total") or 0.0)

            # 2) If the message includes a $-amount (explicit base), use it.
            # Require a literal $ to avoid grabbing the '12' in '12% margin'.
            base_dollar = re.search(rf"\$\s*{NUM}", msg)
            if base_dollar:
                base_amt = _num(base_dollar.group(1))

            if base_amt <= 0:
                return None  # no trustworthy base

            setup_fee = _num(setup_re.group(1)) if setup_re else 0.0
            markup_pct = float((markup_re and markup_re.group(1)) or 0.0)

            marked = base_amt * (1.0 + markup_pct / 100.0) if markup_pct else base_amt
            total = marked + setup_fee

            sess["last_adjustment"] = {"base": base_amt, "setup_fee": setup_fee, "markup_pct": markup_pct, "total": total}

            lines = [f"Starting from ${base_amt:,.2f}"]
            if markup_pct:
                lines.append(f"Markup ({markup_pct:.0f}%): +${(marked - base_amt):,.2f} → ${marked:,.2f}")
            if setup_fee:
                lines.append(f"Setup Fee: +${setup_fee:,.2f} → ${total:,.2f}")
            lines.append(f"✅ Adjusted Total: ${total:,.2f}")
            return "\n".join(lines)

        adj_reply = _try_local_adjustment_reply(user_message, sess)
        if adj_reply:
            if dn:
                return jsonify({"reply": adj_reply, "state": {"dealer_number": dn, "dealer_name": sess.get("dealer_name")}})
            return jsonify({"reply": adj_reply})

        # ---- Resolve pending multiple-choice answers ----
        pending = sess.get("pending_question")
        if pending and not started_new_quote:
            cwids = pending.get("choices_with_ids") or []
            labels = (
                [str(c.get("label", "")).strip() for c in cwids]
                if cwids else
                [str(c).strip() for c in (pending.get("choices") or [])]
            )

            if not labels:
                # Driveline: present defaults
                pname = (pending.get("name") or "").strip().lower()
                if pname in {"driveline", "bw_driveline", "ds_driveline"}:
                    return jsonify({
                        "reply": pending.get("question", "Which driveline do you want?"),
                        "ui": {"type": "choices", "name": pending.get("name") or "driveline",
                               "question": pending.get("question") or "Which driveline do you want?",
                               "choices": [
                                   {"id": "540", "label": "540 RPM"},
                                   {"id": "1000", "label": "1000 RPM"},
                               ]},
                        "state": {"dealer_number": dn, "dealer_name": sess.get("dealer_name")},
                    })
                return jsonify({
                    "reply": pending.get("question", "Please choose:"),
                    "state": {"dealer_number": dn, "dealer_name": sess.get("dealer_name")},
                })

            idx = letter_or_number_choice_to_index(user_message, len(labels))

            if idx is None:
                low = user_message.strip().lower()
                # exact label
                for i, lab in enumerate(labels):
                    if lab.lower() == low:
                        idx = i
                        break
                # contains
                if idx is None:
                    for i, lab in enumerate(labels):
                        if low in lab.lower():
                            idx = i
                            break
                # exact id match
                if idx is None and cwids:
                    ids = [str(c.get("id", "")).strip().lower() for c in cwids]
                    for i, _id in enumerate(ids):
                        if _id and _id == low:
                            idx = i
                            break

            if idx is None:
                letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                opts = "\n".join(f"{letters[i]}. {labels[i]}" for i in range(len(labels)))
                return jsonify({
                    "reply": f"{pending.get('question', 'Please choose:')}\n\n{opts}",
                    "ui": {"type": "choices", "name": pending.get("name"), "question": pending.get("question"),
                           "choices": cwids or [{"id": str(c), "label": str(c)} for c in (pending.get("choices") or [])]},
                    "state": {"dealer_number": dn, "dealer_name": sess.get("dealer_name")},
                })

            # Build merged params with both provided and canonical names
            value = cwids[idx].get("id") if cwids else (pending.get("choices") or [None])[idx]
            name = (pending.get("name") or "").strip()
            lname = name.lower() if name else ""
            canonical = infer_field_from_question(pending.get("question"), sess.get("family"))

            merged: Dict[str, Any] = {}
            merged.update(sess.get("in_progress_params", {}) or {})
            merged.update(sess.get("params", {}) or {})

            def _maybe_int(k: str, v: Any) -> Any:
                if isinstance(v, str) and k and k.endswith("_qty") and v.isdigit():
                    return int(v)
                return v

            if lname:
                merged[lname] = _maybe_int(lname, value)
            if canonical and canonical != lname:
                merged[canonical] = _maybe_int(canonical, value)

            merged["dealer_number"] = dn  # always send dealer on answer turns

            sess.pop("pending_question", None)
            sess.pop("in_progress_params", None)
            sess["params"] = {}

            return do_quote(sess, merged)

        # ---- No pending question ----
        if dn:
            # Decide if this is a new quote
            if model_in_text or fam_in_text:
                parsed = extract_params_from_text(user_message)
                if parsed.get("family"):
                    sess["family"] = parsed["family"]
                extra = parsed if parsed else {"q": user_message}
                return do_quote(sess, extra)

            # Otherwise, let the planner talk (AI feel)
            plan = call_openai_for_plan(user_message)
            action = plan.get("action")
            reply = plan.get("reply") or ""
            params = plan.get("params", {})

            if action == "dealer_lookup" and params.get("dealer_number"):
                info = call_dealer_discount(params["dealer_number"])
                rd = info if isinstance(info, dict) else {}
                if "discount" in rd:
                    sess["dealer_discount"] = float(rd["discount"])
                    sess["dealer_name"] = rd.get("dealer_name", "")
                    sess["dealer_number"] = params["dealer_number"]
                    msg = (
                        f"Using {sess['dealer_name']} (Dealer #{sess['dealer_number']}) with a "
                        f"{int(sess['dealer_discount'] * 100)}% dealer discount for this session. ✅\n\n"
                        "What would you like me to quote?"
                    )
                    return jsonify({
                        "reply": msg,
                        "state": {"dealer_number": sess["dealer_number"], "dealer_name": sess["dealer_name"]},
                    })

            if action == "quote":
                if not params.get("dealer_number"):
                    params["dealer_number"] = dn
                if params.get("family"):
                    sess["family"] = params["family"]
                return do_quote(sess, params)

            # smalltalk / ask
            return jsonify({
                "reply": reply or "How can I help with your Woods quote today?",
                "state": {"dealer_number": dn, "dealer_name": sess.get("dealer_name")},
            })

        # ---- No dealer known yet ----
        # Still allow follow-up totals after a prior adjustment (no dealer required)
        if re.search(r"\b(total|out\s*the\s*door|all\s*in)\b", user_message.lower()) and sess.get("last_adjustment"):
            last_adj = sess["last_adjustment"]
            return jsonify({"reply": f"✅ Adjusted Total: ${float(last_adj.get('total', 0.0)):,.2f}"})

        # Planner to request dealer nicely or handle smalltalk
        plan = call_openai_for_plan(user_message)
        action = plan.get("action")
        reply = plan.get("reply") or ""
        params = plan.get("params", {})

        if action == "dealer_lookup":
            if not params.get("dealer_number"):
                return jsonify({"reply": "Please provide your dealer number (e.g., dealer #178200)."})
            info = call_dealer_discount(params["dealer_number"])
            rd = info if isinstance(info, dict) else {}
            if "discount" in rd:
                sess["dealer_discount"] = float(rd["discount"])
                sess["dealer_name"] = rd.get("dealer_name", "")
                sess["dealer_number"] = params["dealer_number"]
                msg = (
                    f"Using {sess['dealer_name']} (Dealer #{sess['dealer_number']}) with a "
                    f"{int(sess['dealer_discount'] * 100)}% dealer discount for this session. ✅\n\n"
                    "What would you like me to quote?"
                )
                return jsonify({
                    "reply": msg,
                    "state": {"dealer_number": sess["dealer_number"], "dealer_name": sess["dealer_name"]},
                })

        if action == "quote":
            if not params.get("dealer_number"):
                return jsonify({"reply": "Please provide your dealer number to begin (e.g., dealer #178200)."})
            sess["dealer_number"] = params["dealer_number"]
            if params.get("family"):
                sess["family"] = params["family"]
            return do_quote(sess, params)

        if action == "ask":
            return jsonify({"reply": reply or "Please provide your dealer number to begin (e.g., dealer #178200)."})

        return jsonify({"reply": reply or "Please provide your dealer number to begin (e.g., dealer #178200)."})

    except Exception as e:
        logging.exception("Unhandled error in /chat")
        # Always return JSON so the frontend never sees an HTML error page
        return jsonify({
            "reply": "There was a system error while handling your request. Please try again.",
            "error": str(e),
        })

# =========================
# Entrypoint
# =========================
if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    app.run(host="0.0.0.0", port=port, debug=True)
