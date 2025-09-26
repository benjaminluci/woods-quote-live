# app.py — AI-first Woods chatbot with tool-calling, with stable session memory
# - Persists tool calls/results to session (prevents forgetting steps)
# - Maps A/B/C/1/2/3 to the actual label shown last turn
# - Injects SESSION_STATE each turn (dealer + last choices)

import os, re, json, time, logging, traceback
from typing import Any, Dict, List, Tuple

import requests
from flask import Flask, request, jsonify
try:
    from flask_cors import CORS
except Exception:
    CORS = None

# ---------------- Config ----------------
QUOTE_API_BASE = os.environ.get("QUOTE_API_BASE", "https://woods-quote-api.onrender.com").rstrip("/")
OPENAI_MODEL   = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

client = None
try:
    from openai import OpenAI
    if os.environ.get("OPENAI_API_KEY"):
        client = OpenAI()
except Exception:
    client = None

app = Flask(__name__)
if CORS:
    allow = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "").split(",") if o.strip()]
    if allow:
        CORS(app, resources={r"/chat": {"origins": allow}}, supports_credentials=False)
    else:
        CORS(app, supports_credentials=False)

logging.basicConfig(level=logging.INFO)
SESSIONS: Dict[str, Dict[str, Any]] = {}

# ---------------- Knowledge ----------------
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
- Always format multiple options vertically with lettered lists (A., B., C., …) for clarity.
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

Additional Guidance
- If the API requests a driveline but provides no choices, present these defaults:
  • 540 RPM
  • 1000 RPM
"""

# ---------------- Family-trees ----------------
FAMILY_TREE_GUIDE = r"""
Family-tree guidance — use ONLY when the API does NOT specify the next step (i.e., when /quote does not return required_questions). Ask exactly one question at a time and wait for the answer. Use the exact parameter names shown.

BrushFighter
  1) width_ft (5/6/7 if not implied by model)
  2) bf_choice_id or bf_choice (exact ID or label provided by API)
  3) Optional drive hint: drive ("Slip Clutch" | "Shear Pin") — only if ambiguous

BrushBull
  1) bb_shielding ("Belt" | "Chain" | "Single Row" | "Double Row")
  2) Optional: bb_tailwheel ("Single" | "Dual") if the API asks via required_questions

Dual Spindle (DS/MDS)
  1) ds_mount ("mounted" | "pull")
  2) ds_shielding (e.g., "Belt" | "Chain") when asked
  3) ds_driveline ("540" | "1000")
  4) tire_id (part ID) — ask only if API doesn’t return choices automatically
  5) tire_qty (integer)

Batwing (BW)
  1) bw_duty (series-specific) — ask only if needed
  2) bw_driveline ("540" | "1000")
  3) shielding_rows ("Single Row" | "Double Row") if supported
  4) deck_rings ("Yes" | "No") if supported
  5) tire_id (choices_with_ids from API)
  6) bw_tire_qty (choices from API / width rules)

Turf Batwing (TBW)
  1) tbw_duty ("Residential (.20)" | "Commercial (.40)")
  2) width_ft (12/15/17 — residential 12 only)
  3) front_rollers ("Yes" | "No")
  4) chains ("Yes" | "No")

Rear Discharge Finish Mower
  1) finish_choice ("tk" | "tkp" | "rd990x")
  2) For TK/TKP: front_rollers ("Yes"/"No"), chains ("Yes"/"No") if user asks

Box Scraper (BS)   [valid widths 48/60/72/84 in]
  1) bs_width_in ("48" | "60" | "72" | "84")
  2) bs_duty ("Light Duty (.20)" | "Medium Duty (.30)" | "Heavy Duty (.40)") if offered
  3) bs_choice_id or bs_choice

Grading Scraper (GS)
  1) gs_width_in
  2) gs_choice_id or gs_choice

Landscape Rake (LRS)
  1) lrs_width_in
  2) lrs_grade ("Standard" | "Premium (P)") when both exist
  3) lrs_choice_id or lrs_choice

Rear Blade (RB)
  1) rb_width_in
  2) rb_duty ("Standard" | "Standard (Premium P)" | "Heavy Duty" | "Extreme Duty")
  3) rb_choice_id or rb_choice

Disc Harrow (DHS/DHM)  [Loop Fix applies]
  1) dh_width_in (48/64/80/96 or as offered)
  2) dh_duty ("Standard (DHS)" | "Heavy Duty (DHM)")
  3) dh_blade ("Combo (C)" | "Notched (N)")
  4) dh_spacing_id or dh_spacing

Post Hole Digger (PD)
  1) pd_model (PD25.21 | PD35.31 | PD95.51) if not selected yet
  2) auger_id or auger_choice — REQUIRED to finish

Tillers (DB/RT/RTR)
  1) tiller_series ("DB" | "RT")
  2) tiller_width_in
  3) For RT only: tiller_rotation ("Forward" | "Reverse")
  4) tiller_choice_id or tiller_choice

Bale Spear
  1) bspear_choice_id or bspear_choice

Pallet Fork
  1) pf_choice_id or pf_choice

Quick Hitch
  1) qh_choice_id or qh_choice

Stump Grinder (TSG)
  1) hydraulics_id or hydraulics_choice

General:
- If /quote returns required_questions, ALWAYS use those exact names and choices next; do not use the family-tree step if the API specified the next parameter.
- If a driveline question appears with no choices, present: 540 RPM, 1000 RPM.
- When the user explicitly requests an accessory/option, call /quote with accessory_id or accessory_desc and add it as a separate line, unless no price is returned (then show escalation message).
"""

# ---------------- Synonyms / Normalization hints ----------------
SYNONYMS = r"""
Normalize families from phrasing and typos:
- "box blade", "boxblade" → box_scraper
- "brush bull", "brushbull" → brushbull
- "dual spindle" → dual_spindle
- "batwing", "bat wimg", "batwng" → batwing
- "turf batwing" → turf_batwing
- "rear finish" → rear_finish
- "grading scraper" → grading_scraper
- "landscape rake" → landscape_rake
- "rear blade" → rear_blade
- "disc harrow" → disc_harrow
- "post hole digger" → post_hole_digger
- "tiller" → tiller
- "bale spear" → bale_spear
- "pallet fork" → pallet_fork
- "quick hitch" → quick_hitch
- "stump grinder" → stump_grinder
"""

# ---------------- Tool usage summary ----------------
OPENAPI_HINT = r"""
Woods Quote Tool (HTTP GET):
- /health
- /dealer-discount?dealer_number=178647
- /quote?{ dealer_number, model, family, width_ft, bw_driveline, bb_shielding, tire_id, bw_tire_qty,
           ds_mount, ds_shielding, ds_driveline, shielding_rows, deck_rings, tbw_duty, finish_choice,
           *_choice_id, *_choice, *_width_in, *_duty, *_grade, *_rotation, tire_qty, auger_id,
           accessory_id (repeatable), accessory_ids, accessory, accessory_desc, part_id, part_no, q }

Rules:
- Before pricing, call /dealer-discount to get the discount for the dealer.
- To progress config, call /quote with known params. If {mode:"questions"} is returned, ask exactly ONE question (lettered list) and wait.
- Never fabricate prices; all numbers come from /quote. If a part can’t be priced, show the escalation message.
- For Disc Harrow spacing loop: if API keeps repeating the same spacing prompt after you posted it back, stop and show the loop fix message.
"""

PLANNER_SYSTEM = (
    KNOWLEDGE
    + "\n\n--- FAMILY TREES ---\n"
    + FAMILY_TREE_GUIDE
    + "\n\n--- SYNONYMS ---\n"
    + SYNONYMS
    + "\n\n--- TOOL SPEC SUMMARY ---\n"
    + OPENAPI_HINT
    + r"""

You are an AI orchestrator with tool access. On every turn:
- Use plain English to talk to the user.
- When you need data, call the HTTP tool to /dealer-discount or /quote.
- Ask ONE configuration question at a time with A./B./C. lettered options.
- When final data is available, format the customer-ready quote per Knowledge.

Your output to the user must be plain text; tool calls are separate.
"""
)

# ---------------- Choice helpers ----------------
CHOICE_LINE_RE = re.compile(r"^[A-Z]\.\s+(.+)$", re.M)
DEALER_RE_1 = re.compile(r"dealer\s*#\s*(\d{4,7})", re.I)
DEALER_RE_2 = re.compile(r"\b(\d{5,7})\b")

def parse_lettered_choices(text: str) -> List[str]:
    if not text:
        return []
    return [m.group(1).strip() for m in CHOICE_LINE_RE.finditer(text)]

def letter_or_number_to_index(s: str, n: int):
    if not s or n <= 0:
        return None
    t = s.strip().lower()
    if len(t) == 1 and "a" <= t <= "z":
        i = ord(t) - ord("a")
        return i if 0 <= i < n else None
    if t.isdigit():
        i = int(t) - 1
        return i if 0 <= i < n else None
    return None

def find_dealer_number(text: str):
    if not text:
        return None
    m = DEALER_RE_1.search(text)
    if m:
        return m.group(1)
    m = DEALER_RE_2.search(text)
    return m.group(1) if m else None

def trim_messages(history: List[Dict[str, Any]], keep_last: int = 60) -> List[Dict[str, Any]]:
    """Keep system + last N messages to control token growth."""
    if not history:
        return history
    # keep first system message if present
    sys = history[0] if history and history[0].get("role") == "system" else None
    tail = history[1:] if sys else history[:]
    if len(tail) <= keep_last:
        return history
    trimmed = ( [sys] if sys else [] ) + tail[-keep_last:]
    return trimmed

# ---------------- OpenAI tool (function) ----------------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "http_get",
            "description": "Call Woods Quote API via HTTP GET. Allowed paths: /health, /dealer-discount, /quote",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "enum": ["/health", "/dealer-discount", "/quote"],
                        "description": "Endpoint path to call",
                    },
                    "params": {
                        "type": "object",
                        "description": "Query parameters to send (e.g., dealer_number, model, family, etc.)",
                        "additionalProperties": True,
                    },
                },
                "required": ["path", "params"],
                "additionalProperties": False,
            },
        },
    }
]

# ---------------- HTTP executor with retry ----------------
def woods_http_get(path: str, params: Dict[str, Any]) -> Tuple[Dict[str, Any], int, str]:
    url = f"{QUOTE_API_BASE}{path}"
    tries, last_exc = 0, None
    while tries < 2:
        try:
            r = requests.get(url, params=params, timeout=30)
            status = r.status_code
            ctype = r.headers.get("content-type", "")
            if "application/json" in (ctype or "").lower():
                return r.json(), status, r.url
            return {"raw_text": r.text}, status, r.url
        except Exception as e:
            last_exc = e
            tries += 1
            if tries >= 2: break
            time.sleep(0.35)
    logging.error("woods_http_get failed %s params=%s error=%s", url, params, last_exc)
    return {"raw_text": str(last_exc)}, 599, url

# ---------------- AI loop (returns text + full history) ----------------
def run_ai(messages: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
    if not client:
        return "Planner unavailable (missing OPENAI_API_KEY).", messages

    completion = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0.2,
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
    )

    loop_guard = 0
    history = messages[:]  # we will append to this and return it
    while True:
        loop_guard += 1
        if loop_guard > 8:
            # Append final assistant notice so history stays consistent
            history.append({"role": "assistant", "content": "Tool loop exceeded. Please try again."})
            return "Tool loop exceeded. Please try again.", history

        choice = completion.choices[0].message
        tool_calls = getattr(choice, "tool_calls", None)

        if not tool_calls:
            # append the assistant's final content to history before returning
            history.append({"role": "assistant", "content": choice.content or ""})
            return (choice.content or ""), history

        # Record one assistant message containing all tool_calls
        assistant_msg = {
            "role": "assistant",
            "content": choice.content or "",
            "tool_calls": []
        }
        for tc in tool_calls:
            assistant_msg["tool_calls"].append({
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments or "{}",
                },
            })
        history.append(assistant_msg)

        # Execute tools and append results
        for tc in tool_calls:
            fn = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except Exception:
                args = {}
            result = {"ok": False, "status": 0, "url": "", "body": {}}
            if fn == "http_get":
                path = args.get("path") or ""
                params = args.get("params") or {}
                body, status, used = woods_http_get(path, params)
                result = {"ok": (status == 200), "status": status, "url": used, "body": body}
            else:
                result = {"ok": False, "status": 0, "url": "", "body": {"error": "Unknown tool"}}

            history.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "name": fn,
                "content": json.dumps(result),
            })

        # Ask the model to continue after tool results
        completion = client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0.2,
            messages=history,
            tools=TOOLS,
            tool_choice="auto",
        )

# ---------------- Routes ----------------
@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json(force=True, silent=True) or {}
        user_message = (data.get("message") or "").strip()
        if not user_message:
            return jsonify({"error": "Missing message"}), 400

        # Session
        session_id = request.headers.get("X-Session-Id") or data.get("session_id") or f"anon-{int(time.time()*1000)}"
        sess = SESSIONS.setdefault(session_id, {})
        convo: List[Dict[str, Any]] = sess.get("messages") or []
        if not convo:
            convo = [{"role": "system", "content": PLANNER_SYSTEM}]

        # Dealer capture from user text (keeps context robust even if prior turns roll off)
        dn_in_text = find_dealer_number(user_message)
        if dn_in_text:
            sess["dealer_number"] = dn_in_text

        # Translate A/B/C/1/2/3 into explicit selection if we have a stored list
        last_choices: List[str] = sess.get("last_choices") or []
        idx = letter_or_number_to_index(user_message, len(last_choices)) if last_choices else None
        if idx is not None:
            choice_label = last_choices[idx]
            user_message = f"Selection: {choice_label}"

        # Inject a tiny session snapshot so the model always knows dealer + last choices
        snapshot = {
            "dealer_number": sess.get("dealer_number"),
            "dealer_name": sess.get("dealer_name"),
            "last_choices": last_choices or None,
        }
        convo.append({"role": "system", "content": "SESSION_STATE: " + json.dumps(snapshot)})

        # Append user turn
        convo.append({"role": "user", "content": user_message})

        # Run AI; get BOTH text and full updated history (with tool calls/results)
        reply_text, updated_history = run_ai(convo)

        # Store updated history (trimmed)
        updated_history = trim_messages(updated_history, keep_last=60)
        sess["messages"] = updated_history

        # Update last_choices based on the *final* assistant text (for the next turn)
        if reply_text and "**Woods Equipment Quote**" in reply_text:
            # Final quote shown — clear lettered list context
            sess.pop("last_choices", None)
        else:
            # Capture the latest lettered options the assistant showed
            choices_list = parse_lettered_choices(reply_text or "")
            if choices_list:
                # keep max 26
                sess["last_choices"] = choices_list[:26]

        return jsonify({"reply": reply_text})

    except Exception as e:
        logging.exception("Unhandled error in /chat")
        # Always JSON (prevents "<!doctype ..." errors in client)
        return jsonify({
            "reply": "There was a system error while handling your request. Please try again.",
            "error": str(e),
            "trace": traceback.format_exc(limit=1),
        }), 200

@app.route("/health", methods=["GET"])
def health():
    try:
        body, status, _ = woods_http_get("/health", {})
        return jsonify({
            "ok": status == 200,
            "planner": bool(client),
            "quote_api_base": QUOTE_API_BASE,
            "api_health": body if isinstance(body, dict) else {},
        }), 200
    except Exception:
        return jsonify({
            "ok": False,
            "planner": bool(client),
            "quote_api_base": QUOTE_API_BASE,
            "api_health": {},
        }), 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    app.run(host="0.0.0.0", port=port, debug=False)
