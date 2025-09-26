# app.py — Minimal AI-first Woods chatbot
# - Only KNOWLEDGE + FAMILY_TREE in the system prompt
# - One tool: http_get (GET /dealer-discount, /quote, /health)
# - Persist full tool history in session so the model remembers config steps
# - No extra UX glue, no choice mapping, no snapshots

import os, json, time, logging, traceback
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

# ---------------- Knowledge (as provided) ----------------
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

# ---------------- Family Tree guidance (as provided) ----------------
FAMILY_TREE_GUIDE = r"""
Use this family-tree guidance ONLY when /quote does not return required_questions. Ask exactly one question at a time and wait for the answer. Use the exact parameter names.

BrushFighter
  1) width_ft (5/6/7)
  2) bf_choice_id or bf_choice
  3) Optional drive hint: drive ("Slip Clutch" | "Shear Pin")

BrushBull
  1) bb_shielding ("Belt" | "Chain" | "Single Row" | "Double Row")
  2) bb_tailwheel ("Single" | "Dual") if the API asks

Dual Spindle (DS/MDS)
  1) ds_mount ("mounted" | "pull")
  2) ds_shielding
  3) ds_driveline ("540" | "1000")
  4) tire_id (part ID) if needed
  5) tire_qty (integer)

Batwing (BW)
  1) bw_duty (series-specific) if needed
  2) bw_driveline ("540" | "1000")
  3) shielding_rows ("Single Row" | "Double Row") if supported
  4) deck_rings ("Yes" | "No") if supported
  5) tire_id (from choices_with_ids)
  6) bw_tire_qty (from choices/width rules)

Turf Batwing (TBW)
  1) tbw_duty ("Residential (.20)" | "Commercial (.40)")
  2) width_ft (12/15/17; residential=12)
  3) front_rollers ("Yes" | "No")
  4) chains ("Yes" | "No")

Rear Discharge Finish Mower
  1) finish_choice ("tk" | "tkp" | "rd990x")
  2) TK/TKP: front_rollers / chains if asked

Box Scraper (BS)
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

Disc Harrow (DHS/DHM)
  1) dh_width_in (48/64/80/96)
  2) dh_duty ("Standard (DHS)" | "Heavy Duty (DHM)")
  3) dh_blade ("Combo (C)" | "Notched (N)")
  4) dh_spacing_id or dh_spacing  [Stop if spacing prompt loops per Knowledge]

Post Hole Digger (PD)
  1) pd_model (PD25.21 | PD35.31 | PD95.51)
  2) auger_id or auger_choice (REQUIRED)

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
- If /quote returns required_questions, ALWAYS use those names/choices next; do not use the family tree step.
- If a driveline question has no choices, present: 540 RPM, 1000 RPM.
- If the user requests an accessory or option, call /quote with accessory_id or accessory_desc; add as separate line if priced.
"""

SYSTEM_PROMPT = KNOWLEDGE + "\n\n--- FAMILY TREE GUIDE ---\n" + FAMILY_TREE_GUIDE

# ---------------- One tool definition ----------------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "http_get",
            "description": "HTTP GET to Woods Quote API. Use for /dealer-discount, /quote, /health.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "enum": ["/health", "/dealer-discount", "/quote"]},
                    "params": {"type": "object", "additionalProperties": True}
                },
                "required": ["path", "params"],
                "additionalProperties": False
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
            if tries >= 2:
                break
            time.sleep(0.35)
    logging.error("woods_http_get failed %s params=%s error=%s", url, params, last_exc)
    return {"raw_text": str(last_exc)}, 599, url

# ---------------- AI loop (persists tool calls/results in history) ----------------
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
    history = messages[:]
    while True:
        loop_guard += 1
        if loop_guard > 8:
            history.append({"role": "assistant", "content": "Tool loop exceeded. Please try again."})
            return "Tool loop exceeded. Please try again.", history

        choice = completion.choices[0].message
        tool_calls = getattr(choice, "tool_calls", None)

        if not tool_calls:
            history.append({"role": "assistant", "content": choice.content or ""})
            return (choice.content or ""), history

        # Log the assistant message that invoked tools
        assistant_msg = {
            "role": "assistant",
            "content": choice.content or "",
            "tool_calls": []
        }
        for tc in tool_calls:
            assistant_msg["tool_calls"].append({
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments or "{}"},
            })
        history.append(assistant_msg)

        # Execute each tool call and append its result
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

        # Ask the model to continue after seeing tool results
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

        session_id = request.headers.get("X-Session-Id") or data.get("session_id") or f"anon-{int(time.time()*1000)}"
        sess = SESSIONS.setdefault(session_id, {})

        # Seed system message once
        convo: List[Dict[str, Any]] = sess.get("messages") or [{"role": "system", "content": SYSTEM_PROMPT}]

        # Append user turn and run AI
        convo.append({"role": "user", "content": user_message})
        reply_text, updated_history = run_ai(convo)

        # Persist the full updated history (so tool results are remembered)
        # Optionally trim to avoid unlimited growth
        MAX_KEEP = 80
        if len(updated_history) > MAX_KEEP:
            # keep first system + last N-1
            head = updated_history[0:1] if updated_history and updated_history[0].get("role") == "system" else []
            updated_history = head + updated_history[-(MAX_KEEP - len(head)):]
        sess["messages"] = updated_history

        return jsonify({"reply": reply_text})
    except Exception as e:
        logging.exception("Unhandled error in /chat")
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
