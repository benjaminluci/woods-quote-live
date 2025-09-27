# app.py — AI-first Woods quoting backend (less strict, GPT-driven)
# - Mirrors your previous file’s shape (Flask, sessions, http_get tool, OpenAI planner)
# - Minimal guardrails; the model has “free will” guided by your knowledge + family trees
# - One tool: http_get -> GET /dealer-discount, /quote, /health (with retry)
# - Sessions persist full message + tool history; model remembers config Q&A per session
#
# Endpoints:
#   POST /chat   {"session_id":"<id>","message":"<text>"} -> {"reply": "..."}
#   GET  /health -> {"ok": bool, "planner": bool, "quote_api_base": str, "api_health": {...}}
#
# Env:
#   QUOTE_API_BASE (default: https://woods-quote-tool.onrender.com)
#   OPENAI_API_KEY (optional; if omitted, planner is disabled and you’ll get a fallback message)
#   OPENAI_MODEL   (default: gpt-4o-mini)
#   ALLOWED_ORIGINS (CSV, optional CORS whitelist for /chat)
#   SYSTEM_PROMPT_APPEND (optional text appended to the system prompt)
#
# Run:
#   pip install flask flask-cors requests openai
#   python app.py

from __future__ import annotations
import os, json, time, logging, traceback
from typing import Any, Dict, List, Tuple

import requests
from flask import Flask, request, jsonify
try:
    from flask_cors import CORS
except Exception:
    CORS = None

# ---------------- Config ----------------
QUOTE_API_BASE = os.environ.get("QUOTE_API_BASE", "https://woods-quote-tool.onrender.com").rstrip("/")
OPENAI_MODEL   = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
HTTP_TIMEOUT   = float(os.environ.get("HTTP_TIMEOUT", "25"))
RETRY_ATTEMPTS = 2

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("woods-ai")

# ---------------- OpenAI client (optional) ----------------
client = None
try:
    # Modern SDK
    from openai import OpenAI
    if os.environ.get("OPENAI_API_KEY"):
        client = OpenAI()
except Exception:
    client = None

# ---------------- Flask app + CORS ----------------
app = Flask(__name__)
if CORS:
    allow = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "").split(",") if o.strip()]
    if allow:
        CORS(app, resources={r"/chat": {"origins": allow}}, supports_credentials=False)
    else:
        CORS(app, supports_credentials=False)

# ---------------- Sessions (simple in-memory) ----------------
SESSIONS: Dict[str, Dict[str, Any]] = {}
SESSION_TTL_SECONDS = 60 * 60 * 12  # 12h

# ---------------- Knowledge & Family Trees (model-guided) ----------------
KNOWLEDGE = r"""
You are a quoting assistant for Woods Equipment dealership staff. Your primary job is to retrieve part numbers,
list prices, dealer discounts, and configuration requirements exclusively from the Woods Pricing API. You never
fabricate data, never infer pricing, and only ask configuration questions when required.

Core Rules
- A dealer number is required before quoting. Use the Pricing API to look up the dealer’s discount. Do not begin quotes or give pricing without it.
- Dealer numbers may be remembered within a session and across multiple quotes for the same user unless the dealer provides a new number.
- All model, accessory, and pricing data must be pulled directly from the API. Never invent, infer, reuse, or cache data.
- Every quote must pull fresh pricing from the API for all items — including list prices and accessories.
- If a valid part number returns no price, quoting must stop and inform the dealer to escalate the issue.

API Error Handling
- Retry any connector error once automatically before showing an error.
- If retry succeeds, proceed normally.
- If retry fails, show: “There was a system error while retrieving data. Please try again shortly. If the issue persists, escalate to Benjamin Luci at 615-516-8802.”

Pricing Logic
1) Retrieve list price for each part number from API
2) Apply dealer discount from lookup
3) Unless the dealer discount is exactly 5%, apply an additional 12% cash discount
4) Format quote as plain text, customer-ready
⚠️ Cash discount must always be applied unless dealer discount is exactly 5%. Never skip this.

Quote Output Format
- Begin with "**Woods Equipment Quote**"
- Include dealer name and dealer number
- Bold the final dealer net with ✅
- No "Subtotal" section
- Always include: List Price → Discount → Cash Discount → Final Net
- Add: “Cash discount included only if paid within terms.”
- If any part cannot be priced: “Unable to find pricing... contact Benjamin Luci at 615-516-8802.”

Session Handling
- Remember dealer number across quotes in the same session
- Remember selected model/config only within a single quote
- Always re-pull prices between quotes
- Never say “API says…” — present info as system output

Access Control
- Never disclose pricing from one dealer to another
- If dealer needs help finding dealer number, direct them to the Woods dealer portal

Accessory Handling
- If a dealer requests a specific accessory (e.g., tires, chains, dual hub):
  - Attempt API lookup
  - If priced, add as separate line item
  - If not priced, stop and show the escalation message
- Never treat a dealer-requested accessory as included by default

Interaction Style
- Ask one config question at a time
- Never combine multiple questions into a single message
- Format multiple options as lettered vertical lists (A., B., C., …)
- Wait for user response before proceeding

Box Scraper Correction
- Valid widths: 48 in (4 ft), 60 in (5 ft), 72 in (6 ft), 84 in (7 ft)

Disc Harrow Fix
- If the system repeatedly requires disc spacing for the same selection:
  - Detect the loop
  - Stop quoting
  - Say: “The system is stuck on a required disc spacing selection. Please escalate to Benjamin Luci at 615-516-8802.”
  - Do not retry endlessly

Correction Enforcement
- Do not stop quotes after dealer discount
- If dealer discount ≠ 5%, 12% cash discount must be shown
- If cash discount is missing from final output, quote is invalid and must be corrected
"""

FAMILY_TREE = r"""
Use this guide only when /quote does not already ask. Ask exactly one question at a time.

BrushFighter
  width_ft (5/6/7) → bf_choice_id or bf_choice → optional drive ("Slip Clutch" | "Shear Pin")

BrushBull
  bb_shielding ("Belt" | "Chain" | "Single Row" | "Double Row") → bb_tailwheel ("Single" | "Dual") if asked

Dual Spindle (DS/MDS)
  ds_mount ("mounted" | "pull") → ds_shielding → ds_driveline ("540" | "1000") → tire_id/tire_qty if needed

Batwing
  width_ft (12/15/20) → bw_duty → bw_driveline ("540" | "1000") → shielding_rows ("Single Row" | "Double Row")
  deck_rings ("Yes" | "No") if model supports R → bw_tire_qty valid options by width

Turf Batwing (TBW)
  width_ft (12/15/17) → tbw_duty ("Residential (.20)" | "Commercial (.40)") → front_rollers ("Yes" | "No", qty 3)
  chains ("Yes" | "No")

Finish Mowers
  finish_choice (tk | tkp | rd990x) → front_rollers/chains as supported

Box Scraper
  bs_width_in (48/60/72/84) → bs_duty → bs_choice_id/bs_choice

Grading Scraper
  gs_width_in → gs_choice_id/gs_choice

Landscape Rake
  lrs_width_in → lrs_grade if width offers both → lrs_choice_id/lrs_choice

Rear Blade
  rb_width_in → rb_duty → rb_choice_id/rb_choice

Post Hole Digger
  pd_model (PD25.21/PD35.31/PD95.51) → auger_id/auger_choice (required)

Disc Harrow (DHS/DHM)
  dh_width_in → dh_duty ("Standard (DHS)" | "Heavy Duty (DHM)") → dh_blade ("Notched (N)" | "Combo (C)") → dh_spacing_id

Tillers (DB/RT)
  tiller_series ("DB" | "RT") → tiller_width_in → (RT only) tiller_rotation ("Forward" | "Reverse") → tiller_choice_id

Bale Spear / Pallet Fork / Quick Hitch / Stump Grinder
  choose by part ID via choices; use hydraulics_id for stump grinder.

If a driveline question has no choices, present: 540 RPM, 1000 RPM.
"""

SYSTEM_PROMPT = KNOWLEDGE + "\n\n--- FAMILY TREE GUIDE ---\n" + FAMILY_TREE + \
                ("\n\n" + os.environ["SYSTEM_PROMPT_APPEND"] if os.environ.get("SYSTEM_PROMPT_APPEND") else "")

# ---------------- Tools schema (for the LLM) ----------------
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

# ---------------- HTTP executor with retry (tool impl) ----------------
def woods_http_get(path: str, params: Dict[str, Any]) -> Tuple[Dict[str, Any], int, str]:
    url = f"{QUOTE_API_BASE}{path}"
    tries, last_exc = 0, None
    while tries < RETRY_ATTEMPTS:
        try:
            r = requests.get(url, params=params, timeout=HTTP_TIMEOUT)
            ctype = (r.headers.get("content-type") or "").lower()
            body = r.json() if "application/json" in ctype else {"raw_text": r.text}
            return body, r.status_code, r.url
        except Exception as e:
            last_exc = e
            tries += 1
            if tries >= RETRY_ATTEMPTS:
                break
            time.sleep(0.35)
    log.error("woods_http_get failed %s params=%s error=%s", url, params, last_exc)
    return {"error": str(last_exc)}, 599, url

# ---------------- Planner loop (LLM + tools) ----------------
def run_ai(messages: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
    """Return assistant text and updated message history."""
    if not client:
        return "Planner unavailable (missing OPENAI_API_KEY).", messages

    completion = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0.2,          # light creativity; 'free will' but consistent
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
    )

    history = messages[:]
    loop_guard = 0

    while True:
        loop_guard += 1
        if loop_guard > 8:
            history.append({"role": "assistant", "content": "Tool loop exceeded. Please try again."})
            return "Tool loop exceeded. Please try again.", history

        msg = completion.choices[0].message
        tool_calls = getattr(msg, "tool_calls", None)

        if not tool_calls:
            # assistant text reply
            history.append({"role": "assistant", "content": msg.content or ""})
            return msg.content or "", history

        # record assistant tool invocations
        assistant_msg = {
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [],
        }
        for tc in tool_calls:
            assistant_msg["tool_calls"].append({
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments or "{}"},
            })
        history.append(assistant_msg)

        # execute each tool call and append results
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

        # next step with new tool results
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

        # GC expired sessions
        now = time.time()
        to_del = [sid for sid, s in SESSIONS.items() if (now - s.get("updated_at", now)) > SESSION_TTL_SECONDS]
        for sid in to_del:
            SESSIONS.pop(sid, None)

        session_id = request.headers.get("X-Session-Id") or data.get("session_id") or f"anon-{int(now*1000)}"
        sess = SESSIONS.setdefault(session_id, {"messages": [{"role": "system", "content": SYSTEM_PROMPT}]})
        sess["updated_at"] = now

        # append user turn
        convo: List[Dict[str, Any]] = list(sess["messages"])
        convo.append({"role": "user", "content": user_message})

        # run the planner
        reply_text, updated_history = run_ai(convo)

        # keep history compact (preserve the system prompt; otherwise cap ~80 messages)
        MAX_KEEP = 80
        if len(updated_history) > MAX_KEEP:
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
