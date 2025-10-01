# app.py — AI bot with GPT-style Actions (function tools) that call YOUR app.py API
# - Tools:
#     woods_dealer_discount(dealer_number)
#     woods_quote(**query params mirrored from your OpenAPI)
#     woods_health()
# - The LLM plans freely (system prompt + family trees) and calls tools as needed.
# - Auto-triggers:
#     * Detects dealer number in user text -> calls woods_dealer_discount immediately
#     * Detects quote intent (and known dealer) -> calls woods_quote(q=<message>, dealer_number=<known>)
#
# Endpoints:
#   POST /chat   {"session_id":"<id>","message":"..."} -> {"reply":"..."}
#   GET  /health -> {"ok":bool,"planner":bool,"quote_api_base":str,"api_health":{...}}
#
# Env (optional):
#   OPENAI_API_KEY   (if missing, planner disabled and you’ll get a fallback text reply)
#   OPENAI_MODEL     (default: gpt-4o-mini)
#   ALLOWED_ORIGINS  (CSV for CORS)
#   SYSTEM_PROMPT_APPEND (extra rules appended to the system prompt)
#
# Run:
#   pip install flask flask-cors requests openai
#   python app.py

from __future__ import annotations
import os, re, json, time, logging, traceback
from typing import Any, Dict, List, Tuple

import requests
from flask import Flask, request, jsonify
try:
    from flask_cors import CORS
except Exception:
    CORS = None

# ---------------- Config ----------------
QUOTE_API_BASE = "https://woods-quote-api.onrender.com".rstrip("/")  # your service
OPENAI_MODEL   = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
HTTP_TIMEOUT   = float(os.environ.get("HTTP_TIMEOUT", "60"))
RETRY_ATTEMPTS = 2

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("woods-actions")

# ---------------- OpenAI client (optional) ----------------
client = None
try:
    from openai import OpenAI
    if os.environ.get("OPENAI_API_KEY"):
        client = OpenAI()
except Exception:
    client = None

# ---------------- Flask + CORS ----------------
app = Flask(__name__)
if CORS:
    allow = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "").split(",") if o.strip()]
    if allow:
        CORS(app, resources={r"/chat": {"origins": allow}}, supports_credentials=False)
    else:
        CORS(app, supports_credentials=False)

# ---------------- Sessions ----------------
SESSIONS: Dict[str, Dict[str, Any]] = {}
SESSION_TTL_SECONDS = 60 * 30  # 30 min

# ---------------- System prompt (rules + family trees) ----------------
KNOWLEDGE = r"""
You are a quoting assistant for Woods Equipment dealership staff. Retrieve part numbers, list prices, dealer discounts,
and configuration requirements exclusively from the Woods Pricing API (the tools in this chat). Never fabricate data,
never infer pricing, and only ask configuration questions when required.

Core Rules
- A dealer number is required before quoting. Look it up via woods_dealer_discount. Don’t present pricing without it.
- Remember the dealer number within the session; forget model/config after a quote is complete.
- Always pull fresh pricing for every line (models & accessories). No caching across quotes.
- If a real part number returns no price, stop and show the escalation message.

API Error Handling
- Connector errors are retried once. If still failing, say:
  “There was a system error while retrieving data. Please try again shortly. If the issue persists, escalate to Benjamin Luci at 615-516-8802.”

Pricing Logic
1) Retrieve list price for each part
2) Apply dealer discount from lookup
3) If dealer discount ≠ 5%, apply an additional 12% cash discount (always)
4) Output must be plain text, customer-ready

Quote Output Format
- Begin with "**Woods Equipment Quote**"
- Include dealer name and dealer number
- Bold the final dealer net with ✅
- No "Subtotal" section
- Always include: List Price → Discount → Cash Discount → Final Net
- Add: “Cash discount included only if paid within terms.”
- If any part cannot be priced: “Unable to find pricing... contact Benjamin Luci at 615-516-8802.”

Accessory Handling
- If the user requests an accessory, attempt API lookup and add as a separate line if priced. If not priced: escalate.
- Never assume accessories are included by default.

Interaction Style
- Ask one configuration question at a time
- Use lettered vertical lists (A., B., C., …) for options
- Wait for the dealer’s single response before proceeding
"""

FAMILY_TREE = r"""
Use this only when /quote does not already ask. Ask exactly one config question at a time.

BrushFighter: width_ft (5/6/7) → bf_choice_id/bf_choice → drive if needed
BrushBull: bb_shielding → bb_tailwheel if asked
Dual Spindle (DS/MDS): ds_mount → ds_shielding → ds_driveline (540/1000) → tire_id/tire_qty if needed
Batwing: width_ft (12/15/20) → bw_duty → bw_driveline (540/1000) → shielding_rows → deck_rings → bw_tire_qty
Turf Batwing (TBW): width_ft (12/15/17) → tbw_duty (Residential/Commercial) → front_rollers (qty 3) → chains
Finish Mowers: finish_choice (tk/tkp/rd990x) → rollers/chains as supported
Box Scraper: bs_width_in (48/60/72/84) → bs_duty → bs_choice_id
Grading Scraper: gs_width_in → gs_choice_id
Landscape Rake: lrs_width_in → lrs_grade (if both) → lrs_choice_id
Rear Blade: rb_width_in → rb_duty → rb_choice_id
Post Hole Digger: pd_model → auger_id (required)
Disc Harrow: dh_width_in → dh_duty (DHS/DHM) → dh_blade (N/C) → dh_spacing_id
Tillers (DB/RT): tiller_series → tiller_width_in → (RT) tiller_rotation → tiller_choice_id
Bale Spear / Pallet Fork / Quick Hitch / Stump Grinder: choose by part ID; stump grinder uses hydraulics_id.

If driveline choices are missing, present: 540 RPM and 1000 RPM.
"""

SYSTEM_PROMPT = KNOWLEDGE + "\n\n--- FAMILY TREE GUIDE ---\n" + FAMILY_TREE + \
                ("\n\n" + os.environ["SYSTEM_PROMPT_APPEND"] if os.environ.get("SYSTEM_PROMPT_APPEND") else "")

# ---------------- Helper: HTTP GET with one retry ----------------
def http_get(path: str, params: Dict[str, Any]) -> Tuple[Dict[str, Any], int, str]:
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
    log.error("http_get failed %s params=%s error=%s", url, params, last_exc)
    return {"error": str(last_exc)}, 599, url

# ---------------- GPT-style Actions (function tools) ----------------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "woods_dealer_discount",
            "description": "Look up a dealer’s discount and (if available) name by dealer_number.",
            "parameters": {
                "type": "object",
                "properties": {
                    "dealer_number": {"type": "string", "description": "Dealer number (e.g., 179269)."}
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
            "description": "Unified quoting action for Woods families. Send any of the documented /quote query params.",
            "parameters": {
                "type": "object",
                "properties": {
                    # generic routing
                    "q": {"type": "string", "description": "Free-text like 'quote a 7 ft cutter'."},
                    "family": {"type": "string"},
                    "family_choice": {"type": "string"},
                    "model": {"type": "string"},
                    "width": {"type": "string"},
                    "width_ft": {"type": "string"},
                    # pricing context
                    "dealer_number": {"type": "string"},
                    "dealer_discount": {"type": "number"},
                    "freight": {"type": "string"},
                    # accessories (generic)
                    "list_accessories": {"type": "boolean"},
                    "accessory_id": {"type": "array", "items": {"type": "string"}},
                    "accessory_ids": {"type": "string"},
                    "accessory": {"type": "array", "items": {"type": "string"}},
                    "accessory_desc": {"type": "array", "items": {"type": "string"}},
                    # examples of follow-ups (the API supports many; we pass-through)
                    "bf_choice_id": {"type": "string"},
                    "bb_duty": {"type": "string"},
                    "bw_driveline": {"type": "string"},
                    "shielding_rows": {"type": "string"},
                    "deck_rings": {"type": "string"},
                    "bw_tire_qty": {"type": "integer"},
                    "tbw_duty": {"type": "string"},
                    "front_rollers": {"type": "string"},
                    "chains": {"type": "string"},
                    "finish_choice": {"type": "string"},
                    "bs_width_in": {"type": "string"},
                    "bs_duty": {"type": "string"},
                    "bs_choice_id": {"type": "string"},
                    "gs_width_in": {"type": "string"},
                    "gs_choice_id": {"type": "string"},
                    "lrs_width_in": {"type": "string"},
                    "lrs_grade": {"type": "string"},
                    "lrs_choice_id": {"type": "string"},
                    "rb_width_in": {"type": "string"},
                    "rb_duty": {"type": "string"},
                    "rb_choice_id": {"type": "string"},
                    "pd_model": {"type": "string"},
                    "auger_id": {"type": "string"},
                    "dh_width_in": {"type": "string"},
                    "dh_duty": {"type": "string"},
                    "dh_blade": {"type": "string"},
                    "dh_spacing_id": {"type": "string"},
                    "tiller_series": {"type": "string"},
                    "tiller_width_in": {"type": "string"},
                    "tiller_rotation": {"type": "string"},
                    "tiller_choice_id": {"type": "string"},
                    "bspear_choice_id": {"type": "string"},
                    "pf_choice_id": {"type": "string"},
                    "qh_choice_id": {"type": "string"},
                    "hydraulics_id": {"type": "string"},
                    "part_id": {"type": "string"},
                    "part_no": {"type": "string"}
                },
                "additionalProperties": True
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "woods_health",
            "description": "Quick API health/status.",
            "parameters": {"type": "object", "properties": {}}
        },
    },
]

# ---------------- Tool implementations ----------------
def tool_woods_dealer_discount(args: Dict[str, Any]) -> Dict[str, Any]:
    dealer_number = str(args.get("dealer_number") or "").strip()
    body, status, used = http_get("/dealer-discount", {"dealer_number": dealer_number})
    return {"ok": status == 200, "status": status, "url": used, "body": body}

def tool_woods_quote(args: Dict[str, Any]) -> Dict[str, Any]:
    # Pass-through of whatever fields the model supplies
    params = {k: v for (k, v) in (args or {}).items() if v is not None}
    body, status, used = http_get("/quote", params)
    return {"ok": status == 200, "status": status, "url": used, "body": body}

def tool_woods_health(args: Dict[str, Any]) -> Dict[str, Any]:
    body, status, used = http_get("/health", {})
    return {"ok": status == 200, "status": status, "url": used, "body": body}

# ---------------- Intent detection (auto triggers) ----------------
DEALER_RE = re.compile(r"\b(\d{4,9})\b")
FAMILY_HINTS = [
    "brushfighter","brush bull","brushbull","dual spindle","batwing","turf batwing","rear discharge",
    "finish mower","box scraper","grading scraper","landscape rake","rear blade","disc harrow",
    "post hole digger","tiller","bale spear","pallet fork","quick hitch","stump grinder",
    "bf","bb","ds","mds","bw","tbw","rd990x","lrs","rb","dhs","dhm","pd","rt","rtr","db",
]
QUOTE_HINTS = ["quote","price","pricing","cost","how much","list price"]

def has_quote_intent(text: str) -> bool:
    t = text.lower()
    return any(k in t for k in QUOTE_HINTS) or any(k in t for k in FAMILY_HINTS)

def inject_tool_result(history: List[Dict[str, Any]], name: str, args: Dict[str, Any], result: Dict[str, Any]) -> None:
    call_id = f"{name}-{int(time.time()*1000)}"
    history.append({
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": call_id,
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)},
        }]
    })
    history.append({
        "role": "tool",
        "tool_call_id": call_id,
        "name": name,
        "content": json.dumps(result),
    })

# ---------------- Planner loop (LLM + tools) ----------------
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

        # execute calls
        for tc in tool_calls:
            fn = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except Exception:
                args = {}
            if fn == "woods_dealer_discount":
                result = tool_woods_dealer_discount(args)
            elif fn == "woods_quote":
                result = tool_woods_quote(args)
            elif fn == "woods_health":
                result = tool_woods_health(args)
            else:
                result = {"ok": False, "status": 0, "url": "", "body": {"error": "Unknown tool"}}

            history.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "name": fn,
                "content": json.dumps(result),
            })

        # next turn
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

        # Build convo with prior history
        convo: List[Dict[str, Any]] = list(sess["messages"])

        # -------- AUTO-TRIGGERS (pre-inject tool results) --------
        # 1) dealer number detection -> woods_dealer_discount
        m = DEALER_RE.search(user_message)
        if m:
            dn = m.group(1)
            result = tool_woods_dealer_discount({"dealer_number": dn})
            inject_tool_result(convo, "woods_dealer_discount", {"dealer_number": dn}, result)

        # 2) quote intent -> woods_quote (if dealer known in history)
        dealer_number = None
        # Look for last dealer_number in previous tool results (dealer lookup or quote calls)
        for m in reversed(convo):
            if m.get("role") == "tool":
                try:
                    payload = json.loads(m.get("content") or "{}")
                    body = payload.get("body") or {}
                    # try to infer dealer_number from body or URL
                    dn = body.get("dealer_number") or body.get("number") or None
                    if not dn:
                        url = payload.get("url") or ""
                        mm = re.search(r"dealer_number=(\d+)", url)
                        dn = mm.group(1) if mm else None
                    if dn:
                        dealer_number = str(dn)
                        break
                except Exception:
                    pass

        if has_quote_intent(user_message) and dealer_number:
            args = {"q": user_message, "dealer_number": dealer_number}
            result = tool_woods_quote(args)
            inject_tool_result(convo, "woods_quote", args, result)

        # Append user turn and run the planner
        convo.append({"role": "user", "content": user_message})
        reply_text, updated_history = run_ai(convo)

        # Keep history compact (preserve system)
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
        result = tool_woods_health({})
        return jsonify({
            "ok": result.get("ok", False),
            "planner": bool(client),
            "quote_api_base": QUOTE_API_BASE,
            "api_health": result.get("body") or {},
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
