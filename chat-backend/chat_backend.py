# app.py — GPT Actions style with full knowledge + family trees
# - Tools: woods_dealer_discount, woods_quote, woods_health (proxy to YOUR API)
# - Free-will LLM (rules in system prompt) + dealer auto-attach + model normalization (BW12.40)
# - Sticky in-memory sessions keyed by X-Session-Id or session_id in body
# - Point your frontend to POST /chat on this server (e.g., https://woods-quote-backend.onrender.com/chat)

from __future__ import annotations
import os, re, json, time, logging, traceback
from typing import Any, Dict, List, Tuple

import requests
from flask import Flask, request, jsonify
try:
    from flask_cors import CORS
except Exception:
    CORS = None

# ---------- Config ----------
QUOTE_API_BASE = "https://woods-quote-api.onrender.com".rstrip("/")   # <— your app.py API base
OPENAI_MODEL   = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
HTTP_TIMEOUT   = float(os.environ.get("HTTP_TIMEOUT", "25"))
RETRY_ATTEMPTS = 2
MAX_TOOL_STEPS = 8

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("woods-actions")

# ---------- OpenAI client (optional) ----------
client = None
try:
    from openai import OpenAI
    if os.environ.get("OPENAI_API_KEY"):
        client = OpenAI()
except Exception:
    client = None

# ---------- Flask + CORS ----------
app = Flask(__name__)
if CORS:
    allow = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "").split(",") if o.strip()]
    if allow:
        CORS(app, resources={r"/chat": {"origins": allow}}, supports_credentials=False)
    else:
        CORS(app, supports_credentials=False)

# ---------- Sessions (in-memory) ----------
SESSIONS: Dict[str, Dict[str, Any]] = {}
SESSION_TTL_SECONDS = 60 * 60 * 12  # 12h

# ---------- Knowledge (full rules) ----------
KNOWLEDGE = r"""
You are a quoting assistant for Woods Equipment dealership staff. Your primary job is to retrieve part numbers, list prices,
dealer discounts, and configuration requirements exclusively from the Woods Pricing API via the provided tools.
You never fabricate data, never infer pricing, and only ask configuration questions when required.

---
Core Rules
- A dealer number is required before quoting. Use woods_dealer_discount to look up the dealer’s discount.
  Do not begin quotes or give pricing without it.
- Dealer numbers may be remembered within a session and across multiple quotes for the same user unless the dealer provides a new number.
- All model, accessory, and pricing data must be pulled directly from the API. Never invent, infer, reuse, or cache data.
- Every quote must pull fresh pricing from the API for all items — including list prices and accessories.
- If a valid part number returns no price, quoting must stop and inform the dealer to escalate the issue.

---
API Error Handling
- Retry any connector error once automatically (the backend does this for you) before showing an error.
- If retry still fails, show: “There was a system error while retrieving data. Please try again shortly.
  If the issue persists, escalate to Benjamin Luci at 615-516-8802.”

---
Pricing Logic
1) Retrieve list price for each part number from API
2) Apply dealer discount from lookup
3) Unless the dealer discount is exactly 5%, apply an additional 12% cash discount
4) Format quote as plain text, customer-ready
⚠️ Cash discount must always be applied unless dealer discount is exactly 5%. Never skip this.

---
Quote Output Format
- Begin with "**Woods Equipment Quote**"
- Include dealer name and dealer number
- Bold the final dealer net with ✅
- No "Subtotal" section
- Always include: List Price → Discount → Cash Discount → Final Net
- Add: “Cash discount included only if paid within terms.”
- If any part cannot be priced: “Unable to find pricing... contact Benjamin Luci at 615-516-8802.”

---
Session Handling
- Remember dealer number across quotes in the same session
- Remember selected model/config only within a single quote
- Always re-pull prices between quotes
- Never say “API says…” — present info as system output

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
  A. option
  B. option
  C. option
- Wait for user response before proceeding

---
Box Scraper Correction
- Valid widths: 48 in (4 ft), 60 in (5 ft), 72 in (6 ft), 84 in (7 ft)

---
Disc Harrow Fix
- If API returns the same required spacing prompt repeatedly:
  - Detect the loop
  - Stop quoting
  - Say: “The system is stuck on a required disc spacing selection. Please escalate to Benjamin Luci at 615-516-8802.”
  - Do not retry endlessly

---
Correction Enforcement
- Do not stop quotes after dealer discount
- If dealer discount ≠ 5%, 12% cash discount must be shown
- If cash discount is missing from final output, quote is invalid and must be corrected
"""

# ---------- Family Trees (detailed, mirrors your OpenAPI 1.2.0) ----------
FAMILY_TREES = r"""
Use these trees ONLY when woods_quote doesn’t already ask. Ask EXACTLY one follow-up at a time using lettered options.

Families & key params (pass as woods_quote params):

Generic routing
- q: free-text like "quote a 7 ft cutter"
- family / family_choice: one of [brushfighter, brushbull, dual_spindle, batwing, turf_batwing, rear_finish, box_scraper,
  grading_scraper, landscape_rake, rear_blade, disc_harrow, post_hole_digger, tiller, bale_spear, pallet_fork, quick_hitch, stump_grinder]
- model: exact code (e.g., BF5.20, BB84.50, DS8.30, BW15.72, TBW12.40, BS84.30, RB84.50, DHS64C, PD25.21, DB60, RT72/RTR72, TQH1/TQH2)
- width / width_ft: user asked size (e.g., "7 ft") or normalized width_ft "7", "12", "15", "17", "20"
- Always include dealer_number after dealer is known (the backend will auto-attach it if you forget)

Accessories (generic)
- list_accessories: boolean
- accessory_id: array of part IDs (repeat param)
- accessory_ids: comma-separated part IDs
- accessory: array of tokens pid or pid:qty
- accessory_desc: array of fuzzy description terms

BrushFighter (BF)
- Follow-ups: bf_choice_id or bf_choice; drive ("Slip Clutch" | "Shear Pin") if needed

BrushBull (BB)
- bb_duty: one of ["Standard Duty (.30)","Standard Plus (.40)","Heavy Duty (.50)","Extreme Duty (.60)"]
- bb_shielding: ["Belt","Chain","Single Row","Double Row"]
- bb_tailwheel: ["Single","Dual"]

Dual Spindle (DS/MDS)
- ds_mount: ["mounted","pull"]
- ds_shielding: label or option as API returns (e.g., Belt/Chain)
- ds_driveline (alias: driveline): ["540","1000"]
- tire_id: part ID for tire
- tire_qty: integer (valid by model/width)

Batwing (BW)
- width_ft: "12" | "15" | "20"
- bw_duty: ["Standard Duty (.40)","Standard Duty (.51)","Medium Duty (.61)","Heavy Duty (.71)","Standard Duty (.52)","Heavy Duty (.72)"]
- bw_driveline: ["540","1000"]
- shielding_rows: ["Single Row","Double Row"]
- deck_rings: ["Yes","No"]
- bw_tire_qty: integer (12→4 or 6; 15→4/6/8; 20→6/8)

Turf Batwing (TBW)
- tbw_duty: ["Residential (.20)","Commercial (.40)"]
- front_rollers: ["Yes","No"] (if Yes, qty 3)
- chains: ["Yes","No"]

Rear Discharge Finish Mower
- finish_choice: ["tk","tkp","rd990x"]
  - TK/TKP: front_rollers (qty 1) and chains supported
  - RD990X: front roller only if explicitly requested

Box Scraper (BS)
- bs_width_in: "48" | "60" | "72" | "84" (post back the ID from choices_with_ids.id)
- bs_duty: ["Light Duty (.20)","Medium Duty (.30)","Heavy Duty (.40)"]
- bs_choice_id or bs_choice

Grading Scraper (GS)
- gs_width_in (inches) → gs_choice_id or gs_choice

Landscape Rake (LRS)
- lrs_width_in (inches)
- lrs_grade: ["Standard","Premium (P)"] when both exist
- lrs_choice_id or lrs_choice

Rear Blade (RB)
- rb_width_in (inches)
- rb_duty: ["Standard","Standard (Premium P)","Heavy Duty","Extreme Duty"]
- rb_choice_id or rb_choice

Post Hole Digger (PD)
- pd_model: "PD25.21" | "PD35.31" | "PD95.51"
- auger_id (required) or auger_choice (fallback)

Disc Harrow (DHS/DHM)
- dh_width_in (inches)
- dh_duty: ["Standard (DHS)","Heavy Duty (DHM)"]
- dh_blade: ["Notched (N)","Combo (C)"]
- dh_spacing_id (required when prompted) or dh_spacing (exact label fallback)
LOOP FIX: If the same spacing prompt repeats, stop and output: “The system is stuck on a required disc spacing selection. Please escalate to Benjamin Luci at 615-516-8802.”

Tillers (DB/RT)
- tiller_series: ["DB","RT"]
- tiller_width_in (inches)
- (RT only) tiller_rotation: ["Forward","Reverse"]
- tiller_choice_id or tiller_choice

Bale Spear
- bspear_choice_id (select by part ID; labels carry capacity/loader/mount)

Pallet Fork
- pf_choice_id (select by part ID; labels carry capacity/class/width)

Quick Hitch
- qh_choice_id (or pick between TQH1 and TQH2 when asked)

Stump Grinder (TSG)
- hydraulics_id (e.g., standard or high-flow) or hydraulics_choice (fallback)
"""

# Helpful hint to the model about endpoints and behavior
OPENAPI_HINT = r"""
Endpoints to use via tools:
- woods_dealer_discount(dealer_number) -> GET {QUOTE_API_BASE}/dealer-discount
- woods_quote({...params...}) -> GET {QUOTE_API_BASE}/quote
- woods_health() -> GET {QUOTE_API_BASE}/health

Behavioral reminders:
- Always include dealer_number on quotes (the server will auto-attach the known dealer if you forget).
- If a driveline question has no choices, present: 540 RPM and 1000 RPM.
- Ask exactly one configuration question at a time with lettered options.
""".replace("{QUOTE_API_BASE}", QUOTE_API_BASE)

SYSTEM_PROMPT = KNOWLEDGE + "\n\n--- FAMILY TREES ---\n" + FAMILY_TREES + "\n\n--- OPENAPI HINT ---\n" + OPENAPI_HINT + \
                ("\n\n" + os.environ["SYSTEM_PROMPT_APPEND"] if os.environ.get("SYSTEM_PROMPT_APPEND") else "")

# ---------- HTTP helper (retry once) ----------
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

# ---------- Tools schema ----------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "woods_dealer_discount",
            "description": "Look up a dealer’s discount (and possibly name) by dealer_number.",
            "parameters": {
                "type": "object",
                "properties": {"dealer_number": {"type": "string", "description": "Dealer number, e.g., 179269"}},
                "required": ["dealer_number"],
                "additionalProperties": False
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "woods_quote",
            "description": "Unified quoting; pass any /quote query params (model, width_ft, etc.).",
            "parameters": {
                "type": "object",
                "properties": { "dealer_number": {"type": "string"} },
                "additionalProperties": True
            },
        },
    },
    {
        "type": "function",
        "function": {"name": "woods_health", "description": "API health check", "parameters": {"type": "object"}},
    },
]

# ---------- Tool implementations (dealer auto-attach) ----------
def tool_woods_dealer_discount(sess: Dict[str, Any], args: Dict[str, Any]) -> Dict[str, Any]:
    dealer_number = str(args.get("dealer_number") or "").strip()
    body, status, used = http_get("/dealer-discount", {"dealer_number": dealer_number})
    if status == 200:
        sess["dealer_number"] = dealer_number
        if isinstance(body, dict):
            sess["dealer_name"] = str(body.get("dealer_name") or body.get("name") or body.get("dealer") or "") or None
            # Keep raw discount if present (model will compute pricing presentation)
            sess["dealer_discount_raw"] = body.get("discount") or body.get("dealer_discount") or body.get("percent")
    return {"ok": status == 200, "status": status, "url": used, "body": body}

def tool_woods_quote(sess: Dict[str, Any], args: Dict[str, Any]) -> Dict[str, Any]:
    params = {k: v for (k, v) in (args or {}).items() if v is not None}
    if "dealer_number" not in params and sess.get("dealer_number"):
        params["dealer_number"] = sess["dealer_number"]
    body, status, used = http_get("/quote", params)
    return {"ok": status == 200, "status": status, "url": used, "body": body}

def tool_woods_health(_: Dict[str, Any], __: Dict[str, Any]) -> Dict[str, Any]:
    body, status, used = http_get("/health", {})
    return {"ok": status == 200, "status": status, "url": used, "body": body}

# ---------- Intent detection / normalization ----------
DEALER_RE = re.compile(r"\b(\d{4,9})\b")
MODEL_RE  = re.compile(r"\b([a-z]{2,3}\d{2}\.\d{2})\b", re.I)  # e.g., bw12.40, bb60.30

FAMILY_HINTS = ["brushfighter","brush bull","brushbull","dual spindle","batwing","turf batwing","rear discharge",
                "finish mower","box scraper","grading scraper","landscape rake","rear blade","disc harrow",
                "post hole digger","tiller","bale spear","pallet fork","quick hitch","stump grinder",
                "bf","bb","ds","mds","bw","tbw","rd990x","lrs","rb","dhs","dhm","pd","rt","rtr","db"]
QUOTE_HINTS = ["quote","price","pricing","cost","how much","list price"]

def has_quote_intent(text: str) -> bool:
    t = text.lower()
    return any(k in t for k in QUOTE_HINTS) or any(k in t for k in FAMILY_HINTS) or bool(MODEL_RE.search(t))

def normalize_model_arg(text: str) -> Dict[str, Any]:
    m = MODEL_RE.search(text)
    if not m:
        return {}
    return {"model": m.group(1).upper()}

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

# ---------- Tool-run loop with session context ----------
def model_step(messages: List[Dict[str, Any]]) -> Tuple[str, Any]:
    if not client:
        return "Planner unavailable (missing OPENAI_API_KEY).", None
    resp = client.chat.completions.create(
        model=OPENAI_MODEL, temperature=0.2, messages=messages, tools=TOOLS, tool_choice="auto",
    )
    return (resp.choices[0].message, resp)

def run_with_tools(sess: Dict[str, Any], convo: List[Dict[str, Any]]) -> str:
    steps = 0
    reply_text: str | None = None
    while steps < MAX_TOOL_STEPS:
        steps += 1
        msg, raw = model_step(convo)
        tool_calls = getattr(msg, "tool_calls", None)

        if not tool_calls:
            reply_text = (msg.content or "").strip()
            if reply_text:
                convo.append({"role": "assistant", "content": reply_text})
            break

        # Record assistant message that invoked tools
        assistant_msg = {
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [{
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments or "{}"},
            } for tc in tool_calls]
        }
        convo.append(assistant_msg)

        # Execute each tool with session context
        for tc in tool_calls:
            fn = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except Exception:
                args = {}

            if fn == "woods_dealer_discount":
                result = tool_woods_dealer_discount(sess, args)
            elif fn == "woods_quote":
                # Normalize if user gave a precise model and the model only sent q
                if "model" not in args and "q" in args:
                    args |= normalize_model_arg(args.get("q") or "")
                result = tool_woods_quote(sess, args)
            elif fn == "woods_health":
                result = tool_woods_health(sess, args)
            else:
                result = {"ok": False, "status": 0, "url": "", "body": {"error": f"Unknown tool {fn}"}}

            convo.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "name": fn,
                "content": json.dumps(result),
            })

        # continue loop to let model react to tool results
    return reply_text or "Understood. Continuing with your quote."

# ---------- Routes ----------
@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json(force=True, silent=True) or {}
        user_message = (data.get("message") or "").strip()
        if not user_message:
            return jsonify({"error": "Missing message"}), 400

        # Sessions
        now = time.time()
        # GC
        for sid in [sid for sid, s in SESSIONS.items() if (now - s.get("updated_at", now)) > SESSION_TTL_SECONDS]:
            SESSIONS.pop(sid, None)
        session_id = request.headers.get("X-Session-Id") or data.get("session_id") or f"anon-{int(now*1000)}"
        sess = SESSIONS.setdefault(session_id, {"messages": [{"role": "system", "content": SYSTEM_PROMPT}]})
        sess["updated_at"] = now

        # Build convo from history
        convo: List[Dict[str, Any]] = list(sess["messages"])

        # ---- Auto-triggers BEFORE appending the user message ----
        # A) Dealer detection -> woods_dealer_discount
        m = DEALER_RE.search(user_message)
        if m:
            dn = m.group(1)
            result = tool_woods_dealer_discount(sess, {"dealer_number": dn})
            inject_tool_result(convo, "woods_dealer_discount", {"dealer_number": dn}, result)

        # B) Quote intent -> woods_quote (dealer auto-attach happens in tool)
        if has_quote_intent(user_message):
            args: Dict[str, Any] = normalize_model_arg(user_message) or {"q": user_message}
            result = tool_woods_quote(sess, args)
            inject_tool_result(convo, "woods_quote", args, result)

        # Append user turn
        convo.append({"role": "user", "content": user_message})

        # Run planner with tools
        reply = run_with_tools(sess, convo)

        # Persist history (cap ~80, keep system)
        MAX_KEEP = 80
        if len(convo) > MAX_KEEP:
            head = convo[0:1] if convo and convo[0].get("role") == "system" else []
            convo = head + convo[-(MAX_KEEP - len(head)):]
        sess["messages"] = convo

        return jsonify({"reply": reply})
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
        result = tool_woods_health({}, {})
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
