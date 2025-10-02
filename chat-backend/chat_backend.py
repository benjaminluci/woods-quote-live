# chat_backend.py — Woods Quoting Assistant (strict query filter for /quote)
# - GET /quote?model=...&dealer_number=... (+ options) — unchanged
# - Dealer remembered; dealer_number auto-injected for quotes
# - Cash-discount enforcement (12% unless dealer discount == 5%) via _enforced_totals
# - CRITICAL CHANGE: do NOT send dealer_discount or unknown keys to the Pricing API.
#   We strictly filter query params to a known allow-list the API accepts.

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

# ---------------- OpenAI (planner optional) ----------------
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
- A dealer number is required before quoting. Use the Pricing API to look up the dealer’s discount. Do not begin quotes or give pricing without it.
- Dealer numbers may be remembered within a session and across multiple quotes for the same user unless the dealer provides a new number.
- All model, accessory, and pricing data must be pulled directly from the API. Never invent, infer, reuse, or cache data.
- Every quote must pull fresh pricing from the API for all items — including list prices and accessories.
- If a valid part number returns no price, quoting must stop and inform the dealer to escalate the issue.

---
API Error Handling
- Retry any connector error once automatically before showing an error.
- If retry succeeds, proceed normally.
- If retry fails, show: “There was a system error while retrieving data. Please try again shortly. If the issue persists, escalate to Benjamin Luci at 615-516-8802.”

---
Pricing Logic
1. Retrieve list price for each part number from API
2. Apply dealer discount from lookup
3. Unless the dealer discount is exactly 5%, apply an additional 12% cash discount
4. Format quote as plain text, customer-ready

⚠️ Cash discount must always be applied unless dealer discount is exactly 5%. Never skip this.

---
Quote Output Format
- Begin with "**Woods Equipment Quote**"
- Include the dealer name and dealer number below the title
- Final Dealer Net shown boldly with ✅
- Omit the "Subtotal" section
- Include: List Price → Discount → Cash Discount → Final Net
- Include: “Cash discount included only if paid within terms.”
- If a model or part cannot be priced, say: “Unable to find pricing... contact Benjamin Luci at 615-516-8802.”

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
  - Say: “The system is stuck on a required disc spacing selection. Please escalate to Benjamin Luci at 615-516-8802.”
  - Do not retry endlessly

---
Correction Enforcement
- Do not stop quotes after dealer discount
- If dealer discount ≠ 5%, 12% cash discount **must** be shown
- If cash discount is missing from final output, quote is invalid and must be corrected
"""

# Let the assistant prefer server-enforced totals if present
ENFORCER_NOTE = r"""
---
Server Enforcement
If a tool result contains an "_enforced_totals" object, you must use those numbers when presenting:
- list total, dealer discount amount, cash discount amount, and the Final Net.
Never recompute these yourself; use the provided values.
"""

SYSTEM_PROMPT = KNOWLEDGE + ENFORCER_NOTE

# ---------------- HTTP helper ----------------
def http_get(path: str, params: Dict[str, Any]) -> Tuple[Dict[str, Any], int, str]:
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

# ---------------- Model + dealer detection ----------------
MODEL_RE = re.compile(r"\b([A-Za-z]{2,3}\d{1,2}\.\d{2})\b")
DEALER_NUM_RE = re.compile(r"\b(\d{5,9})\b")

def detect_model(text: str) -> str | None:
    m = MODEL_RE.search(text or "")
    return m.group(1).upper() if m else None

def extract_dealer(text: str) -> str | None:
    m = DEALER_NUM_RE.search(text or "")
    return m.group(1) if m else None

# ---------------- Tools (schemas) ----------------
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
                    # Provide dealer_discount only for server-side math; NOT sent to API.
                    "dealer_discount": {"type": "number", "description": "Dealer discount rate (e.g., 0.24)."},
                    # -------- OPTIONS (broad; pass-through allowed) --------
                    "quantity": {"type": "integer"},
                    "part_id": {"type": "string"},
                    "accessory_id": {"type": "string"},
                    "family": {"type": "string"},
                    "width": {"type": "string"},
                    "width_ft": {"type": "string"},
                    "width_in": {"type": "string"},
                    "bb_shielding": {"type": "string"},
                    "bb_tailwheel": {"type": "string"},
                    "bw_duty": {"type": "string"},
                    "bw_driveline": {"type": "string"},
                    "bw_tire_qty": {"type": "integer"},
                    "shielding_rows": {"type": "string"},
                    "deck_rings": {"type": "string"},
                    "tbw_duty": {"type": "string"},
                    "front_rollers": {"type": "string"},
                    "chains": {"type": "string"},
                    "ds_mount": {"type": "string"},
                    "ds_shielding": {"type": "string"},
                    "ds_driveline": {"type": "string"},
                    "tire_id": {"type": "string"},
                    "tire_qty": {"type": "integer"},
                    "dh_width_in": {"type": "string"},
                    "dh_duty": {"type": "string"},
                    "dh_spacing_id": {"type": "string"},
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
                    "tiller_series": {"type": "string"},
                    "tiller_width_in": {"type": "string"},
                    "tiller_rotation": {"type": "string"},
                    "tiller_choice_id": {"type": "string"},
                    "finish_choice": {"type": "string"},
                    "rollers": {"type": "string"},
                    "hydraulics_id": {"type": "string"},
                    "choice_id": {"type": "string"},
                },
                "required": ["dealer_number"],
                "additionalProperties": True
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

# --------- Strict allow-list for /quote query params ---------
ALLOWED_QUERY_KEYS = {
    # core
    "model", "dealer_number",
    # generic
    "quantity", "part_id", "accessory_id", "family", "width", "width_ft", "width_in",
    # BrushBull
    "bb_shielding", "bb_tailwheel",
    # Batwing / Turf Batwing
    "bw_duty", "bw_driveline", "bw_tire_qty", "shielding_rows", "deck_rings",
    "tbw_duty", "front_rollers", "chains",
    # Dual Spindle
    "ds_mount", "ds_shielding", "ds_driveline", "tire_id", "tire_qty",
    # Disc Harrow
    "dh_width_in", "dh_duty", "dh_spacing_id",
    # Box / Grading Scrapers
    "bs_width_in", "bs_duty", "bs_choice_id", "gs_width_in", "gs_choice_id",
    # Landscape Rake
    "lrs_width_in", "lrs_grade", "lrs_choice_id",
    # Rear Blade
    "rb_width_in", "rb_duty", "rb_choice_id",
    # Post Hole Digger
    "pd_model", "auger_id",
    # Tillers
    "tiller_series", "tiller_width_in", "tiller_rotation", "tiller_choice_id",
    # Finish Mowers
    "finish_choice", "rollers",
    # Stump Grinder / misc
    "hydraulics_id", "choice_id",
}

# ---------------- Tool implementations ----------------
def tool_woods_dealer_discount(args: Dict[str, Any]) -> Dict[str, Any]:
    dn = str(args.get("dealer_number") or "").strip()
    body, status, used = http_get("/dealer-discount", {"dealer_number": dn})
    return {"ok": status == 200, "status": status, "url": used, "body": body}

def _num(x):
    try:
        if isinstance(x, (int, float)): return float(x)
        if isinstance(x, str): return float(x.replace(",", "").strip())
    except Exception:
        return None
    return None

def _sum_list_total(body: Dict[str, Any]) -> float | None:
    totals = body.get("totals") if isinstance(body, dict) else None
    if isinstance(totals, dict):
        lt = _num(totals.get("list_total"))
        if lt is not None: return lt
    for key in ("items", "line_items"):
        arr = body.get(key)
        if isinstance(arr, list) and arr:
            s, any_found = 0.0, False
            for it in arr:
                if not isinstance(it, dict): continue
                price = _num(it.get("list_price")) or _num(it.get("list")) or (_num(it.get("list_price_cents")) or 0)/100.0
                qty = _num(it.get("qty")) or _num(it.get("quantity")) or 1.0
                if price is not None:
                    any_found = True
                    s += price * float(qty)
            if any_found:
                return s
    lt = _num(body.get("list_price")) or _num(body.get("list"))
    return lt

def _enforce_cash_totals(body: Dict[str, Any], dealer_discount: float | None) -> Dict[str, Any] | None:
    if dealer_discount is None:
        dd = _num((body.get("dealer") or {}).get("discount") if isinstance(body.get("dealer"), dict) else body.get("dealer_discount"))
        dealer_discount = dd if dd is not None else None
    list_total = _sum_list_total(body)
    if list_total is None or dealer_discount is None:
        return None
    dd_rate = float(dealer_discount)
    net_after_dealer = list_total * (1.0 - dd_rate)
    cash_rate = 0.0 if abs(dd_rate - 0.05) < 1e-9 else 0.12
    cash_amount = net_after_dealer * cash_rate
    final_net = net_after_dealer * (1.0 - cash_rate)
    def r2(x): return float(f"{x:.2f}")
    return {
        "list_total": r2(list_total),
        "dealer_discount_rate": r2(dd_rate),
        "dealer_discount_amount": r2(list_total * dd_rate),
        "net_after_dealer": r2(net_after_dealer),
        "cash_discount_rate": r2(cash_rate),
        "cash_discount_amount": r2(cash_amount),
        "final_net_enforced": r2(final_net),
        "note": "12% cash discount applied unless dealer discount is exactly 5%.",
    }

def tool_woods_quote(args: Dict[str, Any]) -> Dict[str, Any]:
    # Keep dealer_discount locally for math, but do NOT send it to the API
    dd_local = args.get("dealer_discount")
    safe_params = {}
    for k, v in (args or {}).items():
        if v in (None, ""): 
            continue
        if k == "dealer_discount":
            continue  # do not send to API
        if k in ALLOWED_QUERY_KEYS:
            safe_params[k] = v
    body, status, used = http_get("/quote", safe_params)
    try:
        if status == 200 and isinstance(body, dict):
            enforced = _enforce_cash_totals(body, _num(dd_local) if dd_local is not None else None)
            if enforced:
                body["_enforced_totals"] = enforced
    except Exception as e:
        log.warning("enforcer failed: %s", e)
    return {"ok": status == 200, "status": status, "url": used, "body": body}

def tool_woods_health(args: Dict[str, Any]) -> Dict[str, Any]:
    body, status, used = http_get("/health", {})
    return {"ok": status == 200, "status": status, "url": used, "body": body}

# ---------------- Planner ----------------
def run_ai(messages: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
    if not client:
        return "Planner unavailable (missing OPENAI_API_KEY).", messages
    completion = client.chat.completions.create(
        model=OPENAI_MODEL, temperature=0.2, messages=messages, tools=TOOLS, tool_choice="auto"
    )
    history = messages[:]
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

        history.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments or "{}"},
                } for tc in tool_calls
            ]
        })

        for tc in tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except Exception:
                args = {}

            # ALWAYS inject dealer_number and dealer_discount for woods_quote if missing
            if name == "woods_quote":
                dn = args.get("dealer_number")
                dd_rate = args.get("dealer_discount")
                for m in reversed(history):
                    if dn and dd_rate is not None:
                        break
                    if m.get("role") == "tool" and m.get("name") == "woods_dealer_discount":
                        try:
                            payload = json.loads(m.get("content") or "{}")
                            b = payload.get("body") or {}
                            if not dn:
                                dn0 = b.get("dealer_number") or b.get("number")
                                if dn0: dn = str(dn0)
                            if dd_rate is None and (b.get("discount") is not None):
                                try: dd_rate = float(b.get("discount"))
                                except Exception: pass
                        except Exception:
                            pass
                if dn and not args.get("dealer_number"):
                    args["dealer_number"] = dn
                if (dd_rate is not None) and (args.get("dealer_discount") is None):
                    args["dealer_discount"] = dd_rate

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

        completion = client.chat.completions.create(
            model=OPENAI_MODEL, temperature=0.2, messages=history, tools=TOOLS, tool_choice="auto"
        )

# ---------------- Utilities ----------------
def add_tool_exchange(convo: List[Dict[str, Any]], name: str, args: Dict[str, Any], result: Dict[str, Any]) -> None:
    call_id = f"{name}-{int(time.time()*1000)}"
    convo.append({
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": call_id,
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)},
        }]
    })
    convo.append({
        "role": "tool",
        "tool_call_id": call_id,
        "name": name,
        "content": json.dumps(result),
    })

# ---------------- Routes ----------------
@app.post("/chat")
def chat():
    try:
        payload = request.get_json(silent=True) or {}
        user_message = str(payload.get("message") or "").strip()
        if not user_message:
            return jsonify({"error": "Missing message"}), 400

        now = time.time()
        sid = request.headers.get("X-Session-Id") or payload.get("session_id") or f"anon-{int(now*1000)}"
        for k in list(SESS.keys()):
            if now - SESS[k].get("updated_at", now) > SESSION_TTL:
                SESS.pop(k, None)
        sess = SESS.setdefault(sid, {"messages": [], "dealer": None, "updated_at": now})
        sess["updated_at"] = now

        convo: List[Dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        convo.extend(sess["messages"])

        # Auto: dealer detection
        dn_from_msg = extract_dealer(user_message)
        if dn_from_msg:
            res = tool_woods_dealer_discount({"dealer_number": dn_from_msg})
            add_tool_exchange(convo, "woods_dealer_discount", {"dealer_number": dn_from_msg}, res)
            if res.get("ok"):
                body = res.get("body") or {}
                sess["dealer"] = {
                    "dealer_number": body.get("dealer_number") or dn_from_msg,
                    "dealer_name": body.get("dealer_name") or "",
                    "discount": body.get("discount"),
                }

        # Determine dealer for quoting (session first, else scan convo)
        dealer_num, dealer_disc = None, None
        if sess.get("dealer"):
            dealer_num = str(sess["dealer"].get("dealer_number") or "") or None
            dealer_disc = sess["dealer"].get("discount")
        if (dealer_num is None) or (dealer_disc is None):
            for m in reversed(convo):
                if m.get("role") == "tool" and m.get("name") == "woods_dealer_discount":
                    try:
                        payload2 = json.loads(m.get("content") or "{}")
                        b = payload2.get("body") or {}
                        if dealer_num is None:
                            dn = b.get("dealer_number") or b.get("number")
                            if dn: dealer_num = str(dn)
                        if dealer_disc is None and (b.get("discount") is not None):
                            dealer_disc = float(b.get("discount"))
                        if dealer_num and (dealer_disc is not None):
                            break
                    except Exception:
                        pass

        # Auto: quote trigger ONLY when model detected AND dealer present
        model = detect_model(user_message)
        if model and dealer_num:
            args = {"model": model, "dealer_number": dealer_num}
            if dealer_disc is not None:
                args["dealer_discount"] = float(dealer_disc)  # used locally; filtered out for API
            res = tool_woods_quote(args)
            add_tool_exchange(convo, "woods_quote", args, res)

        # Append user & run planner
        convo.append({"role": "user", "content": user_message})
        reply, hist = run_ai(convo)

        # trim + persist (keep system reinjected each turn)
        MAX_KEEP = 80
        trimmed = [m for m in hist if m.get("role") != "system"]
        if len(trimmed) > MAX_KEEP:
            trimmed = trimmed[-MAX_KEEP:]
        sess["messages"] = trimmed

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
            "trace": traceback.format_exc(limit=1),
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
