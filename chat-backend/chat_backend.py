# chat_backend.py — Minimal AI chat bot backend with Woods Quote Tool actions
#
# What this is:
# - Tiny Flask server exposing:
#     POST /chat   -> { reply: "...", routing?: {...}, quote?: {...}, dealer?: {...} }
#     GET  /health -> { ok, planner, model, action_base }
# - LLM chat first; optionally executes *actions* when the model returns routing JSON.
# - Sticky sessions via X-Session-Id (or session_id in JSON).
# - Seeded with your FULL knowledge block so the model can already reason about quoting flows.
#
# Quick start
#   pip install flask flask-cors openai requests
#   export OPENAI_API_KEY=sk-...   # required for real replies
#   python chat_backend.py
#
# Frontend example
#   fetch("/chat", { method: "POST", headers: {"content-type":"application/json","X-Session-Id": sid}, body: JSON.stringify({message}) })
#
# Config (env vars)
#   OPENAI_API_KEY   -> enables the planner (OpenAI client)
#   OPENAI_MODEL     -> default: gpt-4o-mini
#   TEMPERATURE      -> default: 0.3
#   ALLOWED_ORIGINS  -> CSV of origins for CORS (optional)
#   PORT             -> default: 8000
#   ACTION_BASE_URL  -> default: https://woods-quote-tool.onrender.com
#   INCLUDE_ROUTING  -> if truthy, include parsed routing JSON in response (default: true)
#
from __future__ import annotations
import os, time, logging, traceback, json, re
from typing import Any, Dict, List, Optional, Tuple

from flask import Flask, request, jsonify
try:
    from flask_cors import CORS
except Exception:
    CORS = None

import requests

# -------------- Config --------------
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
TEMPERATURE  = float(os.environ.get("TEMPERATURE", "0.3"))
SESSION_TTL_SECONDS = 60 * 30  # 30 minutes
ACTION_BASE_URL = os.environ.get("ACTION_BASE_URL", "https://woods-quote-tool.onrender.com")
INCLUDE_ROUTING = str(os.environ.get("INCLUDE_ROUTING", "true")).lower() not in ("0", "false", "no")
ACTION_BASE_URL = "https://woods-quote-api.onrender.com"


# =========================
# FULL Knowledge Block (your rules)
# =========================
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

ROUTING_RULES = r"""
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

SYSTEM_PROMPT = os.environ.get("SYSTEM_PROMPT") or (KNOWLEDGE + "\n\n" + PARAM_HINTS + "\n\n" + ROUTING_RULES)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("minibot")

# -------------- OpenAI client (optional) --------------
client = None
try:
    from openai import OpenAI
    if os.environ.get("OPENAI_API_KEY"):
        client = OpenAI()
        log.info("OpenAI planner enabled (model=%s)", OPENAI_MODEL)
    else:
        log.warning("OPENAI_API_KEY not set — running in fallback mode (static replies)")
except Exception as e:
    log.error("Failed to init OpenAI client: %s", e)
    client = None

# -------------- Flask + CORS --------------
app = Flask(__name__)
if CORS:
    allow = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "").split(",") if o.strip()]
    if allow:
        CORS(app, resources={r"/chat": {"origins": allow}}, supports_credentials=False)
    else:
        CORS(app, supports_credentials=False)

# -------------- Sessions (in-memory) --------------
# Session structure:
# {
#   "messages": [...],
#   "updated_at": <ts>,
#   "dealer_number": "179269" | None
# }
SESSIONS: Dict[str, Dict[str, Any]] = {}

# -------------- Helpers --------------
def get_session(session_id: str) -> Dict[str, Any]:
    now = time.time()
    # GC expired
    expired = [sid for sid, s in SESSIONS.items() if (now - s.get("updated_at", now)) > SESSION_TTL_SECONDS]
    for sid in expired:
        SESSIONS.pop(sid, None)

    sess = SESSIONS.setdefault(session_id, {
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}],
        "updated_at": now,
        "dealer_number": None,
    })
    sess["updated_at"] = now
    return sess

def trim_history(messages: List[Dict[str, str]], max_keep: int = 60) -> List[Dict[str, str]]:
    if len(messages) <= max_keep:
        return messages
    head = messages[:1] if messages and messages[0].get("role") == "system" else []
    return head + messages[-(max_keep - len(head)) :]

def extract_routing_json(text: str) -> Optional[Dict[str, Any]]:
    """Try to parse routing JSON from the model reply."""
    # direct JSON?
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    # fenced code blocks ```json ... ```
    fence = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        try:
            obj = json.loads(fence.group(1))
            if isinstance(obj, dict):
                return obj
        except Exception:
            return None
    # loose { ... } grab (best-effort)
    brace = re.search(r"(\{.*\})", text, flags=re.DOTALL)
    if brace:
        try:
            obj = json.loads(brace.group(1))
            if isinstance(obj, dict):
                return obj
        except Exception:
            return None
    return None

def is_valid_routing(obj: Dict[str, Any]) -> bool:
    if not isinstance(obj, dict): return False
    if obj.get("action") not in {"dealer_lookup", "quote", "ask", "smalltalk"}: return False
    if "reply" not in obj: return False
    if "params" not in obj or not isinstance(obj["params"], dict): return False
    return True

def http_get_with_retry(url: str, params: Dict[str, Any], retries: int = 1, timeout: int = 60) -> Tuple[int, Any, Optional[str]]:
    """GET with one retry on connector errors; returns (status_code, json_or_text, error)."""
    try:
        r = requests.get(url, params=params, timeout=timeout)
        ct = r.headers.get("content-type", "")
        data: Any
        if "application/json" in ct:
            try:
                data = r.json()
            except Exception:
                data = r.text
        else:
            data = r.text
        if r.status_code >= 500 and retries > 0:
            # retry once on server error
            r2 = requests.get(url, params=params, timeout=timeout)
            ct2 = r2.headers.get("content-type", "")
            if "application/json" in ct2:
                try:
                    data2 = r2.json()
                except Exception:
                    data2 = r2.text
            else:
                data2 = r2.text
            return r2.status_code, data2, None if r2.ok else f"HTTP {r2.status_code}"
        return r.status_code, data, None if r.ok else f"HTTP {r.status_code}"
    except Exception as e:
        if retries > 0:
            try:
                r = requests.get(url, params=params, timeout=timeout)
                ct = r.headers.get("content-type", "")
                if "application/json" in ct:
                    try:
                        data = r.json()
                    except Exception:
                        data = r.text
                else:
                    data = r.text
                return r.status_code, data, None if r.ok else f"HTTP {r.status_code}"
            except Exception as e2:
                return 0, None, str(e2)
        return 0, None, str(e)

def run_action(sess: Dict[str, Any], routing: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[str]]:
    """
    Execute dealer_lookup or quote against ACTION_BASE_URL.
    Returns (dealer_result, quote_result, error_message)
    """
    action = routing.get("action")
    params = dict(routing.get("params") or {})

    # auto-fill dealer_number from session if missing for quotes
    if action == "quote" and not params.get("dealer_number") and sess.get("dealer_number"):
        params["dealer_number"] = sess["dealer_number"]

    # Map to endpoints
    if action == "dealer_lookup":
        dn = params.get("dealer_number")
        if not dn:
            return None, None, "Missing dealer_number for dealer lookup."
        url = f"{ACTION_BASE_URL}/dealer-discount"
        code, data, err = http_get_with_retry(url, {"dealer_number": dn})
        if code == 200 and isinstance(data, (dict, list)):
            # Remember dealer number on success
            sess["dealer_number"] = dn
            return data, None, None
        if err:
            return None, None, err
        return None, None, f"Lookup failed (status {code})."

    if action == "quote":
        url = f"{ACTION_BASE_URL}/quote"
        code, data, err = http_get_with_retry(url, params)
        if code == 200 and isinstance(data, (dict, list, str)):
            # If the response includes dealer_number, remember it
            dn = params.get("dealer_number")
            if dn:
                sess["dealer_number"] = dn
            return None, data, None
        if err:
            return None, None, err
        return None, None, f"Quote failed (status {code})."

    # For "ask" and "smalltalk" we don't call anything.
    return None, None, None

# -------------- Routes --------------
@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json(force=True, silent=True) or {}
        text = (data.get("message") or "").strip()
        if not text:
            return jsonify({"error": "Missing message"}), 400

        session_id = request.headers.get("X-Session-Id") or data.get("session_id") or f"anon-{int(time.time()*1000)}"

        # --- Manual session reset trigger (case-insensitive) ---
        if text.lower() == "ben luci reset":
            SESSIONS.pop(session_id, None)
            return jsonify({"reply": "✅ Session reset."}), 200

        sess = get_session(session_id)

        # Build conversation
        convo: List[Dict[str, str]] = list(sess["messages"]) + [{"role": "user", "content": text}]
        convo = trim_history(convo)

        # If no OpenAI key, return a friendly fallback
        if not client:
            reply = "(Backend missing OPENAI_API_KEY) — echo: " + text
            # Persist
            sess["messages"] = convo + [{"role": "assistant", "content": reply}]
            return jsonify({"reply": reply})

        # Call OpenAI
        try:
            resp = client.chat.completions.create(
                model=OPENAI_MODEL,
                temperature=TEMPERATURE,
                messages=convo,
            )
            reply = resp.choices[0].message.content or ""
        except Exception as e:
            log.error("OpenAI error: %s", e)
            reply = "There was an error generating a reply. Please try again."
            sess["messages"] = trim_history(convo + [{"role": "assistant", "content": reply}])
            return jsonify({"reply": reply})

        # Try to parse routing JSON from the model reply
        routing = extract_routing_json(reply)
        dealer_result: Optional[Dict[str, Any]] = None
        quote_result: Optional[Dict[str, Any]] = None
        action_error: Optional[str] = None

        if routing and is_valid_routing(routing):
            # Execute action if applicable
            try:
                dealer_result, quote_result, action_error = run_action(sess, routing)
            except Exception as e:
                action_error = str(e)

            # If action error matches your escalation rule, transform the message accordingly
            if action_error:
                # Follow your API Error Handling message
                reply = ("There was a system error while retrieving data. Please try again shortly. "
                         "If the issue persists, escalate to Benjamin Luci at 615-516-8802.")
        else:
            routing = None  # not valid

        # Persist convo
        sess["messages"] = trim_history(convo + [{"role": "assistant", "content": reply}])

        # Build response
        out: Dict[str, Any] = {"reply": reply}
        if INCLUDE_ROUTING and routing:
            out["routing"] = routing
        if dealer_result is not None:
            out["dealer"] = dealer_result
        if quote_result is not None:
            out["quote"] = quote_result
        if sess.get("dealer_number"):
            out["dealer_number"] = sess["dealer_number"]
        if action_error:
            out["action_error"] = action_error

        return jsonify(out)

    except Exception as e:
        log.exception("Unhandled error in /chat")
        return jsonify({
            "reply": "There was a system error while handling your request. Please try again.",
            "error": str(e),
            "trace": traceback.format_exc(limit=1),
        }), 200

@app.route("/health", methods=["GET"])
def health():
    try:
        return jsonify({
            "ok": True,
            "planner": bool(client),
            "model": OPENAI_MODEL,
            "action_base": ACTION_BASE_URL,
        }), 200
    except Exception:
        return jsonify({"ok": False, "planner": bool(client), "model": OPENAI_MODEL, "action_base": ACTION_BASE_URL}), 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    app.run(host="0.0.0.0", port=port, debug=False)
