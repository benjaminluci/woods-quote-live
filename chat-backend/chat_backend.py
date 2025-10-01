# chat_backend.py — AI chat backend with Woods actions + family-tree Q&A
#
# Endpoints
#   POST /chat   -> { reply: "...", quote?: {...}, dealer?: {...}, dealer_number?: "..." }
#   GET  /health -> { ok, planner, model, action_base, allowed_origins, include_routing }
#
# Env (Render):
#   OPENAI_API_KEY=...
#   ALLOWED_ORIGINS=https://woodsequipment-quote.onrender.com
#   QUOTE_API_BASE=https://woods-quote-api.onrender.com      # or ACTION_BASE_URL
#   (optional) INCLUDE_ROUTING=false
#   (optional) OPENAI_MODEL=gpt-4o-mini
#   (optional) TEMPERATURE=0.3
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

# Quoting API base (NOT the chat backend host) — supports either env name
ACTION_BASE_URL = (
    os.environ.get("ACTION_BASE_URL")
    or os.environ.get("QUOTE_API_BASE")
    or "https://woods-quote-api.onrender.com"
)

# Hide routing JSON from HTTP responses by default
INCLUDE_ROUTING = str(os.environ.get("INCLUDE_ROUTING", "false")).lower() not in ("0", "false", "no")

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("minibot")

# =========================
# Knowledge + Routing rules (system prompt for the LLM)
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
Accessory Handling
- If a dealer requests a specific accessory (e.g., tires, chains, dual hub):
  - Attempt API lookup
  - If priced, add as separate line item
  - If not priced, stop and show the escalation message

---
Interaction Style
- Ask one config question at a time
- Never combine multiple questions into a single message
- Format multiple options as lettered vertical lists (A., B., C.)
- Wait for user response before proceeding

---
Disc Harrow Fix
- If API returns the same required spacing prompt repeatedly:
  - Detect the loop
  - Stop quoting
  - Say: “The system is stuck on a required disc spacing selection. Please escalate to Benjamin Luci at 615-516-8802.”
"""

PARAM_HINTS = r"""
Valid families: brushfighter, brushbull, dual_spindle, batwing, turf_batwing, rear_finish,
box_scraper, grading_scraper, landscape_rake, rear_blade, disc_harrow, post_hole_digger,
tiller, bale_spear, pallet_fork, quick_hitch, stump_grinder.

Common params:
- dealer_number (string)
- model (e.g., BB60.30, BW12.40)
- family (see list above)
- width_ft ("12", "15", "17", "20")
- plus *_id / *_choice / duty / driveline / shielding params as needed by family
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
- If the message contains both a dealer number and a quote request, use:
  { "action":"quote", "params": { "dealer_number": "...", ... } }.
- If the message is a quote request without a dealer number, return:
  { "action":"ask", "reply":"Please provide your dealer number to begin." }.
- Map obvious natural language to params (e.g., “12 foot batwing, 540 driveline” → family=batwing, width_ft=12, bw_driveline=540).
- Only ask for missing fields via { "action":"ask", "reply":"<one question at a time>" }.
"""

SYSTEM_PROMPT = os.environ.get("SYSTEM_PROMPT") or (KNOWLEDGE + "\n\n" + PARAM_HINTS + "\n\n" + ROUTING_RULES)

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
        CORS(app, resources={r"/chat": {"origins": allow}, r"/health": {"origins": allow}}, supports_credentials=False)
    else:
        CORS(app, supports_credentials=False)

# Ensure JSON body + no-store on every response
@app.after_request
def ensure_non_empty_json(resp):
    try:
        if resp.status_code == 200 and (not resp.data or resp.data == b""):
            resp = app.response_class(
                response=json.dumps({"reply": "There was a system error while handling your request. Please try again."}),
                status=200,
                mimetype="application/json",
            )
        resp.headers["Cache-Control"] = "no-store"
        return resp
    except Exception:
        return app.response_class(
            response='{"reply":"There was a system error while handling your request. Please try again."}',
            status=200,
            mimetype="application/json",
            headers={"Cache-Control": "no-store"},
        )

@app.errorhandler(Exception)
def on_error(e):
    log.exception("Unhandled exception")
    return jsonify({"reply": "There was a system error while handling your request. Please try again."}), 200

# -------------- Sessions (in-memory) --------------
SESSIONS: Dict[str, Dict[str, Any]] = {}

def get_session(session_id: str) -> Dict[str, Any]:
    now = time.time()
    expired = [sid for sid, s in SESSIONS.items() if (now - s.get("updated_at", now)) > SESSION_TTL_SECONDS]
    for sid in expired:
        SESSIONS.pop(sid, None)

    sess = SESSIONS.setdefault(session_id, {
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}],
        "updated_at": now,
        "dealer_number": None,
        "accum_params": {},
        "pending_question": None,
    })
    sess["updated_at"] = now
    return sess

def trim_history(messages: List[Dict[str, str]], max_keep: int = 60) -> List[Dict[str, str]]:
    if len(messages) <= max_keep:
        return messages
    head = messages[:1] if messages and messages[0].get("role") == "system" else []
    return head + messages[-(max_keep - len(head)) :]

# ---------- Routing JSON handling ----------
def _looks_jsonish(text: str) -> bool:
    if not isinstance(text, str):
        return False
    t = text.strip()
    return t.startswith("{") or t.startswith("[") or "```json" in t.lower()

def _sanitize_to_ai_reply(text: str) -> str:
    if not isinstance(text, str):
        return ""
    out = re.sub(r"```json[\s\S]*?```", "", text, flags=re.IGNORECASE).strip()
    if _looks_jsonish(out):
        return "Got it — handled that selection. What would you like next?"
    return out or "Okay."

def extract_routing_json(text: str) -> Optional[Dict[str, Any]]:
    """Parse routing JSON even if the model wrapped it in quotes or code fences."""
    if not text:
        return None
    t = text.strip()

    m = re.search(r"```json\s*([\s\S]*?)\s*```", t, flags=re.IGNORECASE)
    if m:
        t = m.group(1).strip()

    if (t.startswith('"') and t.endswith('"')) or (t.startswith("'") and t.endswith("'")):
        inner = t[1:-1]
        try:
            t = bytes(inner, "utf-8").decode("unicode_escape")
        except Exception:
            t = inner

    try:
        obj = json.loads(t)
        if isinstance(obj, dict):
            return obj
        if isinstance(obj, str):
            obj2 = json.loads(obj)
            if isinstance(obj2, dict):
                return obj2
    except Exception:
        pass

    m = re.search(r"\{[\s\S]*\}", t)
    if m:
        chunk = m.group(0)
        try:
            obj = json.loads(chunk)
            if isinstance(obj, dict):
                return obj
        except Exception:
            chunk2 = chunk.replace("“", '"').replace("”", '"').replace("’", "'")
            try:
                obj = json.loads(chunk2)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                return None
    return None

def is_valid_routing(obj: Dict[str, Any]) -> bool:
    """Accept actions even if 'reply' is missing; default it later."""
    if not isinstance(obj, dict):
        return False
    if obj.get("action") not in {"dealer_lookup", "quote", "ask", "smalltalk"}:
        return False
    if not isinstance(obj.get("params"), dict):
        obj["params"] = {}
    if "reply" not in obj or not isinstance(obj.get("reply"), str):
        obj["reply"] = ""
    return True

# ---------- HTTP util ----------
def http_get_with_retry(url: str, params: Dict[str, Any], retries: int = 1, timeout: int = 30) -> Tuple[int, Any, Optional[str]]:
    try:
        r = requests.get(url, params=params, timeout=timeout)
        data = r.json() if "application/json" in (r.headers.get("content-type","")) else r.text
        if r.status_code >= 500 and retries > 0:
            r2 = requests.get(url, params=params, timeout=timeout)
            data2 = r2.json() if "application/json" in (r2.headers.get("content-type","")) else r2.text
            return r2.status_code, data2, None if r2.ok else f"HTTP {r2.status_code}"
        return r.status_code, data, None if r.ok else f"HTTP {r.status_code}"
    except Exception as e:
        if retries > 0:
            try:
                r = requests.get(url, params=params, timeout=timeout)
                data = r.json() if "application/json" in (r.headers.get("content-type","")) else r.text
                return r.status_code, data, None if r.ok else f"HTTP {r.status_code}"
            except Exception as e2:
                return 0, None, str(e2)
        return 0, None, str(e)

# ---------- Family-tree Q&A interpreter ----------
def _normalize_question(q: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(q, dict):
        return None
    question_text = q.get("question") or q.get("prompt") or q.get("label") or "Please select an option:"
    param = q.get("param") or q.get("name") or q.get("answer_param") or "choice"
    choices = []
    use_id = False

    if isinstance(q.get("choices_with_ids"), list) and q["choices_with_ids"]:
        use_id = True
        for it in q["choices_with_ids"]:
            if isinstance(it, dict):
                label = it.get("label") or it.get("text") or it.get("name") or str(it.get("id"))
                choices.append({"label": label, "id": it.get("id"), "value": it.get("value")})
    elif isinstance(q.get("choices"), list) and q["choices"]:
        for it in q["choices"]:
            if isinstance(it, dict):
                label = it.get("label") or it.get("text") or it.get("name") or str(it)
                choices.append({"label": label, "id": it.get("id"), "value": it.get("value")})
            else:
                choices.append({"label": str(it), "id": None, "value": str(it)})

    if not choices:
        return None

    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    out_choices = []
    for idx, it in enumerate(choices):
        out_choices.append({
            "letter": letters[idx] if idx < len(letters) else str(idx+1),
            **it
        })

    return {
        "param": param,
        "question": question_text,
        "use_id": use_id,
        "choices": out_choices,
        "raw": q,
    }

def _first_required_question(payload: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(payload, dict):
        return None
    rq = payload.get("required_questions")
    if isinstance(rq, list) and rq:
        return _normalize_question(rq[0])
    if "question" in payload and ("choices" in payload or "choices_with_ids" in payload):
        return _normalize_question(payload)
    nq = payload.get("next_question")
    if isinstance(nq, dict) and ("choices" in nq or "choices_with_ids" in nq):
        return _normalize_question(nq)
    return None

def _format_lettered(qnorm: Dict[str, Any]) -> str:
    lines = [qnorm["question"], ""]
    for it in qnorm["choices"]:
        lines.append(f"{it['letter']}. {it['label']}")
    return "\n".join(lines)

def _guess_param_names(base_param: str, prefer_id: bool) -> List[str]:
    names = []
    if prefer_id:
        if base_param.endswith("_id"):
            names.append(base_param)
        else:
            names.append(base_param + "_id")
    names.append(base_param)
    if base_param.endswith("_choice") and prefer_id:
        names.append(base_param.replace("_choice", "_id"))
    if base_param.endswith("_width") and prefer_id:
        names.append(base_param + "_id")
    return list(dict.fromkeys(names))

def _apply_user_answer_to_params(user_text: str, pending: Dict[str, Any], accum: Dict[str, Any]) -> Dict[str, Any]:
    t = (user_text or "").trim() if hasattr(str, "trim") else (user_text or "").strip()
    choice = None

    for it in pending["choices"]:
        if t.lower() == it["letter"].lower():
            choice = it
            break
    if not choice and t.isdigit():
        idx = int(t) - 1
        if 0 <= idx < len(pending["choices"]):
            choice = pending["choices"][idx]

    if not choice:
        for it in pending["choices"]:
            if t.lower() == (it["label"] or "").lower():
                choice = it
                break
        if not choice:
            for it in pending["choices"]:
                if (it["label"] or "").lower().startswith(t.lower()):
                    choice = it
                    break

    if not choice:
        return accum

    prefer_id = pending.get("use_id", False)
    names_to_try = _guess_param_names(pending["param"], prefer_id)

    value_id = choice.get("id")
    value_label = choice.get("value") or choice.get("label")

    new_params = dict(accum)
    applied = False
    if prefer_id and value_id:
        for name in names_to_try:
            if name.endswith("_id"):
                new_params[name] = value_id
                applied = True
                break
    if not applied:
        new_params[names_to_try[-1]] = value_label

    base = pending["param"].rstrip("_id")
    new_params.setdefault(base + "_choice", choice.get("label"))

    return new_params

def _detect_disc_harrow_loop(prev_pending: Optional[Dict[str, Any]], new_pending: Optional[Dict[str, Any]]) -> bool:
    if not prev_pending or not new_pending:
        return False
    return (
        prev_pending.get("param") == new_pending.get("param")
        and prev_pending.get("question") == new_pending.get("question")
        and len(prev_pending.get("choices", [])) == len(new_pending.get("choices", []))
    )

# ---------- Action runners ----------
def run_dealer_lookup(sess: Dict[str, Any], dealer_number: str) -> Tuple[Optional[Any], Optional[str]]:
    url = f"{ACTION_BASE_URL}/dealer-discount"
    code, data, err = http_get_with_retry(url, {"dealer_number": dealer_number})
    if code == 200:
        sess["dealer_number"] = dealer_number
        return data, None
    return None, err or f"Lookup failed (status {code})."

def run_quote(sess: Dict[str, Any], params: Dict[str, Any]) -> Tuple[Optional[Any], Optional[Dict[str, Any]], Optional[str]]:
    p = dict(params)
    if not p.get("dealer_number") and sess.get("dealer_number"):
        p["dealer_number"] = sess["dealer_number"]
    p.update(sess.get("accum_params") or {})

    url = f"{ACTION_BASE_URL}/quote"
    code, data, err = http_get_with_retry(url, p)
    if code != 200:
        return None, None, err or f"Quote failed (status {code})."

    pending = _first_required_question(data)
    return data, pending, None

# ---------- User-facing formatting ----------
def stringify(obj: Any, limit: int = 12000) -> str:
    try:
        s = json.dumps(obj, ensure_ascii=False, indent=2)
    except Exception:
        s = str(obj)
    return s if len(s) <= limit else s[:limit] + "\n... (truncated) ..."

def build_quote_text_from_payload(payload: Any, dealer_number: Optional[str], params_sent: Dict[str, Any]) -> str:
    if client:
        try:
            messages = [
                {"role":"system","content":
                 "You are a Woods quoting assistant. Follow the 'Quote Output Format' exactly. "
                 "Use ONLY the provided payload values. If any required price is missing, state the escalation message exactly. "
                 "Apply the cash discount rule strictly."},
                {"role":"user","content":
                 f"Dealer number: {dealer_number}\n"
                 f"Params sent: {json.dumps(params_sent, ensure_ascii=False)}\n"
                 f"Raw quote payload:\n```json\n{stringify(payload)}\n```"}
            ]
            resp = client.chat.completions.create(model=OPENAI_MODEL, temperature=0.1, messages=messages)
            text = (resp.choices[0].message.content or "").strip()
            if text:
                return text
        except Exception as e:
            log.warning("Formatter pass failed: %s", e)
    return "Here is your quote. (Details are attached in the response payload.)"

# -------------- Routes --------------
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True, silent=True) or {}
    text = (data.get("message") or "").strip()
    if not text:
        return jsonify({"error": "Missing message"}), 400

    session_id = request.headers.get("X-Session-Id") or data.get("session_id") or f"anon-{int(time.time()*1000)}"

    # Manual session reset
    if text.lower() == "ben luci reset":
        SESSIONS.pop(session_id, None)
        return jsonify({"reply": "✅ Session reset."}), 200

    sess = get_session(session_id)

    # If waiting on a pending question, interpret user's answer first
    if sess.get("pending_question"):
        pending = sess["pending_question"]
        new_params = _apply_user_answer_to_params(text, pending, sess.get("accum_params") or {})
        if new_params == (sess.get("accum_params") or {}):
            prompt = _format_lettered(pending) + "\n\nPlease reply with a letter (A, B, C) or the exact option."
            prompt = _sanitize_to_ai_reply(prompt)
            return jsonify({"reply": prompt, "pending": pending})
        sess["accum_params"] = new_params
        quote_payload, new_pending, err = run_quote(sess, params={})
        if err:
            sess["pending_question"] = None
            return jsonify({"reply":
                "There was a system error while retrieving data. Please try again shortly. "
                "If the issue persists, escalate to Benjamin Luci at 615-516-8802.",
                "action_error": err
            })
        if _detect_disc_harrow_loop(pending, new_pending):
            sess["pending_question"] = None
            return jsonify({"reply":
                "The system is stuck on a required disc spacing selection. Please escalate to Benjamin Luci at 615-516-8802."
            })
        if new_pending:
            sess["pending_question"] = new_pending
            prompt = _sanitize_to_ai_reply(_format_lettered(new_pending))
            return jsonify({"reply": prompt, "pending": new_pending})
        # Done — final quote reply
        sess["pending_question"] = None
        quote_text = build_quote_text_from_payload(quote_payload, sess.get("dealer_number"), sess.get("accum_params") or {})
        quote_text = _sanitize_to_ai_reply(quote_text)
        out = {"reply": quote_text, "quote": quote_payload}
        if sess.get("dealer_number"):
            out["dealer_number"] = sess["dealer_number"]
        return jsonify(out)

    # Not waiting — proceed with LLM to decide routing
    convo: List[Dict[str, str]] = list(sess["messages"]) + [{"role": "user", "content": text}]
    convo = trim_history(convo)

    if not client:
        reply = "(Backend missing OPENAI_API_KEY) — echo: " + text
        reply = _sanitize_to_ai_reply(reply)
        sess["messages"] = convo + [{"role": "assistant", "content": reply}]
        return jsonify({"reply": reply})

    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=TEMPERATURE,
            messages=convo,
        )
        model_reply = resp.choices[0].message.content or ""
    except Exception as e:
        log.error("OpenAI error: %s", e)
        reply = "There was an error generating a reply. Please try again."
        reply = _sanitize_to_ai_reply(reply)
        sess["messages"] = trim_history(convo + [{"role": "assistant", "content": reply}])
        return jsonify({"reply": reply})

    routing = extract_routing_json(model_reply)
    if not routing and _looks_jsonish(model_reply):
        maybe = extract_routing_json(model_reply)
        if maybe and is_valid_routing(maybe):
            routing = maybe

    dealer_result: Optional[Any] = None
    quote_result: Optional[Any] = None
    action_error: Optional[str] = None

    if routing and is_valid_routing(routing):
        action = routing["action"]
        params = dict(routing.get("params") or {})

        if action == "dealer_lookup":
            dn = params.get("dealer_number")
            if not dn:
                user_reply = "Please provide your dealer number to begin."
            else:
                dealer_result, action_error = run_dealer_lookup(sess, dn)
                user_reply = f"Dealer #{dn} verified. What would you like to quote next?" if not action_error else (
                    "There was a system error while retrieving data. Please try again shortly. "
                    "If the issue persists, escalate to Benjamin Luci at 615-516-8802."
                )

        elif action == "quote":
            sess["accum_params"] = {}
            quote_result, pending, action_error = run_quote(sess, params)
            if action_error:
                user_reply = ("There was a system error while retrieving data. Please try again shortly. "
                              "If the issue persists, escalate to Benjamin Luci at 615-516-8802.")
            elif pending:
                sess["pending_question"] = pending
                user_reply = _format_lettered(pending)
            else:
                sess["pending_question"] = None
                user_reply = build_quote_text_from_payload(quote_result, sess.get("dealer_number"), params)

        elif action == "ask":
            user_reply = (routing.get("reply") or "What do you need next?").strip()

        else:  # smalltalk
            user_reply = (routing.get("reply") or "How can I help?").strip()
    else:
        routing = None
        user_reply = model_reply.strip() or "How can I help?"

    user_reply = _sanitize_to_ai_reply(user_reply)
    sess["messages"] = trim_history(convo + [{"role": "assistant", "content": user_reply}])

    out: Dict[str, Any] = {"reply": user_reply}
    if dealer_result is not None:
        out["dealer"] = dealer_result
    if quote_result is not None:
        out["quote"] = quote_result
    if sess.get("dealer_number"):
        out["dealer_number"] = sess["dealer_number"]
    if INCLUDE_ROUTING and routing:
        out["routing"] = routing
    if action_error:
        out["action_error"] = action_error
    if sess.get("pending_question"):
        out["pending"] = sess["pending_question"]

    return jsonify(out)

@app.route("/health", methods=["GET"])
def health():
    allow = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "").split(",") if o.strip()]
    return jsonify({
        "ok": True,
        "planner": bool(client),
        "model": OPENAI_MODEL,
        "action_base": ACTION_BASE_URL,
        "allowed_origins": allow,
        "include_routing": INCLUDE_ROUTING,
    }), 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    app.run(host="0.0.0.0", port=port, debug=False)
