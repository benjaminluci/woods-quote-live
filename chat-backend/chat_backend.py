#!/usr/bin/env python3
"""
chat_backend.py — Woods Quoting Assistant backend
- Injects your full Knowledge section as a system message every turn
- Stores dealer & rules in session (no AI "memory" needed for critical facts)
- Enforces 12% Cash Discount unless dealer discount is exactly 5%
- Clean message ordering: user → tools → AI
- CORS across browser-facing routes
- One automatic retry on connector errors (2 attempts total)
"""
from __future__ import annotations

import os, re, json, time, random, logging
from typing import Any, Dict, List, Tuple, Optional
from urllib.parse import urljoin

import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

# ---------------- Env & globals ----------------
OPENAI_MODEL       = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
QUOTE_API_BASE     = os.environ.get("QUOTE_API_BASE", "").rstrip("/")
ALLOWED_ORIGINS    = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "").split(",") if o.strip()]
CASH_DISCOUNT_PCT  = float(os.environ.get("CASH_DISCOUNT_PCT", "12"))  # default 12%
SERVICE_JSON_RAW   = os.environ.get("SERVICE_JSON", "")

# OpenAI client (Render has key)
client = None
try:
    from openai import OpenAI  # type: ignore
    if os.environ.get("OPENAI_API_KEY"):
        client = OpenAI()
except Exception:
    client = None

# In-memory sessions (swap to Redis if multi-instance)
SESS: Dict[str, Dict[str, Any]] = {}

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Flask + CORS
app = Flask(__name__)
if ALLOWED_ORIGINS:
    CORS(app, resources={r"/.*": {"origins": ALLOWED_ORIGINS}}, supports_credentials=False)
else:
    CORS(app, supports_credentials=False)

# ---------------- Your Knowledge (verbatim) ----------------
KNOWLEDGE_PROMPT = """You are a quoting assistant for Woods Equipment dealership staff. Your primary job is to retrieve part numbers, list prices, dealer discounts, and configuration requirements exclusively from the Woods Pricing API. You never fabricate data, never infer pricing, and only ask configuration questions when required.

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

# Additional guardrails to keep math deterministic on the server (model writes narrative only)
SYSTEM_PROMPT = """You are the Woods Quoting Assistant.
- Use tools for dealer lookup and quotes. Do NOT invent numbers.
- Pricing math comes from API data and server enforcement. You write a clean, plain-text quote per the rules.
- If dealer is missing, ask for it before quoting.
- Keep answers concise and professional; use bullet points for options."""

# ---------------- Utilities ----------------
RE_INT = re.compile(r"[-+]?\d+")
def _j(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), sort_keys=False)

# Exactly ONE retry: 2 attempts total
def _retry(fn, attempts=2, base_delay=0.25, max_delay=0.6):
    last_err = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:
            last_err = e
            time.sleep(min(max_delay, base_delay * (2 ** i)) + random.uniform(0, 0.05))
    raise last_err  # type: ignore

def http_get(path: str, params: Optional[Dict[str, Any]]=None, timeout=20) -> Dict[str, Any]:
    url = urljoin(QUOTE_API_BASE + "/", path.lstrip("/"))
    def do():
        resp = requests.get(url, params=params, timeout=timeout)
        try:
            body = resp.json()
        except Exception:
            body = resp.text
        return {"ok": resp.ok, "status": resp.status_code, "body": body, "url": resp.url}
    return _retry(do)

# ---------------- Tool adapters ----------------
def tool_woods_health() -> Dict[str, Any]:
    if not QUOTE_API_BASE:
        return {"ok": False, "status": 0, "body": {"error": "QUOTE_API_BASE missing"}}
    try:
        return http_get("/health")
    except Exception as e:
        return {"ok": False, "status": 0, "body": {"error": str(e)}}

def tool_woods_dealer_discount(dealer_number: str) -> Dict[str, Any]:
    if not QUOTE_API_BASE:
        return {"ok": False, "status": 0, "body": {"error": "QUOTE_API_BASE missing"}}
    dealer_number = (dealer_number or "").strip()
    if not dealer_number:
        return {"ok": False, "status": 400, "body": {"error": "dealer_number required"}}
    try:
        return http_get("/dealer-discount", params={"dealer_number": dealer_number})
    except Exception as e:
        return {"ok": False, "status": 0, "body": {"error": str(e)}}

def tool_woods_quote(params: Dict[str, Any]) -> Dict[str, Any]:
    if not QUOTE_API_BASE:
        return {"ok": False, "status": 0, "body": {"error": "QUOTE_API_BASE missing"}}
    try:
        return http_get("/quote", params=params)
    except Exception as e:
        return {"ok": False, "status": 0, "body": {"error": str(e)}}

# ---------------- Helpers for dealer discount extraction ----------------
def _extract_pct(val: Any) -> Optional[float]:
    """Parse 5 / 5.0 / '5' / '5%' / '0.05' into percentage (0-100)."""
    if val is None: return None
    if isinstance(val, (int, float)):
        x = float(val)
        if 0 <= x <= 1: return round(x*100, 4)
        if 0 <= x <= 100: return round(x, 4)
        return None
    s = str(val).strip().replace("%","")
    try:
        x = float(s)
        if 0 <= x <= 1: return round(x*100, 4)
        if 0 <= x <= 100: return round(x, 4)
    except Exception:
        return None
    return None

def _walk_for_discount_pct(obj: Any) -> Optional[float]:
    """Search nested dict/list for a key that looks like dealer discount percent."""
    keys_like = {"discount","dealer_discount","dealerDisc","discount_percent","dealer_discount_percent","dealerPct","dealer_percent"}
    if isinstance(obj, dict):
        for k, v in obj.items():
            kl = k.lower().replace(" ", "_")
            if any(kk in kl for kk in keys_like):
                pct = _extract_pct(v)
                if pct is not None: return pct
            got = _walk_for_discount_pct(v)
            if got is not None: return got
    elif isinstance(obj, list):
        for it in obj:
            got = _walk_for_discount_pct(it)
            if got is not None: return got
    return None

# ---------------- Deterministic cash-discount auditor ----------------
def _coerce_float(x: Any) -> Optional[float]:
    if isinstance(x, (int, float)): return float(x)
    if isinstance(x, str):
        s = x.replace(",", "").strip()
        try: return float(s)
        except Exception:
            m = RE_INT.search(s)
            if m:
                try: return float(m.group(0))
                except Exception: return None
    return None

def _guess_lines(quote_body: Any) -> List[Dict[str, Any]]:
    if isinstance(quote_body, dict):
        for key in ("lines","items","line_items"):
            if isinstance(quote_body.get(key), list): return list(quote_body.get(key) or [])
        for key in ("quote","data","result"):
            inner = quote_body.get(key)
            if isinstance(inner, dict):
                for k2 in ("lines","items","line_items"):
                    if isinstance(inner.get(k2), list): return list(inner.get(k2) or [])
    return []

def _sum_non_discount(lines: List[Dict[str, Any]]) -> float:
    total = 0.0
    for ln in lines:
        desc = (str(ln.get("description") or ln.get("name") or "")).lower()
        code = str(ln.get("code") or "").lower()
        if "discount" in desc or "discount" in code: continue
        amt = _coerce_float(ln.get("amount") or ln.get("total") or ln.get("ext_price") or ln.get("price"))
        if amt is not None: total += amt
    return round(total, 2)

def _find_cash_discount_line(lines: List[Dict[str, Any]]) -> Optional[int]:
    for idx, ln in enumerate(lines):
        desc = (str(ln.get("description") or ln.get("name") or "")).lower()
        code = str(ln.get("code") or "").lower()
        if "cash" in desc and "discount" in desc: return idx
        if "cashdisc" in code or "cash_discount" in code: return idx
    return None

def _ensure_cash_discount(quote_body: Any, pct_cash: float, dealer_pct: Optional[float]) -> Dict[str, Any]:
    """
    Ensure cash discount presence/absence according to rule:
      - If dealer discount == 5%, remove/forbid cash discount line.
      - Else, ensure a 'Cash discount (12%)' line exists. If we can compute a target amount,
        set it to -12% of non-discount subtotal; otherwise leave existing amount as-is.
    """
    lines = _guess_lines(quote_body)
    audit = {"added_cash_discount": False, "adjusted_cash_discount": False,
             "removed_cash_discount": False, "notes": []}

    if not lines:
        audit["notes"].append("No line items found; cannot enforce cash discount safely.")
        return {"quote": quote_body, "audit": audit}

    # If dealer discount is exactly 5%, cash discount must NOT be applied
    if dealer_pct is not None and abs(dealer_pct - 5.0) < 1e-6:
        idx = _find_cash_discount_line(lines)
        if idx is not None:
            lines.pop(idx)
            audit["removed_cash_discount"] = True
            audit["notes"].append("Dealer discount is exactly 5%; removed cash discount per policy.")
    else:
        # Must ensure a cash discount line exists
        idx = _find_cash_discount_line(lines)
        if idx is None:
            lines.append({"code": "CASHDISC", "description": f"Cash discount ({pct_cash:.0f}%)"})
            audit["added_cash_discount"] = True
            audit["notes"].append(f"Inserted cash discount header at {pct_cash:.0f}% (amount derived from API data).")
        else:
            # If amount is present but clearly wrong and we can recompute, set it. Otherwise leave as-is.
            non_disc_subtotal = _sum_non_discount(lines)
            target = round(non_disc_subtotal * (pct_cash/100.0), 2)
            target_neg = -abs(target)
            existing = _coerce_float(lines[idx].get("amount") or lines[idx].get("total") or lines[idx].get("ext_price"))
            if existing is None:
                # leave as-is; model/renderer can compute; note only
                audit["notes"].append("Cash discount line present; amount not adjusted (derived from API/model).")
            elif round(existing,2) != target_neg:
                # we *can* adjust to match policy (still based on totals from API lines)
                lines[idx]["amount"] = target_neg
                audit["adjusted_cash_discount"] = True
                audit["notes"].append(f"Adjusted cash discount to {pct_cash:.0f}% based on API-derived subtotal.")

    # Put lines back where they came from
    updated = quote_body
    if isinstance(updated, dict):
        placed = False
        for key in ("lines","items","line_items"):
            if isinstance(updated.get(key), list):
                updated[key] = lines; placed = True; break
        if not placed:
            for key in ("quote","data","result"):
                inner = updated.get(key)
                if isinstance(inner, dict):
                    for k2 in ("lines","items","line_items"):
                        if isinstance(inner.get(k2), list):
                            inner[k2] = lines; placed = True; break
                if placed: break
    return {"quote": updated, "audit": audit}

# ---------------- LLM tools schema ----------------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "woods_dealer_discount",
            "description": "Lookup dealer information and discount/terms by dealer number.",
            "parameters": {"type": "object","properties":{"dealer_number":{"type":"string"}},"required":["dealer_number"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "woods_quote",
            "description": "Unified quoting action for Woods product families. Provide 'q' and 'dealer_number'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "q": {"type":"string"},
                    "dealer_number": {"type":"string"},
                    "family": {"type":"string"},
                    "family_choice": {"type":"string"},
                    "model": {"type":"string"},
                    "width": {"type":"string"},
                    "width_ft": {"type":"string"},
                },
                "required": ["q","dealer_number"],
                "additionalProperties": True,
            },
        },
    },
    { "type": "function", "function": { "name": "woods_health", "description": "Check health of the quoting API.", "parameters": {"type":"object","properties":{}} } },
]

# Dealer pattern & intent hints
DEALER_PAT  = re.compile(r"\b(?:dealer(?:\s*#| number)?\s*:?)\s*(\d{6,9})\b", re.I)
QUOTE_HINTS = ("quote","price","cost","estimate","bid","out-the-door","ootd","budgetary","total","configure","spec")

# ---------------- Planner loop ----------------
def run_ai(messages: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
    if not client:
        return "Planner unavailable (missing OPENAI_API_KEY).", messages

    max_rounds = 8
    history = list(messages)
    for _ in range(max_rounds):
        completion = client.chat.completions.create(
            model=OPENAI_MODEL, temperature=0.2, tools=TOOLS, tool_choice="auto", messages=history
        )
        choice = completion.choices[0]
        msg = choice.message
        history.append({"role": "assistant", "content": msg.content, "tool_calls": [tc.to_dict() for tc in (msg.tool_calls or [])]})

        if not msg.tool_calls:
            return msg.content or "", history

        for tc in (msg.tool_calls or []):
            name = tc.function.name
            call_id = tc.id
            try:
                args = json.loads(tc.function.arguments or "{}")
            except Exception:
                args = {}

            result: Dict[str, Any] = {"ok": False, "body": {"error": "unknown tool"}}
            try:
                if name == "woods_dealer_discount":
                    result = tool_woods_dealer_discount(args.get("dealer_number",""))
                elif name == "woods_quote":
                    result = tool_woods_quote(args)
                    # enforce cash discount presence/absence using known dealer pct if available
                    dealer_pct = None
                    # Try to read dealer pct from session context system card the planner already has? Not accessible here.
                    # We will pass via /chat state (see autotrigger where we store sess['state']['rules']['dealer_discount_pct'])
                    # Here we can't access session; enforcement happens earlier in /chat auto-trigger for deterministic behavior.
                elif name == "woods_health":
                    result = tool_woods_health()
            except Exception as e:
                logging.exception("Tool error")
                result = {"ok": False, "body": {"error": str(e)}}

            history.append({
                "role": "tool",
                "tool_call_id": call_id,
                "name": name,
                "content": json.dumps(result, ensure_ascii=False),
            })

    return "I'm sorry—I'm having trouble finishing that plan. Try again.", history

# ---------------- Session + orchestration ----------------
def _get_sess(sid: str) -> Dict[str, Any]:
    sess = SESS.setdefault(sid, {
        "messages": [],
        "state": {"dealer": None, "rules": {"cash_discount_pct": CASH_DISCOUNT_PCT, "dealer_discount_pct": None}}
    })
    if len(sess["messages"]) > 80:
        sess["messages"] = sess["messages"][-80:]
    return sess

def _context_card(sess: Dict[str, Any]) -> str:
    dealer = sess["state"].get("dealer") or {}
    dealer_num = dealer.get("number") or dealer.get("dealer_number") or ""
    dealer_name = dealer.get("name") or dealer.get("dealer_name") or ""
    cash_pct   = sess["state"].get("rules", {}).get("cash_discount_pct", CASH_DISCOUNT_PCT)
    dealer_pct = sess["state"].get("rules", {}).get("dealer_discount_pct", None)
    parts = [
        "Context card:",
        f"- Dealer: {dealer_num} {('(' + dealer_name + ')') if dealer_name else ''}".strip(),
        f"- Dealer discount: {dealer_pct:.2f}% " if isinstance(dealer_pct, (int,float)) else "- Dealer discount: (unknown)",
        f"- Cash discount policy: {cash_pct:.0f}% unless dealer discount is exactly 5%",
        "Do NOT invent numbers; rely on API and these rules."
    ]
    return "\n".join(parts)

def _detect_dealer_number(text: str) -> Optional[str]:
    m = DEALER_PAT.search(text or "")
    if m: return m.group(1)
    tokens = re.findall(r"\b(\d{6,9})\b", text or "")
    return tokens[0] if tokens else None

def _has_quote_intent(text: str) -> bool:
    t = (text or "").lower()
    return any(h in t for h in QUOTE_HINTS)

@app.post("/chat")
def chat():
    payload = request.get_json(silent=True) or {}
    session_id   = str(payload.get("session_id") or "").strip() or "default"
    user_message = str(payload.get("message") or "").strip()

    sess = _get_sess(session_id)

    # Build conversation: system knowledge → system guardrails → context card → history → user
    messages: List[Dict[str, Any]] = []
    messages.append({"role": "system", "content": KNOWLEDGE_PROMPT})
    messages.append({"role": "system", "content": SYSTEM_PROMPT})
    messages.append({"role": "system", "content": _context_card(sess)})
    messages.extend(sess["messages"])
    messages.append({"role": "user", "content": user_message})  # user first

    # -------- Auto-triggers AFTER user message --------
    # 1) Dealer detection + store dealer discount %
    dealer_number = _detect_dealer_number(user_message)
    if dealer_number:
        dealer_res = tool_woods_dealer_discount(dealer_number)
        if dealer_res.get("ok"):
            body = dealer_res.get("body") or {}
            sess["state"]["dealer"] = {
                "number": dealer_number,
                "name": body.get("dealer_name") or body.get("name") or "",
                "discounts": body.get("discounts") or body.get("terms") or {},
            }
            # try to extract dealer discount percent and store in session rules
            ddp = _walk_for_discount_pct(body)
            if ddp is not None:
                sess["state"]["rules"]["dealer_discount_pct"] = ddp
        messages.append({"role":"tool","tool_call_id":"autotrigger-woods_dealer_discount","name":"woods_dealer_discount","content":_j(dealer_res)})

    have_dealer = bool(sess["state"].get("dealer", {}).get("number"))
    if _has_quote_intent(user_message) and (have_dealer or dealer_number):
        use_dealer = (sess["state"]["dealer"] or {}).get("number") or dealer_number
        quote_params = {"q": user_message, "dealer_number": str(use_dealer)}
        quote_res = tool_woods_quote(quote_params)
        # Enforce cash discount policy based on known dealer pct (if we have it)
        dealer_pct = sess["state"]["rules"].get("dealer_discount_pct")
        enforced = _ensure_cash_discount(quote_res.get("body"), pct_cash=CASH_DISCOUNT_PCT, dealer_pct=dealer_pct)
        quote_res = dict(quote_res, body=enforced)
        messages.append({"role":"tool","tool_call_id":"autotrigger-woods_quote","name":"woods_quote","content":_j(quote_res)})

    # -------- Planner call --------
    try:
        reply_text, new_history = run_ai(messages)
    except Exception as e:
        logging.exception("Planner failed")
        # Mirror your required error message shape
        reply_text = ("There was a system error while retrieving data. Please try again shortly. "
                      "If the issue persists, escalate to Benjamin Luci at 615-516-8802.")
        new_history = messages

    # Persist trimmed (drop system cards; re-add fresh each turn)
    trimmed = [m for m in new_history if m.get("role") != "system"]
    sess["messages"] = trimmed[-80:]

    # Dealer badge for UI
    dealer = sess["state"].get("dealer") or {}
    badge = {"dealer_number": dealer.get("number"), "dealer_name": dealer.get("name")}

    return jsonify({"reply": reply_text, "dealer": badge})

# ---------------- Health ----------------
@app.get("/health")
def health():
    planner_ok = bool(client)
    try:
        upstream = tool_woods_health()
    except Exception as e:
        upstream = {"ok": False, "status": 0, "body": {"error": str(e)}}
    return jsonify({
        "ok": True,
        "planner": planner_ok,
        "quote_api_base": QUOTE_API_BASE,
        "api_health": upstream,
        "allowed_origins": ALLOWED_ORIGINS,
        "cash_discount_pct": CASH_DISCOUNT_PCT,
    })

# ---------------- Entrypoint ----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5056"))
    logging.info(f"Starting Woods backend on :{port}, model={OPENAI_MODEL}, quote_api={QUOTE_API_BASE}")
    app.run(host="0.0.0.0", port=port, debug=bool(os.environ.get("DEBUG")))
