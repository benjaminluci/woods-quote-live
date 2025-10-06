import os
import re
import time
import json
import pandas as pd
from flask import Flask, request, jsonify
import gspread
from google.oauth2.service_account import Credentials

app = Flask(__name__)

# =========================
# Config / Auth
# =========================
SHEET_ID = os.getenv("SHEET_ID", "1xAIONQSDWz95u97XLfAIb5gt6ddR8OWg3nN0MPN2LGw")
SHEET_NAME = os.getenv("GOOGLE_SHEET_NAME", "Woods ChatGPT Master Price File")
DEFAULT_DISCOUNT = float(os.getenv("DEALER_DISCOUNT", 0.0))
DEFAULT_FREIGHT = float(os.getenv("FREIGHT", 0.0))
SERVICE_FILE = os.getenv("SERVICE_FILE", "speedy-lattice-469816-m7-d33c619c53ee.json")

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

def gsheet_client():
    raw = os.getenv("SERVICE_JSON")
    if raw:
        info = json.loads(raw) if isinstance(raw, str) else raw
        creds = Credentials.from_service_account_info(info, scopes=SCOPES)
        return gspread.authorize(creds)

    if not os.path.exists(SERVICE_FILE):
        raise RuntimeError(
            "Missing Google credentials. Set SERVICE_JSON with your key JSON "
            f"or place the key file at: {SERVICE_FILE}"
        )
    creds = Credentials.from_service_account_file(SERVICE_FILE, scopes=SCOPES)
    return gspread.authorize(creds)

# =========================
# Data loading (cached)
# =========================
DF_CACHE = {"df": None, "loaded_at": 0.0}

def load_df(force=False):
    if DF_CACHE["df"] is None or force:
        gc = gsheet_client()
        sh = None
        if SHEET_ID:
            try:
                sh = gc.open_by_key(SHEET_ID)
            except Exception:
                sh = None
        if sh is None:
            sh = gc.open(SHEET_NAME)
        ws = sh.sheet1
        records = ws.get_all_records()
        df = pd.DataFrame(records)
        # Trim headers and cells
        df.columns = [str(c).strip() for c in df.columns]
        for c in df.columns:
            if df[c].dtype == object:
                df[c] = df[c].astype(str).str.strip()
        DF_CACHE["df"] = df
        DF_CACHE["loaded_at"] = time.time()
    return DF_CACHE["df"]
# =========================
# Columns
# =========================
COL_CATEGORY = "Category"  # A
COL_MODEL = "Model"  # B
COL_WIDTH_FT = "Width (ft)"  # C
COL_DRIVE = "Hitch Type / Drive"  # D
COL_TAILWHEEL = "Laminated Tail Wheel"  # E (Single/Dual)
COL_SHIELDING = "Shielding Type"  # F
COL_USED_ON = "Used On"  # I
COL_SUSPENSION = "Suspension"  # J (for other families later)
COL_DESC = "Description"  # X
COL_TIRES_REQ = "Tires Required"  # Z
COL_LIST_PRICE = "List Price"  # AD
COL_DUTY_CLASS = "Duty Class"  # if present

PART_ID_COLS = ["Woods Part No.", "Part No."]

# =========================
# Helpers
# =========================
def _price_number(val):
    if val is None:
        return 0.0
    s = str(val).replace("$", "").replace(",", "").strip()
    m = re.search(r"(\d+(?:\.\d+)?)", s)
    return float(m.group(1)) if m else 0.0

def _make_line(desc, qty, unit, **extra):
    qty = int(qty or 1)
    unit = float(unit or 0.0)
    line = {
        "desc": str(desc),
        "qty": qty,
        "unit_price": round(unit, 2),
        "subtotal": round(qty * unit, 2),
    }
    if extra:
        line.update(extra)  # e.g., part_id="639920A"
    return line

def _totals_payload(lines, notes=None, category="", model_name="",
                    dealer_discount_override=None, dealer_meta=None):
    if dealer_discount_override is None:
        dealer_discount = float(request.args.get("dealer_discount", DEFAULT_DISCOUNT))
    else:
        dealer_discount = float(dealer_discount_override)

    freight_val = _price_number(request.args.get("freight")) if request.args.get("freight") else DEFAULT_FREIGHT

    subtotal = round(sum(li["subtotal"] for li in lines), 2)
    discount_amt = round(subtotal * dealer_discount, 2)
    total = round(subtotal - discount_amt + freight_val, 2)

    payload = {
        "found": True,
        "mode": "quote",
        "category": category,
        "model": model_name,
        "items": lines,
        "notes": notes or [],
        "summary": {
            "subtotal_list": subtotal,
            "dealer_discount_rate": dealer_discount,
            "dealer_discount_amt": discount_amt,
            "freight": freight_val,
            "total": total
        }
    }
    if dealer_meta:
        payload["dealer"] = dealer_meta
    return jsonify(payload)

def _first_part_id(row):
    for c in PART_ID_COLS:
        if c in row.index and str(row[c]).strip():
            return str(row[c]).strip()
    return ""

def _find_by_part_id(df, pid):
    if not pid:
        return df.iloc[0:0]
    pid = str(pid).strip().upper()
    hits = []
    for c in PART_ID_COLS:
        if c in df.columns:
            sub = df[df[c].astype(str).str.strip().str.upper() == pid]
            if not sub.empty:
                hits.append(sub)
    return pd.concat(hits, ignore_index=True) if hits else df.iloc[0:0]
# =========================
# width helpers
# =========================
def _ft_str(val):
    if val is None:
        return ""
    s = str(val).strip()
    m = re.search(r"(\d+(?:\.\d+)?)", s)
    if not m:
        return ""
    try:
        num = float(m.group(1))
        return str(int(round(num)))
    except:
        return ""

def _normalize_width_ft_input(width_txt):
    """
    Convert user width (e.g., '5', '5ft', '60', '60"') to feet as an int string.
    Heuristic: number <= 10 => feet; else inches->feet.
    'ft' forces feet; '"' forces inches.
    """
    if not width_txt:
        return None
    s = str(width_txt).strip().lower()
    m = re.search(r"(\d+(?:\.\d+)?)", s)
    if not m:
        return None
    num = float(m.group(1))
    if any(x in s for x in ['"', "in", "inch", "inches"]):
        return str(int(round(num / 12.0)))
    if any(x in s for x in ["ft", "foot", "feet"]):
        return str(int(round(num)))
    if num <= 10:
        return str(int(round(num)))  # feet
    return str(int(round(num / 12.0)))  # inches -> feet

def _normalize_width_ft_input_feet_only(width_txt):
    """Strict feet parsing for Batwings (no inch interpretation)."""
    if not width_txt:
        return None
    s = str(width_txt).strip().lower()
    m = re.search(r"(\d+(?:\.\d+)?)", s)
    if not m:
        return None
    num = float(m.group(1))
    return str(int(round(num)))  # always feet

def _ensure_width_norm_col(frame):
    col = "_WIDTH_FT_NORM"
    if col not in frame.columns:
        frame[col] = frame[COL_WIDTH_FT].apply(_ft_str)
    return col

def _ensure_width_in_norm_col(frame):
    """
    Create/return a normalized inches column for Box Scrapers:
    - Prefer an explicit 'Width (in)' column if present.
    - Else derive from 'Width (ft)' or numeric-looking width:
        * <= 10 → treat as feet, convert to inches
        * > 10  → treat as inches already
    """
    col = "_WIDTH_IN_NORM"
    if col in frame.columns:
        return col

    width_in_col = None
    for cname in frame.columns:
        if str(cname).strip().lower() in {"width (in)", "width (inch)", "width (inches)", "width-in"}:
            width_in_col = cname
            break

    if width_in_col:
        def _to_int_in(v):
            s = str(v).strip()
            m = re.search(r"(\d+(?:\.\d+)?)", s)
            if not m: return ""
            return str(int(round(float(m.group(1)))))
        frame[col] = frame[width_in_col].apply(_to_int_in)
        return col

    def _derive_in(v):
        s = str(v).strip().lower()
        m = re.search(r"(\d+(?:\.\d+)?)", s)
        if not m: return ""
        num = float(m.group(1))
        if any(x in s for x in ["ft", "foot", "feet"]) or num <= 10:
            return str(int(round(num * 12.0)))
        return str(int(round(num)))

    src = "Width (ft)" if "Width (ft)" in frame.columns else None
    if src is None:
        try:
            src = frame.columns[2]  # fallback to column C
        except:
            src = None
    if src:
        frame[col] = frame[src].apply(_derive_in)
    else:
        frame[col] = ""
    return col

def _inches_to_feet_label(in_str):
    try:
        n = int(in_str)
        ft = n / 12.0
        if abs(ft - round(ft)) < 1e-6:
            return f"{int(round(ft))} ft"
        return f"{ft:.1f} ft"
    except:
        return ""

def _dh_width_in_from_model(model: str):
    """Pull inches from model like DHS72N / DHM84C -> 72 / 84."""
    s = (model or "").upper()
    m = re.search(r'\bDH[SM]?\s*0*?(\d{2,3})', s)
    if m:
        try:
            return int(m.group(1))
        except:
            pass
    return None

def _dh_series_from_model(model: str):
    """Return 'DHS' or 'DHM' (Standard vs Heavy Duty) if detectable."""
    s = (model or "").upper()
    if s.startswith("DHM"):
        return "DHM"
    if s.startswith("DHS"):
        return "DHS"
    # fallback: if only DH??, infer later from row/model list
    return ""

def _dh_blade_from_model(model: str):
    """Return 'N' (Notched) or 'C' (Combo) if model ends with it."""
    s = (model or "").upper().strip()
    if s.endswith("N"):
        return "N"
    if s.endswith("C"):
        return "C"
    return ""

def _dh_label(r, duty_text=None, blade_text=None, spacing_pair=None, include_desc=True):
    model = str(r.get(COL_MODEL, "")).strip()
    bits = [model]
    if duty_text:
        bits.append(duty_text)
    if blade_text:
        bits.append(f"Blades: {blade_text}")
    if spacing_pair:
        f, b = spacing_pair
        if f or b:
            bits.append(f"Spacing: Front {f or '?'} / Rear {b or '?'}")
    if include_desc:
        dsc = str(r.get(COL_DESC, "") or "").strip()
        if dsc:
            bits.append(dsc)
    return " — ".join(bits)

def _dh_find_spacing_cols(frame):
    """Return (front_col, rear_col) for disc spacing, if present."""
    lc = [str(c).strip().lower() for c in frame.columns]
    def pick(*names):
        for want in names:
            want_l = want.lower()
            for c, cl in zip(frame.columns, lc):
                if cl == want_l:
                    return c
        return None

    # Try the obvious names first, then loose variants
    front = pick("Front Disc Spacing", "Front Spacing")
    rear  = pick("Rear Disc Spacing", "Rear Spacing")

    if front is None:
        for c, cl in zip(frame.columns, lc):
            if "front" in cl and "spacing" in cl:
                front = c; break
    if rear is None:
        for c, cl in zip(frame.columns, lc):
            if "rear" in cl and "spacing" in cl:
                rear = c; break
    return front, rear


def _dh_spacing_label(row, front_col, rear_col):
    """Human label like '7 in front / 9 in rear' or '9 in' when single value."""
    f = str(row.get(front_col, "") or "").strip() if front_col else ""
    r = str(row.get(rear_col, "") or "").strip() if rear_col else ""
    # normalize to digits if they include units
    def norm(x):
        m = re.search(r"(\d+(\.\d+)?)", x)
        return m.group(1) if m else x
    f, r = norm(f), norm(r)
    if f and r and f != r:
        return f"{f} in front / {r} in rear"
    if f or r:
        return f"{f or r} in"
    return "Standard spacing"

def _dh_width_in_from_row(row):
    """Return working width in *inches* as a string, best-effort."""
    # 1) Prefer an explicit inches column
    for cname in row.index:
        cl = str(cname).strip().lower()
        if cl in {"width (in)", "width in", "width inches", "width-in"}:
            s = str(row.get(cname) or "").strip()
            m = re.search(r'(\d+(?:\.\d+)?)', s)
            if m:
                return str(int(round(float(m.group(1)))))
    # 2) Fall back: feet -> inches
    if COL_WIDTH_FT in row.index and str(row.get(COL_WIDTH_FT) or "").strip():
        s = str(row.get(COL_WIDTH_FT)).strip()
        m = re.search(r'(\d+(?:\.\d+)?)', s)
        if m:
            return str(int(round(float(m.group(1)) * 12)))
    # 3) Last resort: parse from model code (e.g., DHS72N -> 72)
    w = _dh_width_in_from_model(str(row.get(COL_MODEL) or ""))
    return str(w) if w else ""



# =========================
# regex for family models
# =========================
BF_MODEL_RE = re.compile(r'\b(BF\d+(?:\.\d+)?)\b', re.I)
BB_MODEL_RE = re.compile(r'\b(BB\d+(?:\.\d+)?D?)\b', re.I)
DS_MODEL_RE = re.compile(r'\b((?:MDS|DS)\d+(?:\.\d+)?Q?)\b', re.I)
BW_MODEL_RE = re.compile(r'\b(BW\d+(?:\.\d+)?R?Q?)\b', re.I)  # BWxx.yy maybe with R or Q
TBW_MODEL_RE = re.compile(r'\b(TBW\d+(?:\.\d+)?)\b', re.I)
TK_MODEL_RE   = re.compile(r'\b(TK\d+(?:\.\d+)?)\b', re.I)
TKP_MODEL_RE  = re.compile(r'\b(TKP\d+(?:\.\d+)?)\b', re.I)
RD990X_MODEL_RE = re.compile(r'\b(RD990X)\b', re.I)
BS_MODEL_RE = re.compile(r'\b(BS\d+(?:\.\d{2})?)\b', re.I)
GS_MODEL_RE = re.compile(r'\b(GS\d+(?:\.\d+)?)\b', re.I)
LRS_MODEL_RE = re.compile(r'\b(LRS\d+(?:\.\d+)?)\b', re.I)
PD_MODEL_RE = re.compile(r'\b(PD\d+\.\d+)\b', re.I)
RB_MODEL_RE = re.compile(r'\b(RB\d+(?:\.\d+)?P?)\b', re.I)  # optional Premium "P" suffix
DH_MODEL_RE = re.compile(r'\b(DH[SM]?\d+(?:[NC])?)\b', re.I)
TILLER_MODEL_RE = re.compile(r'\b((?:RTR|RT|DB)\s*\d+(?:\.\d+)?)\b', re.I)
TQH_MODEL_RE = re.compile(r'\b(TQH[12])\b', re.I)
TSG_MODEL_RE = re.compile(r'\b(TSG50)\b', re.I)





def _extract_bf_code(text):
    s = (text or "").strip()
    m = BF_MODEL_RE.search(s)
    return m.group(1).upper() if m else s.upper()

def _extract_bb_code(text):
    s = (text or "").strip()
    m = BB_MODEL_RE.search(s)
    return m.group(1).upper() if m else s.upper()

def _extract_ds_code(text):
    s = (text or "").strip()
    m = DS_MODEL_RE.search(s)
    return m.group(1).upper() if m else s.upper()

def _extract_bw_code(text):
    s = (text or "").strip()
    m = BW_MODEL_RE.search(s)
    return m.group(1).upper() if m else s.upper()

def _extract_bs_code(text):
    s = (text or "").strip()
    m = BS_MODEL_RE.search(s)
    return m.group(1).upper() if m else s.upper()

def _extract_pd_code(text):
    s = (text or "").strip()
    m = PD_MODEL_RE.search(s)
    return m.group(1).upper() if m else s.upper()

def _extract_dh_code(text):
    s = (text or "").strip()
    m = DH_MODEL_RE.search(s)
    return m.group(1).upper() if m else s.upper()

# --- Param normalization & safe row pick ---
PARAM_ALIASES = {
    "q": ["q", "query"],
    "family": ["family", "family_choice"],
    "width_ft": ["width_ft", "width"],
    "model": ["model"],
    "dealer_number": ["dealer_number", "dealer"],

    # BrushFighter
    "bf_choice": ["bf_choice", "choice", "choice_label"],
    "bf_choice_id": ["bf_choice_id", "choice_id", "part_id", "part_no"],

    "drive": ["drive", "drive_type", "driveline"],

    # BrushBull
    "bb_duty": ["bb_duty", "duty"],
    "bb_shielding": ["bb_shielding", "shielding", "shielding_type"],
    "bb_tailwheel": ["bb_tailwheel", "tailwheel"],

    # Dual Spindle
    "ds_mount": ["ds_mount", "mount", "mount_type"],  # Mounted / Pull-type
    "ds_shielding": ["ds_shielding", "shielding", "shielding_type"],
    "ds_driveline": ["ds_driveline", "driveline", "drive"],
    "tire_choice": ["tire_choice", "tire_desc", "tire"],
    "tire_id": ["tire_id", "accessory_id"],  # tires via part id
    "tire_qty": ["tire_qty"],

    # Batwing
    "bw_duty": ["bw_duty", "duty"],
    "bw_driveline": ["bw_driveline", "driveline", "drive"],
    "deck_rings": ["deck_rings", "rings"],
    "shielding_rows":["shielding_rows", "shielding_option"],  # Single Row / Double Row

    # Finish Mowers
    "finish_choice": ["finish_choice", "finish_type"],

    # Box Scraper
    "bs_width_in": ["bs_width_in", "bs_width", "width_in", "width_inch", "width_inches"],
    "bs_duty": ["bs_duty", "duty"],
    "bs_choice_id": ["bs_choice_id", "choice_id", "part_id", "part_no"],
    "bs_choice": ["bs_choice", "choice", "choice_label"],

    # Grading Scraper
    "gs_choice": ["gs_choice", "choice", "choice_label"],
    "gs_choice_id": ["gs_choice_id", "choice_id", "part_id", "part_no"],

    # Post Hole Digger
    "pd_model": ["pd_model", "model"],
    "auger_id": ["auger_id", "accessory_id"],
    "auger_choice": ["auger_choice", "accessory_desc"],

    # Rear Blade (RB)
    "rb_width_in": ["rb_width_in", "rb_width", "width_in"],
    "rb_duty": ["rb_duty", "duty"],
    "rb_choice_id": ["rb_choice_id", "choice_id", "part_id", "part_no"],
    "rb_choice": ["rb_choice", "choice", "choice_label"],

    # Disc Harrow
    "dh_width_in": ["dh_width_in"],
    "dh_duty": ["dh_duty", "duty"],
    "dh_blade": ["dh_blade", "blade"],
    "dh_choice_id": ["dh_choice_id", "choice_id", "dh_spacing_id", "spacing_id"],
    "dh_choice":    ["dh_choice", "choice", "dh_spacing", "spacing"],

    # --- Tillers ---
    "tiller_family":   ["tiller_family", "tiller_duty", "db_or_rt", "tiller_type", "family_choice"],
    "tiller_rotation": ["tiller_rotation", "rotation"],
    "tiller_width_in": ["tiller_width_in", "width_in", "width"],
    "tiller_choice_id":["tiller_choice_id", "choice_id", "part_id", "part_no"],
    "tiller_choice":   ["tiller_choice", "choice", "choice_label"],
    "tiller_series":   ["tiller_series", "tiller_family", "tiller_duty", "db_or_rt", "tiller_type", "family_choice"],

    # --- Bale Spear ---
    "balespear_choice_id": ["balespear_choice_id", "choice_id", "part_id", "part_no"],
    "balespear_choice":    ["balespear_choice", "choice", "choice_label"],

    # Pallet Fork
    "pf_choice_id": ["pf_choice_id", "part_id", "part_no", "choice_id"],
    "pf_choice":    ["pf_choice", "choice", "choice_label"],

    # Quick Hitch
    "qh_choice_id": ["qh_choice_id", "part_id", "part_no", "choice_id"],
    "qh_choice":    ["qh_choice", "choice", "choice_label"],

    # Stump Grinder
    "hydraulics_id":     ["hydraulics_id", "accessory_id", "part_id", "part_no"],
    "hydraulics_choice": ["hydraulics_choice", "choice", "choice_label", "accessory_desc"],



    # Generic accessories
    "accessory_id": ["accessory_id", "part_id", "part_no"],
    "accessory_ids": ["accessory_ids"],
    "accessory": ["accessory"],  # pid:qty tokens
    "accessory_desc":["accessory_desc"],  # desc contains
}

def getp(*keys, default=""):
    for k in keys:
        real_keys = PARAM_ALIASES.get(k, [k])
        for rk in real_keys:
            v = request.args.get(rk)
            if v is not None and str(v).strip() != "":
                return str(v).strip()
    return default

def pick_one_or_400(rows, where_msg, context=None):
    if rows is None or len(rows.index) == 0:
        msg = f"No matching rows for: {where_msg}"
        if context:
            msg += f" | context={context}"
        return None, (jsonify({"found": False, "mode": "error", "message": msg}), 400)
    return rows.iloc[0], None
# =========================
# Accessories/Tires
# =========================
# --- extend accessories pattern to include Augers (so they’re treated like accessories) ---
ACC_TIRE_PATTERN = re.compile(r"(Accessories|Tire|Auger)", re.I)  # ← add “Auger”

def _used_on_exact_mask(series, model_code: str):
    """
    True when 'Used On' contains the exact model code as a whole token.
    Example: model_code='RBS72' will NOT match 'RBS72P' or 'RBS72.50'.
    """
    if not model_code:
        return series == "__no_match__"  # always False
    pat = r'(?<![A-Z0-9])' + re.escape(str(model_code).upper().strip()) + r'(?![A-Z0-9])'
    return series.astype(str).str.upper().str.contains(pat, regex=True, na=False)

def accessories_for_model(df, model_code):
    """Return accessories/tires whose 'Category' mentions Accessories/Tire and 'Used On' contains model_code."""
    if not model_code:
        return df.iloc[0:0]
    cat_mask = df[COL_CATEGORY].astype(str).str.contains(ACC_TIRE_PATTERN, na=False)
    used_on_mask = df[COL_USED_ON].astype(str).str.upper().str.contains(model_code.upper(), na=False)
    acc = df[cat_mask & used_on_mask].copy()

    # keep only rows with a usable part id
    has_id_mask = None
    for c in PART_ID_COLS:
        if c in acc.columns:
            acc[c] = acc[c].astype(str).str.strip()
            m = acc[c].astype(bool)
            has_id_mask = m if has_id_mask is None else (has_id_mask | m)
    if has_id_mask is None:
        return acc
    return acc[has_id_mask]

def accessory_choices_payload(acc_df, family_label, model_label,
                              multi=True, required=False, name="accessory_ids",
                              question="Select accessories to add (optional)."):
    choices = []
    choices_with_ids = []
    for _, r in acc_df.iterrows():
        desc = (r.get(COL_DESC) or "").strip()
        if not desc:
            continue
        pid = _first_part_id(r)
        if not pid:
            continue
        choices.append(desc)
        choices_with_ids.append({"id": pid, "label": desc})

    return jsonify({
        "found": True,
        "mode": "questions",
        "category": family_label,
        "model": model_label,
        "required_questions": [{
            "name": name,
            "question": question,
            "choices": choices,
            "choices_with_ids": choices_with_ids,
            "multiple": bool(multi),
            "required": bool(required),
        }]
    })

def _read_accessory_params():
    ids = set()
    qty_map = {}

    # single id for raw
    for raw in request.args.getlist("accessory_id"):
        tok = str(raw).strip()
        if tok:
            ids.add(tok)

    # csv ids
    csv = (request.args.get("accessory_ids") or "").strip()
    if csv:
        for tok in csv.split(","):
            tok = tok.strip()
            if tok:
                ids.add(tok)

    # pid[:qty] tokens
    for raw in request.args.getlist("accessory"):
        tok = str(raw).strip()
        if not tok:
            continue
        if ":" in tok:
            pid, q = tok.split(":", 1)
            pid = pid.strip()
            if pid:
                ids.add(pid)
                try:
                    qty_map[pid] = max(1, int(float(q)))
                except:
                    qty_map[pid] = 1
        else:
            ids.add(tok)

    # desc terms
    desc_terms = [t for t in request.args.getlist("accessory_desc") if str(t).strip()]
    return ids, qty_map, desc_terms

def accessory_lines_from_selection(acc_df, ids, qty_map, desc_terms, default_qty=1):
    lines = []
    if acc_df.empty:
        return lines

    # ID-based selection
    if ids:
        ids_upper = {str(x).strip().upper() for x in ids if str(x).strip()}
        by_id = []
        for c in PART_ID_COLS:
            if c in acc_df.columns:
                sub = acc_df[acc_df[c].astype(str).str.strip().str.upper().isin(ids_upper)]
                if not sub.empty:
                    by_id.append(sub)
        if by_id:
            acc_hit = pd.concat(by_id).drop_duplicates()
            for _, r in acc_hit.iterrows():
                pid = _first_part_id(r)
                desc = (r.get(COL_DESC) or "").strip() or "Accessory"
                qty = max(qty_map.get(pid, default_qty), 1)
                lines.append(_make_line(
                    f"{desc} ({pid})",
                    qty,
                    _price_number(r.get(COL_LIST_PRICE)),
                    part_id=pid
                ))

    # Description-based selection
    if desc_terms:
        for term in desc_terms:
            sub = acc_df[acc_df[COL_DESC].astype(str).str.contains(term, case=False, na=False)]
            if not sub.empty:
                r = sub.iloc[0]
                pid = _first_part_id(r)
                desc = (r.get(COL_DESC) or "").strip() or f"Accessory — {term}"
                lines.append(_make_line(
                    f"{desc} ({pid})",
                    default_qty,
                    _price_number(r.get(COL_LIST_PRICE)),
                    part_id=pid
                ))

    return lines

def _rb_model_tokens(model_code: str):
    s = (model_code or "").upper().strip()
    toks = set()
    if not s:
        return toks
    # exact + no-P variant
    toks.add(s)                             # e.g., RB84P
    toks.add(re.sub(r'P$', '', s))          # e.g., RB84

    # width forms: RB84 / RB 84 / RB-84 / 84
    m = re.search(r'RB\s*0*?(\d+)', s)
    if m:
        w = m.group(1)
        for sep in ["", " ", "-"]:
            toks.add(f"RB{sep}{w}")
        toks.add(w)

    # family catch-all
    toks.add("RB")
    return toks

def _rb_accessories_for_model(df, model_code):
    """
    Return ONLY accessories that explicitly list the exact model in 'Used On'.

    - Strict match: exact model token (uppercased, punctuation/spaces removed).
    - Category must look like Accessories (so base units aren't included).
    - No fuzzy fallbacks like 'RB' or width numbers.
    """
    import re

    if not model_code:
        return df.iloc[0:0]

    def _norm_token(s: str) -> str:
        # keep letters/digits/dots; drop spaces, hyphens, etc.
        return re.sub(r'[^A-Z0-9\.]+', '', str(s or '').upper())

    target = _norm_token(model_code)

    # Accessory-only rows (catch common labels: Accessory/Accessories/Kit, etc.)
    cat = df[COL_CATEGORY].astype(str)
    acc_only = df[
        cat.str.contains(r'\bAccessories?\b', case=False, na=False) |
        cat.str.contains(r'\bAccessory\b',   case=False, na=False) |
        cat.str.contains(r'\bKit\b',         case=False, na=False)
    ].copy()

    if acc_only.empty:
        return acc_only  # nothing to show

    # Keep rows whose 'Used On' explicitly lists the exact model
    used_on_series = acc_only[COL_USED_ON].astype(str)

    def _used_on_has_exact(cell: str) -> bool:
        if not cell.strip():
            return False
        # split on common list delimiters
        parts = re.split(r'[,\;/\|\n]+', cell)
        for p in parts:
            if _norm_token(p) == target:
                return True
        return False

    acc = acc_only[used_on_series.apply(_used_on_has_exact)].copy()

    # Keep only rows with a usable part id
    has_id = None
    for c in PART_ID_COLS:
        if c in acc.columns:
            acc[c] = acc[c].astype(str).str.strip()
            m = acc[c].astype(bool)
            has_id = m if has_id is None else (has_id | m)

    if has_id is None:
        return acc
    return acc[has_id]

def _col_by_letter(df, letter: str):
    """Return the column name at spreadsheet letter (A=1). Safe if missing."""
    letter = letter.strip().upper()
    # Single-letter only is needed here (K/L/M/P/Q/R)
    idx = ord(letter) - ord('A')  # 0-based
    try:
        return df.columns[idx]
    except Exception:
        return None

def _details_from_letters(row, letters):
    """Build 'Header: value' chunks from lettered columns if present."""
    parts = []
    df = row.to_frame().T  # for header lookup
    for L in letters:
        cname = _col_by_letter(df, L)
        if cname:
            val = str(row.get(cname, "") or "").strip()
            if val:
                parts.append(f"{cname}: {val}")
    return " — ".join(parts)


# =========================
# Diagnostics
# =========================
@app.get("/health")
def health():
    try:
        df = load_df()
        bf = df[df[COL_CATEGORY].astype(str).str.contains("BrushFighter", case=False, na=False)]
        bb = df[df[COL_CATEGORY].astype(str).str.contains("BrushBull", case=False, na=False)]
        ds = df[df[COL_CATEGORY].astype(str).str.contains("Dual", case=False, na=False)]
        bw = df[df[COL_CATEGORY].astype(str).str.contains("Batwing", case=False, na=False)]
        return {
            "ok": True,
            "rows": int(len(df)),
            "columns": df.columns.tolist(),
            "brushfighter_rows": int(len(bf)),
            "brushbull_rows": int(len(bb)),
            "dualspindle_rows": int(len(ds)),
            "batwing_rows": int(len(bw)),
            "loaded_at": DF_CACHE["loaded_at"],
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}, 500

@app.get("/reload")
def reload_data():
    try:
        load_df(force=True)
        return {"ok": True, "reloaded_at": DF_CACHE["loaded_at"]}
    except Exception as e:
        return {"ok": False, "error": str(e)}, 500

# =========================
# Dealer Discounts
# =========================
DEALER_DISCOUNTS = {
    "6003250": ("Cain Equipment", 0.21),
    "178035": ("Claiborne County Coop", 0.18),
    "178841": ("Dooley Tractor", 0.23),
    "178275": ("First Choice Farm & Lawn", 0.26),
    "178492": ("Haney Equipment", 0.21),
    "178200": ("Lynch Equipment", 0.21),
    "178644": ("Macon-Trousdale Coop", 0.18),
    "179269": ("Mountain Farm", 0.24),
    "201965": ("Smoky MTN Farmers Coop", 0.23),
    "200781": ("Smoky MTN Farmers Coop", 0.23),
    "201967": ("Smoky MTN Farmers Coop", 0.23),
    "201966": ("Smoky MTN Farmers Coop", 0.23),
    "201939": ("Waters Equipment", 0.23),
    "6295647": ("First Choice Farm & Lawn", 0.26),
    "178097": ("First Choice Farm & Lawn", 0.26),
    "200252": ("First Choice Farm & Lawn", 0.26),
    "178274": ("First Choice Farm & Lawn", 0.26),
    "198061": ("First Choice Farm & Lawn", 0.26),
    "6113621": ("First Choice Farm & Lawn", 0.26),
    "177774": ("Henson Tractor", 0.05),
    "178055": ("J & J Sales", 0.24),
    "178697": ("Oktbbeha County Coop", 0.21),
    "178647": ("Sanford & Sons", 0.24),
    "178223": ("Sid's Trading Co.", 0.22),
    "178023": ("Stateline Turf a& Tractor", 0.24),
    "179492": ("Tennessee Tractor", 0.26),
    "179495": ("Tennessee Tractor", 0.26),
    "179225": ("Tennessee Tractor", 0.26),
    "179491": ("Tennessee Tractor", 0.26),
    "179226": ("Tennessee Tractor", 0.26),
    "179227": ("Tennessee Tractor", 0.26),
    "179494": ("Tennessee Tractor", 0.26),
    "179228": ("Tennessee Tractor", 0.26),
    "179490": ("Tennessee Tractor", 0.26),
    "179493": ("Tennessee Tractor", 0.26),
    "6260420": ("Tupelo Farm & Ranch", 0.22),
    "178409": ("Ayres-Delta Implement", 0.05),
    "205396": ("Triple Crown Equipment", 0.25),
}

@app.get("/dealer-discount")
def get_dealer_discount():
    dealer_number = request.args.get("dealer_number")
    if not dealer_number:
        return {"error": "Missing dealer_number"}, 400
    dealer = DEALER_DISCOUNTS.get(dealer_number)
    if dealer:
        name, discount = dealer
        return {
            "dealer_number": dealer_number,
            "dealer_name": name,
            "discount": discount
        }
    return {"error": f"Dealer number {dealer_number} not found"}, 404
# =========================
# Family utilities
# =========================
def _family_base(df, family_pattern, exclude_turf=False):
    allrows = df[df[COL_CATEGORY].astype(str).str.contains(family_pattern, case=False, na=False)]
    if exclude_turf:
        allrows = allrows[~allrows[COL_MODEL].astype(str).str.upper().str.startswith("TBW")]
        allrows = allrows[~allrows[COL_CATEGORY].astype(str).str.contains("Turf", case=False, na=False)]
    # Exclude accessories/tires for base
    return allrows[~allrows[COL_CATEGORY].astype(str).str.contains(ACC_TIRE_PATTERN, na=False)]

# =========================
# Quote router
# =========================
@app.get("/quote")
def quote():
    df = load_df()
    qtext = getp("q")
    family_param = getp("family").lower()
    model_raw = getp("model")
    width_txt = getp("width_ft")  # generic width

    # Family routing (explicit)
    if family_param in ("brushfighter", "bf"):
        return _quote_brushfighter(df)
    if family_param in ("brushbull", "bb"):
        return _quote_brushbull(df)
    if family_param in ("dual_spindle", "ds", "mds"):
        return _quote_dual_spindle(df)
    if family_param in ("batwing", "bw"):
        return _quote_batwing(df)
    if family_param in ("turf_batwing", "tbw", "turf"):
        return _quote_turf_batwing(df)
    if family_param in ("rear_finish", "rear_discharge", "finish_rear", "rear_mower"):
        return _quote_rear_finish(df)
    if family_param in ("box_scraper", "bs", "box"):
        return _quote_box_scraper(df)
    if family_param in ("grading_scraper", "gs", "grading"):
        return _quote_grading_scraper(df)   # ← add this
    if family_param in ("landscape_rake", "lrs", "rake", "landscape"):
        return _quote_landscape_rake(df)
    if family_param in ("post_hole_digger", "pd", "posthole", "post_hole", "digger"):
        return _quote_post_hole_digger(df)
    if family_param in ("rear_blade", "rb", "blade", "rearblade"):
        return _quote_rear_blade(df)
    if family_param in ("disc_harrow", "dh", "disc_harrows"):
        return _quote_disc_harrow(df)
    if family_param in ("tiller", "tillers", "rotary_tiller", "rotary-tiller", "db", "rt"):
        return _quote_tiller(df)
    if family_param in ("bale_spear", "balespear", "spear", "bale spear"):
        return _quote_bale_spear(df)
    if family_param in ("pallet_fork", "palletfork", "pf", "forks", "fork"):
        return _quote_pallet_fork(df)
    if family_param in ("quick_hitch", "quickhitch", "qh", "tqh"):
        return _quote_quick_hitch(df)
    if family_param in ("stump_grinder", "stump", "stumpgrinder", "tsg"):
        return _quote_stump_grinder(df)







    # Heuristic by model token
    if BF_MODEL_RE.search(model_raw or ""):
        return _quote_brushfighter(df)
    if BB_MODEL_RE.search(model_raw or ""):
        return _quote_brushbull(df)
    if DS_MODEL_RE.search(model_raw or ""):
        return _quote_dual_spindle(df)
    if BW_MODEL_RE.search(model_raw or ""):
        return _quote_batwing(df)
    if TBW_MODEL_RE.search(model_raw or ""):
        return _quote_turf_batwing(df)
    if TK_MODEL_RE.search(model_raw or "") or TKP_MODEL_RE.search(model_raw or "") or RD990X_MODEL_RE.search(model_raw or ""):
        return _quote_rear_finish(df)
    if BS_MODEL_RE.search(model_raw or ""):
        return _quote_box_scraper(df)
    if GS_MODEL_RE.search(model_raw or ""):
        return _quote_grading_scraper(df)
    if LRS_MODEL_RE.search(model_raw or ""):
        return _quote_landscape_rake(df)
    if PD_MODEL_RE.search(model_raw or ""):
        return _quote_post_hole_digger(df)
    if RB_MODEL_RE.search(model_raw or ""):
        return _quote_rear_blade(df)
    if DH_MODEL_RE.search(model_raw or ""):
        return _quote_disc_harrow(df)
    if DB_MODEL_RE.search(model_raw or "") or RT_MODEL_RE.search(model_raw or ""):
        return _quote_tiller(df)
    if re.search(r"\bspear\b", (qtext or ""), re.I):
        return _quote_bale_spear(df)
    if re.search(r"\b(pallet\s*fork|forks?)\b", (qtext or ""), re.I) or (model_raw or "").strip().upper().startswith("PF"):
        return _quote_pallet_fork(df)
    if re.search(r"\bquick\s*hitch\b", (qtext or ""), re.I) or TQH_MODEL_RE.search(model_raw or ""):
        return _quote_quick_hitch(df)
    if TSG_MODEL_RE.search(model_raw or "") or re.search(r"\bstump\s*grind", (qtext or ""), re.I):
        return _quote_stump_grinder(df)












    # Fallback: if query mentions "cutter"/"bush hog", ask family at width (if width known)
    GENERIC_CUTTER_RE = re.compile(r"\b(cutter|bush\s*hogg?)\b", re.I)
    if GENERIC_CUTTER_RE.search((qtext or "")) or family_param in {"cutter", "bushhog", "bush hog", "bush_hog"}:
        norm_ft = _normalize_width_ft_input(width_txt) if width_txt else None
        if not norm_ft:
            fam_choices = [
                {"id": "brushfighter", "label": "BrushFighter"},
                {"id": "brushbull", "label": "BrushBull"},
                {"id": "dual_spindle", "label": "Dual Spindle"},
                {"id": "batwing", "label": "Batwing"},
            ]
            return jsonify({
                "found": True,
                "mode": "questions",
                "category": "Cutter",
                "model": "",
                "required_questions": [{
                    "name": "family_choice",
                    "question": "Which family do you want?",
                    "choices": [c["label"] for c in fam_choices],
                    "choices_with_ids": fam_choices
                }]
            })

        # width known → offer families with SKUs at that width
        fams = []
        fam_ids = []
        for fid, pat in {"brushfighter": "BrushFighter", "brushbull": "BrushBull", "dual_spindle": "Dual", "batwing": "Batwing"}.items():
            base = _family_base(df, pat, exclude_turf=(fid == "batwing"))
            if base.empty:
                continue
            wn = _ensure_width_norm_col(base)
            at_w = base[base[wn].astype(str) == norm_ft]
            if not at_w.empty:
                models = sorted({str(m).strip() for m in at_w[COL_MODEL].tolist() if str(m).strip()})
                sample = " / ".join(models[:3])
                label = f"{fid.replace('_',' ').title()} — {sample}" if sample else fid.replace('_',' ').title()
                fams.append(label)
                fam_ids.append({"id": fid, "label": label})
        if fams:
            return jsonify({
                "found": True,
                "mode": "questions",
                "category": "Cutter",
                "model": f"{norm_ft} ft",
                "required_questions": [{
                    "name": "family_choice",
                    "question": f"Which family at {norm_ft} ft?",
                    "choices": fams,
                    "choices_with_ids": fam_ids
                }]
            })
        return jsonify({"found": False, "mode": "error", "message": f"No cutter families have {norm_ft} ft models."}), 404

    # Default to BrushFighter
    return _quote_brushfighter(df)
# =========================
# BrushFighter
# =========================
def _bf_label(r, include_desc=False):
    base = str(r[COL_MODEL])
    drv = str(r.get(COL_DRIVE, "") or "").strip()
    lbl = f"{base} — {drv}" if drv else base
    if include_desc:
        dsc = str(r.get(COL_DESC, "") or "").strip()
        if dsc:
            lbl += f" — {dsc}"
    return lbl

def _quote_brushfighter(df):
    bf_base = _family_base(df, "BrushFighter")
    if bf_base.empty:
        return jsonify({"found": False, "mode": "error", "message": "No BrushFighter base rows found."}), 404

    # Dealer
    dealer_number = getp("dealer_number")
    dealer_meta, dealer_rate_override = None, None
    if dealer_number and dealer_number in DEALER_DISCOUNTS:
        name, rate = DEALER_DISCOUNTS[dealer_number]
        dealer_meta = {"dealer_number": dealer_number, "dealer_name": name, "applied_discount": rate}
        dealer_rate_override = rate

    model_code = _extract_bf_code(getp("model"))
    width_txt = getp("width_ft")
    part_param = getp("bf_choice_id", "accessory_id")  # allow direct part id too
    choice_id = getp("bf_choice_id")
    choice_label = getp("bf_choice")
    drive_hint = getp("drive").lower()

    # accessories selection
    list_access = (request.args.get("list_accessories") or "").strip() not in ("", "0", "false", "False")
    acc_ids, acc_qty_map, acc_desc_terms = _read_accessory_params()

    # direct by part id if passed
    if part_param and not model_code:
        hit = _find_by_part_id(bf_base, part_param)
        r, err = pick_one_or_400(hit, "BrushFighter part id", {"part_id": part_param})
        if err:
            return err
        model = str(r[COL_MODEL]).strip()
        pid = _first_part_id(r)
        label = _bf_label(r, include_desc=True) + (f" ({pid})" if pid else "")
        lines = [_make_line(label, 1, _price_number(r.get(COL_LIST_PRICE)), part_id=pid)]

        acc_df = accessories_for_model(df, model)
        if list_access and not acc_ids and not acc_desc_terms:
            return accessory_choices_payload(acc_df, "BrushFighter", model)
        lines += accessory_lines_from_selection(acc_df, acc_ids, acc_qty_map, acc_desc_terms)

        return _totals_payload(
            lines,
            ["Comes standard with Belt Shielding; Optional Chain Shielding available."],
            "BrushFighter",
            model,
            dealer_rate_override,
            dealer_meta,
        )

    wn = _ensure_width_norm_col(bf_base)

    if model_code.startswith("BF"):
        rows = bf_base[bf_base[COL_MODEL].astype(str).str.upper() == model_code.upper()]
    else:
        if not width_txt:
            sizes = sorted({s for s in bf_base[wn].astype(str).tolist() if s}, key=lambda x: int(x))
            return jsonify({
                "found": True,
                "mode": "questions",
                "category": "BrushFighter",
                "model": "",
                "required_questions": [{
                    "name": "width_ft",
                    "question": "What size BrushFighter do you want (in feet)?",
                    "choices": sizes
                }]
            })
        norm_ft = _normalize_width_ft_input(width_txt)
        rows = bf_base[bf_base[wn].astype(str) == (norm_ft or "")]

    if drive_hint and not rows.empty:
        rows = rows[rows[COL_DRIVE].astype(str).str.lower().str.contains(drive_hint)]

    if rows.empty:
        return jsonify({"found": False, "mode": "error", "message": "No BrushFighter rows matched model/width."}), 404

    # multiple matches → ask choice
    if len(rows.index) > 1 and not (choice_id or choice_label):
        str_choices, id_choices = [], []
        for _, r in rows.iterrows():
            lbl = _bf_label(r)
            rid = _first_part_id(r) or lbl
            str_choices.append(lbl)
            id_choices.append({"id": rid, "label": lbl})
        return jsonify({
            "found": True,
            "mode": "questions",
            "category": "BrushFighter",
            "model": model_code or (f"{width_txt} BrushFighter" if width_txt else ""),
            "required_questions": [{
                "name": "bf_choice",
                "question": "Which BrushFighter configuration would you like?",
                "choices": str_choices,
                "choices_with_ids": id_choices
            }]
        })

    # finalize selection
    chosen = rows
    if choice_id or choice_label:
        chosen = _select_by_id_or_label(rows, choice_id, choice_label, label_fn=_bf_label)

    r, err = pick_one_or_400(chosen, "BrushFighter final selection", {"choice_id": choice_id, "choice_label": choice_label})
    if err:
        return err

    model = str(r[COL_MODEL]).strip()
    pid = _first_part_id(r)
    label = _bf_label(r, include_desc=True) + (f" ({pid})" if pid else "")

    acc_df = accessories_for_model(df, model)
    if list_access and not acc_ids and not acc_desc_terms:
        return accessory_choices_payload(acc_df, "BrushFighter", model)

    lines = [_make_line(label, 1, _price_number(r.get(COL_LIST_PRICE)), part_id=pid)]
    lines += accessory_lines_from_selection(acc_df, acc_ids, acc_qty_map, acc_desc_terms)

    return _totals_payload(
        lines,
        ["Comes standard with Belt Shielding; Optional Chain Shielding available."],
        "BrushFighter",
        model,
        dealer_rate_override,
        dealer_meta,
    )
# =========================
# BrushBull
# =========================
def _bb_duty_sort_key(label):
    order = {
        "Standard Duty (.30)": 1,
        "Standard Plus (.40)": 2,
        "Heavy Duty (.50)": 3,
        "Extreme Duty (.60)": 4,
    }
    return order.get(label, 99)

def _bb_duty_label_from_row(r):
    """Return standardized BrushBull duty label. Handles D-suffix and maps sheet synonyms."""
    m = str(r.get(COL_MODEL, "") or "").upper().strip()
    m_noD = re.sub(r'D$', '', m)
    if m_noD.endswith(".30"):
        return "Standard Duty (.30)"
    if m_noD.endswith(".40"):
        return "Standard Plus (.40)"
    if m_noD.endswith(".50"):
        return "Heavy Duty (.50)"
    if m_noD.endswith(".60"):
        return "Extreme Duty (.60)"

    # Fallback to Duty Class text, map to standard four labels
    dc = (str(r.get(COL_DUTY_CLASS, "") or "")).strip().lower()
    if not dc:
        return ""
    if "standard plus" in dc or "deluxe" in dc:
        return "Standard Plus (.40)"
    if "premium" in dc or "heavy" in dc:
        return "Heavy Duty (.50)"
    if "extreme" in dc:
        return "Extreme Duty (.60)"
    if "standard" in dc:
        return "Standard Duty (.30)"
    return dc.title()

def _bb_norm_shielding(val):
    s = (val or "").strip().lower()
    if "chain" in s:
        return "Chain"
    if "belt" in s:
        return "Belt"
    return (val or "").strip()

def _bb_tailwheel_norm(val):
    s = (val or "").strip().lower()
    if "dual" in s:
        return "Dual"
    if "single" in s:
        return "Single"
    return ""  # unknown/blank

def _bb_model_has_dual_code(model_str):
    s = (model_str or "").upper().strip()
    return bool(re.search(r"D\b", s)) or s.endswith("D")

def _bb_label(r, include_desc=False, force_tailwheel=None):
    model = str(r[COL_MODEL])
    duty = _bb_duty_label_from_row(r)
    sh = _bb_norm_shielding(r.get(COL_SHIELDING, ""))
    tw = _bb_tailwheel_norm(r.get(COL_TAILWHEEL, "")) or ("Dual" if _bb_model_has_dual_code(model) else "")
    if force_tailwheel:
        tw = force_tailwheel

    bits = []
    if duty:
        bits.append(duty)
    if sh:
        bits.append(f"{sh} shielding")
    if tw:
        bits.append(f"{tw} tailwheel")

    label = f"{model} — {', '.join(bits)}" if bits else model
    if include_desc:
        dsc = str(r.get(COL_DESC, "") or "").strip()
        if dsc:
            label += f" — {dsc}"
    return label

def _quote_brushbull(df):
    bb_base = _family_base(df, "BrushBull")
    if bb_base.empty:
        return jsonify({"found": False, "mode": "error", "message": "No BrushBull base rows found."}), 404

    # Dealer
    dealer_number = getp("dealer_number")
    dealer_meta, dealer_rate_override = None, None
    if dealer_number and dealer_number in DEALER_DISCOUNTS:
        name, rate = DEALER_DISCOUNTS[dealer_number]
        dealer_meta = {"dealer_number": dealer_number, "dealer_name": name, "applied_discount": rate}
        dealer_rate_override = rate

    model_code = _extract_bb_code(getp("model"))
    width_txt = getp("width_ft")
    choice_id = getp("bb_choice_id", "accessory_id")
    choice_label = getp("bb_choice")
    duty_param = getp("bb_duty")
    shielding_param = getp("bb_shielding")
    tailwheel_param = getp("bb_tailwheel")

    list_access = (request.args.get("list_accessories") or "").strip() not in ("", "0", "false", "False")
    acc_ids, acc_qty_map, acc_desc_terms = _read_accessory_params()

    wn = _ensure_width_norm_col(bb_base)

    # MODEL-FIRST with D-sibling awareness
    if model_code.startswith("BB"):
        base_noD = re.sub(r'D$', '', model_code.upper())
        rows = bb_base[bb_base[COL_MODEL].astype(str).str.upper().isin([base_noD, base_noD + "D"])]
        if rows.empty:
            return jsonify({"found": False, "mode": "error", "message": f"BrushBull model not found: {model_code}"}), 404

        # Shielding question if both present
        sh_set = sorted({_bb_norm_shielding(v) for v in rows[COL_SHIELDING].astype(str).tolist() if v})
        if len(sh_set) > 1 and not shielding_param:
            return jsonify({
                "found": True,
                "mode": "questions",
                "category": "BrushBull",
                "model": model_code,
                "required_questions": [{
                    "name": "bb_shielding",
                    "question": f"Which shielding for {model_code}?",
                    "choices": sh_set
                }]
            })
        if shielding_param:
            norm_sh = _bb_norm_shielding(shielding_param)
            rows = rows[rows[COL_SHIELDING].astype(str).str.lower().str.contains(norm_sh.lower())]
            if rows.empty:
                return jsonify({"found": False, "mode": "error", "message": f"No {norm_sh} shielding row found for {model_code}"}), 404

        # Tailwheel if both exist (and add synthetic Dual if D-sibling exists)
        tw_set = sorted({_bb_tailwheel_norm(v) for v in rows[COL_TAILWHEEL].astype(str).tolist() if v})
        has_dual_code = any(_bb_model_has_dual_code(m) for m in rows[COL_MODEL].astype(str))
        if has_dual_code and "Dual" not in tw_set:
            tw_set.append("Dual")
        tw_set = [x for x in tw_set if x]

        if len(set(tw_set)) > 1 and not tailwheel_param:
            return jsonify({
                "found": True,
                "mode": "questions",
                "category": "BrushBull",
                "model": model_code,
                "required_questions": [{
                    "name": "bb_tailwheel",
                    "question": f"Single or Dual tailwheel for {model_code}?",
                    "choices": sorted(set(tw_set))
                }]
            })
        if tailwheel_param:
            tw = tailwheel_param.strip().lower()
            if "dual" in tw:
                rows = rows[
                    rows[COL_TAILWHEEL].astype(str).str.contains("dual", case=False, na=False) |
                    rows[COL_MODEL].astype(str).str.upper().str.endswith("D")
                ]
            elif "single" in tw:
                rows = rows[
                    rows[COL_TAILWHEEL].astype(str).str.contains("single", case=False, na=False) |
                    (~rows[COL_MODEL].astype(str).str.upper().str.endswith("D"))
                ]
            if rows.empty:
                return jsonify({"found": False, "mode": "error", "message": f"No {tailwheel_param} tailwheel row for {model_code}"}), 404
    else:
        # WIDTH-FIRST
        if not width_txt:
            sizes = sorted({s for s in bb_base[wn].astype(str).tolist() if s}, key=lambda x: int(x))
            return jsonify({
                "found": True,
                "mode": "questions",
                "category": "BrushBull",
                "model": "",
                "required_questions": [{
                    "name": "width_ft",
                    "question": "What size BrushBull do you want (in feet)?",
                    "choices": sizes
                }]
            })
        norm_ft = _normalize_width_ft_input(width_txt)
        rows = bb_base[bb_base[wn].astype(str) == norm_ft]
        if rows.empty:
            return jsonify({"found": False, "mode": "error", "message": f"No BrushBull rows at {norm_ft} ft."}), 404

        # duties that exist at this width
        available_duties = sorted(
            {_bb_duty_label_from_row(r) for _, r in rows.iterrows() if _bb_duty_label_from_row(r)},
            key=_bb_duty_sort_key
        )
        if len(available_duties) > 1 and not duty_param:
            return jsonify({
                "found": True,
                "mode": "questions",
                "category": "BrushBull",
                "model": f"{norm_ft} ft BrushBull",
                "required_questions": [{
                    "name": "bb_duty",
                    "question": f"Which duty class for {norm_ft} ft BrushBull?",
                    "choices": available_duties
                }]
            })
        if duty_param:
            dp = duty_param.strip().lower()
            rows = rows[rows.apply(lambda r: _bb_duty_label_from_row(r).strip().lower() == dp, axis=1)]
            if rows.empty:
                return jsonify({"found": False, "mode": "error", "message": f"No BrushBull rows for {norm_ft} ft with duty '{duty_param}'."}), 404

        # shielding choice
        sh_set = sorted({_bb_norm_shielding(v) for v in rows[COL_SHIELDING].astype(str).tolist() if v})
        if len(sh_set) > 1 and not shielding_param:
            return jsonify({
                "found": True,
                "mode": "questions",
                "category": "BrushBull",
                "model": f"{norm_ft} ft BrushBull",
                "required_questions": [{
                    "name": "bb_shielding",
                    "question": f"Belt or Chain shielding for {norm_ft} ft BrushBull?",
                    "choices": sh_set
                }]
            })
        if shielding_param:
            norm_sh = _bb_norm_shielding(shielding_param)
            rows = rows[rows[COL_SHIELDING].astype(str).str.lower().str.contains(norm_sh.lower())]
            if rows.empty:
                return jsonify({"found": False, "mode": "error", "message": f"No {norm_sh} shielding rows for {norm_ft} ft BrushBull."}), 404

        # tailwheel choice
        tw_set = sorted({_bb_tailwheel_norm(v) for v in rows[COL_TAILWHEEL].astype(str).tolist() if v})
        has_dual_code = any(_bb_model_has_dual_code(m) for m in rows[COL_MODEL].astype(str))
        if has_dual_code and "Dual" not in tw_set:
            tw_set.append("Dual")
        tw_set = [x for x in tw_set if x]

        if len(set(tw_set)) > 1 and not tailwheel_param:
            return jsonify({
                "found": True,
                "mode": "questions",
                "category": "BrushBull",
                "model": f"{norm_ft} ft BrushBull",
                "required_questions": [{
                    "name": "bb_tailwheel",
                    "question": f"Single or Dual tailwheel for {norm_ft} ft BrushBull?",
                    "choices": sorted(set(tw_set))
                }]
            })
        if tailwheel_param:
            tw = tailwheel_param.strip().lower()
            if "dual" in tw:
                rows = rows[
                    rows[COL_TAILWHEEL].astype(str).str.contains("dual", case=False, na=False) |
                    rows[COL_MODEL].astype(str).str.upper().str.endswith("D")
                ]
            elif "single" in tw:
                rows = rows[
                    rows[COL_TAILWHEEL].astype(str).str.contains("single", case=False, na=False) |
                    (~rows[COL_MODEL].astype(str).str.upper().str.endswith("D"))
                ]
            if rows.empty:
                return jsonify({"found": False, "mode": "error", "message": f"No {tailwheel_param} tailwheel rows for {norm_ft} ft BrushBull."}), 404

    # Final pick or ask exact
    if len(rows.index) > 1 and not (choice_id or choice_label):
        str_choices, id_choices = [], []
        for _, r in rows.iterrows():
            lbl = _bb_label(r)
            rid = _first_part_id(r) or lbl
            str_choices.append(lbl)
            id_choices.append({"id": rid, "label": lbl})
        return jsonify({
            "found": True,
            "mode": "questions",
            "category": "BrushBull",
            "model": model_code or (width_txt and f"{width_txt} BrushBull") or "",
            "required_questions": [{
                "name": "bb_choice",
                "question": "Which BrushBull configuration would you like?",
                "choices": str_choices,
                "choices_with_ids": id_choices
            }]
        })

    chosen = rows
    if choice_id or choice_label:
        chosen = _select_by_id_or_label(rows, choice_id, choice_label, label_fn=_bb_label)

    r, err = pick_one_or_400(chosen, "BrushBull final selection", {"choice_id": choice_id, "choice_label": choice_label})
    if err:
        return err

    model = str(r[COL_MODEL]).strip()
    pid = _first_part_id(r)

    force_tw = None
    if tailwheel_param:
        twn = _bb_tailwheel_norm(tailwheel_param)
        if twn:
            force_tw = twn

    label = _bb_label(r, include_desc=True, force_tailwheel=force_tw) + (f" ({pid})" if pid else "")
    acc_df = accessories_for_model(df, model)
    if list_access and not acc_ids and not acc_desc_terms:
        return accessory_choices_payload(acc_df, "BrushBull", model)

    lines = [_make_line(label, 1, _price_number(r.get(COL_LIST_PRICE)), part_id=pid)]
    lines += accessory_lines_from_selection(acc_df, acc_ids, acc_qty_map, acc_desc_terms)

    return _totals_payload(
        lines,
        ["Duty, shielding, and tailwheel as selected."],
        "BrushBull",
        model,
        dealer_rate_override,
        dealer_meta,
    )
# =========================
# Dual Spindle (MDS / DS / DSO)
# =========================
def _quote_dual_spindle(df):
    import re

    ds_base = _family_base(df, "Dual", exclude_turf=False)
    if ds_base.empty:
        return jsonify({"found": False, "mode": "error", "message": "No Dual Spindle base rows found."}), 404

    # --- Dealer context ------------------------------------------------------
    dealer_number = getp("dealer_number")
    dealer_meta, dealer_rate_override = None, None
    if dealer_number and dealer_number in DEALER_DISCOUNTS:
        name, rate = DEALER_DISCOUNTS[dealer_number]
        dealer_meta = {"dealer_number": dealer_number, "dealer_name": name, "applied_discount": rate}
        dealer_rate_override = rate

    # --- Inputs (null-safe) --------------------------------------------------
    raw_model   = (getp("model") or "").strip()
    model_code  = _extract_ds_code(raw_model)              # e.g., DS10.50Q / MDS8.30 / DSO10.50 / DS8
    width_txt   = (getp("width_ft") or "").strip()
    ds_offset   = (getp("ds_offset") or "").strip().lower()   # "yes" / "no"
    ds_mount    = (getp("ds_mount") or "").strip().lower()    # "mounted" / "pull" / "ds" / "mds"
    ds_duty     = (getp("ds_duty") or "").strip()             # "Standard Duty (.30/.40)" or "Heavy Duty (.50)"
    shielding   = (getp("ds_shielding") or "").strip()
    ds_drive    = (getp("ds_driveline") or "").strip()        # "540" / "1000"
    tire_id     = getp("tire_id")
    tire_choice = getp("tire_choice")

    # Normalized width column NAME (use df[wn], not wn.astype)
    wn_base = _ensure_width_norm_col(ds_base)

    # --- Derive / confirm width ---------------------------------------------
    width_val = ""
    if width_txt:
        width_val = width_txt
    if not width_val and model_code:
        # Prefer explicit digits in model like DS8 / DS10
        m_digit = re.search(r'(?:^|[^\d])(8|10)(?:[^\d]|$)', model_code)
        if m_digit:
            width_val = m_digit.group(1)
    if not width_val and model_code:
        # Try exact model row first
        m_exact = ds_base[ds_base[COL_MODEL].astype(str).str.upper().eq(model_code.upper())]
        if not m_exact.empty:
            wn_exact = _ensure_width_norm_col(m_exact)
            width_val = str(m_exact[wn_exact].iloc[0]).strip()
        else:
            # Fallback to family prefix rows (DS/MDS/DSO) for a width hint
            pref = model_code[:3].upper()
            m_pref = ds_base[ds_base[COL_MODEL].astype(str).str.upper().str.startswith(pref)]
            if not m_pref.empty:
                wn_pref = _ensure_width_norm_col(m_pref)
                width_val = str(m_pref[wn_pref].iloc[0]).strip()
    width_val = _normalize_width_ft_input(width_val) if width_val else ""

    # --- Offset: skip the question if model is explicit ----------------------
    if model_code.startswith("DSO"):
        ds_offset = "yes"
    elif model_code.startswith(("DS", "MDS")):
        ds_offset = "no"

    # If no model was given, we need width (8/10) first
    if not model_code:
        if not width_val:
            widths_avail = sorted({str(v).strip() for v in ds_base[wn_base].astype(str) if str(v).strip()})
            if len(widths_avail) > 1:
                return jsonify({
                    "found": True,
                    "mode": "questions",
                    "category": "Dual Spindle",
                    "model": "Dual Spindle",
                    "required_questions": [{
                        "name": "width_ft",
                        "question": "What cutting width?",
                        "choices": widths_avail
                    }]
                })
            elif widths_avail:
                width_val = widths_avail[0]

        # After width, ask offset only if not already decided
        if ds_offset not in ("yes", "no"):
            return jsonify({
                "found": True,
                "mode": "questions",
                "category": "Dual Spindle",
                "model": f"{width_val} ft Dual Spindle",
                "required_questions": [{
                    "name": "ds_offset",
                    "question": "Is this an offset cutter?",
                    "choices": ["Yes", "No"]
                }]
            })

    # --- OFFSET PATH ---------------------------------------------------------
    if ds_offset == "yes":
        m_dso = ds_base[COL_MODEL].astype(str).str.upper().str.startswith("DSO")
        m_w   = ds_base[wn_base].astype(str) == width_val
        rows  = ds_base[(m_dso) & (m_w)]
        if rows.empty:
            return jsonify({"found": False, "mode": "error", "message": f"No Offset (DSO) rows found for {width_val} ft."}), 404

        # If a concrete DSO model was provided, narrow to it (±Q)
        if model_code.startswith("DSO"):
            base_noQ = re.sub(r'Q$', '', model_code.upper())
            is_model = rows[COL_MODEL].astype(str).str.upper().isin([base_noQ, base_noQ + "Q"])
            rows = rows[is_model] if is_model.any() else rows

        r, err = pick_one_or_400(rows, "Dual Spindle Offset final selection", {"width_ft": width_val, "offset": "yes"})
        if err:
            return err

        model = str(r[COL_MODEL]).strip()
        pid   = _first_part_id(r)
        label = f"{model} — Offset Cutter (includes 2 laminated tires)" + (f" ({pid})" if pid else "")
        lines = [_make_line(label, 1, _price_number(r.get(COL_LIST_PRICE)), part_id=pid)]

        # Accessories optional
        acc_df = accessories_for_model(df, model)
        list_access = (request.args.get("list_accessories") or "").strip() not in ("", "0", "false", "False")
        acc_ids, acc_qty_map, acc_desc_terms = _read_accessory_params()
        if list_access and not acc_ids and not acc_desc_terms:
            return accessory_choices_payload(acc_df, "Dual Spindle (Offset)", model)
        lines += accessory_lines_from_selection(acc_df, acc_ids, acc_qty_map, acc_desc_terms)

        return _totals_payload(
            lines,
            ["Offset model includes 2 laminated tires (no tire selection required)."],
            "Dual Spindle (Offset)",
            model,
            dealer_rate_override,
            dealer_meta,
        )

    # --- NON-OFFSET PATH (standard DS/MDS) ----------------------------------
    if not ds_mount:
        return jsonify({
            "found": True,
            "mode": "questions",
            "category": "Dual Spindle",
            "model": f"{width_val} ft Dual Spindle",
            "required_questions": [{
                "name": "ds_mount",
                "question": "Mounted (MDS) or Pull-type (DS)?",
                "choices": ["Mounted (MDS)", "Pull-type (DS)"]
            }]
        })

    mount_is_mds = ("mds" in ds_mount) or ("mount" in ds_mount)
    mount_is_ds  = (("ds" in ds_mount) and ("mds" not in ds_mount)) or ("pull" in ds_mount)

    # Filter by mount (exclude DSO when DS) and by width
    if mount_is_mds:
        mask_mds = ds_base[COL_MODEL].astype(str).str.upper().str.startswith("MDS")
        rows = ds_base[mask_mds]
    else:
        m_ds  = ds_base[COL_MODEL].astype(str).str.upper().str.startswith("DS")
        m_dso = ds_base[COL_MODEL].astype(str).str.upper().str.startswith("DSO")
        rows  = ds_base[(m_ds) & (~m_dso)]
    wn_rows = _ensure_width_norm_col(rows)
    rows = rows[rows[wn_rows].astype(str) == width_val]

    # If a specific non-DSO model was provided, only narrow when it contains a duty suffix (.30/.40/.50 ±Q).
    if model_code.startswith(("MDS", "DS")) and not model_code.startswith("DSO"):
        m = re.search(r"\.(30|40|50)(Q)?$", model_code, re.IGNORECASE)
        if m:
            base_noQ = re.sub(r'Q$', '', model_code.upper())
            is_model = rows[COL_MODEL].astype(str).str.upper().isin([base_noQ, base_noQ + "Q"])
            if is_model.any():
                rows = rows[is_model]

    if rows.empty:
        return jsonify({"found": False, "mode": "error", "message": f"No Dual Spindle rows match {width_val} ft and mount {('MDS' if mount_is_mds else 'DS')}."}), 404

    # --- DUTY by width -------------------------------------------------------
    # 8 ft: .30 Standard, .50 Heavy
    # 10 ft: .40 Standard, .50 Heavy
    want_token = None
    m = re.search(r"\.(30|40|50)(Q)?$", model_code, re.IGNORECASE)
    if m:
        want_token = m.group(1)

    if want_token:
        rows = rows[rows[COL_MODEL].astype(str).str.contains(rf"\.{want_token}(?:Q)?$", regex=True)]
        if want_token == "30": ds_duty = "Standard Duty (.30)"
        elif want_token == "40": ds_duty = "Standard Duty (.40)"
        elif want_token == "50": ds_duty = "Heavy Duty (.50)"
    else:
        duty_choices = []
        if width_val == "8":
            duty_choices = ["Standard Duty (.30)", "Heavy Duty (.50)"]
        elif width_val == "10":
            duty_choices = ["Standard Duty (.40)", "Heavy Duty (.50)"]

        if duty_choices:
            if not ds_duty:
                return jsonify({
                    "found": True,
                    "mode": "questions",
                    "category": "Dual Spindle",
                    "model": f"{width_val} ft Dual Spindle — {'MDS' if mount_is_mds else 'DS'}",
                    "required_questions": [{
                        "name": "ds_duty",
                        "question": "Which duty class?",
                        "choices": duty_choices  # already smallest → largest
                    }]
                })
            chosen = ds_duty.lower()
            duty_token = "30" if ".30" in chosen else ("40" if ".40" in chosen else ("50" if ".50" in chosen else None))
            if duty_token:
                rows = rows[rows[COL_MODEL].astype(str).str.contains(rf"\.{duty_token}(?:Q)?$", regex=True)]
            if rows.empty:
                return jsonify({"found": False, "mode": "error", "message": f"No Dual Spindle rows match duty '{ds_duty}'."}), 404

    # --- DRIVELINE (540/1000) only if real choice ---------------------------
    ends_q = rows[COL_MODEL].astype(str).str.upper().str.endswith("Q")
    has_q  = bool(ends_q.any())
    if not has_q:
        ds_drive = "540"
    elif not ds_drive:
        return jsonify({
            "found": True,
            "mode": "questions",
            "category": "Dual Spindle",
            "model": f"{width_val} ft Dual Spindle — {'MDS' if mount_is_mds else 'DS'}",
            "required_questions": [{
                "name": "ds_driveline",
                "question": "Which driveline?",
                "choices": ["540", "1000"]
            }]
        })

    if ds_drive == "1000":
        rows = rows[ends_q]
    elif ds_drive == "540":
        rows = rows[~ends_q]
    # else: leave rows unchanged if driveline not applicable

    if rows.empty:
        return jsonify({"found": False, "mode": "error", "message": "No Dual Spindle rows after driveline selection."}), 404

    # --- SHIELDING (Belt/Chain) only if real choice --------------------------
    sh_set = []
    if COL_SHIELDING in rows.columns:
        sh_col = rows[COL_SHIELDING].astype(str)
        sh_set = sorted({v.strip() for v in sh_col.tolist() if v})
    if sh_set and len(sh_set) > 1 and not shielding:
        order = {"belt": 0, "chain": 1}
        sh_set = sorted(sh_set, key=lambda s: order.get(s.lower(), 99))
        return jsonify({
            "found": True,
            "mode": "questions",
            "category": "Dual Spindle",
            "model": f"{width_val} ft Dual Spindle — {'MDS' if mount_is_mds else 'DS'}",
            "required_questions": [{
                "name": "ds_shielding",
                "question": "Belt or Chain shielding?",
                "choices": sh_set
            }]
        })
    if shielding and COL_SHIELDING in rows.columns:
        rows = rows[rows[COL_SHIELDING].astype(str).str.lower().str.contains(shielding.lower())]
        if rows.empty:
            return jsonify({"found": False, "mode": "error", "message": f"No Dual Spindle rows with shielding '{shielding}'."}), 404

    # --- PICK THE UNIT -------------------------------------------------------
    r, err = pick_one_or_400(
        rows,
        "Dual Spindle final selection",
        {
            "width_ft": width_val,
            "offset": "no",
            "ds_mount": ("MDS" if mount_is_mds else "DS"),
            "duty": ds_duty,
            "shielding": shielding or (sh_set[0] if sh_set and len(sh_set) == 1 else ""),
            "driveline": ds_drive or "540",
        }
    )
    if err:
        return err

    model = str(r[COL_MODEL]).strip()
    pid   = _first_part_id(r)
    label_bits = [model]
    if ds_duty:
        label_bits.append(ds_duty)  # optional: drop if redundant with .30/.40/.50 in model
    if shielding:
        label_bits.append(f"Shielding: {shielding}")
    if ds_drive:
        label_bits.append(f"{ds_drive} RPM")
    label = " — ".join([b for b in label_bits if b]) + (f" ({pid})" if pid else "")

    lines = [_make_line(label, 1, _price_number(r.get(COL_LIST_PRICE)), part_id=pid)]

    # --- TIRES ---------------------------------------------------------------
    notes = ["Mounted (MDS) and Offset (DSO) include 2 laminated tires; Pull-type (DS) requires tires (quoted as accessories, qty 2)."]
    mount_is_mds_final = model.upper().startswith("MDS")
    if not mount_is_mds_final:  # DS pull-type
        acc_df = accessories_for_model(df, model)
        tire_opts = acc_df[acc_df[COL_CATEGORY].astype(str).str.contains("Tire", case=False, na=False)]
        if tire_opts is None or tire_opts.empty:
            notes.append("No tire options were listed for this DS model; dealer may need to supply tires.")
        else:
            if not (tire_id or tire_choice):
                choices = []
                choices_with_ids = []
                for _, tr in tire_opts.iterrows():
                    desc = (tr.get(COL_DESC) or "").strip()
                    pidt = _first_part_id(tr)
                    if desc and pidt:
                        choices.append(desc)
                        choices_with_ids.append({"id": pidt, "label": desc})
                return jsonify({
                    "found": True,
                    "mode": "questions",
                    "category": "Dual Spindle",
                    "model": model,
                    "required_questions": [{
                        "name": "tire_id",
                        "question": "Which tire option would you like (DS requires two tires)?",
                        "choices": choices,
                        "choices_with_ids": choices_with_ids,
                        "multiple": False,
                        "required": True
                    }]
                })
            # add selected tires (qty 2)
            tire_sel = None
            if tire_id:
                tire_sel = _find_by_part_id(tire_opts, tire_id)
            if (tire_sel is None or tire_sel.empty) and tire_choice:
                tire_sel = tire_opts[tire_opts[COL_DESC].astype(str).str.contains(tire_choice, case=False, na=False)]
            if tire_sel is not None and not tire_sel.empty:
                tr = tire_sel.iloc[0]
                tpid = _first_part_id(tr)
                tdesc = (tr.get(COL_DESC) or "").strip() or "Tires"
                lines.append(_make_line(f"{tdesc} ({tpid})", 2, _price_number(tr.get(COL_LIST_PRICE)), part_id=tpid))

    # --- ACCESSORIES (optional) ----------------------------------------------
    acc_df = accessories_for_model(df, model)
    list_access = (request.args.get("list_accessories") or "").strip() not in ("", "0", "false", "False")
    acc_ids, acc_qty_map, acc_desc_terms = _read_accessory_params()
    if list_access and not acc_ids and not acc_desc_terms:
        return accessory_choices_payload(acc_df, "Dual Spindle", model)
    lines += accessory_lines_from_selection(acc_df, acc_ids, acc_qty_map, acc_desc_terms)

    # --- TOTALS --------------------------------------------------------------
    return _totals_payload(
        lines,
        notes,
        "Dual Spindle",
        model,
        dealer_rate_override,
        dealer_meta,
    )


# =========================
# Batwing
# =========================
def _bw_duty_from_model(model: str, width_ft: str):
    """Return standardized duty label for Batwing based on model suffix and width."""
    m = (model or "").upper()
    if width_ft == "12":
        return "Standard Duty (.40)"  # BW12.40 typical single class
    if width_ft == "15":
        if m.endswith(".52") or m.endswith("52R") or m.endswith("52Q") or m.endswith("52RQ"):
            return "Standard Duty (.52)"
        if m.endswith(".72") or m.endswith("72R") or m.endswith("72Q") or m.endswith("72RQ"):
            return "Heavy Duty (.72)"
    if width_ft == "20":
        if m.endswith(".51") or m.endswith("51Q"):
            return "Standard Duty (.51)"
        if m.endswith(".61") or m.endswith("61Q"):
            return "Medium Duty (.61)"
        if m.endswith(".71") or m.endswith("71Q"):
            return "Heavy Duty (.71)"

    # fallback: try to read two-digit suffix
    m2 = re.search(r'\.(\d{2})', m)
    suff = m2.group(1) if m2 else ""
    mapping = {
        "40": "Standard Duty (.40)",
        "52": "Standard Duty (.52)",
        "72": "Heavy Duty (.72)",
        "51": "Standard Duty (.51)",
        "61": "Medium Duty (.61)",
        "71": "Heavy Duty (.71)",
    }
    return mapping.get(suff, "")


def _normalize_bw_duty_label(width_ft: str, duty_raw: str) -> str:
    """
    Map shorthand duty labels to the exact canonical strings your filter expects.
    Only normalizes where multiple duty classes exist (e.g., 15 ft).
    Leaves known canonical strings unchanged; otherwise returns the original.
    """
    if not duty_raw:
        return duty_raw
    duty_clean = duty_raw.strip().lower()

    # Already canonical? Leave it.
    if duty_raw in (
        "Standard Duty (.52)",
        "Heavy Duty (.72)",
        "Standard Duty (.40)",
        "Standard Duty (.51)",
        "Medium Duty (.61)",
        "Heavy Duty (.71)",
    ):
        return duty_raw

    # Normalize by width (only where multiple classes exist)
    if str(width_ft) == "15":
        if duty_clean in {"standard", "std", "light", "standard duty"}:
            return "Standard Duty (.52)"
        if duty_clean in {"heavy", "hd", "heavy duty"}:
            return "Heavy Duty (.72)"

    # For widths with a single class (e.g., 12 ft .40) nothing to do.
    return duty_raw


def _bw_label(r, duty=None, driveline=None, deck_rings=None, shielding_rows=None):
    model = str(r[COL_MODEL]).strip()
    bits = []
    if duty:
        bits.append(duty)
    if driveline:
        bits.append(f"{driveline} RPM")
    # Chain shielding is standard; only call out single/double if specified
    if shielding_rows:
        bits.append(f"{shielding_rows} chain shielding")
    if deck_rings:
        bits.append("Deck Rings")

    dsc = str(r.get(COL_DESC, "") or "").strip()
    label = model
    if bits:
        label += " — " + ", ".join(bits)
    if dsc:
        label += f" — {dsc}"
    return label


def _quote_batwing(df):
    bw_base = _family_base(df, "Batwing", exclude_turf=True)
    if bw_base.empty:
        return jsonify({"found": False, "mode": "error", "message": "No Batwing base rows found."}), 404

    dealer_number = getp("dealer_number")
    dealer_meta, dealer_rate_override = None, None
    if dealer_number and dealer_number in DEALER_DISCOUNTS:
        name, rate = DEALER_DISCOUNTS[dealer_number]
        dealer_meta = {"dealer_number": dealer_number, "dealer_name": name, "applied_discount": rate}
        dealer_rate_override = rate

    model_code = _extract_bw_code(getp("model"))
    width_txt = getp("width_ft")
    bw_duty = getp("bw_duty")
    bw_drive = getp("bw_driveline")
    deck_rings = getp("deck_rings")
    shield_rows = getp("shielding_rows")

    tire_id = getp("tire_id")
    tire_choice = getp("tire_choice")
    # Accept either value or id to avoid loops when choices_with_ids is present
    tire_qty_arg = getp("bw_tire_qty", "tire_qty") or getp("bw_tire_qty_id")

    wn = _ensure_width_norm_col(bw_base)
    rows = pd.DataFrame()

    if model_code.startswith("BW"):
        base_noQR = re.sub(r'[RQ]$', '', model_code.upper())
        rows = bw_base[bw_base[COL_MODEL].astype(str).str.upper().isin(
            [base_noQR, base_noQR + "R", base_noQR + "Q", base_noQR + "RQ"]
        )]
        if rows.empty:
            return jsonify({"found": False, "mode": "error", "message": f"Batwing model not found: {model_code}"}), 404
        width_ft = str(rows.iloc[0][COL_WIDTH_FT]).strip()
        width_ft = _ft_str(width_ft) or _normalize_width_ft_input_feet_only(width_txt) or ""
    else:
        if not width_txt:
            sizes = sorted({s for s in bw_base[wn].astype(str).tolist() if s}, key=lambda x: int(x))
            return jsonify({
                "found": True,
                "mode": "questions",
                "category": "Batwing",
                "model": "",
                "required_questions": [{
                    "name": "width_ft",
                    "question": "What size Batwing do you want (in feet)?",
                    "choices": sizes
                }]
            })
        width_ft = _normalize_width_ft_input_feet_only(width_txt)
        rows = bw_base[bw_base[wn].astype(str) == width_ft]

    if rows.empty:
        return jsonify({"found": False, "mode": "error", "message": f"No Batwing rows at {width_ft} ft."}), 404

    def duty_of_row(rr):
        return _bw_duty_from_model(str(rr[COL_MODEL]), width_ft)

    rows = rows[rows[wn].astype(str) == width_ft]

    _NUM_RE = re.compile(r'(\d+(?:\.\d+)?|\.\d+)')             # matches 72, 12.5, .52, etc.
    _DUTY_PRECEDENCE = {"standard": 0, "medium": 1, "heavy": 2}

    duties_avail = sorted(
        {duty_of_row(r) for _, r in rows.iterrows() if duty_of_row(r)},
        key=lambda s: (
            (0, float(_NUM_RE.search(s).group(1)))            # if a number exists, sort numerically
            if _NUM_RE.search(s) else
            (1, next((rank for k, rank in _DUTY_PRECEDENCE.items() if k in s.lower()), 99), s.lower())
        )
    )

    # Normalize any shorthand duty to exact canonical label for this width
    if bw_duty:
        bw_duty = _normalize_bw_duty_label(width_ft, bw_duty)

    if len(duties_avail) > 1 and not bw_duty:
        return jsonify({
            "found": True,
            "mode": "questions",
            "category": "Batwing",
            "model": f"{width_ft} ft Batwing",
            "required_questions": [{
                "name": "bw_duty",
                "question": f"Which duty class for {width_ft} ft Batwing?",
                "choices": duties_avail
            }]
        })

    if bw_duty:
        rows = rows[rows.apply(lambda r: duty_of_row(r).lower() == bw_duty.strip().lower(), axis=1)]
        if rows.empty:
            return jsonify({"found": False, "mode": "error", "message": f"No Batwing rows for {width_ft} ft with duty '{bw_duty}'."}), 404

    has_Q = any(str(m).upper().endswith("Q") or str(m).upper().endswith("RQ") for m in rows[COL_MODEL].astype(str))

    if not has_Q:
        bw_drive = "540"
    elif not bw_drive:
        return jsonify({
            "found": True,
            "mode": "questions",
            "category": "Batwing",
            "model": f"{width_ft} ft Batwing",
            "required_questions": [{
                "name": "bw_driveline",
                "question": "Which driveline do you want?",
                "choices": ["540", "1000"] if has_Q else []
            }]
        })
    if bw_drive.strip() == "1000":
        rows = rows[rows[COL_MODEL].astype(str).str.upper().str.endswith(("Q", "RQ"))]
    else:
        rows = rows[~rows[COL_MODEL].astype(str).str.upper().str.endswith(("Q", "RQ"))]

    has_R = any(str(m).upper().endswith("R") or str(m).upper().endswith("RQ") for m in rows[COL_MODEL].astype(str))
    has_nonR = any(not (str(m).upper().endswith("R") or str(m).upper().endswith("RQ")) for m in rows[COL_MODEL].astype(str))
    if has_R and has_nonR and not deck_rings:
        return jsonify({
            "found": True,
            "mode": "questions",
            "category": "Batwing",
            "model": f"{width_ft} ft Batwing",
            "required_questions": [{
                "name": "deck_rings",
                "question": "Add Deck Rings?",
                "choices": ["Yes", "No"]
            }]
        })
    if deck_rings:
        want_R = deck_rings.strip().lower().startswith("y")
        rows = rows[rows[COL_MODEL].astype(str).str.upper().str.endswith(("R", "RQ"))] if want_R else rows[~rows[COL_MODEL].astype(str).str.upper().str.endswith(("R", "RQ"))]

    # Shielding Rows
    sh_variants = sorted({str(s).strip().lower() for s in rows[COL_SHIELDING].astype(str).tolist() if s})
    has_double = any("double" in s for s in sh_variants)
    has_single = any("single" in s for s in sh_variants)
    if (has_single and has_double) and not shield_rows:
        return jsonify({
            "found": True,
            "mode": "questions",
            "category": "Batwing",
            "model": f"{width_ft} ft Batwing",
            "required_questions": [{
                "name": "shielding_rows",
                "question": "Single Row or Double Row chain shielding?",
                "choices": ["Single Row", "Double Row"]
            }]
        })

    if shield_rows:
        sr = shield_rows.strip().lower()
        if "double" in sr:
            rows = rows[rows[COL_SHIELDING].astype(str).str.lower().str.contains("double")]
        elif "single" in sr:
            rows = rows[rows[COL_SHIELDING].astype(str).str.lower().str.contains("single")]

    r, err = pick_one_or_400(rows, "Batwing final base selection", {
        "width_ft": width_ft,
        "duty": bw_duty,
        "driveline": bw_drive,
        "rings": deck_rings,
        "shield_rows": shield_rows
    })
    if err:
        return err

    model = str(r[COL_MODEL]).strip()
    pid = _first_part_id(r)
    duty_label = _bw_duty_from_model(model, width_ft)
    label = _bw_label(
        r,
        duty=duty_label,
        driveline=bw_drive or ("1000" if model.upper().endswith("Q") or model.upper().endswith("RQ") else "540"),
        deck_rings=(deck_rings.strip().lower().startswith("y") if deck_rings else (model.upper().endswith("R") or model.upper().endswith("RQ"))),
        shielding_rows=shield_rows
    ) + (f" ({pid})" if pid else "")
    lines = [_make_line(label, 1, _price_number(r.get(COL_LIST_PRICE)), part_id=pid)]

    acc_df = accessories_for_model(df, model)
    tire_opts = acc_df[acc_df[COL_CATEGORY].astype(str).str.contains("Tire", case=False, na=False)]

    if not tire_opts.empty:
        if not (tire_id or tire_choice):
            choices, choices_with_ids = [], []
            for _, tr in tire_opts.iterrows():
                desc = (tr.get(COL_DESC) or "").strip()
                pidt = _first_part_id(tr)
                if desc and pidt:
                    choices.append(desc)
                    choices_with_ids.append({"id": pidt, "label": desc})
            return jsonify({
                "found": True,
                "mode": "questions",
                "category": "Batwing",
                "model": model,
                "required_questions": [{
                    "name": "tire_id",
                    "question": "Select a tire option (required for Batwing):",
                    "choices": choices,
                    "choices_with_ids": choices_with_ids,
                    "multiple": False,
                    "required": True
                }]
            })

        tire_sel = None
        if tire_id:
            tire_sel = _find_by_part_id(tire_opts, tire_id)
        if (tire_sel is None or tire_sel.empty) and tire_choice:
            tire_sel = tire_opts[tire_opts[COL_DESC].astype(str).str.contains(tire_choice, case=False, na=False)]

        if tire_sel is not None and not tire_sel.empty:
            tr = tire_sel.iloc[0]
            tpid = _first_part_id(tr)
            tdesc = (tr.get(COL_DESC) or "").strip() or "Tires"

            tire_qty = None
            if tire_qty_arg not in (None, "", "0"):
                try:
                    tire_qty = int(float(tire_qty_arg))
                except:
                    tire_qty = None

            valid_counts = []
            if width_ft == "12":
                valid_counts = [4, 6]
            elif width_ft == "10":
                if "72" in model:
                    valid_counts = [5, 6]
                else:
                    valid_counts = [3, 5, 6]
            elif width_ft == "13":
                valid_counts = [5, 6]
            elif width_ft == "15":
                if "52" in model:
                    valid_counts = [4, 6, 8]
                elif "72" in model:
                    valid_counts = [6, 8]
            elif width_ft == "20":
                valid_counts = [6, 8]

            if tire_qty not in valid_counts:
                return jsonify({
                    "found": True,
                    "mode": "questions",
                    "category": "Batwing",
                    "model": model,
                    "required_questions": [{
                        "name": "bw_tire_qty",
                        "question": f"How many tires do you want on the {model}?",
                        "choices": [str(v) for v in valid_counts],
                        "choices_with_ids": [{"id": str(v), "label": f"{v} tires"} for v in valid_counts],
                        "multiple": False,
                        "required": True
                    }]
                })

            lines.append(_make_line(
                f"{tdesc} ({tpid})",
                tire_qty,
                _price_number(tr.get(COL_LIST_PRICE)),
                part_id=tpid
            ))

            # --- Dual Hub Kit Logic ---
            model_upper = model.upper()
            hub_kit_needs = []
            if "12" in model_upper:
                if tire_qty == 6:
                    hub_kit_needs = ["center"]
            if "10.52" in model_upper:
                if tire_qty == 5:
                    hub_kit_needs = ["center"]
                elif tire_qty == 6:
                    hub_kit_needs = ["center", "wing"]
            elif "10.72" in model_upper:
                if tire_qty == 6:
                    hub_kit_needs = ["wing"]
            elif "13" in model_upper:
                if tire_qty == 6:
                    hub_kit_needs = ["wing"]
            elif "15.52" in model_upper:
                if tire_qty == 6:
                    hub_kit_needs = ["center"]
                elif tire_qty == 8:
                    hub_kit_needs = ["center", "wing"]
            elif "15.72" in model_upper:
                if tire_qty == 8:
                    hub_kit_needs = ["wing"]
            elif "20" in model_upper:
                if tire_qty == 8:
                    hub_kit_needs = ["wing"]

            if hub_kit_needs:
                hub_kits = acc_df[acc_df[COL_DESC].astype(str).str.lower().str.contains("hub kit")]
                added_needs = set()
                for need in hub_kit_needs:
                    for _, hrow in hub_kits.iterrows():
                        desc = str(hrow.get(COL_DESC, "")).lower()
                        if need in desc and need not in added_needs:
                            part_id = _first_part_id(hrow)
                            if not part_id:
                                continue
                            lines.append(_make_line(
                                f"{hrow.get(COL_DESC)} ({part_id})",
                                1,
                                _price_number(hrow.get(COL_LIST_PRICE)),
                                part_id=part_id
                            ))
                            added_needs.add(need)
                            break

    list_access = (request.args.get("list_accessories") or "").strip() not in ("", "0", "false", "False")
    acc_ids2, acc_qty_map2, acc_desc_terms2 = _read_accessory_params()
    lines += accessory_lines_from_selection(
        acc_df[~acc_df[COL_CATEGORY].astype(str).str.contains("Tire", case=False, na=False)],
        acc_ids2, acc_qty_map2, acc_desc_terms2
    )

    return _totals_payload(
        lines,
        ["Chain shielding is standard. Tires are quoted as separate line items."],
        "Batwing",
        model,
        dealer_rate_override,
        dealer_meta,
    )

# =====================
# Turf Batwing (TBW)
# =====================

def _quote_turf_batwing(df):
    """Turf Batwings (TBWxx.yy)
    Duty classes:
      - .20 → Residential (12 ft only)
      - .40 → Commercial (12, 15, 17 ft)
    Flow:
      - Model-first, else duty/width questions (like other families).
      - Ask: Front Roller Kits? (recommended) → if Yes, add qty 3.
      - Ask: Chains? (optional; unit passes safety standards without chains) → if Yes, add by accessory rows/qty.
    """
    # ---- Base rows (TBW only; exclude accessories/tires) ----
    tbw_base = _family_base(df, "Turf")
    tbw_base = tbw_base[tbw_base[COL_MODEL].astype(str).str.upper().str.startswith("TBW")]
    if tbw_base.empty:
        return jsonify({"found": False, "mode": "error", "message": "No Turf Batwing base rows found."}), 404

    # ---- Dealer (same pattern as others) ----
    dealer_number = getp("dealer_number")
    dealer_meta, dealer_rate_override = None, None
    if dealer_number and dealer_number in DEALER_DISCOUNTS:
        name, rate = DEALER_DISCOUNTS[dealer_number]
        dealer_meta = {"dealer_number": dealer_number, "dealer_name": name, "applied_discount": rate}
        dealer_rate_override = rate

    # ---- Params ----
    model_raw    = getp("model")
    width_txt    = getp("width_ft")
    tbw_duty_in  = getp("tbw_duty", "duty")  # accepts either
    want_rollers = getp("front_rollers")     # "Yes"/"No"
    want_chains  = getp("chains")            # "Yes"/"No"

    # ---- Helpers ----
    def _norm_duty_label(s: str) -> str:
        s = (s or "").strip().lower()
        if not s:
            return ""
        if s in {".20", "20", "residential", "res", "light", "residential (.20)"} or "20" in s or "res" in s:
            return "Residential (.20)"
        if s in {".40", "40", "commercial", "comm", "heavy", "commercial (.40)"} or "40" in s or "com" in s or "heavy" in s:
            return "Commercial (.40)"
        return ""

    def _duty_from_model(m: str) -> str:
        mu = (m or "").upper().strip()
        if mu.endswith(".20"):
            return "Residential (.20)"
        if mu.endswith(".40"):
            return "Commercial (.40)"
        return ""

    def _width_norm(s: str) -> str:
        return _normalize_width_ft_input_feet_only(s) if s else None

    def _qty_from_row(r, default=1) -> int:
        # scan any column with 'qty' / 'quantity'
        for c in r.index:
            cl = str(c).lower()
            if "qty" in cl or "quantity" in cl:
                v = str(r.get(c) or "").strip()
                m = re.search(r"(\d+)", v)
                if m:
                    try:
                        n = int(m.group(1))
                        if n > 0:
                            return n
                    except:
                        pass
        return int(default)

    def _tbw_label(row):
        m = str(row[COL_MODEL]).strip()
        d = _duty_from_model(m)
        dsc = str(row.get(COL_DESC, "") or "").strip()
        bits = [m]
        if d:
            bits.append(d)
        if dsc:
            bits.append(dsc)
        return " — ".join(bits)

    # ---- Model-first or duty/width selection ----
    model_code = None
    mm = re.search(r'\b(TBW\d+(?:\.\d+)?)\b', model_raw or "", re.I)
    if mm:
        model_code = mm.group(1).upper()
        rows = tbw_base[tbw_base[COL_MODEL].astype(str).str.upper() == model_code]
        if rows.empty:
            return jsonify({"found": False, "mode": "error", "message": f"Turf Batwing model not found: {model_code}"}), 404
        width_ft = _ft_str(str(rows.iloc[0].get(COL_WIDTH_FT, "")))
        tbw_duty_label = _duty_from_model(model_code) or ""
    else:
        wn = _ensure_width_norm_col(tbw_base)
        width_ft = _width_norm(width_txt)
        rows = tbw_base

        # Width pre-filter if provided
        if width_ft:
            rows = rows[rows[wn].astype(str) == width_ft]
            if rows.empty:
                return jsonify({"found": False, "mode": "error", "message": f"No Turf Batwing rows at {width_ft} ft."}), 404

        # Duty resolution
        duties_avail = sorted({_duty_from_model(str(r[COL_MODEL])) for _, r in rows.iterrows() if _duty_from_model(str(r[COL_MODEL]))})
        if len(duties_avail) > 1 and not tbw_duty_in:
            return jsonify({
                "found": True,
                "mode": "questions",
                "category": "Turf Batwing",
                "model": (width_ft and f"{width_ft} ft TBW") or "",
                "required_questions": [{
                    "name": "tbw_duty",
                    "question": "Which duty class for Turf Batwing?",
                    "choices": duties_avail
                }]
            })
        tbw_duty_label = _norm_duty_label(tbw_duty_in) if tbw_duty_in else (duties_avail[0] if duties_avail else "")

        if tbw_duty_label:
            rows = rows[rows.apply(lambda r: _duty_from_model(str(r[COL_MODEL])) == tbw_duty_label, axis=1)]
        if rows.empty:
            return jsonify({"found": False, "mode": "error", "message": "No TBW rows after duty/width filtering."}), 404

        # Residential default to 12 ft if no width chosen
        if not width_ft and tbw_duty_label.lower().startswith("residential"):
            width_ft = "12"

        # Width question (if not supplied and multiple remain)
        if not width_ft:
            wn2 = _ensure_width_norm_col(rows)
            widths = sorted({s for s in rows[wn2].astype(str).tolist() if s}, key=lambda x: int(x))
            if len(widths) > 1:
                return jsonify({
                    "found": True,
                    "mode": "questions",
                    "category": "Turf Batwing",
                    "model": tbw_duty_label,
                    "required_questions": [{
                        "name": "width_ft",
                        "question": f"Which width for {tbw_duty_label}?",
                        "choices": widths
                    }]
                })
            width_ft = widths[0]

        # ✅ FIX: reduce to chosen width using the column name returned by _ensure_width_norm_col
        wn3 = _ensure_width_norm_col(rows)
        rows = rows[rows[wn3].astype(str) == str(width_ft)]

    # ---- Final base selection ----
    r, err = pick_one_or_400(rows, "Turf Batwing final base selection", {"width_ft": width_ft, "duty": tbw_duty_label})
    if err:
        return err

    model = str(r[COL_MODEL]).strip()
    pid = _first_part_id(r)
    label = _tbw_label(r) + (f" ({pid})" if pid else "")
    lines = [_make_line(label, 1, _price_number(r.get(COL_LIST_PRICE)), part_id=pid)]
    notes = []

    # ---- Accessories for TBW model ----
    acc_df = accessories_for_model(df, model)

    # 1) Front Roller Kits? (recommended) -> qty 3
    if want_rollers == "":
        return jsonify({
            "found": True,
            "mode": "questions",
            "category": "Turf Batwing",
            "model": model,
            "required_questions": [{
                "name": "front_rollers",
                "question": "Add front roller kits? (Recommended)",
                "choices": ["Yes", "No"]
            }]
        })
    if want_rollers.lower().startswith("y"):
        fr = acc_df[acc_df[COL_DESC].astype(str).str.contains(r"front\s*roller", case=False, na=False)]
        if not fr.empty:
            rr = fr.iloc[0]
            fr_pid = _first_part_id(rr)
            fr_desc = (rr.get(COL_DESC) or "Front Roller Kit").strip()
            lines.append(_make_line(f"{fr_desc} ({fr_pid})", 3, _price_number(rr.get(COL_LIST_PRICE)), part_id=fr_pid))
        notes.append("Front roller kits are recommended.")

    # 2) Chains? (optional; passes safety standards without)
    if want_chains == "":
        return jsonify({
            "found": True,
            "mode": "questions",
            "category": "Turf Batwing",
            "model": model,
            "required_questions": [{
                "name": "chains",
                "question": "Add chains? (Note: TBW passes all safety standards without chains.)",
                "choices": ["Yes", "No"]
            }]
        })
    if want_chains.lower().startswith("y"):
        ch = acc_df[acc_df[COL_DESC].astype(str).str.contains("chain", case=False, na=False)]
        for _, cr in ch.iterrows():
            ch_pid = _first_part_id(cr)
            ch_desc = (cr.get(COL_DESC) or "Chain Kit").strip()
            qty_needed = _qty_from_row(cr, default=1)
            lines.append(_make_line(f"{ch_desc} ({ch_pid})", qty_needed, _price_number(cr.get(COL_LIST_PRICE)), part_id=ch_pid))
        notes.append("Chains added (optional — unit passes safety standards without chains).")
    else:
        notes.append("Unit passes safety standards without chains.")

    return _totals_payload(
        lines,
        notes or [],
        "Turf Batwing",
        model,
        dealer_rate_override,
        dealer_meta,
    )

def _quote_rear_finish(df):
    """
    Rear Discharge Finish Mowers:
      Category mapping:
        - Residential Finish Mowers  -> Turf Keeper (TK)
        - Commercial Finish Mowers   -> Turf Keeper Pro (TKP)
        - RD990X Finish Mower        -> RD990X
    Logic:
      - Ask sub-category with descriptive labels if unknown.
      - TK/TKP: resolve model/width, then IF both chain + no-chain variants exist,
        ask "Chains?" and filter rows to the proper variant (W-suffix or col F='Chain').
        DO NOT add chain accessories for TK/TKP.
      - RD990X: base unit only; add front roller only if explicitly asked.
    """
    # --- Dealer context
    dealer_number = getp("dealer_number")
    dealer_meta, dealer_rate_override = None, None
    if dealer_number and dealer_number in DEALER_DISCOUNTS:
        name, rate = DEALER_DISCOUNTS[dealer_number]
        dealer_meta = {"dealer_number": dealer_number, "dealer_name": name, "applied_discount": rate}
        dealer_rate_override = rate

    # --- Params
    model_raw     = getp("model")
    width_txt     = getp("width_ft")
    sub_choice_in = getp("finish_choice")   # tk / tkp / rd990x or label
    want_rollers  = getp("front_rollers")   # "Yes"/"No" or ""
    want_chains   = getp("chains")          # "Yes"/"No" or ""

    # --- Base by CATEGORY (primary) with fallback to MODEL prefix; exclude accessories/tires
    cat = df[COL_CATEGORY].astype(str).str.strip()
    model_series = df[COL_MODEL].astype(str).str.upper().str.strip()
    is_accessory = df[COL_CATEGORY].astype(str).str.contains(ACC_TIRE_PATTERN, na=False)

    is_finish_category = (
        cat.str.contains(r"\bResidential Finish Mowers\b", case=False, na=False) |
        cat.str.contains(r"\bCommercial Finish Mowers\b", case=False, na=False) |
        cat.str.contains(r"\bRD990X Finish Mower\b", case=False, na=False)
    )
    fin_base = df[is_finish_category & (~is_accessory)].copy()

    if fin_base.empty:
        # Fallback to model prefixes if category text isn't present
        is_finish_model = (
            model_series.str.startswith("TKP") |
            (model_series.str.startswith("TK") & (~model_series.str.startswith("TKP"))) |
            model_series.eq("RD990X")
        )
        fin_base = df[is_finish_model & (~is_accessory)].copy()

    if fin_base.empty:
        return jsonify({"found": False, "mode": "error",
                        "message": "No Rear Discharge Finish Mower base rows found (check Category or Model prefixes)."}), 404

    # --- Helpers
    def _sub_from_text(s: str) -> str:
        s = (s or "").strip().lower()
        if not s:
            return ""
        if s in {"tk", "turf keeper", "turf keeper (residential mower)", "residential"}:
            return "tk"
        if s in {"tkp", "turf keeper pro", "turf keeper pro (commercial duty)", "commercial"}:
            return "tkp"
        if "rd990x" in s or "hybrid" in s:
            return "rd990x"
        return ""

    def _rear_label(r, include_desc=True):
        m = str(r[COL_MODEL]).strip()
        drv = str(r.get(COL_DRIVE, "") or "").strip()
        bits = [m]
        if drv:
            bits.append(drv)
        if include_desc:
            dsc = str(r.get(COL_DESC, "") or "").strip()
            if dsc:
                bits.append(dsc)
        return " — ".join(bits)

    def _qty_from_row(row, default=1) -> int:
        for c in row.index:
            cl = str(c).lower()
            if "qty" in cl or "quantity" in cl:
                v = str(row.get(c) or "").strip()
                m = re.search(r"(\d+)", v)
                if m:
                    try:
                        n = int(m.group(1))
                        if n > 0:
                            return n
                    except:
                        pass
        return int(default)

    def _is_chain_row(row) -> bool:
        m = str(row.get(COL_MODEL, "") or "").strip().upper()
        sh = str(row.get(COL_SHIELDING, "") or "").strip().lower()
        if m.endswith("W"):
            return True
        if "chain" in sh:
            return True
        return False

    # --- Determine sub-category from model or user choice
    sub_choice = ""
    if model_raw:
        up = model_raw.strip().upper()
        if up.startswith("TKP"):
            sub_choice = "tkp"
        elif up.startswith("TK"):
            sub_choice = "tk"
        elif "RD990X" in up:
            sub_choice = "rd990x"
    if not sub_choice:
        sub_choice = _sub_from_text(sub_choice_in)

    # Ask sub-category (with descriptive labels) if unknown
    if not sub_choice:
        return jsonify({
            "found": True,
            "mode": "questions",
            "category": "Rear Discharge Finish Mower",
            "model": "",
            "required_questions": [{
                "name": "finish_choice",
                "question": "Which type of finish mower do you want quoted?",
                "choices": [
                    "Turf Keeper (Residential Mower)",
                    "Turf Keeper Pro (Commercial Duty)",
                    "RD990X (Finish Mower + Brush Cutter Hybrid)"
                ],
                "choices_with_ids": [
                    {"id": "tk",     "label": "Turf Keeper (Residential Mower)"},
                    {"id": "tkp",    "label": "Turf Keeper Pro (Commercial Duty)"},
                    {"id": "rd990x", "label": "RD990X (Finish Mower + Brush Cutter Hybrid)"}
                ]
            }]
        })

    # --- Filter base by chosen sub-category
    if sub_choice == "tk":
        base = fin_base[
            cat.str.contains(r"\bResidential Finish Mowers\b", case=False, na=False) |
            (model_series.str.startswith("TK") & (~model_series.str.startswith("TKP")))
        ]
    elif sub_choice == "tkp":
        base = fin_base[
            cat.str.contains(r"\bCommercial Finish Mowers\b", case=False, na=False) |
            model_series.str.startswith("TKP")
        ]
    else:  # rd990x
        base = fin_base[
            cat.str.contains(r"\bRD990X Finish Mower\b", case=False, na=False) |
            model_series.eq("RD990X")
        ]

    if base.empty:
        return jsonify({"found": False, "mode": "error",
                        "message": f"No rows found for finish sub-category '{sub_choice}'. "
                                   f"Verify Category labels or Model prefixes."}), 404

    # --- Resolve model/width set (do NOT commit to an exact chain/non-chain row yet)
    # Prefer model if explicitly provided; otherwise go by width.
    wn = _ensure_width_norm_col(base)
    rows = pd.DataFrame()

    if model_raw:
        # Collect all rows sharing this model's WIDTH, so we can flip to chain/no-chain sibling if needed
        mdl_hit = base[base[COL_MODEL].astype(str).str.upper().str.strip() == model_raw.strip().upper()]
        if not mdl_hit.empty:
            this_w = _ft_str(mdl_hit.iloc[0].get(COL_WIDTH_FT))
            rows = base[base[wn].astype(str) == this_w]
        else:
            # Fall back to width-first if the exact model wasn't found
            pass

    if rows.empty:
        if not width_txt:
            sizes = sorted({s for s in base[wn].astype(str).tolist() if s}, key=lambda x: int(x))
            return jsonify({
                "found": True,
                "mode": "questions",
                "category": "Rear Discharge Finish Mower",
                "model": {"tk": "Turf Keeper", "tkp": "Turf Keeper Pro", "rd990x": "RD990X"}[sub_choice],
                "required_questions": [{
                    "name": "width_ft",
                    "question": "What cutting width (in feet)?",
                    "choices": sizes
                }]
            })
        norm_ft = _normalize_width_ft_input(width_txt)
        rows = base[base[wn].astype(str) == (norm_ft or "")]
        if rows.empty:
            return jsonify({"found": False, "mode": "error",
                            "message": f"No finish mower rows at {norm_ft} ft for {sub_choice}."}), 404

    # --- TK/TKP ONLY: resolve Chain vs No-Chain by picking the correct model variant
    if sub_choice in {"tk", "tkp"}:
        has_chain     = any(_is_chain_row(r) for _, r in rows.iterrows())
        has_no_chain  = any(not _is_chain_row(r) for _, r in rows.iterrows())

        if has_chain and has_no_chain and not want_chains:
            # Ask once, before any final pick, and DO NOT add chain accessories later
            return jsonify({
                "found": True,
                "mode": "questions",
                "category": "Rear Discharge Finish Mower",
                "model": {"tk": "Turf Keeper", "tkp": "Turf Keeper Pro"}[sub_choice],
                "required_questions": [{
                    "name": "chains",
                    "question": "Do you want the chain-shielded model? (Note: passes all safety standards without chains.)",
                    "choices": ["Yes", "No"]
                }]
            })

        if want_chains:
            yn = want_chains.strip().lower()
            if yn.startswith("y"):
                rows = rows[rows.apply(lambda r: _is_chain_row(r), axis=1)]
            elif yn.startswith("n"):
                rows = rows[~rows.apply(lambda r: _is_chain_row(r), axis=1)]
            # If filtering removed everything (data issue), keep original set
            if rows.empty:
                rows = base[base[wn].astype(str) == rows.iloc[0][wn] if not rows.empty else base[wn].astype(str)]

    # --- If multiple configs remain → ask to pick one
    choice_id  = getp("rear_choice_id", "finish_choice_id", "accessory_id")
    choice_lbl = getp("rear_choice", "finish_choice")
    if len(rows.index) > 1 and not (choice_id or choice_lbl):
        str_choices, id_choices = [], []
        for _, r in rows.iterrows():
            lbl = _rear_label(r)
            rid = _first_part_id(r) or lbl
            str_choices.append(lbl)
            id_choices.append({"id": rid, "label": lbl})
        return jsonify({
            "found": True,
            "mode": "questions",
            "category": "Rear Discharge Finish Mower",
            "model": {"tk": "Turf Keeper", "tkp": "Turf Keeper Pro", "rd990x": "RD990X"}[sub_choice],
            "required_questions": [{
                "name": "rear_choice",
                "question": "Which configuration would you like?",
                "choices": str_choices,
                "choices_with_ids": id_choices
            }]
        })

    # --- Final pick
    chosen = rows
    if choice_id or choice_lbl:
        chosen = _select_by_id_or_label(rows, choice_id, choice_lbl, label_fn=_rear_label)

    r, err = pick_one_or_400(chosen, "Rear Finish final selection",
                             {"choice_id": choice_id, "choice_label": choice_lbl})
    if err:
        return err

    model = str(r[COL_MODEL]).strip()
    pid   = _first_part_id(r)
    label = _rear_label(r, include_desc=True) + (f" ({pid})" if pid else "")
    lines = [_make_line(label, 1, _price_number(r.get(COL_LIST_PRICE)), part_id=pid)]
    notes = []

    # --- Accessories
    acc_df = accessories_for_model(df, model)

    if sub_choice in {"tk", "tkp"}:
        # Front Roller Kit (recommended) → qty 1
        if want_rollers == "":
            return jsonify({
                "found": True,
                "mode": "questions",
                "category": "Rear Discharge Finish Mower",
                "model": {"tk": "Turf Keeper", "tkp": "Turf Keeper Pro"}[sub_choice],
                "required_questions": [{
                    "name": "front_rollers",
                    "question": "Add front roller kit? (Recommended)",
                    "choices": ["Yes", "No"]
                }]
            })
        if want_rollers.lower().startswith("y"):
            fr = acc_df[acc_df[COL_DESC].astype(str).str.contains(r"front\s*roller", case=False, na=False)]
            if not fr.empty:
                rr = fr.iloc[0]
                fr_pid = _first_part_id(rr)
                fr_desc = (rr.get(COL_DESC) or "Front Roller Kit").strip()
                lines.append(_make_line(f"{fr_desc} ({fr_pid})", 1, _price_number(rr.get(COL_LIST_PRICE)), part_id=fr_pid))
            notes.append("Front roller kit is recommended.")

        # NOTE: No chain accessory lines for TK/TKP; chain handled via model variant above.
        if want_chains:
            if want_chains.lower().startswith("y"):
                notes.append("Chain-shielded model selected.")
            else:
                notes.append("Non-chain model selected (passes all safety standards).")
        else:
            # If we never had to ask (only one variant existed), add a neutral note:
            notes.append("Passes all safety standards without chains.")

    else:
        # RD990X: base unit only; add front roller only if explicitly asked (no prompt)
        if want_rollers and want_rollers.lower().startswith("y"):
            fr = acc_df[acc_df[COL_DESC].astype(str).str.contains(r"front\s*roller", case=False, na=False)]
            if not fr.empty:
                rr = fr.iloc[0]
                fr_pid = _first_part_id(rr)
                fr_desc = (rr.get(COL_DESC) or "Front Roller Kit").strip()
                lines.append(_make_line(f"{fr_desc} ({fr_pid})", 1, _price_number(rr.get(COL_LIST_PRICE)), part_id=fr_pid))

    # Optional generic accessories (still allowed)
    list_access = (request.args.get("list_accessories") or "").strip() not in ("", "0", "false", "False")
    acc_ids, acc_qty_map, acc_desc_terms = _read_accessory_params()
    if list_access and not acc_ids and not acc_desc_terms:
        return accessory_choices_payload(acc_df, "Rear Discharge Finish Mower", model)
    lines += accessory_lines_from_selection(acc_df, acc_ids, acc_qty_map, acc_desc_terms)

    return _totals_payload(
        lines,
        notes,
        "Rear Discharge Finish Mower",
        model,
        dealer_rate_override,
        dealer_meta,
    )

# =========================
# Box Scraper (BS)
# =========================

def _bs_duty_from_model(model: str) -> str:
    m = (model or "").upper().strip()
    m2 = re.search(r"\.(\d{2})$", m)
    suff = m2.group(1) if m2 else ""
    mapping = {
        "20": "Light Duty (.20)",
        "30": "Medium Duty (.30)",
        "40": "Heavy Duty (.40)",
    }
    return mapping.get(suff, "")

def _bs_label(r):
    model = str(r[COL_MODEL]).strip()
    duty = _bs_duty_from_model(model)
    dsc = str(r.get(COL_DESC, "") or "").strip()
    bits = [model]
    if duty: bits.append(duty)
    if dsc: bits.append(dsc)
    return " — ".join(bits)

def _quote_box_scraper(df):
    # Base rows: strictly BS models; exclude accessories/tires
    bs_base = df[df[COL_MODEL].astype(str).str.upper().str.startswith("BS")].copy()
    bs_base = bs_base[~bs_base[COL_CATEGORY].astype(str).str.contains(ACC_TIRE_PATTERN, na=False)]
    if bs_base.empty:
        return jsonify({"found": False, "mode": "error", "message": "No Box Scraper rows found."}), 404

    # Dealer context
    dealer_number = getp("dealer_number")
    dealer_meta, dealer_rate_override = None, None
    if dealer_number and dealer_number in DEALER_DISCOUNTS:
        name, rate = DEALER_DISCOUNTS[dealer_number]
        dealer_meta = {"dealer_number": dealer_number, "dealer_name": name, "applied_discount": rate}
        dealer_rate_override = rate

    # Params
    model_raw = getp("model")
    bs_width_in = getp("bs_width_in")  # normalized inches string like "72"
    bs_duty = getp("bs_duty")          # Light/Medium/Heavy labels
    choice_id = getp("bs_choice_id", "accessory_id")
    choice_lbl = getp("bs_choice")

    # Model-first
    model_code = _extract_bs_code(model_raw)
    if model_code.startswith("BS"):
        rows = bs_base[bs_base[COL_MODEL].astype(str).str.upper() == model_code.upper()]
        if rows.empty:
            return jsonify({"found": False, "mode": "error", "message": f"Box Scraper model not found: {model_code}"}), 404
        r, err = pick_one_or_400(rows, "Box Scraper final (model)", {"model": model_code})
        if err: return err
        model = str(r[COL_MODEL]).strip()
        pid = _first_part_id(r)
        label = _bs_label(r) + (f" ({pid})" if pid else "")
        lines = [_make_line(label, 1, _price_number(r.get(COL_LIST_PRICE)), part_id=pid)]
        return _totals_payload(lines, ["Selected exact model."], "Box Scraper", model, dealer_rate_override, dealer_meta)

    # Width-first flow
    win = _ensure_width_in_norm_col(bs_base)

    # Ask width if not provided
    if not bs_width_in:
        widths_in = sorted({w for w in bs_base[win].astype(str).tolist() if w and w.isdigit()}, key=lambda x: int(x))
        if not widths_in:
            return jsonify({"found": False, "mode": "error", "message": "No Box Scraper widths available."}), 404
        choices = [f'{w} in ({_inches_to_feet_label(w)})' for w in widths_in]
        choices_with_ids = [{"id": w, "label": f'{w} in ({_inches_to_feet_label(w)})'} for w in widths_in]
        return jsonify({
            "found": True,
            "mode": "questions",
            "category": "Box Scraper",
            "model": "",
            "required_questions": [{
                "name": "bs_width_in",
                "question": "What width do you want for the box scraper?",
                "choices": choices,
                "choices_with_ids": choices_with_ids
            }]
        })

    # Filter by chosen width (exact inches string)
    rows = bs_base[bs_base[win].astype(str) == str(bs_width_in).strip()]
    if rows.empty:
        return jsonify({"found": False, "mode": "error", "message": f"No Box Scraper rows at {bs_width_in} in."}), 404

    # Duty choices available at this width
    duties_avail = sorted({ _bs_duty_from_model(str(r[COL_MODEL])) for _, r in rows.iterrows() if _bs_duty_from_model(str(r[COL_MODEL])) })
    if len(duties_avail) > 1 and not bs_duty:
        return jsonify({
            "found": True,
            "mode": "questions",
            "category": "Box Scraper",
            "model": f'{bs_width_in} in ({_inches_to_feet_label(bs_width_in)})',
            "required_questions": [{
                "name": "bs_duty",
                "question": f"Which duty class for the {bs_width_in} in box scraper?",
                "choices": duties_avail
            }]
        })
    # If only one duty exists, lock it in
    duty_chosen = bs_duty.strip() if bs_duty else (duties_avail[0] if duties_avail else "")
    if duty_chosen:
        rows = rows[rows.apply(lambda r: _bs_duty_from_model(str(r[COL_MODEL])).lower() == duty_chosen.lower(), axis=1)]
        if rows.empty:
            return jsonify({"found": False, "mode": "error", "message": f"No Box Scraper rows for {bs_width_in} in with duty '{duty_chosen}'."}), 404

    # If more than one config (rare), ask which
    if len(rows.index) > 1 and not (choice_id or choice_lbl):
        str_choices, id_choices = [], []
        for _, rr in rows.iterrows():
            lbl = _bs_label(rr)
            rid = _first_part_id(rr) or lbl
            str_choices.append(lbl)
            id_choices.append({"id": rid, "label": lbl})
        return jsonify({
            "found": True,
            "mode": "questions",
            "category": "Box Scraper",
            "model": f'{bs_width_in} in ({_inches_to_feet_label(bs_width_in)}) — {duty_chosen}' if duty_chosen else f'{bs_width_in} in',
            "required_questions": [{
                "name": "bs_choice",
                "question": "Which configuration would you like?",
                "choices": str_choices,
                "choices_with_ids": id_choices
            }]
        })

    # Final pick
    chosen = rows
    if choice_id or choice_lbl:
        chosen = _select_by_id_or_label(rows, choice_id, choice_lbl, label_fn=_bs_label)
    r, err = pick_one_or_400(chosen, "Box Scraper final selection", {"width_in": bs_width_in, "duty": duty_chosen, "choice_id": choice_id, "choice_label": choice_lbl})
    if err: return err

    model = str(r[COL_MODEL]).strip()
    pid = _first_part_id(r)
    label = _bs_label(r) + (f" ({pid})" if pid else "")
    lines = [_make_line(label, 1, _price_number(r.get(COL_LIST_PRICE)), part_id=pid)]
    return _totals_payload(lines, [f"Width: {bs_width_in} in ({_inches_to_feet_label(bs_width_in)})", f"Duty: {duty_chosen}" if duty_chosen else ""], "Box Scraper", model, dealer_rate_override, dealer_meta)

# =========================
# Grading Scraper (GS)
# =========================
def _quote_grading_scraper(df):
    """
    Grading Scraper (GSxx): Single decision = width (inches). If model is given, quote directly.
    No accessories. Uses whichever width column exists (inches or feet), normalizing to inches.
    """

    # --- Build a clean GS base set (no accessories/tires) ---
    gs_base = _family_base(df, "Grading")  # matches "Grading Scraper" / "Grading Scrapers" etc.
    gs_base = gs_base[gs_base[COL_MODEL].astype(str).str.upper().str.startswith("GS")]
    if gs_base.empty:
        return jsonify({"found": False, "mode": "error", "message": "No Grading Scraper base rows found."}), 404

    # --- Dealer context (same pattern as other families) ---
    dealer_number = getp("dealer_number")
    dealer_meta, dealer_rate_override = None, None
    if dealer_number and dealer_number in DEALER_DISCOUNTS:
        name, rate = DEALER_DISCOUNTS[dealer_number]
        dealer_meta = {"dealer_number": dealer_number, "dealer_name": name, "applied_discount": rate}
        dealer_rate_override = rate

    model_raw = getp("model")
    width_txt = getp("gs_width_in", "width", "width_ft")

    # --- Local inches parser: accepts "84", '84 in', '7 ft', 7, 6.0, etc. → returns "84" ---
    def _to_inches_str(val) -> str:
        s = str(val or "").strip().lower()
        if not s:
            return ""
        m = re.search(r'(\d+(?:\.\d+)?)', s)
        if not m:
            return ""
        num = float(m.group(1))
        # explicit unit hints
        if any(tok in s for tok in ['"', "in", "inch"]):
            return str(int(round(num)))
        if any(tok in s for tok in ["ft", "foot", "feet"]):
            return str(int(round(num * 12)))
        # heuristic: <= 10 => feet, otherwise inches
        return str(int(round(num * 12))) if num <= 10 else str(int(round(num)))

    # --- Create a normalized inches column from whatever the sheet provides ---
    WIN = "_WIDTH_IN_NORM"
    if WIN not in gs_base.columns:
        cand_cols = ["Width (in)", "Width in", "Width Inches", "Width", COL_WIDTH_FT]
        def _pick_source(row):
            for c in cand_cols:
                if c in row.index:
                    v = row.get(c)
                    if str(v).strip():
                        return v
            return ""
        gs_base[WIN] = gs_base.apply(lambda r: _to_inches_str(_pick_source(r)), axis=1)

    # Guard: if we still couldn't derive any widths, fail clearly
    if gs_base[WIN].astype(str).str.strip().eq("").all():
        return jsonify({"found": False, "mode": "error", "message": "No usable width data for Grading Scrapers."}), 404

    # --- MODEL-FIRST (exact GS code) ---
    if model_raw:
        mm = GS_MODEL_RE.search(model_raw)
        if mm:
            model_code = mm.group(1).upper()
            rows = gs_base[gs_base[COL_MODEL].astype(str).str.upper() == model_code]
            if rows.empty:
                return jsonify({"found": False, "mode": "error", "message": f"Grading Scraper model not found: {model_code}"}), 404
            r, err = pick_one_or_400(rows, "Grading Scraper model", {"model": model_code})
            if err:
                return err
            model = str(r[COL_MODEL]).strip()
            pid = _first_part_id(r)
            label = model
            dsc = (r.get(COL_DESC) or "").strip()
            if dsc:
                label += f" — {dsc}"
            if pid:
                label += f" ({pid})"
            lines = [_make_line(label, 1, _price_number(r.get(COL_LIST_PRICE)), part_id=pid)]
            return _totals_payload(lines, [], "Grading Scraper", model, dealer_rate_override, dealer_meta)

    # --- SIZE-FIRST (width in inches) ---
    # Build available widths from the data
    widths_in = sorted({w for w in gs_base[WIN].astype(str).tolist() if w.isdigit()}, key=lambda x: int(x))

    def _fmt_label(inches_str: str) -> str:
        n = int(inches_str)
        ft = n / 12.0
        # show 1 or 2 decimals only if needed
        ft_str = f"{ft:.2f}".rstrip("0").rstrip(".")
        return f"{n} in ({ft_str} ft)"

    # Ask for size if not provided
    if not width_txt:
        choices = [_fmt_label(w) for w in widths_in]
        return jsonify({
            "found": True,
            "mode": "questions",
            "category": "Grading Scraper",
            "model": "",
            "required_questions": [{
                "name": "gs_width_in",
                "question": "Which Grading Scraper size do you want?",
                "choices": choices,
                "choices_with_ids": [{"id": w, "label": _fmt_label(w)} for w in widths_in],
                "multiple": False,
                "required": True
            }]
        })

    want_in = _to_inches_str(width_txt)
    if not want_in or want_in not in widths_in:
        # re-ask with valid choices
        choices = [_fmt_label(w) for w in widths_in]
        return jsonify({
            "found": True,
            "mode": "questions",
            "category": "Grading Scraper",
            "model": "",
            "required_questions": [{
                "name": "gs_width_in",
                "question": "Please pick one of the available sizes:",
                "choices": choices,
                "choices_with_ids": [{"id": w, "label": _fmt_label(w)} for w in widths_in],
                "multiple": False,
                "required": True
            }]
        })

    # Filter to chosen width and quote
    rows = gs_base[gs_base[WIN].astype(str) == want_in]
    if rows.empty:
        return jsonify({"found": False, "mode": "error", "message": f"No Grading Scraper rows at {want_in} inches."}), 404

    r, err = pick_one_or_400(rows, "Grading Scraper width selection", {"gs_width_in": want_in})
    if err:
        return err

    model = str(r[COL_MODEL]).strip()
    pid = _first_part_id(r)
    label = model
    dsc = (r.get(COL_DESC) or "").strip()
    if dsc:
        label += f" — {dsc}"
    if pid:
        label += f" ({pid})"

    lines = [_make_line(label, 1, _price_number(r.get(COL_LIST_PRICE)), part_id=pid)]
    return _totals_payload(lines, [], "Grading Scraper", model, dealer_rate_override, dealer_meta)

# =========================
# Landscake Rake (LRS)
# =========================

def _quote_landscape_rake(df):
    """
    Landscape Rake (LRSxx): size-first flow.
    - If exact model (e.g., LRS72 or LRS72P) is provided, quote directly.
    - Otherwise ask for width; if both Standard and Premium exist at that width,
      ask which version; then quote.
    - Accessories: use list_accessories=true to return selectable accessory list.
    """

    # --- Base rows (LRS only; exclude accessories/tires) ---
    lrs_base = _family_base(df, "Landscape")  # matches Landscape Rake family
    lrs_base = lrs_base[lrs_base[COL_MODEL].astype(str).str.upper().str.startswith("LRS")]
    if lrs_base.empty:
        return jsonify({"found": False, "mode": "error", "message": "No Landscape Rake base rows found."}), 404

    # --- Dealer context (standard pattern) ---
    dealer_number = getp("dealer_number")
    dealer_meta, dealer_rate_override = None, None
    if dealer_number and dealer_number in DEALER_DISCOUNTS:
        name, rate = DEALER_DISCOUNTS[dealer_number]
        dealer_meta = {"dealer_number": dealer_number, "dealer_name": name, "applied_discount": rate}
        dealer_rate_override = rate

    model_raw   = getp("model")
    width_txt   = getp("lrs_width_in", "width", "width_ft")  # accept inches or feet
    lrs_grade   = getp("lrs_grade")  # "Standard" or "Premium"
    want_access = (request.args.get("list_accessories") or "").strip().lower() in {"1","true","yes"}

    # --- Local inches parser (robust) ---
    def _to_inches_str(val) -> str:
        s = str(val or "").strip().lower()
        if not s:
            return ""
        m = re.search(r'(\d+(?:\.\d+)?)', s)
        if not m:
            return ""
        num = float(m.group(1))
        if any(tok in s for tok in ['"', "in", "inch"]):  # explicit inches
            return str(int(round(num)))
        if any(tok in s for tok in ["ft", "foot", "feet"]):  # explicit feet
            return str(int(round(num * 12)))
        # heuristic: <=10 -> feet, otherwise inches
        return str(int(round(num * 12))) if num <= 10 else str(int(round(num)))

    # --- Build a normalized inches column from available width columns ---
    WIN = "_WIDTH_IN_NORM"
    if WIN not in lrs_base.columns:
        cand_cols = ["Width (in)", "Width in", "Width Inches", "Width", COL_WIDTH_FT]
        def _pick_source(row):
            for c in cand_cols:
                if c in row.index:
                    v = row.get(c)
                    if str(v).strip():
                        return v
            return ""
        lrs_base[WIN] = lrs_base.apply(lambda r: _to_inches_str(_pick_source(r)), axis=1)

    if lrs_base[WIN].astype(str).str.strip().eq("").all():
        return jsonify({"found": False, "mode": "error", "message": "No usable width data for Landscape Rakes."}), 404

    # --- MODEL-FIRST ---
    if model_raw:
        mm = LRS_MODEL_RE.search(model_raw)
        if mm:
            model_code = mm.group(1).upper()
            rows = lrs_base[lrs_base[COL_MODEL].astype(str).str.upper() == model_code]
            if rows.empty:
                return jsonify({"found": False, "mode": "error", "message": f"Landscape Rake model not found: {model_code}"}), 404
            r, err = pick_one_or_400(rows, "LRS model", {"model": model_code})
            if err: return err

            model = str(r[COL_MODEL]).strip()
            pid   = _first_part_id(r)
            label = model
            dsc   = (r.get(COL_DESC) or "").strip()
            if dsc: label += f" — {dsc}"
            if pid: label += f" ({pid})"
            lines = [_make_line(label, 1, _price_number(r.get(COL_LIST_PRICE)), part_id=pid)]

            # Accessory follow-up path: if list_accessories true and nothing selected, return choices
            acc_df = accessories_for_model(df, model)
            if want_access and request.args.getlist("accessory_id") == [] and (request.args.get("accessory_ids") or "") == "" and request.args.getlist("accessory") == [] and request.args.getlist("accessory_desc") == []:
                return accessory_choices_payload(acc_df, "Landscape Rake", model)

            # Add any selected accessories
            acc_ids, acc_qty_map, acc_desc_terms = _read_accessory_params()
            lines += accessory_lines_from_selection(acc_df, acc_ids, acc_qty_map, acc_desc_terms)

            return _totals_payload(lines, [], "Landscape Rake", model, dealer_rate_override, dealer_meta)

    # --- SIZE-FIRST (width) ---
    widths_in = sorted({w for w in lrs_base[WIN].astype(str).tolist() if w.isdigit()}, key=lambda x: int(x))

    def _fmt_label(inches_str: str) -> str:
        n = int(inches_str); ft = n / 12.0
        ft_str = f"{ft:.2f}".rstrip("0").rstrip(".")
        return f"{n} in ({ft_str} ft)"

    if not width_txt:
        return jsonify({
            "found": True,
            "mode": "questions",
            "category": "Landscape Rake",
            "model": "",
            "required_questions": [{
                "name": "lrs_width_in",
                "question": "Which Landscape Rake size do you want?",
                "choices": [_fmt_label(w) for w in widths_in],
                "choices_with_ids": [{"id": w, "label": _fmt_label(w)} for w in widths_in],
                "multiple": False,
                "required": True
            }]
        })

    want_in = _to_inches_str(width_txt)
    if not want_in or want_in not in widths_in:
        return jsonify({
            "found": True,
            "mode": "questions",
            "category": "Landscape Rake",
            "model": "",
            "required_questions": [{
                "name": "lrs_width_in",
                "question": "Please pick one of the available Landscape Rake sizes:",
                "choices": [_fmt_label(w) for w in widths_in],
                "choices_with_ids": [{"id": w, "label": _fmt_label(w)} for w in widths_in],
                "multiple": False,
                "required": True
            }]
        })

    rows_at_width = lrs_base[lrs_base[WIN].astype(str) == want_in]
    if rows_at_width.empty:
        return jsonify({"found": False, "mode": "error", "message": f"No Landscape Rake rows at {want_in} inches."}), 404

    # Premium disambiguation (model ending with 'P')
    prem_rows = rows_at_width[rows_at_width[COL_MODEL].astype(str).str.upper().str.endswith("P")]
    std_rows  = rows_at_width[~rows_at_width[COL_MODEL].astype(str).str.upper().str.endswith("P")]

    if not lrs_grade:
        if (not prem_rows.empty) and (not std_rows.empty):
            return jsonify({
                "found": True,
                "mode": "questions",
                "category": "Landscape Rake",
                "model": _fmt_label(want_in),
                "required_questions": [{
                    "name": "lrs_grade",
                    "question": "Standard or Premium?",
                    "choices": ["Standard", "Premium"],
                    "multiple": False,
                    "required": True
                }]
            })
        # only one exists → infer it
        lrs_grade = "Premium" if not prem_rows.empty else "Standard"

    # Choose the row by grade
    chosen_rows = prem_rows if lrs_grade.strip().lower().startswith("prem") else std_rows
    r, err = pick_one_or_400(chosen_rows, "Landscape Rake final selection", {"lrs_width_in": want_in, "lrs_grade": lrs_grade})
    if err: return err

    model = str(r[COL_MODEL]).strip()
    pid   = _first_part_id(r)
    label = model
    dsc   = (r.get(COL_DESC) or "").strip()
    if dsc: label += f" — {dsc}"
    if pid: label += f" ({pid})"
    lines = [_make_line(label, 1, _price_number(r.get(COL_LIST_PRICE)), part_id=pid)]

    # Accessory follow-up (opt-in like the other families)
    acc_df = accessories_for_model(df, model)
    if want_access and request.args.getlist("accessory_id") == [] and (request.args.get("accessory_ids") or "") == "" and request.args.getlist("accessory") == [] and request.args.getlist("accessory_desc") == []:
        return accessory_choices_payload(acc_df, "Landscape Rake", model)

    acc_ids, acc_qty_map, acc_desc_terms = _read_accessory_params()
    lines += accessory_lines_from_selection(acc_df, acc_ids, acc_qty_map, acc_desc_terms)

    return _totals_payload(lines, [], "Landscape Rake", model, dealer_rate_override, dealer_meta)


# =========================
# Post Hole Diggers (PD)
# =========================

def _pd_duty_for_model(model_code: str) -> str:
    m = (model_code or "").upper().strip()
    if m == "PD25.21":
        return "Standard Duty (15–25 HP)"
    if m == "PD35.31":
        return "Standard Duty (20–35 HP)"
    if m == "PD95.51":
        return "Heavy Duty (35–95 HP)"
    return ""

def _pd_label(r, duty_text=""):
    model = str(r[COL_MODEL]).strip()
    dsc = str(r.get(COL_DESC, "") or "").strip()
    bits = [model]
    if duty_text:
        bits.append(duty_text)
    if dsc:
        bits.append(dsc)
    return " — ".join(bits)

def _quote_post_hole_digger(df):
    """
    Flow:
      - If pd_model (or exact model) given -> use it
      - Else ask which PD model (show HP ranges)
      - Then REQUIRE an auger selection (from accessories for that model)
      - Quote base + auger. Add note: must be ordered in packs of 2.
    """
    # ---- Dealer context
    dealer_number = getp("dealer_number")
    dealer_meta, dealer_rate_override = None, None
    if dealer_number and dealer_number in DEALER_DISCOUNTS:
        name, rate = DEALER_DISCOUNTS[dealer_number]
        dealer_meta = {"dealer_number": dealer_number, "dealer_name": name, "applied_discount": rate}
        dealer_rate_override = rate

    # ---- Base rows: PD models only, excluding accessories/tires/augers
    df = load_df()
    model_col = df[COL_MODEL].astype(str).str.upper()
    cat_col = df[COL_CATEGORY].astype(str)
    # Base PD rows: Model starts with PD and Category does NOT look like accessories/tires/auger
    pd_base = df[ model_col.str.startswith("PD") & (~cat_col.str.contains(ACC_TIRE_PATTERN, na=False)) ]
    if pd_base.empty:
        return jsonify({"found": False, "mode": "error", "message": "No Post Hole Digger base rows found."}), 404

    # ---- Params
    pd_model = _extract_pd_code(getp("pd_model", "model"))
    auger_id = getp("auger_id")
    auger_choice = getp("auger_choice")

    # ---- Ask PD model if not provided or not found
    if not pd_model or pd_base[pd_base[COL_MODEL].astype(str).str.upper() == pd_model].empty:
        choices = [
            {"id": "PD25.21", "label": "PD25.21 — Standard Duty (15–25 HP)"},
            {"id": "PD35.31", "label": "PD35.31 — Standard Duty (20–35 HP)"},
            {"id": "PD95.51", "label": "PD95.51 — Heavy Duty (35–95 HP)"},
        ]
        return jsonify({
            "found": True,
            "mode": "questions",
            "category": "Post Hole Digger",
            "model": "",
            "required_questions": [{
                "name": "pd_model",
                "question": "Which Post Hole Digger model?",
                "choices": [c["label"] for c in choices],
                "choices_with_ids": choices,
                "multiple": False,
                "required": True
            }]
        })

    # ---- Resolve the chosen PD base row
    rows = pd_base[pd_base[COL_MODEL].astype(str).str.upper() == pd_model]
    r, err = pick_one_or_400(rows, "Post Hole Digger model", {"pd_model": pd_model})
    if err:
        return err

    model = str(r[COL_MODEL]).strip()
    pid = _first_part_id(r)
    duty_text = _pd_duty_for_model(model)
    base_label = _pd_label(r, duty_text=duty_text) + (f" ({pid})" if pid else "")

    # ---- Augers (REQUIRED) – from accessories for this PD model
    acc_df = accessories_for_model(df, model)
    # Keep only auger-like rows
    augers = acc_df[
        acc_df[COL_CATEGORY].astype(str).str.contains("Auger", case=False, na=False) |
        acc_df[COL_DESC].astype(str).str.contains(r"\bauger\b", case=False, na=False)
    ].copy()

    if augers.empty:
        return jsonify({"found": False, "mode": "error", "message": f"No auger accessories found for {model}."}), 404

    # If no auger chosen yet -> ask
    if not auger_id and not auger_choice:
        choices, choices_with_ids = [], []
        for _, ar in augers.iterrows():
            apid = _first_part_id(ar)
            albl = (ar.get(COL_DESC) or "").strip()
            if apid and albl:
                choices.append(albl)
                choices_with_ids.append({"id": apid, "label": albl})
        return jsonify({
            "found": True,
            "mode": "questions",
            "category": "Post Hole Digger",
            "model": model,
            "required_questions": [{
                "name": "auger_id",
                "question": "Select an auger for the Post Hole Digger (required):",
                "choices": choices,
                "choices_with_ids": choices_with_ids,
                "multiple": False,
                "required": True
            }]
        })

    # Resolve auger selection
    auger_row = None
    if auger_id:
        hit = _find_by_part_id(augers, auger_id)
        if not hit.empty:
            auger_row = hit.iloc[0]
    if auger_row is None and auger_choice:
        sub = augers[augers[COL_DESC].astype(str).str.contains(auger_choice, case=False, na=False)]
        if not sub.empty:
            auger_row = sub.iloc[0]
    if auger_row is None:
        return jsonify({"found": False, "mode": "error", "message": "Selected auger not found for this PD model."}), 404

    aug_pid = _first_part_id(auger_row)
    aug_desc = (auger_row.get(COL_DESC) or "Auger").strip()

    # ---- Build lines
    lines = [
        _make_line(base_label, 1, _price_number(r.get(COL_LIST_PRICE)), part_id=pid),
        _make_line(f"{aug_desc} ({aug_pid})", 1, _price_number(auger_row.get(COL_LIST_PRICE)), part_id=aug_pid),
    ]

    notes = [
        "Note: Post Hole Diggers must be ordered in packs of 2."
    ]

    return _totals_payload(
        lines,
        notes,
        "Post Hole Digger",
        model,
        dealer_rate_override,
        dealer_meta
    )

# =========================
# Rear Blades (RB)
# =========================

def _rb_duty_from_row(r):
    """Map sheet category + model suffix to a clean duty label."""
    cat = (str(r.get(COL_CATEGORY, "") or "")).strip().lower()
    model = (str(r.get(COL_MODEL, "") or "")).strip().upper()
    # Base from Category
    if "extreme" in cat:
        duty = "Extreme Duty"
    elif "heavy" in cat:
        duty = "Heavy Duty"
    else:
        duty = "Standard"
    # Premium flag on Standard only (P suffix)
    if duty == "Standard" and model.endswith("P"):
        return "Standard (Premium)"
    return duty

def _rb_label(r, include_desc=True):
    m = (str(r.get(COL_MODEL, "") or "")).strip()
    d = _rb_duty_from_row(r)
    bits = [m, d]
    if include_desc:
        dsc = (str(r.get(COL_DESC, "") or "")).strip()
        if dsc:
            bits.append(dsc)
    return " — ".join([b for b in bits if b])

def _inches_from_ft_str(ft_str):
    try:
        f = int(ft_str)
        return str(f * 12)
    except:
        return ""

def _feet_from_inches_str(inches_str):
    try:
        i = float(re.search(r"(\d+(?:\.\d+)?)", str(inches_str)).group(1))
        return str(int(round(i / 12.0)))
    except:
        return ""

def _quote_rear_blade(df):
    """Rear Blades (RBxx / RBxxP). Flow:
       - Model-first → quote that exact model.
       - Else ask width (as inches IDs), then ask duty if multiple at that width:
           Standard, Standard (Premium), Heavy Duty, Extreme Duty.
       - If multiple models remain (different option packages), ask to choose.
       - Support optional accessories list/selection like other families.
    """
    # Base rows (exclude accessories/tires)
    rb_base = _family_base(df, "Rear Blade")
    if rb_base.empty:
        return jsonify({"found": False, "mode": "error", "message": "No Rear Blade base rows found."}), 404

    # Dealer context
    dealer_number = getp("dealer_number")
    dealer_meta, dealer_rate_override = None, None
    if dealer_number and dealer_number in DEALER_DISCOUNTS:
        name, rate = DEALER_DISCOUNTS[dealer_number]
        dealer_meta = {"dealer_number": dealer_number, "dealer_name": name, "applied_discount": rate}
        dealer_rate_override = rate

    # Params
    model_raw = getp("model")
    rb_width_in = getp("rb_width_in")
    rb_duty = getp("rb_duty")
    choice_id = getp("rb_choice_id", "accessory_id")
    choice_label = getp("rb_choice")
    list_access = (request.args.get("list_accessories") or "").strip() not in ("", "0", "false", "False")
    acc_ids, acc_qty_map, acc_desc_terms = _read_accessory_params()

    wn = _ensure_width_norm_col(rb_base)

    # --- MODEL-FIRST ---
    model_code = ""
    m = RB_MODEL_RE.search(model_raw or "")
    if m:
        model_code = m.group(1).upper()
    if model_code:
        rows = rb_base[rb_base[COL_MODEL].astype(str).str.upper() == model_code]
        if rows.empty:
            return jsonify({"found": False, "mode": "error", "message": f"Rear Blade model not found: {model_code}"}), 404
    else:
        # --- WIDTH-FIRST ---
        if not rb_width_in:
            # Offer widths from the sheet (use feet col → display inches)
            widths_ft = sorted({s for s in rb_base[wn].astype(str).tolist() if s}, key=lambda x: int(x))
            if not widths_ft:
                return jsonify({"found": False, "mode": "error", "message": "No valid widths found for Rear Blades."}), 404
            choices, choices_with_ids = [], []
            for f in widths_ft:
                inches = _inches_from_ft_str(f)
                if inches:
                    label = f"{inches} in ({f} ft)"
                    choices.append(label)
                    choices_with_ids.append({"id": inches, "label": label})
            return jsonify({
                "found": True,
                "mode": "questions",
                "category": "Rear Blade",
                "model": "",
                "required_questions": [{
                    "name": "rb_width_in",
                    "question": "Which Rear Blade width do you want?",
                    "choices": choices,
                    "choices_with_ids": choices_with_ids
                }]
            })
        # Normalize inches→feet for filtering
        width_ft = _feet_from_inches_str(rb_width_in)
        rows = rb_base[rb_base[wn].astype(str) == width_ft]
        if rows.empty:
            return jsonify({"found": False, "mode": "error",
                            "message": f"No Rear Blade rows at {rb_width_in} in ({width_ft} ft)."}), 404

        # Duty stage (only if multiple duties exist at this width)
        duties_avail = sorted({ _rb_duty_from_row(r) for _, r in rows.iterrows() })
        if len(duties_avail) > 1 and not rb_duty:
            return jsonify({
                "found": True,
                "mode": "questions",
                "category": "Rear Blade",
                "model": f"{rb_width_in} in",
                "required_questions": [{
                    "name": "rb_duty",
                    "question": f"Which duty class for {rb_width_in} in Rear Blade?",
                    "choices": duties_avail
                }]
            })
        if rb_duty:
            want = rb_duty.strip().lower()
            def duty_match(rr):
                d = _rb_duty_from_row(rr).strip().lower()
                return d == want
            rows = rows[rows.apply(duty_match, axis=1)]
            if rows.empty:
                return jsonify({"found": False, "mode": "error",
                                "message": f"No Rear Blade rows at {rb_width_in} in for duty '{rb_duty}'."}), 404

    # If more than one row remains (different option packages), ask to choose
    if len(rows.index) > 1 and not (choice_id or choice_label):
        str_choices, id_choices = [], []
        for _, r in rows.iterrows():
            lbl = _rb_label(r, include_desc=True)
            rid = _first_part_id(r) or lbl
            str_choices.append(lbl)
            id_choices.append({"id": rid, "label": lbl})
        return jsonify({
            "found": True,
            "mode": "questions",
            "category": "Rear Blade",
            "model": model_code or (rb_width_in and f"{rb_width_in} in") or "",
            "required_questions": [{
                "name": "rb_choice",
                "question": "Which Rear Blade configuration would you like?",
                "choices": str_choices,
                "choices_with_ids": id_choices
            }]
        })

    # Finalize selection
    chosen = rows
    if choice_id or choice_label:
        chosen = _select_by_id_or_label(rows, choice_id, choice_label, label_fn=_rb_label)
    r, err = pick_one_or_400(chosen, "Rear Blade final selection",
                             {"rb_width_in": rb_width_in, "rb_duty": rb_duty,
                              "choice_id": choice_id, "choice_label": choice_label})
    if err:
        return err

    model = str(r[COL_MODEL]).strip()
    pid = _first_part_id(r)
    label = _rb_label(r, include_desc=True) + (f" ({pid})" if pid else "")
    lines = [_make_line(label, 1, _price_number(r.get(COL_LIST_PRICE)), part_id=pid)]

    # Accessories
    acc_df = accessories_for_model(df, model)
    if acc_df.empty:
        acc_df = _rb_accessories_for_model(df, model)
    if list_access and not acc_ids and not acc_desc_terms:
        return accessory_choices_payload(acc_df, "Rear Blade", model, multi=True, required=False,
                                         name="accessory_ids",
                                         question="Select accessories to add (optional).")
    lines += accessory_lines_from_selection(acc_df, acc_ids, acc_qty_map, acc_desc_terms)

    return _totals_payload(
        lines,
        ["Select optional accessories as needed."],
        "Rear Blade",
        model,
        dealer_rate_override,
        dealer_meta,
    )

# =========================
# Disc Harrow (DHS / DHM)
# =========================

def _dh_primary_value(qname: str, picked_label: str) -> str:
    """Normalize Disc Harrow labels into the exact values the backend expects."""
    q = (qname or "").strip().lower()
    label = (picked_label or "").strip()

    # Width: keep just the number of inches
    if q == "dh_width_in":
        m = re.search(r"\d+(?:\.\d+)?", label)
        return m.group(0) if m else label

    # Blade: standardize to "Notched (N)" or "Combo (C)"
    if q == "dh_blade":
        l = label.lower()
        if "notched" in l or l.endswith("(n)") or l.endswith(" n"):
            return "Notched (N)"
        if "combo" in l or l.endswith("(c)") or l.endswith(" c"):
            return "Combo (C)"
        return label

    # Everything else: pass through as-is
    return label


def build_dh_answer_params(qname: str, picked_label: str, picked_id: str | None) -> dict:
    """
    For Disc Harrow only:
      - Always send qname with a normalized value.
      - If an ID exists, also send qname_id=<id>.
    """
    base = (qname or "").strip()
    val = _dh_primary_value(base, picked_label or "")
    params = {base: val}
    if picked_id:
        params[f"{base}_id"] = str(picked_id)
    return params


def _quote_disc_harrow(df):
    """
    Disc Harrow (DHS / DHM)
    Flow:
      1) Width (inches)
      2) Duty (Standard=DHS / Heavy Duty=DHM)
      3) Blade style (Notched N / Combo C)
      4) Disc spacing (if multiple spacing rows remain)
    Then quote, with optional accessories list/selection.
    """

    # ---- Base rows: DHS / DHM only, exclude accessories/tires ----
    allrows = df[df[COL_MODEL].astype(str).str.upper().str.startswith(("DHS", "DHM"))]
    base = allrows[~allrows[COL_CATEGORY].astype(str).str.contains(ACC_TIRE_PATTERN, na=False)]
    if base.empty:
        return jsonify({"found": False, "mode": "error", "message": "No Disc Harrow rows found."}), 404

    # ---- Dealer context ----
    dealer_number = getp("dealer_number")
    dealer_meta, dealer_rate_override = None, None
    if dealer_number and dealer_number in DEALER_DISCOUNTS:
        name, rate = DEALER_DISCOUNTS[dealer_number]
        dealer_meta = {"dealer_number": dealer_number, "dealer_name": name, "applied_discount": rate}
        dealer_rate_override = rate

    # ---- Params ----
    model_raw   = getp("model")
    dh_width_in = getp("dh_width_in")
    dh_duty     = getp("dh_duty")
    dh_blade    = getp("dh_blade")

    # Accept both correct and legacy spacing names  # <<< NEW
    choice_id   = getp("dh_choice_id", "choice_id", "dh_spacing_id")  # <<< NEW
    choice_lbl  = getp("dh_choice", "choice")                          # (label stays the same)

    # If width missing, accept dh_width_in_id as value                    # <<< NEW
    if not dh_width_in:
        dh_width_in = getp("dh_width_in_id") or dh_width_in
    # Normalize width to just digits                                      # <<< NEW
    if dh_width_in:
        m = re.search(r"\d+(?:\.\d+)?", str(dh_width_in))
        if m:
            dh_width_in = m.group(0)

    list_access = (request.args.get("list_accessories") or "").strip() not in ("", "0", "false", "False")
    acc_ids, acc_qty_map, acc_desc_terms = _read_accessory_params()

    # ---- Normalize blade input so "N"/"C" (or aliases) work everywhere ----
    blade_raw = (str(dh_blade or "").strip().lower())
    if blade_raw in {"n", "notched", "notch", "notched (n)"}:
        dh_blade = "Notched (N)"
    elif blade_raw in {"c", "combo", "combination", "combo (c)"}:
        dh_blade = "Combo (C)"
    else:
        # also accept id-style params: dh_blade_id=N/C
        blade_id = (str(getp("dh_blade_id") or "").strip().lower())
        if not blade_raw and blade_id in {"n", "c"}:
            dh_blade = "Notched (N)" if blade_id == "n" else "Combo (C)"
    # ---- END normalize blade ----

    # ---- local helpers (self-contained) ----
    def _to_inches_str(val) -> str:
        s = str(val or "").strip().lower()
        if not s: return ""
        m = re.search(r'(\d+(?:\.\d+)?)', s)
        if not m: return ""
        num = float(m.group(1))
        if any(t in s for t in ['"', "in", "inch", "inches"]): return str(int(round(num)))
        if any(t in s for t in ["ft", "foot", "feet"]):        return str(int(round(num * 12)))
        return str(int(round(num))) if num > 10 else str(int(round(num * 12)))

    def width_in_from_row(r) -> str:
        # Prefer explicit inches column if present; else derive from ft; else parse model
        for cname in r.index:
            cl = str(cname).strip().lower()
            if cl in {"width (in)", "width in", "width inches", "width-in"}:
                return _to_inches_str(r.get(cname))
        if COL_WIDTH_FT in r.index and str(r.get(COL_WIDTH_FT) or "").strip():
            # treat as feet -> inches
            s = str(r.get(COL_WIDTH_FT)).strip()
            m = re.search(r'(\d+(?:\.\d+)?)', s)
            if m:
                try:
                    return str(int(round(float(m.group(1)) * 12)))
                except:
                    pass
        # last resort: parse from model (DHS72N -> 72)
        w = _dh_width_in_from_model(str(r.get(COL_MODEL) or ""))
        return str(w) if w else ""

    def duty_label_from_model(m: str) -> str:
        mu = (m or "").upper().strip()
        if mu.startswith("DHS"): return "Standard (DHS)"
        if mu.startswith("DHM"): return "Heavy Duty (DHM)"
        return ""

    def blade_label_from_model(m: str) -> str:
        mu = (m or "").upper().strip()
        if mu.endswith("N"): return "Notched (N)"
        if mu.endswith("C"): return "Combo (C)"
        return ""

    def _dh_find_spacing_cols(frame):
        lc = [str(c).strip().lower() for c in frame.columns]
        def pick(*names):
            for want in names:
                want_l = want.lower()
                for c, cl in zip(frame.columns, lc):
                    if cl == want_l:
                        return c
            return None
        front = pick("Front Disc Spacing", "Front Spacing")
        rear  = pick("Rear Disc Spacing", "Rear Spacing")
        if front is None:
            for c, cl in zip(frame.columns, lc):
                if "front" in cl and "spacing" in cl:
                    front = c; break
        if rear is None:
            for c, cl in zip(frame.columns, lc):
                if "rear" in cl and "spacing" in cl:
                    rear = c; break
        return front, rear

    def _dh_spacing_label(row, front_col, rear_col):
        f = str(row.get(front_col, "") or "").strip() if front_col else ""
        r = str(row.get(rear_col, "") or "").strip() if rear_col else ""
        def norm(x):
            m = re.search(r"(\d+(\.\d+)?)", x)
            return m.group(1) if m else x
        f, r = norm(f), norm(r)
        if f and r and f != r:
            return f"{f} in front / {r} in rear"
        if f or r:
            return f"{f or r} in"
        return "Standard spacing"

    # Two label forms:
    # 1) choice label used for asking/matching (must be identical both times)
    # 2) invoice label used on the final quote line item (adds duty/blade/desc)
    def spacing_choice_label(rr):
        fcol, rcol = _dh_find_spacing_cols(base)
        spacing_txt = _dh_spacing_label(rr, fcol, rcol) if (fcol or rcol) else "Standard spacing"
        return f"{str(rr[COL_MODEL]).strip()} — {spacing_txt}"

    def invoice_label(rr):
        fcol, rcol = _dh_find_spacing_cols(base)
        spacing_txt = _dh_spacing_label(rr, fcol, rcol) if (fcol or rcol) else ""
        model = str(rr[COL_MODEL]).strip()
        dsc   = (rr.get(COL_DESC) or "").strip()
        duty  = duty_label_from_model(model)
        blade = blade_label_from_model(model)
        bits  = [model]
        if duty:  bits.append(duty)
        if blade: bits.append(blade)
        if spacing_txt: bits.append(f"Spacing: {spacing_txt}")
        if dsc: bits.append(dsc)
        return " — ".join(bits)

    def ask_width(rows_for_width_choice):
        wset = sorted({
            width_in_from_row(x) for _, x in rows_for_width_choice.iterrows()
        }, key=lambda x: int(re.search(r"\d+", x).group()) if re.search(r"\d+", x) else 9999)
        wset = [w for w in wset if w]
        if not wset:
            return jsonify({"found": False, "mode": "error", "message": "No Disc Harrow widths available."}), 404
        choices = [f'{w} in ({int(w)//12} ft)' if re.match(r"^\d+$", w) else f'{w} in' for w in wset]
        return jsonify({
            "found": True, "mode": "questions", "category": "Disc Harrow", "model": "",
            "required_questions": [{
                "name": "dh_width_in",
                "question": "What working width do you want?",
                "choices": choices,
                "choices_with_ids": [{"id": w, "label": c} for w, c in zip(wset, choices)],
                "multiple": False, "required": True
            }]
        })

    def ask_duty(rows_for_duty, width_label: str):
        duties = sorted({duty_label_from_model(str(m)) for m in rows_for_duty[COL_MODEL].astype(str)},
                        key=lambda s: {"Standard (DHS)": 1, "Heavy Duty (DHM)": 2}.get(s, 99))
        duties = [d for d in duties if d]
        return jsonify({
            "found": True, "mode": "questions", "category": "Disc Harrow",
            "model": f"{width_label} in Disc Harrow",
            "required_questions": [{
                "name": "dh_duty",
                "question": f"Which duty class for {width_label} in?",
                "choices": duties,
                "multiple": False, "required": True
            }]
        })

    def ask_blade(rows_for_blade, width_label: str):
        blades = sorted({blade_label_from_model(str(m)) for m in rows_for_blade[COL_MODEL].astype(str)})
        blades = [b for b in blades if b]
        return jsonify({
            "found": True, "mode": "questions", "category": "Disc Harrow",
            "model": f"{width_label} in Disc Harrow",
            "required_questions": [{
                "name": "dh_blade",
                "question": "Which blade style?",
                "choices": blades,
                "multiple": False, "required": True
            }]
        })

    def ask_spacing(rows_for_spacing, model_label: str):
        choices, choices_with_ids = [], []
        for _, rr in rows_for_spacing.iterrows():
            pid = _first_part_id(rr)
            if not pid:
                continue
            lbl = spacing_choice_label(rr)  # <- EXACT label used for matching
            choices.append(lbl)
            choices_with_ids.append({"id": pid, "label": lbl})
        return jsonify({
            "found": True, "mode": "questions", "category": "Disc Harrow",
            "model": model_label,
            "required_questions": [{
                "name": "dh_choice",
                "question": "Which disc spacing would you like?",
                "choices": choices,
                "choices_with_ids": choices_with_ids,
                "multiple": False, "required": True
            }]
        })

    # ======================
    # MODEL-FIRST
    # ======================
    if model_raw:
        m = DH_MODEL_RE.search(model_raw)
        if m:
            model_code = m.group(1).upper().strip()
            base_no_suffix = re.sub(r"[NC]$", "", model_code)
            rows = base[base[COL_MODEL].astype(str).str.upper().str.startswith(base_no_suffix)]
            if rows.empty:
                return jsonify({"found": False, "mode": "error",
                                "message": f"Disc Harrow model not found: {model_code}"}), 404

            # duty filter (optional)
            if dh_duty:
                if "standard" in dh_duty.lower():
                    rows = rows[rows[COL_MODEL].astype(str).str.upper().str.startswith("DHS")]
                elif "heavy" in dh_duty.lower():
                    rows = rows[rows[COL_MODEL].astype(str).str.upper().str.startswith("DHM")]
                if rows.empty:
                    return jsonify({"found": False, "mode": "error",
                                    "message": f"No {dh_duty} rows for {model_code}."}), 404

            # blade filter (optional)
            if dh_blade:
                bl = dh_blade.strip().lower()
                if "notched" in bl or bl.endswith("(n)"):
                    rows = rows[rows[COL_MODEL].astype(str).str.upper().str.endswith("N")]
                elif "combo" in bl or bl.endswith("(c)"):
                    rows = rows[rows[COL_MODEL].astype(str).str.upper().str.endswith("C")]
                if rows.empty:
                    return jsonify({"found": False, "mode": "error",
                                    "message": f"No rows with blade '{dh_blade}' for {model_code}."}), 404

            # spacing question if multiple
            if len(rows.index) > 1 and not (choice_id or choice_lbl):
                return ask_spacing(rows, model_code)

            # final selection (match by PID or EXACT spacing label)
            chosen = rows
            if choice_id or choice_lbl:
                chosen = _select_by_id_or_label(rows, choice_id, choice_lbl, label_fn=spacing_choice_label)
            r, err = pick_one_or_400(chosen, "Disc Harrow final selection",
                                     {"choice_id": choice_id, "choice_label": choice_lbl})
            if err: return err

            model = str(r[COL_MODEL]).strip()
            pid   = _first_part_id(r)
            final_label = invoice_label(r) + (f" ({pid})" if pid else "")
            lines = [_make_line(final_label, 1, _price_number(r.get(COL_LIST_PRICE)), part_id=pid)]

            # accessories (optional)
            acc_df = accessories_for_model(df, model)
            if list_access and not acc_ids and not acc_desc_terms:
                return accessory_choices_payload(acc_df, "Disc Harrow", model, multi=True, required=False)
            lines += accessory_lines_from_selection(acc_df, acc_ids, acc_qty_map, acc_desc_terms)

            return _totals_payload(lines, [], "Disc Harrow", model, dealer_rate_override, dealer_meta)

    # ======================
    # WIDTH-FIRST
    # ======================
    base = base.copy()
    base["__DH_WIDTH_IN"] = base.apply(width_in_from_row, axis=1)
    wset = sorted({w for w in base["__DH_WIDTH_IN"].astype(str).tolist() if w},
                  key=lambda x: int(re.search(r"\d+", x).group()))

    if not dh_width_in:
        return ask_width(base)

    rows = base[base["__DH_WIDTH_IN"].astype(str) == str(dh_width_in)]
    if rows.empty:
        return jsonify({"found": False, "mode": "error",
                        "message": f"No Disc Harrow rows at {dh_width_in} in."}), 404

    # duty stage
    if not dh_duty:
        return ask_duty(rows, dh_width_in)
    if "standard" in dh_duty.lower():
        rows = rows[rows[COL_MODEL].astype(str).str.upper().str.startswith("DHS")]
    elif "heavy" in dh_duty.lower():
        rows = rows[rows[COL_MODEL].astype(str).str.upper().str.startswith("DHM")]
    if rows.empty:
        return jsonify({"found": False, "mode": "error",
                        "message": f"No {dh_duty} rows at {dh_width_in} in."}), 404

    # blade stage
    if not dh_blade:
        return ask_blade(rows, dh_width_in)

    bl = dh_blade.strip().lower()
    if "notched" in bl or bl.endswith("(n)"):
        rows = rows[rows[COL_MODEL].astype(str).str.upper().str.endswith("N")]
    elif "combo" in bl or bl.endswith("(c)"):
        rows = rows[rows[COL_MODEL].astype(str).str.upper().str.endswith("C")]

    if isinstance(rows, pd.Series):  # safety if previous line produced a mask
        rows = base[rows]

    if rows.empty:
        return jsonify({"found": False, "mode": "error",
                        "message": f"No rows with blade '{dh_blade}' at {dh_width_in} in."}), 404

    # spacing question if multiple remain
    if len(rows.index) > 1 and not (choice_id or choice_lbl):
        return ask_spacing(rows, f"{dh_width_in} in Disc Harrow")

    # final selection (match by PID or EXACT spacing label)
    chosen = rows
    if choice_id or choice_lbl:
        chosen = _select_by_id_or_label(rows, choice_id, choice_lbl, label_fn=spacing_choice_label)
    r, err = pick_one_or_400(chosen, "Disc Harrow final selection",
                             {"width_in": dh_width_in, "duty": dh_duty, "blade": dh_blade,
                              "choice_id": choice_id, "choice_label": choice_lbl})
    if err: return err

    model = str(r[COL_MODEL]).strip()
    pid   = _first_part_id(r)
    final_label = invoice_label(r) + (f" ({pid})" if pid else "")
    lines = [_make_line(final_label, 1, _price_number(r.get(COL_LIST_PRICE)), part_id=pid)]

    # accessories (optional)
    acc_df = accessories_for_model(df, model)
    if list_access and not acc_ids and not acc_desc_terms:
        return accessory_choices_payload(acc_df, "Disc Harrow", model, multi=True, required=False)
    lines += accessory_lines_from_selection(acc_df, acc_ids, acc_qty_map, acc_desc_terms)

    return _totals_payload(lines, [], "Disc Harrow", model, dealer_rate_override, dealer_meta)

# =========================
# Tillers (DB + RT)
# =========================

def _tiller_rotation_from_model(model: str) -> str:
    s = (model or "").upper().strip()
    if s.startswith("RTR"):
        return "Reverse"
    if s.startswith("RT"):
        return "Forward"
    return ""  # DB has no rotation

def _tiller_width_in_from_model(model: str) -> str:
    s = (model or "").upper()
    m = re.search(r'(\d{2,3})', s)
    return str(int(m.group(1))) if m else ""

def _tiller_label(r):
    model = str(r.get(COL_MODEL, "") or "").strip()
    rot = _tiller_rotation_from_model(model)
    dsc = str(r.get(COL_DESC, "") or "").strip()
    bits = [model]
    if rot:
        bits.append(f"{rot} Rotation")
    if dsc:
        bits.append(dsc)
    return " — ".join(bits)

def _quote_tiller(df):
    """
    Tillers:
      Flow:
        1) Ask DB (Light Duty Dirt Breaker) vs RT (Commercial Duty).
        2) Ask size (inches) for the chosen series.
        3) If RT and both RT/RTR exist at that width → ask Forward vs Reverse.
        4) Quote base. Accessories optional via list_accessories.
    """
    # Base rows: Column A 'Category' contains 'Tiller'; exclude Accessories/Tires
    base_all = df[df[COL_CATEGORY].astype(str).str.contains(r"\btiller\b", case=False, na=False)]
    base = base_all[~base_all[COL_CATEGORY].astype(str).str.contains(ACC_TIRE_PATTERN, na=False)]
    if base.empty:
        return jsonify({"found": False, "mode": "error", "message": "No Tiller rows found."}), 404

    # Dealer context
    dealer_number = getp("dealer_number")
    dealer_meta, dealer_rate_override = None, None
    if dealer_number and dealer_number in DEALER_DISCOUNTS:
        name, rate = DEALER_DISCOUNTS[dealer_number]
        dealer_meta = {"dealer_number": dealer_number, "dealer_name": name, "applied_discount": rate}
        dealer_rate_override = rate

    # Params
    model_raw       = getp("model")
    series          = getp("tiller_series").upper()
    width_in        = getp("tiller_width_in")
    rotation_param  = getp("tiller_rotation")
    choice_id       = getp("tiller_choice_id", "accessory_id")
    choice_label    = getp("tiller_choice")

    list_access = (request.args.get("list_accessories") or "").strip().lower() in {"1","true","yes"}
    acc_ids, acc_qty_map, acc_desc_terms = _read_accessory_params()

    # MODEL-FIRST
    mm = TILLER_MODEL_RE.search(model_raw or "")
    if mm:
        model_code = mm.group(1).upper().replace(" ", "")
        rows = base[base[COL_MODEL].astype(str).str.upper().str.replace(" ", "", regex=False) == model_code]
        if rows.empty:
            return jsonify({"found": False, "mode": "error", "message": f"Tiller model not found: {model_code}"}), 404

        r, err = pick_one_or_400(rows, "Tiller model", {"model": model_code})
        if err: return err

        model = str(r[COL_MODEL]).strip()
        pid   = _first_part_id(r)
        label = _tiller_label(r) + (f" ({pid})" if pid else "")
        lines = [_make_line(label, 1, _price_number(r.get(COL_LIST_PRICE)), part_id=pid)]

        acc_df = accessories_for_model(df, model)
        if list_access and not acc_ids and not acc_desc_terms:
            return accessory_choices_payload(acc_df, "Tiller", model)
        lines += accessory_lines_from_selection(acc_df, acc_ids, acc_qty_map, acc_desc_terms)

        return _totals_payload(lines, [], "Tiller", model, dealer_rate_override, dealer_meta)

    # SERIES-FIRST
    if not series:
        return jsonify({
            "found": True,
            "mode": "questions",
            "category": "Tiller",
            "model": "",
            "required_questions": [{
                "name": "tiller_series",
                "question": "Which type of tiller do you want?",
                "choices": [
                    "DB — Light Duty Dirt Breaker",
                    "RT — Commercial Duty Rotary Tiller"
                ],
                "choices_with_ids": [
                    {"id": "DB", "label": "DB — Light Duty Dirt Breaker"},
                    {"id": "RT", "label": "RT — Commercial Duty Rotary Tiller"}
                ],
                "multiple": False,
                "required": True
            }]
        })

    series = "DB" if series.startswith("DB") else "RT"  # normalize

    if series == "DB":
        rows = base[base[COL_MODEL].astype(str).str.upper().str.startswith("DB")]
    else:  # RT series includes RT (forward) and RTR (reverse)
        ms = base[COL_MODEL].astype(str).str.upper()
        rows = base[ms.str.startswith("RT") | ms.str.startswith("RTR")]

    if rows.empty:
        return jsonify({"found": False, "mode": "error", "message": f"No rows for {series} series."}), 404

    # WIDTH SELECTION (inches)
    win = _ensure_width_in_norm_col(rows)  # uses Width (in) if present else derives
    # backfill blanks from model token (RT72 → 72)
    def _backfill_width(v, m):
        s = str(v).strip()
        if s.isdigit():
            return s
        return _tiller_width_in_from_model(m)

    rows = rows.copy()
    rows[win] = rows.apply(lambda rr: _backfill_width(rr.get(win, ""), rr.get(COL_MODEL, "")), axis=1)

    if not width_in:
        widths = sorted({w for w in rows[win].astype(str).tolist() if w.isdigit()}, key=lambda x: int(x))
        if not widths:
            return jsonify({"found": False, "mode": "error", "message": "No tiller sizes available."}), 404
        def _fmt(w): return f'{w} in ({_inches_to_feet_label(w)})'
        return jsonify({
            "found": True,
            "mode": "questions",
            "category": "Tiller",
            "model": "DB" if series=="DB" else "RT",
            "required_questions": [{
                "name": "tiller_width_in",
                "question": f"What size {series} tiller do you want?",
                "choices": [_fmt(w) for w in widths],
                "choices_with_ids": [{"id": w, "label": _fmt(w)} for w in widths],
                "multiple": False,
                "required": True
            }]
        })

    rows = rows[rows[win].astype(str) == str(width_in).strip()]
    if rows.empty:
        return jsonify({"found": False, "mode": "error", "message": f"No {series} tiller at {width_in} in."}), 404

    # ROTATION (RT only)
    if series == "RT":
        ms = rows[COL_MODEL].astype(str).str.upper()
        has_forward = any(ms.str.startswith("RT") & (~ms.str.startswith("RTR")))
        has_reverse = any(ms.str.startswith("RTR"))

        want_rot = ""
        if rotation_param:
            s = rotation_param.strip().lower()
            if s.startswith("f"): want_rot = "Forward"
            elif s.startswith("r"): want_rot = "Reverse"

        if has_forward and has_reverse and not want_rot:
            return jsonify({
                "found": True,
                "mode": "questions",
                "category": "Tiller",
                "model": f"{width_in} in RT",
                "required_questions": [{
                    "name": "tiller_rotation",
                    "question": "Forward or Reverse rotation?",
                    "choices": ["Forward (RT)", "Reverse (RTR)"],
                    "choices_with_ids": [
                        {"id": "forward", "label": "Forward (RT)"},
                        {"id": "reverse", "label": "Reverse (RTR)"}
                    ],
                    "multiple": False,
                    "required": True
                }]
            })

        # apply rotation filter if needed
        if want_rot == "Forward":
            ms = rows[COL_MODEL].astype(str).str.upper()
            rows = rows[ms.str.startswith("RT") & (~ms.str.startswith("RTR"))]
        elif want_rot == "Reverse":
            rows = rows[rows[COL_MODEL].astype(str).str.upper().str.startswith("RTR")]

        if rows.empty:
            return jsonify({"found": False, "mode": "error",
                            "message": f"No RT tiller at {width_in} in with rotation '{rotation_param}'."}), 404

    # If more than one remains (rare), ask to choose by part id
    if len(rows.index) > 1 and not (choice_id or choice_label):
        str_choices, id_choices = [], []
        for _, rr in rows.iterrows():
            lbl = _tiller_label(rr)
            rid = _first_part_id(rr) or lbl
            str_choices.append(lbl)
            id_choices.append({"id": rid, "label": lbl})
        return jsonify({
            "found": True,
            "mode": "questions",
            "category": "Tiller",
            "model": f"{series} — {width_in} in",
            "required_questions": [{
                "name": "tiller_choice",
                "question": "Which configuration would you like?",
                "choices": str_choices,
                "choices_with_ids": id_choices,
                "multiple": False,
                "required": True
            }]
        })

    # Final selection
    chosen = rows
    if choice_id or choice_label:
        chosen = _select_by_id_or_label(rows, choice_id, choice_label, label_fn=_tiller_label)

    r, err = pick_one_or_400(chosen, "Tiller final selection",
                             {"series": series, "width_in": width_in, "rotation": rotation_param,
                              "choice_id": choice_id, "choice_label": choice_label})
    if err: return err

    model = str(r[COL_MODEL]).strip()
    pid   = _first_part_id(r)
    label = _tiller_label(r) + (f" ({pid})" if pid else "")
    lines = [_make_line(label, 1, _price_number(r.get(COL_LIST_PRICE)), part_id=pid)]

    # Optional accessories
    acc_df = accessories_for_model(df, model)
    if list_access and not acc_ids and not acc_desc_terms:
        return accessory_choices_payload(acc_df, "Tiller", model)
    lines += accessory_lines_from_selection(acc_df, acc_ids, acc_qty_map, acc_desc_terms)

    return _totals_payload(lines, [], "Tiller", model, dealer_rate_override, dealer_meta)
# =========================
# Bale Spear (BS)
# =========================

def _quote_bale_spear(df):
    """
    Bale Spears:
      - Base rows: Category contains 'Bale Spear' (case-insensitive), excluding Accessories/Tires.
      - Step 1: Always return a single question listing ALL bale spear options.
                Each label shows the values from columns K, P, Q, R.
      - Step 2: When the user selects one (by part id or label), quote that unit.
    """
    # Base set
    cat = df[COL_CATEGORY].astype(str)
    base = df[cat.str.contains(r"\bbale\s*spear\b", case=False, na=False)]
    base = base[~base[COL_CATEGORY].astype(str).str.contains(ACC_TIRE_PATTERN, na=False)]
    if base.empty:
        return jsonify({"found": False, "mode": "error", "message": "No Bale Spear rows found."}), 404

    # Dealer context
    dealer_number = getp("dealer_number")
    dealer_meta, dealer_rate_override = None, None
    if dealer_number and dealer_number in DEALER_DISCOUNTS:
        name, rate = DEALER_DISCOUNTS[dealer_number]
        dealer_meta = {"dealer_number": dealer_number, "dealer_name": name, "applied_discount": rate}
        dealer_rate_override = rate

    # Selection params
    choice_id  = getp("balespear_choice_id", "accessory_id")
    choice_lbl = getp("balespear_choice")

    # Resolve K, P, Q, R by *letter position*, fallback-safe
    colK = _col_by_letter(base, "K")
    colP = _col_by_letter(base, "P")
    colQ = _col_by_letter(base, "Q")
    colR = _col_by_letter(base, "R")
    detail_cols = [c for c in [colK, colP, colQ, colR] if c and c in base.columns]

    def _detail_values(row):
        vals = []
        for c in detail_cols:
            v = str(row.get(c) or "").strip()
            if v:
                # prefer showing header names if they exist and look meaningful
                header = str(c).strip()
                # If header looks generic like "Unnamed: 10" just show the value
                if re.match(r"^unnamed", header, re.I):
                    vals.append(v)
                else:
                    vals.append(f"{header}: {v}")
        return " — ".join(vals)

    def _option_label(row):
        model = str(row.get(COL_MODEL, "") or "").strip()
        details = _detail_values(row)
        return f"{model}" + (f" — {details}" if details else "")

    # If no selection yet → ask with the full list
    if not choice_id and not choice_lbl:
        choices = []
        choices_with_ids = []
        for _, r in base.iterrows():
            lbl = _option_label(r)
            pid = _first_part_id(r) or lbl  # prefer part id, but label works as fallback
            choices.append(lbl)
            choices_with_ids.append({"id": pid, "label": lbl})
        return jsonify({
            "found": True,
            "mode": "questions",
            "category": "Bale Spear",
            "model": "",
            "required_questions": [{
                "name": "balespear_choice",
                "question": "Select a Bale Spear:",
                "choices": choices,
                "choices_with_ids": choices_with_ids,
                "multiple": False,
                "required": True
            }]
        })

    # A selection was made → resolve it
    chosen = _select_by_id_or_label(base, choice_id, choice_lbl, label_fn=_option_label)
    r, err = pick_one_or_400(chosen, "Bale Spear final selection",
                             {"choice_id": choice_id, "choice_label": choice_lbl})
    if err:
        return err

    model = str(r.get(COL_MODEL, "") or "").strip()
    pid   = _first_part_id(r)
    label = _option_label(r) + (f" ({pid})" if pid else "")

    lines = [_make_line(label, 1, _price_number(r.get(COL_LIST_PRICE)), part_id=pid)]
    return _totals_payload(
        lines,
        [],
        "Bale Spear",
        model,
        dealer_rate_override,
        dealer_meta
    )

# =========================
# Pallet Fork (PF)
# =========================

def _quote_pallet_fork(df):
    """
    Pallet Forks:
      - Return a full list of pallet fork options with details from columns K, L, M.
      - On pf_choice_id/pf_choice, finalize the quote.
    """
    # Base rows: Category mentions Pallet Fork (avoid Accessories/Tires/Auger)
    cat = df[COL_CATEGORY].astype(str)
    base = df[ cat.str.contains(r"\bpallet\s*fork\b", case=False, na=False) ]
    base = base[ ~base[COL_CATEGORY].astype(str).str.contains(ACC_TIRE_PATTERN, na=False) ]
    if base.empty:
        # Fallback: PF* models (for safety)
        model_upper = df[COL_MODEL].astype(str).str.upper()
        base = df[ model_upper.str.startswith("PF") ]
        base = base[ ~base[COL_CATEGORY].astype(str).str.contains(ACC_TIRE_PATTERN, na=False) ]
    if base.empty:
        return jsonify({"found": False, "mode": "error", "message": "No Pallet Fork rows found."}), 404

    # Dealer context
    dealer_number = getp("dealer_number")
    dealer_meta, dealer_rate_override = None, None
    if dealer_number and dealer_number in DEALER_DISCOUNTS:
        name, rate = DEALER_DISCOUNTS[dealer_number]
        dealer_meta = {"dealer_number": dealer_number, "dealer_name": name, "applied_discount": rate}
        dealer_rate_override = rate

    choice_id  = getp("pf_choice_id")
    choice_lbl = getp("pf_choice")

    # If no selection yet, list all Pallet Fork options with K,L,M details
    if not choice_id and not choice_lbl:
        choices, choices_with_ids = [], []
        for _, r in base.iterrows():
            model = str(r.get(COL_MODEL, "") or "").strip()
            pid = _first_part_id(r)
            if not (model and pid):
                continue
            details = _details_from_letters(r, ["K", "L", "M"])
            label = model if not details else f"{model} — {details}"
            choices.append(label)
            choices_with_ids.append({"id": pid, "label": label})
        if not choices:
            return jsonify({"found": False, "mode": "error", "message": "No selectable Pallet Fork rows with part IDs."}), 404
        return jsonify({
            "found": True,
            "mode": "questions",
            "category": "Pallet Fork",
            "model": "",
            "required_questions": [{
                "name": "pf_choice",
                "question": "Select a Pallet Fork:",
                "choices": choices,
                "choices_with_ids": choices_with_ids,
                "multiple": False,
                "required": True
            }]
        })

    # Finalize selection
    rows = base
    if choice_id:
        rows = _find_by_part_id(base, choice_id)
    if (rows is None or rows.empty) and choice_lbl:
        def _lbl_match(rr):
            model = str(rr.get(COL_MODEL, "") or "").strip()
            details = _details_from_letters(rr, ["K", "L", "M"])
            lbl = model if not details else f"{model} — {details}"
            return lbl.strip().lower() == choice_lbl.strip().lower()
        rows = base[base.apply(_lbl_match, axis=1)]

    r, err = pick_one_or_400(rows, "Pallet Fork final selection",
                             {"pf_choice_id": choice_id, "pf_choice": choice_lbl})
    if err:
        return err

    model = str(r[COL_MODEL]).strip()
    pid   = _first_part_id(r)
    details = _details_from_letters(r, ["K", "L", "M"])
    label = model if not details else f"{model} — {details}"
    if pid:
        label += f" ({pid})"

    lines = [_make_line(label, 1, _price_number(r.get(COL_LIST_PRICE)), part_id=pid)]
    return _totals_payload(lines, [], "Pallet Fork", model, dealer_rate_override, dealer_meta)

# =========================
# Quick Hitch (TQH)
# =========================

def _quote_quick_hitch(df):
    """
    Quick Hitch:
      - Build a base set of all items whose Model starts with TQH (or Category contains 'Quick Hitch'),
        excluding Accessories/Tires.
      - If no selection yet, return a single 'questions' payload with all options.
      - When qh_choice_id/qh_choice provided, quote that single unit.
    """
    # Base rows (no misaligned boolean indexing)
    model_u = df[COL_MODEL].astype(str).str.upper().str.strip()
    cat     = df[COL_CATEGORY].astype(str)

    base = df[
        (model_u.str.startswith("TQH") | cat.str.contains(r"\bquick\s*hitch\b", case=False, na=False))
        & (~cat.str.contains(ACC_TIRE_PATTERN, na=False))
    ].copy()

    if base.empty:
        return jsonify({"found": False, "mode": "error", "message": "No Quick Hitch rows found."}), 404

    # Dealer context
    dealer_number = getp("dealer_number")
    dealer_meta, dealer_rate_override = None, None
    if dealer_number and dealer_number in DEALER_DISCOUNTS:
        name, rate = DEALER_DISCOUNTS[dealer_number]
        dealer_meta = {"dealer_number": dealer_number, "dealer_name": name, "applied_discount": rate}
        dealer_rate_override = rate

    # Params
    choice_id  = getp("qh_choice_id", "part_id", "part_no")
    choice_lbl = getp("qh_choice", "choice", "choice_label")

    # Label helpers (reuse PF/BS-style detail extraction)
    def _qh_label(row):
        m = str(row.get(COL_MODEL, "") or "").strip().upper()
        cat_str = "Category 1 (Cat 1)" if m == "TQH1" else ("Category 2 (Cat 2)" if m == "TQH2" else "")
        dsc = (row.get(COL_DESC) or "").strip()
        bits = [m]
        if cat_str: bits.append(cat_str)
        if dsc: bits.append(dsc)
        return " — ".join(bits)

    def _qh_option_label(row):
        # include spreadsheet details like PF/BS flows; pulls whatever exists among K..R
        details = _details_from_letters(row, ["K","L","M","N","O","P","Q","R"])
        base_lbl = _qh_label(row)
        return f"{base_lbl} — {details}" if details else base_lbl

    # If no selection yet → present all options (like Spears/PF)
    if not choice_id and not choice_lbl:
        # Keep deterministic order (TQH1 before TQH2, then by model/part)
        ordered = base.sort_values(
            by=[COL_MODEL] + [c for c in PART_ID_COLS if c in base.columns],
            key=lambda s: s.astype(str).str.upper()
        )
        choices, choices_with_ids = [], []
        for _, r in ordered.iterrows():
            pid = _first_part_id(r)
            if not pid:
                continue
            label = _qh_option_label(r)
            choices.append(label)
            choices_with_ids.append({"id": pid, "label": label})

        if not choices:
            return jsonify({"found": False, "mode": "error", "message": "No selectable Quick Hitch rows with part IDs."}), 404

        return jsonify({
            "found": True,
            "mode": "questions",
            "category": "Quick Hitch",
            "model": "",
            "required_questions": [{
                "name": "qh_choice",
                "question": "Select a Quick Hitch:",
                "choices": choices,
                "choices_with_ids": choices_with_ids,
                "multiple": False,
                "required": True
            }]
        })

    # A selection was made → resolve by part id or label
    chosen = _select_by_id_or_label(base, choice_id, choice_lbl, label_fn=_qh_option_label)
    r, err = pick_one_or_400(chosen, "Quick Hitch final selection",
                             {"qh_choice_id": choice_id, "qh_choice": choice_lbl})
    if err:
        return err

    model = str(r.get(COL_MODEL, "") or "").strip().upper()
    pid   = _first_part_id(r)
    label = _qh_option_label(r) + (f" ({pid})" if pid else "")

    lines = [_make_line(label, 1, _price_number(r.get(COL_LIST_PRICE)), part_id=pid)]
    return _totals_payload(lines, [], "Quick Hitch", model, dealer_rate_override, dealer_meta)

# =========================
# Stump Grinder (TSG50)
# =========================

def _stump_grinder_accessories(df):
    """
    Returns accessories for the TSG50 Stump Grinder.

    Looks for:
      - Category containing 'Stump Grinder Accessories'
      - OR 'Used On' column containing 'TSG50'
    """
    cat = df.get(COL_CATEGORY, pd.Series([""] * len(df))).astype(str)
    used_on = df.get("Used On", pd.Series([""] * len(df))).astype(str)

    return df[
        cat.str.contains("stump grinder accessories", case=False, na=False) |
        used_on.str.contains("TSG50", case=False, na=False)
    ].copy()

def _quote_stump_grinder(df):
    """
    Single-model stump grinder (TSG50).
    Flow:
      - Quote TSG50 base unit.
      - REQUIRED: ask which hydraulics they want by listing *all accessories* for TSG50.
      - Add the selected hydraulics as a separate line item.
    """
    # Base rows: Category mentions Stump/Grinder OR model is TSG50; exclude accessories/tires
    cat = df[COL_CATEGORY].astype(str)
    model_u = df[COL_MODEL].astype(str).str.upper().str.strip()

    base = df[
        ((cat.str.contains(r"\bstump\b", case=False, na=False) & cat.str.contains(r"\bgrind", case=False, na=False))
         | (model_u == "TSG50"))
        & (~cat.str.contains(ACC_TIRE_PATTERN, na=False))
    ].copy()

    if base.empty:
        return jsonify({"found": False, "mode": "error", "message": "No Stump Grinder rows found."}), 404

    # Dealer context
    dealer_number = getp("dealer_number")
    dealer_meta, dealer_rate_override = None, None
    if dealer_number and dealer_number in DEALER_DISCOUNTS:
        name, rate = DEALER_DISCOUNTS[dealer_number]
        dealer_meta = {"dealer_number": dealer_number, "dealer_name": name, "applied_discount": rate}
        dealer_rate_override = rate

    # Pick the TSG50 base row
    rows = base[model_u == "TSG50"]
    if rows.empty:
        rows = base
    r, err = pick_one_or_400(rows, "Stump Grinder base row", {"model": "TSG50"})
    if err:
        return err

    model = str(r[COL_MODEL]).strip() or "TSG50"
    pid   = _first_part_id(r)
    dsc   = (r.get(COL_DESC) or "").strip()
    base_label = f"{model}" + (f" — {dsc}" if dsc else "")
    if pid:
        base_label += f" ({pid})"

    lines = [_make_line(base_label, 1, _price_number(r.get(COL_LIST_PRICE)), part_id=pid)]

    # Accessories for hydraulics selection (required)
    acc_df = _stump_grinder_accessories(df)
    if acc_df.empty:
        # If none found, just return the base unit with a clear note.
        return _totals_payload(
            lines,
            ["No hydraulics accessories were found for this model in the sheet."],
            "Stump Grinder",
            model,
            dealer_rate_override,
            dealer_meta,
        )

    hydraulics_id  = getp("hydraulics_id", "accessory_id")
    hydraulics_lbl = getp("hydraulics_choice")

    # Ask which hydraulics (required) if none chosen yet
    if not hydraulics_id and not hydraulics_lbl:
        choices, choices_with_ids = [], []
        for _, ar in acc_df.iterrows():
            apid = _first_part_id(ar)
            albl = (ar.get(COL_DESC) or "").strip()
            if apid and albl:
                choices.append(albl)
                choices_with_ids.append({"id": apid, "label": albl})
        if not choices:
            return _totals_payload(
                lines,
                ["No hydraulics accessories with part IDs were found."],
                "Stump Grinder",
                model,
                dealer_rate_override,
                dealer_meta,
            )
        return jsonify({
            "found": True,
            "mode": "questions",
            "category": "Stump Grinder",
            "model": model,
            "required_questions": [{
                "name": "hydraulics_id",
                "question": "Which hydraulics option do you want for the TSG50?",
                "choices": choices,
                "choices_with_ids": choices_with_ids,
                "multiple": False,
                "required": True
            }]
        })

    # Resolve selected hydraulics by part id or label contains
    hyd_row = None
    if hydraulics_id:
        hit = _find_by_part_id(acc_df, hydraulics_id)
        if not hit.empty:
            hyd_row = hit.iloc[0]
    if hyd_row is None and hydraulics_lbl:
        sub = acc_df[acc_df[COL_DESC].astype(str).str.contains(hydraulics_lbl, case=False, na=False)]
        if not sub.empty:
            hyd_row = sub.iloc[0]

    if hyd_row is None:
        return jsonify({"found": False, "mode": "error", "message": "Selected hydraulics option not found for TSG50."}), 404

    hpid = _first_part_id(hyd_row)
    hdesc = (hyd_row.get(COL_DESC) or "Hydraulics").strip()
    lines.append(_make_line(f"{hdesc} ({hpid})", 1, _price_number(hyd_row.get(COL_LIST_PRICE)), part_id=hpid))

    return _totals_payload(
        lines,
        ["Hydraulics added as a separate line item."],
        "Stump Grinder",
        model,
        dealer_rate_override,
        dealer_meta,
    )

# =========================
# shared small utilities
# =========================
def _select_by_id_or_label(rows, choice_id, choice_label, label_fn):
    """
    Select a single row from `rows` (a DataFrame) by either:
      1) explicit ID (choice_id), or
      2) label (choice_label) — BUT if the label looks like a part number
         (e.g., '1037171', '640022', 'PFW4448S2'), also try matching it
         against the PART_ID_COLS first.

    This makes families like pallet_fork, bale_spear, disc_harrow robust when the
    chat sends a part number in the '..._choice' label field instead of '..._choice_id'.
    """
    # 1) Try ID match using choice_id (preferred)
    if choice_id:
        cid = choice_id.strip().upper()
        for c in PART_ID_COLS:
            if c in rows.columns:
                hit = rows[rows[c].astype(str).str.strip().str.upper() == cid]
                if not hit.empty:
                    return hit.iloc[[0]]

    # 1a) If no choice_id matched/provided, but choice_label is present,
    #     try treating the label as an ID too (common when UI passes part numbers in the label slot).
    if choice_label:
        cl_id = choice_label.strip().upper()
        for c in PART_ID_COLS:
            if c in rows.columns:
                hit = rows[rows[c].astype(str).str.strip().str.upper() == cl_id]
                if not hit.empty:
                    return hit.iloc[[0]]

    # 2) Fallback: use label_fn to compare human-readable labels
    if choice_label:
        cl = choice_label.strip().lower()
        # exact match
        for idx, r in rows.iterrows():
            if label_fn(r).lower() == cl:
                return rows.loc[[idx]]
        # contains
        for idx, r in rows.iterrows():
            if cl in label_fn(r).lower():
                return rows.loc[[idx]]

    # 3) If only one possible row, pick it
    if len(rows.index) == 1:
        return rows.iloc[[0]]

    # 4) No match
    return rows.iloc[0:0]

# =========================
# Run
# =========================
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=False)
