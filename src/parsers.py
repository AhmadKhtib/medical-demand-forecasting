"""
Per-clinic MEDICAL field parsers.
Each clinic has its own drug lexicon, aliases, and parse_medical_cell() function,
matching exactly the notebook logic.
"""
import re
import numpy as np
import pandas as pd
from collections import OrderedDict

from .text_utils import (
    keyify, fuzzy_key_match, first_token_key, is_empty_like,
    strip_trailing_form_words, slash_to_pipe, norm_text_base,
    explode_multi_med_line, build_output_series, dedup_meds,
    ARABIC_RE, AR_PLACEHOLDER_RE, SCHED_RE, SCHED_WORDS_RE,
    UNIT_OR_FORM_RE, STRENGTH_RE, STRENGTH_IN_NAME_RE, EMPTY_KEYS, NUM,
)

# ---------------------------------------------------------------------------
# DERMATOLOGY
# ---------------------------------------------------------------------------
_DERM_DRUGS = {
    "lorax","lotax","lor","fex","feox","fex180","feox180","histaz","histafs","histafed",
    "augmin","augmentin","amoxicillin","coamoxiclav","clamoxin","zinnat","zinat",
    "fusidin","fusidine","fusidincream","fusdin","fusi","beta","betaneo","betneoc",
    "flucan","fluconazole","miconazole","ketoconazole","terbinifine","terbinafine",
    "nizoral","canest","canestc","canes","canesc","daktaz","daktazc",
    "permethrine","permethrin","permethine","permetheine","permethein",
    "dermalux","dermulx","demalux","dermulux","calamine","calaime","calamaine","clamine","claamine",
    "betacare","panthenol","benzac","clinda","clind","clindex","clindexgel","clindamycin","clinda150",
    "doxal","doxal100","doxal150","azith","azit","azith500",
    "artex","ogmin","ogmi","ogm","coolc","decort","prednisolone","prednisolne","prednisone",
    "famodine","famodin","albendazole","paracetamol","acyclovir","zinc","zincoxide",
    "tic","tac","tictac","tictacshampoo","mv","multivit","multivitamin",
}
_DERM_DRUGS_KEYS = {keyify(x) for x in _DERM_DRUGS}

_DERM_ALIASES = {
    "permethrine":"permethrin","permethrin":"permethrin","permethine":"permethrin",
    "permetheine":"permethrin","permethein":"permethrin",
    "dermalux":"dermalux","dermulx":"dermalux","demalux":"dermalux","dermulux":"dermalux",
    "augmin":"augmin","augmentin":"augmentin","fusidin":"fusidin","fusidine":"fusidin",
    "fusdin":"fusidin","fusi":"fusidin","beta":"betamethasone","betneoc":"bet neo c",
    "betneo":"bet neo","clind":"clindamycin","clinda":"clindamycin","clindex":"clindex gel",
    "micanazol":"miconazole","miccozaole":"miconazole","micnazole":"miconazole",
    "nizoralc":"nizoral c","canestc":"canest c","canesc":"canes c","daktazc":"daktaz c",
    "famodin":"famodine","prednisolne":"prednisolone","tictacshampoo":"tic tac shampoo",
    "tictac":"tic tac shampoo","mv":"mv","multivit":"multivitamin",
}

_DERM_PLAN_ONLY = {"physiotherapy","physio","pt"}
_DERM_DIAG_ONLY = {"lbp","scoliosis","ulcer","limbing","limping","normal"}
_DERM_EMPTY = {"0","n","nan","none","null","na","nil","pop","no","need","noneed"}
_DERM_SCHED_RE = re.compile(
    rf"{NUM}(?:\s*(?:cc|ml|mg))?(?:\s*[*xX&]\s*{NUM}){{1,4}}(?=\D|$)",
    re.IGNORECASE,
)


def _derm_norm(x):
    if pd.isna(x):
        return np.nan
    s = str(x).replace("\r", "\n")
    s = re.sub(r"\bm\s*/\s*v\b", "mv", s, flags=re.I)
    s = re.sub(r"[\\]{2,}", "\n", s)
    s = re.sub(r"/{2,}", "\n", s)
    s = re.sub(r"(?<=\w)\\(?=\w)", "\n", s)
    s = re.sub(r"\*{1,}", "", s)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{2,}", "\n", s).strip()
    s = re.sub(r"([A-Za-z])(\d)", r"\1 \2", s)
    s = re.sub(r"(mg|ml|cc)(\d)", r"\1 \2", s, flags=re.I)
    s = re.sub(r"(\d)\s*(then)\b", r"\1 then", s, flags=re.I)
    s = re.sub(r"\bth\b", "then", s, flags=re.I)
    s = re.sub(r"[ ]{2,}", " ", s).strip()
    return s.lower()


def _derm_is_med(line):
    line = (line or "").strip()
    if not line:
        return False
    line = re.sub(r"[\\/ ]+$", "", line).strip()
    tok = first_token_key(line)
    full = keyify(line)
    if tok in _DERM_EMPTY or full in _DERM_EMPTY:
        return False
    if tok in _DERM_DIAG_ONLY or full in _DERM_DIAG_ONLY:
        return False
    if _DERM_SCHED_RE.search(line) or SCHED_WORDS_RE.search(line) or UNIT_OR_FORM_RE.search(line):
        return True
    if "+" in line:
        if re.search(r"\+\s*$", line):
            return False
        return True
    if tok in _DERM_ALIASES or full in _DERM_ALIASES:
        return True
    if tok in _DERM_DRUGS_KEYS or full in _DERM_DRUGS_KEYS:
        return True
    return False


def _derm_split(line):
    line = (line or "").strip()
    if not line:
        return (None, None)
    line = re.sub(r"[\\/ ]+$", "", line).strip()
    if "+" in line and not _DERM_SCHED_RE.search(line) and not SCHED_WORDS_RE.search(line):
        parts = [p.strip() for p in line.split("+") if p.strip()]
        meds = [_DERM_ALIASES.get(keyify(p), p.lower()) for p in parts]
        return meds, [""] * len(meds)
    starts = []
    m1 = _DERM_SCHED_RE.search(line)
    if m1:
        starts.append(m1.start())
    m2 = SCHED_WORDS_RE.search(line)
    if m2:
        starts.append(m2.start())
    if not starts:
        return [line], [""]
    start = min(starts)
    med_part = re.sub(r"\bthen\b", "", line[:start]).strip()
    sch_part = re.sub(r"\s{2,}", " ", line[start:]).strip()
    return [med_part] if med_part else [line], [sch_part]


def parse_dermatology(x):
    if pd.isna(x):
        return pd.Series([np.nan]*4, index=["diagnosis","plan_text","medication","dose_schedule"])
    s = _derm_norm(x)
    if pd.isna(s) or s == "":
        return pd.Series([np.nan, s, np.nan, np.nan], index=["diagnosis","plan_text","medication","dose_schedule"])
    if ARABIC_RE.search(s) and not re.search(r"[a-z0-9]", s) and not _DERM_SCHED_RE.search(s):
        return pd.Series([np.nan, s, np.nan, np.nan], index=["diagnosis","plan_text","medication","dose_schedule"])
    cell_key = keyify(s)
    if cell_key in _DERM_EMPTY or s.strip() in {"no need","none","nan"}:
        return pd.Series([np.nan, s, np.nan, np.nan], index=["diagnosis","plan_text","medication","dose_schedule"])
    if cell_key in {keyify(t) for t in _DERM_PLAN_ONLY}:
        return pd.Series([np.nan, s, np.nan, np.nan], index=["diagnosis","plan_text","medication","dose_schedule"])
    lines = [ln.strip() for ln in s.split("\n") if ln.strip()]
    if not lines:
        return pd.Series([np.nan, s, np.nan, np.nan], index=["diagnosis","plan_text","medication","dose_schedule"])
    first_med = next((i for i, ln in enumerate(lines) if _derm_is_med(ln)), None)
    if first_med is None:
        return pd.Series([s, s, np.nan, np.nan], index=["diagnosis","plan_text","medication","dose_schedule"])
    diagnosis = "\n".join(lines[:first_med]).strip() or np.nan
    meds_out, sch_out = [], []
    for ln in lines[first_med:]:
        ms, ss = _derm_split(ln)
        for m, sch in zip(ms, ss):
            k = keyify(m or "")
            m = _DERM_ALIASES.get(k, (m or "").lower()).strip()
            if m:
                meds_out.append(m)
                sch_out.append((sch or "").strip())
    meds2, sch2 = dedup_meds(meds_out, sch_out)
    return build_output_series(diagnosis, s, meds2, sch2)


# ---------------------------------------------------------------------------
# ORTHOPEDIC
# ---------------------------------------------------------------------------
_ORTHO_DRUGS = {
    "trufen","acamol","acmol","mv","multivit","multivitamin",
    "diclofen","diclo","diclofenac","naprox","naproxen","naprex","lornox","lornoxicam",
    "prednitab","prednisone","pirox","feldene","tericox","etericox","etorix","coxib","celecoxib","celcox",
    "omeprazol","omeprazole","omepa","omperzol",
    "zinnat","zinaat","clamoxin","azicare","azicre","denacin","deancin",
    "dimra","dirma","rejoint","jeflex","doxycycline",
    "ketofan","ogmin","aspirin","asprin",
    "amoxicillin","coamoxiclav","flagyl","metronidazole","curam",
    "calcium","zinc","histafed",
}
_ORTHO_DRUGS_KEYS = {keyify(x) for x in _ORTHO_DRUGS}

_ORTHO_ALIASES = {
    "truefn":"trufen","trfuen":"trufen","truifen":"trufen","trufern":"trufen","trufne":"trufen",
    "trufwen":"trufen","tryufen":"trufen","ttrufen":"trufen","truen":"trufen","trofen":"trufen",
    "mv":"mv","m/v":"mv","m-v":"mv",
    "celex200":"celecoxib 200mg","celex20":"celecoxib","elcox":"celecoxib",
    "elecoxib200":"celecoxib 200mg","celcox":"celecoxib","celecox":"celecoxib",
    "etrocoxip":"etoricoxib","etricox":"etericox","teriocox":"etericox",
    "dicc":"diclofenac","diclo":"diclofenac",
    "naproxcin":"naproxen","naprexon":"naproxen",
    "curam":"co-amoxiclav","amoxy":"amoxicillin","amoxi":"amoxicillin","amoxicillin":"amoxicillin",
    "flagyl":"metronidazole","clamxoin":"clamoxin",
    "predni":"prednisone","prednison":"prednisone",
    "multi":"multivitamin","calcium":"calcium","zinc":"zinc",
    "vitdd":"vit d-d","vitd":"vit d","vitddrops":"vit d drops",
    "denaicn":"denacin","histafed":"histafed","demra":"dimra","dimra":"dimra","dirma":"dirma",
}
_ORTHO_PLAN_ONLY = {"physiotherapy","physio","pt"}
_ORTHO_DIAG_ONLY = {"lbp","scoliosis","ulcer","limbing","limping","normal"}
_ORTHO_EMPTY = {"0","n","nan","none","null","na","nil","pop"}


def _ortho_norm(x):
    if pd.isna(x):
        return np.nan
    s = str(x).replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{2,}", "\n", s).strip()
    s = re.sub(r"([A-Za-z])(\d)", r"\1 \2", s)
    s = re.sub(r"(mg|ml)(\d)", r"\1 \2", s, flags=re.I)
    s = re.sub(r"(\d)\s*(then)\b", r"\1 then", s, flags=re.I)
    s = re.sub(r"\bth\b", "then", s, flags=re.I)
    s = re.sub(r"[ ]{2,}", " ", s).strip()
    return s.lower()


def _ortho_is_med(line):
    line = (line or "").strip()
    if not line:
        return False
    tok = first_token_key(line)
    full = keyify(line)
    if tok in _ORTHO_EMPTY or full in _ORTHO_EMPTY:
        return False
    if tok in _ORTHO_DIAG_ONLY or full in _ORTHO_DIAG_ONLY:
        return False
    if SCHED_RE.search(line) or SCHED_WORDS_RE.search(line) or UNIT_OR_FORM_RE.search(line):
        return True
    if "+" in line:
        return True
    if tok in _ORTHO_ALIASES or full in _ORTHO_ALIASES:
        return True
    if tok in _ORTHO_DRUGS_KEYS or full in _ORTHO_DRUGS_KEYS:
        return True
    return False


def _ortho_split(line):
    line = line.strip()
    if not line:
        return (None, None)
    if "+" in line and not SCHED_RE.search(line):
        parts = [p.strip() for p in line.split("+") if p.strip()]
        meds = [_ORTHO_ALIASES.get(keyify(p), p.lower()) for p in parts]
        return meds, [""] * len(meds)
    starts = []
    m1 = SCHED_RE.search(line)
    if m1: starts.append(m1.start())
    m2 = SCHED_WORDS_RE.search(line)
    if m2: starts.append(m2.start())
    if not starts:
        return [line], [""]
    start = min(starts)
    med_part = re.sub(r"\bthen\b", "", line[:start]).strip()
    sch_part = re.sub(r"\s{2,}", " ", line[start:]).strip()
    return [med_part] if med_part else [line], [sch_part]


def parse_orthopedic(x):
    if pd.isna(x):
        return pd.Series([np.nan]*4, index=["diagnosis","plan_text","medication","dose_schedule"])
    s = _ortho_norm(x)
    cell_key = keyify(s) if not pd.isna(s) else ""
    if pd.isna(s) or s == "" or cell_key in _ORTHO_EMPTY:
        return pd.Series([np.nan, s, np.nan, np.nan], index=["diagnosis","plan_text","medication","dose_schedule"])
    if cell_key in {keyify(t) for t in _ORTHO_PLAN_ONLY}:
        return pd.Series([np.nan, s, np.nan, np.nan], index=["diagnosis","plan_text","medication","dose_schedule"])
    s_low = s.lower().strip()
    if s_low in {"no need","none","nan"} or "يرجى إدخال التشخيص" in s:
        return pd.Series([np.nan, s, np.nan, np.nan], index=["diagnosis","plan_text","medication","dose_schedule"])
    lines = [ln.strip() for ln in s.split("\n") if ln.strip()]
    if not lines:
        return pd.Series([np.nan, s, np.nan, np.nan], index=["diagnosis","plan_text","medication","dose_schedule"])
    first_med = next((i for i, ln in enumerate(lines) if _ortho_is_med(ln)), None)
    if first_med is None:
        return pd.Series([s, s, np.nan, np.nan], index=["diagnosis","plan_text","medication","dose_schedule"])
    diagnosis = "\n".join(lines[:first_med]).strip() or np.nan
    meds_out, sch_out = [], []
    for ln in lines[first_med:]:
        ms, ss = _ortho_split(ln)
        for m, sch in zip(ms, ss):
            k = keyify(m or "")
            m = _ORTHO_ALIASES.get(k, (m or "").lower()).strip()
            if m:
                meds_out.append(m)
                sch_out.append((sch or "").strip())
    meds2, sch2 = dedup_meds(meds_out, sch_out)
    return build_output_series(diagnosis, s, meds2, sch2)


# ---------------------------------------------------------------------------
# UROLOGY
# ---------------------------------------------------------------------------
_URO_DRUGS = {
    "omnic","tamsulin","tamsulosin","novitropan","oxybutynin","pirox","piroxicam",
    "rowatinex","cyston","cystone","urosolvin","uroclean","urocleanse","uroclen",
    "levox","levofloxacin","floxin","cipro","ciprofloxacin","flagyl","metronidazole",
    "zinnat","zinaat","cefuroxime","cefix","cefixim","cefixime",
    "clamoxin","moclav","curam","amoxicillin","coamoxiclav",
    "diclofen","diclofenac","trufen","acamol","acmol","paracetamol",
    "scobutyl","scoptyl","spasmin","panto","famodin","famodine",
    "denacin","doxy","doxycycline","canesten","canestenv","dacktazol","vermazol","hemorral",
}
_URO_DRUGS_KEYS = {keyify(x) for x in _URO_DRUGS}

_URO_ALIASES = {
    "ominc":"omnic","omnic04":"omnic 0.4mg","omnic04mg":"omnic 0.4mg","omnic0":"omnic",
    "amnic":"omnic","amnic04":"omnic 0.4mg","amnic04mg":"omnic 0.4mg",
    "tamslin":"tamsulin","tamsulin04":"tamsulin 0.4mg","tamsulin04mg":"tamsulin 0.4mg",
    "rawat":"rowatinex","rawatinex":"rowatinex","rwoatinex":"rowatinex","rowatin":"rowatinex",
    "cystone":"cyston","suiiprim":"sulprim",
    "levox500mg":"levox 500mg","levox500":"levox 500mg","levox750":"levox 750mg",
    "ciphex":"cefix","cefixim":"cefixim","cefixime":"cefixim",
    "uroclaen":"uroclean","uroclen":"uroclean",
    "mv":"mv","m/v":"mv","m-v":"mv","m.v":"mv",
    "truefn":"trufen","truifen":"trufen","tryufen":"trufen","ttrufen":"trufen","trofen":"trufen",
    "diclo":"diclofenac","dicc":"diclofenac","famodin":"famodin",
}
_URO_DIAG_ONLY = {
    "uti","cystitis","prostatitis","chronicprostatitis","febrileuti",
    "renalcolic","renalcholic","backache","backpain",
}
_URO_EMPTY = {"0","n","nan","none","null","na","nil","pop"}
_URO_PLAN_ONLY = {"physiotherapy","physio","pt"}


def _uro_strip_arabic(s):
    if not s:
        return s
    m = ARABIC_RE.search(s)
    return s[:m.start()].strip() if m else s


def _uro_norm(x):
    if pd.isna(x):
        return np.nan
    s = str(x).replace("\r", "\n").replace("//", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{2,}", "\n", s).strip()
    s = re.sub(r"([A-Za-z])(\d)", r"\1 \2", s)
    s = re.sub(r"(mg|ml)(\d)", r"\1 \2", s, flags=re.I)
    s = re.sub(r"(\d)\s*(then)\b", r"\1 then", s, flags=re.I)
    s = re.sub(r"\bth\b", "then", s, flags=re.I)
    s = re.sub(r"[ ]{2,}", " ", s).strip()
    return s.lower()


def _uro_is_med(line):
    line = (line or "").strip()
    if not line:
        return False
    tok = first_token_key(line)
    full = keyify(line)
    if tok in _URO_EMPTY or full in _URO_EMPTY:
        return False
    if tok in _URO_DIAG_ONLY or full in _URO_DIAG_ONLY:
        return False
    if SCHED_RE.search(line) or SCHED_WORDS_RE.search(line) or UNIT_OR_FORM_RE.search(line):
        return True
    if "+" in line:
        return True
    if tok in _URO_ALIASES or full in _URO_ALIASES:
        return True
    if tok in _URO_DRUGS_KEYS or full in _URO_DRUGS_KEYS:
        return True
    return False


def _uro_normalize_name(m):
    m = _uro_strip_arabic((m or "").strip().lower())
    if not m:
        return m
    k = keyify(m)
    return _URO_ALIASES.get(k, m)


def _uro_split(line):
    line = _uro_strip_arabic((line or "").strip().lower())
    if not line:
        return [], []
    if "+" in line and not SCHED_RE.search(line):
        parts = [p.strip() for p in line.split("+") if p.strip()]
        meds = [_uro_normalize_name(p) for p in parts]
        return meds, [""] * len(meds)
    starts = []
    m1 = SCHED_RE.search(line)
    if m1: starts.append(m1.start())
    m2 = SCHED_WORDS_RE.search(line)
    if m2: starts.append(m2.start())
    if not starts:
        return [_uro_normalize_name(line)], [""]
    start = min(starts)
    med_part = _uro_normalize_name(re.sub(r"\bthen\b", "", line[:start]).strip())
    sch_part = re.sub(r"\s{2,}", " ", line[start:]).strip()
    return [med_part] if med_part else [_uro_normalize_name(line)], [sch_part]


def parse_urology(x):
    if pd.isna(x):
        return pd.Series([np.nan]*4, index=["diagnosis","plan_text","medication","dose_schedule"])
    s = _uro_norm(x)
    if pd.isna(s) or s == "":
        return pd.Series([np.nan, s, np.nan, np.nan], index=["diagnosis","plan_text","medication","dose_schedule"])
    cell_key = keyify(s)
    if cell_key in _URO_EMPTY or cell_key in {keyify(t) for t in _URO_PLAN_ONLY}:
        return pd.Series([np.nan, s, np.nan, np.nan], index=["diagnosis","plan_text","medication","dose_schedule"])
    if "يرجى إدخال التشخيص" in s:
        return pd.Series([np.nan, s, np.nan, np.nan], index=["diagnosis","plan_text","medication","dose_schedule"])
    lines = [ln.strip() for ln in s.split("\n") if ln.strip()]
    if not lines:
        return pd.Series([np.nan, s, np.nan, np.nan], index=["diagnosis","plan_text","medication","dose_schedule"])
    first_med = next((i for i, ln in enumerate(lines) if _uro_is_med(ln)), None)
    if first_med is None:
        return pd.Series([s, s, np.nan, np.nan], index=["diagnosis","plan_text","medication","dose_schedule"])
    diagnosis = "\n".join(lines[:first_med]).strip() or np.nan
    meds_out, sch_out = [], []
    for ln in lines[first_med:]:
        ms, ss = _uro_split(ln)
        for m, sch in zip(ms, ss):
            m = _uro_normalize_name(m)
            if m:
                meds_out.append(m)
                sch_out.append((sch or "").strip())
    meds2, sch2 = dedup_meds(meds_out, sch_out)
    return build_output_series(diagnosis, s, meds2, sch2)


# ---------------------------------------------------------------------------
# NUTRITION  (scored approach with fuzzy matching)
# ---------------------------------------------------------------------------
_NUT_DRUGS = {
    "vit","vitamin","vit c","vitamin c","vit d","vitamin d",
    "multivit","multivitamin","mv","m.v",
    "zinc","iron","ferrous","folic acid","folate","ors",
    "omega 3","omega3","kirk","paracetamol","famotidine",
}
_NUT_DRUGS_KEYS = {keyify(x) for x in _NUT_DRUGS}

_NUT_ALIASES = {
    keyify("mv"):"multivitamin",keyify("m.v"):"multivitamin",
    keyify("multivit"):"multivitamin",keyify("vitc"):"vit c",
    keyify("vitamin c"):"vit c",keyify("vit d"):"vit d",
    keyify("omega"):"omega 3",keyify("omega3"):"omega 3",
    keyify("kirak"):"kirk",keyify("folicacid"):"folic acid",
    keyify("folic"):"folic acid",keyify("ieron"):"iron",keyify("ir0n"):"iron",
}
_NUT_EMPTY = {"0","n","nan","none","null","na","nil","pop"}
_NUT_NO_NEED = {"noneed","notneed","none","no","nil"}
_NUT_DIAG_ONLY = {"lbp","scoliosis","ulcer","limbing","limping","normal"}
_NUT_PLAN_ONLY = {"physiotherapy","physio","pt"}
_NUT_DIAG_KW = re.compile(
    r"(?i)\b(pain|ache|fever|cough|cold|flu|diarrh|vomit|nausea|headache|"
    r"dizzy|infection|rash|ulcer|scoliosis|lbp|limp|normal)\b"
)
_NUT_DRUG_SUFFIX = re.compile(
    r"(?i)\b(cillin|mycin|azole|prazole|tidine|sartan|pril|statin|dipine|zepam|"
    r"triptan|caine|profen|oxicam|xone|vir|mab|nib|tinib)\b"
)


def _nut_norm(x, lower=True):
    if pd.isna(x):
        return np.nan
    s = str(x).replace("\r", "\n")
    s = re.sub(r"\bm\s*/\s*v\b", "mv", s, flags=re.I)
    s = re.sub(r"[\\]{2,}", "\n", s)
    s = re.sub(r"/{2,}", "\n", s)
    s = re.sub(r"(?<=\w)\\(?=\w)", "\n", s)
    s = re.sub(r"\*{1,}", "", s)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{2,}", "\n", s).strip()
    s = re.sub(r"([A-Za-z])(\d)", r"\1 \2", s)
    s = re.sub(r"(mg|ml|cc)(\d)", r"\1 \2", s, flags=re.I)
    s = re.sub(r"(\d)\s*(then)\b", r"\1 then", s, flags=re.I)
    s = re.sub(r"\bth\b", "then", s, flags=re.I)
    s = re.sub(r"[ ]{2,}", " ", s).strip()
    return s.lower() if lower else s


def _nut_score(line):
    line0 = (line or "").strip()
    if not line0:
        return 0, 0
    line0 = re.sub(r"[\\/ ]+$", "", line0).strip()
    low = line0.lower()
    tok = first_token_key(low)
    full = keyify(low)
    if tok in _NUT_EMPTY or full in _NUT_EMPTY:
        return 0, 0
    med_score = 0
    diag_score = 0
    if tok in _NUT_DIAG_ONLY or full in _NUT_DIAG_ONLY:
        diag_score += 5
    if _NUT_DIAG_KW.search(low):
        diag_score += 2
    if SCHED_RE.search(low) or SCHED_WORDS_RE.search(low):
        med_score += 4
    if UNIT_OR_FORM_RE.search(low):
        med_score += 3
    if "+" in low and not re.search(r"\+\s*$", low):
        med_score += 2
    if tok.startswith("vit") and re.search(r"\b\d+\b", low):
        med_score += 3
    matched = fuzzy_key_match(tok, _NUT_DRUGS_KEYS) or fuzzy_key_match(full, _NUT_DRUGS_KEYS)
    if matched:
        med_score += 3
    if _NUT_DRUG_SUFFIX.search(low):
        med_score += 2
    if re.search(r"\d", low):
        med_score += 1
    return med_score, diag_score


def _nut_is_med(line):
    med, diag = _nut_score(line)
    if diag >= 5 and med < 5:
        return False
    return med >= 3 and med >= diag + 1


def _nut_all_look_like_meds(lines):
    if not lines:
        return False
    for ln in lines:
        _, d = _nut_score(ln)
        if d >= 4:
            return False
    if any(_nut_is_med(ln) for ln in lines):
        return True
    for ln in lines:
        parts = ln.split()
        if len(parts) != 1:
            return False
        k = keyify(parts[0])
        if not k or k in _NUT_EMPTY or k in _NUT_DIAG_ONLY or _NUT_DIAG_KW.search(ln.lower()):
            return False
        if len(k) < 3:
            return False
    return True


def _nut_normalize_name(m):
    m = (m or "").strip()
    if not m:
        return m
    k = keyify(m)
    return _NUT_ALIASES.get(k, m.lower())


def _nut_split(line):
    line = (line or "").strip()
    if not line:
        return None, None
    line = re.sub(r"[\\/ ]+$", "", line).strip()
    if tok := line.split():
        if keyify(tok[0]).startswith("vit") and re.search(r"\b\d+\b", line):
            if not SCHED_RE.search(line) and not SCHED_WORDS_RE.search(line):
                parts = line.split()
                return [parts[0]], [" ".join(parts[1:]).strip()]
    if "+" in line and not SCHED_RE.search(line) and not SCHED_WORDS_RE.search(line):
        parts = [p.strip() for p in line.split("+") if p.strip()]
        meds = [_nut_normalize_name(p) for p in parts]
        return meds, [""] * len(meds)
    starts = []
    m1 = SCHED_RE.search(line)
    if m1: starts.append(m1.start())
    m2 = SCHED_WORDS_RE.search(line)
    if m2: starts.append(m2.start())
    if not starts:
        return [line], [""]
    start = min(starts)
    med_part = re.sub(r"\bthen\b", "", line[:start], flags=re.I).strip()
    sch_part = re.sub(r"\s{2,}", " ", line[start:]).strip()
    return [med_part] if med_part else [line], [sch_part]


def parse_nutrition(x):
    if pd.isna(x):
        return pd.Series([np.nan]*4, index=["diagnosis","plan_text","medication","dose_schedule"])
    s = _nut_norm(x)
    if pd.isna(s) or s.strip() == "":
        return pd.Series([np.nan, s, np.nan, np.nan], index=["diagnosis","plan_text","medication","dose_schedule"])
    if AR_PLACEHOLDER_RE.search(s):
        return pd.Series([np.nan, s, np.nan, np.nan], index=["diagnosis","plan_text","medication","dose_schedule"])
    cell_key = keyify(s)
    if cell_key in _NUT_EMPTY or cell_key in _NUT_NO_NEED or s.strip() in {"no need","not need","none"}:
        return pd.Series([np.nan, s, np.nan, np.nan], index=["diagnosis","plan_text","medication","dose_schedule"])
    if cell_key in {keyify(t) for t in _NUT_PLAN_ONLY}:
        return pd.Series([np.nan, s, np.nan, np.nan], index=["diagnosis","plan_text","medication","dose_schedule"])
    lines = [ln.strip() for ln in s.split("\n") if ln.strip()]
    if not lines:
        return pd.Series([np.nan, s, np.nan, np.nan], index=["diagnosis","plan_text","medication","dose_schedule"])
    if _nut_all_look_like_meds(lines):
        diag_lines, med_lines = [], lines
    else:
        first_med = next((i for i, ln in enumerate(lines) if _nut_is_med(ln)), None)
        if first_med is None:
            return pd.Series([s, s, np.nan, np.nan], index=["diagnosis","plan_text","medication","dose_schedule"])
        diag_lines, med_lines = lines[:first_med], lines[first_med:]
    diagnosis = "\n".join(diag_lines).strip() or np.nan
    med_to_scheds = OrderedDict()
    for ln in med_lines:
        ms, ss = _nut_split(ln)
        if not ms:
            continue
        for m, sch in zip(ms, ss):
            m = _nut_normalize_name((m or "").strip())
            sch = (sch or "").strip()
            if not m:
                continue
            if m not in med_to_scheds:
                med_to_scheds[m] = []
            if sch and sch not in med_to_scheds[m]:
                med_to_scheds[m].append(sch)
    medication = " | ".join(med_to_scheds.keys()) if med_to_scheds else np.nan
    if med_to_scheds:
        dose_schedule = " | ".join("; ".join(v) if v else "" for v in med_to_scheds.values())
        dose_schedule = dose_schedule if dose_schedule.strip() else np.nan
    else:
        dose_schedule = np.nan
    return build_output_series(diagnosis, s, list(med_to_scheds.keys()), ["; ".join(v) if v else "" for v in med_to_scheds.values()])


# ---------------------------------------------------------------------------
# Helper: generic scored parser factory used by Pediatrics, ENT, and later clinics
# ---------------------------------------------------------------------------
def _make_scored_parser(
    drugs_keys, aliases, diag_abbrev_keys, plan_only_keys,
    extra_norm_fn=None, use_per_line_classify=False
):
    """
    Returns a parse_medical_cell function using the scored approach.
    use_per_line_classify=True means ENT style (each line classified separately).
    """
    def _is_diag_abbrev(line):
        return keyify(line) in diag_abbrev_keys

    def _score(line):
        line0 = (line or "").strip()
        if not line0:
            return 0, 0
        low = re.sub(r"[\\/ ]+$", "", line0).strip().lower()
        tok = first_token_key(low)
        full = keyify(low)
        if tok in EMPTY_KEYS or full in EMPTY_KEYS:
            return 0, 0
        med_score = 0
        diag_score = 0
        if _is_diag_abbrev(low):
            diag_score += 5
        if SCHED_RE.search(low) or SCHED_WORDS_RE.search(low):
            med_score += 4
        if UNIT_OR_FORM_RE.search(low):
            med_score += 3
        if "+" in low and not re.search(r"\+\s*$", low):
            med_score += 2
        if re.search(r"\d", low):
            med_score += 1
        matched = fuzzy_key_match(tok, drugs_keys) or fuzzy_key_match(full, drugs_keys)
        if matched or tok in aliases or full in aliases:
            med_score += 3
        return med_score, diag_score

    def _is_med(line):
        med, diag = _score(line)
        if diag >= 5 and med < 5:
            return False
        return med >= 3 and med >= diag + 1

    def _all_look_like_meds(lines):
        if not lines:
            return False
        for ln in lines:
            _, d = _score(ln)
            if d >= 4:
                return False
        if any(_is_med(ln) for ln in lines):
            return True
        for ln in lines:
            parts = ln.split()
            if len(parts) != 1:
                return False
            k = keyify(parts[0])
            if not k or k in EMPTY_KEYS or _is_diag_abbrev(parts[0]) or len(k) < 3:
                return False
        return True

    def _normalize_name(m):
        m = (m or "").strip()
        if not m:
            return m
        k = keyify(m)
        return aliases.get(k, m.lower())

    def _split(line):
        line = (line or "").strip()
        if not line:
            return None, None
        line = re.sub(r"[\\/ ]+$", "", line).strip()
        if "+" in line and not SCHED_RE.search(line) and not SCHED_WORDS_RE.search(line):
            parts = [p.strip() for p in line.split("+") if p.strip()]
            meds = [_normalize_name(p) for p in parts]
            return meds, [""] * len(meds)
        starts = []
        m1 = SCHED_RE.search(line)
        if m1: starts.append(m1.start())
        m2 = SCHED_WORDS_RE.search(line)
        if m2: starts.append(m2.start())
        # trailing single qty
        m_end = re.search(r"^(.*?)(?:\s+)(\d+)\s*$", line)
        if not starts and m_end:
            before = m_end.group(1).strip()
            tail = m_end.group(2)
            if before and (UNIT_OR_FORM_RE.search(before) or keyify(before.split()[0]) in drugs_keys):
                return [before], [tail]
        if not starts:
            return [line], [""]
        start = min(starts)
        med_part = line[:start].strip()
        sch_part = re.sub(r"\s{2,}", " ", line[start:]).strip()
        return [med_part] if med_part else [line], [sch_part]

    def parse_cell(x):
        if pd.isna(x):
            return pd.Series([np.nan]*4, index=["diagnosis","plan_text","medication","dose_schedule"])
        raw_norm = extra_norm_fn if extra_norm_fn else norm_text_base
        s = raw_norm(x)
        if pd.isna(s) or s == "":
            return pd.Series([np.nan, s, np.nan, np.nan], index=["diagnosis","plan_text","medication","dose_schedule"])
        if AR_PLACEHOLDER_RE.search(s):
            return pd.Series([np.nan, s, np.nan, np.nan], index=["diagnosis","plan_text","medication","dose_schedule"])
        cell_key = keyify(s)
        if cell_key in plan_only_keys:
            return pd.Series([np.nan, s, np.nan, np.nan], index=["diagnosis","plan_text","medication","dose_schedule"])
        if cell_key in EMPTY_KEYS or s.strip() in {"no need","not need","none"}:
            return pd.Series([np.nan, s, np.nan, np.nan], index=["diagnosis","plan_text","medication","dose_schedule"])
        lines = [ln.strip() for ln in s.split("\n") if ln.strip() and keyify(ln) not in EMPTY_KEYS]
        if not lines:
            return pd.Series([np.nan, s, np.nan, np.nan], index=["diagnosis","plan_text","medication","dose_schedule"])

        if use_per_line_classify:
            diag_lines = [ln for ln in lines if not _is_med(ln)]
            med_lines  = [ln for ln in lines if _is_med(ln)]
            if not med_lines:
                return pd.Series([s, s, np.nan, np.nan], index=["diagnosis","plan_text","medication","dose_schedule"])
        elif _all_look_like_meds(lines):
            diag_lines, med_lines = [], lines
        else:
            first_med = next((i for i, ln in enumerate(lines) if _is_med(ln)), None)
            if first_med is None:
                return pd.Series([s, s, np.nan, np.nan], index=["diagnosis","plan_text","medication","dose_schedule"])
            diag_lines, med_lines = lines[:first_med], lines[first_med:]

        diagnosis = "\n".join(diag_lines).strip() or np.nan
        med_to_scheds = OrderedDict()
        for ln in med_lines:
            ms, ss = _split(ln)
            if not ms:
                continue
            for m, sch in zip(ms, ss):
                m = _normalize_name((m or "").strip())
                # for pediatrics: slash to pipe
                m = slash_to_pipe(m)
                sch = (sch or "").strip()
                if not m:
                    continue
                if m not in med_to_scheds:
                    med_to_scheds[m] = []
                if sch and sch not in med_to_scheds[m]:
                    med_to_scheds[m].append(sch)
        if not med_to_scheds:
            return pd.Series([diagnosis, s, np.nan, np.nan], index=["diagnosis","plan_text","medication","dose_schedule"])
        return build_output_series(
            diagnosis, s,
            list(med_to_scheds.keys()),
            ["; ".join(v) if v else "" for v in med_to_scheds.values()],
        )

    return parse_cell


# ---------------------------------------------------------------------------
# PEDIATRICS
# ---------------------------------------------------------------------------
_PED_DRUGS = {
    "vit","vitc","vitamin","multivit","multivitamin","multivitamines","mv","omega","omega3",
    "folicacid","folic","kirk","kirak",
    "amoxi","amoxicillin","clamoxin","zinnat","zinat","azicare","zitrocin","sulprim",
    "acamol","paracetamol","adol","trufen","diclofen","flagyl","omeprazol","famotidine",
    "nystatin","normasal","nsdrop","n.s","nirvin","ors","zinc","iron",
    "cool","coolcream","allow","allergan","allergon","albendazol","vermazol","vermazole",
    "permethrin","permetrin","permetrinn","calamin","calamine","betacare","neomycin",
    "clendamycin","clindamycin","cetal",
}
_PED_DRUGS_KEYS = {keyify(x) for x in _PED_DRUGS}

_PED_ALIASES = {
    "mv":"multivitamin","multivit":"multivitamin","multivitamines":"multivitamin",
    "vitc":"vit c","omega3":"omega 3","folicacid":"folic acid","kirak":"kirk",
    "amoxi":"amoxicillin","acamol":"paracetamol","diclofen":"diclofenac",
    "flagyl":"metronidazole","omeprazol":"omeprazole","normasal":"normasal drops",
    "nsdrop":"normasal drops","cool":"cool cream","coolcream":"cool cream",
    "calamin":"calamine lotion","calamine":"calamine lotion",
    "permethrin":"permethrin lotion","permetrin":"permethrin lotion","permetrinn":"permethrin lotion",
    "clendamycin":"clindamycin","albendazol":"albendazole","vermazol":"albendazole",
    "ieron":"iron","cetal":"cetal",
}
_PED_DIAG_ABBREV = {
    "urti","uti","age","at","na","nd","a","n",
    "adysentry","asbronch","asbronchitis","chpox","omedia",
    "stomatitis","impitigo","impetigo","arthralgia","gastritis","dentalpain",
    "skinallergy","skininf",
}
_PED_PLAN_ONLY = {"forderma","fordermatology","dermatology","forderm"}


def _ped_norm(x):
    if pd.isna(x):
        return np.nan
    s = str(x).replace("\r", "\n")
    s = re.sub(r"\bm\s*/\s*v\b", "mv", s, flags=re.I)
    s = re.sub(r"[\\]{2,}", "\n", s)
    s = re.sub(r"/{2,}", "\n", s)
    s = re.sub(r"(?<=\w)\\(?=\w)", "\n", s)
    s = re.sub(r"\*{1,}", "", s)
    s = re.sub(r"\b(urti|uti|age|nd|na|at|asbronch|asbronchitis|chpox|omedia)(?=[a-z])", r"\1\n", s, flags=re.I)
    s = re.sub(r"(\d+)\s*dr\s*\.\s*(\d+)", r"\1dr*\2", s, flags=re.I)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{2,}", "\n", s).strip()
    s = re.sub(r"([A-Za-z])(\d)", r"\1 \2", s)
    s = re.sub(r"(mg|ml|cc)(\d)", r"\1 \2", s, flags=re.I)
    s = re.sub(r"[ ]{2,}", " ", s).strip()
    return s.lower()


parse_pediatrics = _make_scored_parser(
    drugs_keys=_PED_DRUGS_KEYS,
    aliases=_PED_ALIASES,
    diag_abbrev_keys=_PED_DIAG_ABBREV,
    plan_only_keys=_PED_PLAN_ONLY,
    extra_norm_fn=_ped_norm,
    use_per_line_classify=False,
)


# ---------------------------------------------------------------------------
# ENT  (per-line classification)
# ---------------------------------------------------------------------------
_ENT_DRUGS = {
    "acamol","paracetamol","panadol","trufen",
    "amoxi","amoxicillin","augmentin","clamoxin","clamoxi","zinnat","azicare",
    "cipro","ciprocare","levox","moxif","oflox",
    "lorax","lorias","fexodin","fexoden","histafed","hista","histazin",
    "paraflu","paraf","flurest","allergon","allegro","betastin","selegon","solgon",
    "predni","prednisone","prednisolone",
    "otrivin","nosacare","otiblock","otoblock","gentacol","genta",
    "neurovit","ogmin",
}
_ENT_DRUGS_KEYS = {keyify(x) for x in _ENT_DRUGS}

_ENT_ALIASES = {
    "acamol":"paracetamol","panadol":"paracetamol","amoxi":"amoxicillin","clamoxi":"clamoxin",
    "cipro":"ciprofloxacin","ciprocare":"ciprofloxacin","levox":"levofloxacin",
    "moxif":"moxifloxacin","oflox":"ofloxacin","lorias":"lorax","fexoden":"fexodin",
    "paraf":"paraflu","solgon":"selegon","otoblock":"otiblock","predni":"prednisolone",
}
_ENT_DIAG_KEYS = {
    "flu","coldflu","cold","earwax","ear","wax","rhinitis","pharyngitis","otalgia",
    "cough","dns","followup","follow","up","foroutclinic","outclinic",
    "bronchitis","otitismedia","otitis","media",
}
_ENT_PLAN_ONLY_KEYS = {"foroutclinic","followup","follow","up"}

# Build glued-drug split regex for ENT
_ENT_GLUED_TOKENS = sorted(
    {keyify(t) for t in (list(_ENT_DRUGS) + list(_ENT_ALIASES.keys())) if keyify(t)},
    key=len, reverse=True,
)
_ENT_GLUED_RE = re.compile(
    r"(?i)(?<=[A-Za-z])(?=(" + "|".join(map(re.escape, _ENT_GLUED_TOKENS)) + r"))"
)


def _ent_norm(x):
    if pd.isna(x):
        return np.nan
    s = str(x).replace("\r", "\n")
    s = re.sub(r"\bm\s*/\s*v\b", "mv", s, flags=re.I)
    s = re.sub(r"[\\]{2,}", "\n", s)
    s = re.sub(r"/{2,}", "\n", s)
    s = re.sub(r"(?<=\w)\\(?=\w)", "\n", s)
    s = re.sub(r"\*{1,}", "", s)
    s = _ENT_GLUED_RE.sub("\n", s)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{2,}", "\n", s).strip()
    s = re.sub(r"([A-Za-z])(\d)", r"\1 \2", s)
    s = re.sub(r"(\d)\s*(mg|ml|cc|g|dr)\b", r"\1 \2", s, flags=re.I)
    s = re.sub(r"\b(mg|ml|cc|g|dr)\s*(\d)", r"\1 \2", s, flags=re.I)
    s = re.sub(r"[ ]{2,}", " ", s).strip()
    return s.lower()


parse_ent = _make_scored_parser(
    drugs_keys=_ENT_DRUGS_KEYS,
    aliases=_ENT_ALIASES,
    diag_abbrev_keys=_ENT_DIAG_KEYS,
    plan_only_keys=_ENT_PLAN_ONLY_KEYS,
    extra_norm_fn=_ent_norm,
    use_per_line_classify=True,
)


# ---------------------------------------------------------------------------
# GENERAL SURGERY
# ---------------------------------------------------------------------------
_GS_DRUGS = {
    "acamol","paracetamol","trufen","coxib","pirox",
    "amoxi","amoxicillin","clamoxin","zinat","zinnat","zinaxem","zinaxim","zinaxime",
    "cipro","ciproc","ciprocar","ciprocare","levox","floxin","ogmin","ogmain","flagyl",
    "omprazol","omeprazol","omeprazole","famodin","laxidin","laxadin","laxadine","laxedin",
    "scobutyl","allvent","multivit","mv","vit",
    "betacare","betacar","betaca","beta","fusdin","neomycin","miconazol","daktazole","betacorten",
}
_GS_DRUGS_KEYS = {keyify(x) for x in _GS_DRUGS}

_GS_ALIASES = {
    "acamol":"paracetamol","trufen":"trufen","pirox":"piroxicam","amoxi":"amoxicillin",
    "zinat":"zinnat","zinaxim":"zinaxem","zinaxime":"zinaxem","cipro":"ciprofloxacin",
    "ciproc":"ciprofloxacin","ciprocar":"ciprofloxacin","ciprocare":"ciprofloxacin",
    "levox":"levofloxacin","floxin":"ofloxacin","ogmin":"ogmin","ogmain":"ogmin",
    "oogmain":"ogmin","ogmen":"ogmin","flagyl":"metronidazole","omprazol":"omeprazole",
    "omeprazol":"omeprazole","famodin":"famotidine","laxidin":"laxidin","laxadin":"laxadin",
    "laxadine":"laxadin","laxedin":"laxadin","mv":"multivitamin","multivit":"multivitamin",
    "betacare":"betacare","betacar":"betacare","betaca":"betacare","beta":"betamethasone",
    "fusdin":"fusidin","miconazol":"miconazole",
}
_GS_PLAN_ONLY = {"followup","follow","up"}
_GS_DIAG_KEYS = {
    keyify("circumcision"),keyify("hemoral"),keyify("proctacare"),
    keyify("proctacar"),keyify("polycat"),keyify("polycatan"),
}


def _gs_norm(x):
    if pd.isna(x):
        return np.nan
    s = str(x).replace("\r", "\n")
    s = re.sub(r"\bm\s*/\s*v\b", "mv", s, flags=re.I)
    s = re.sub(r"[\\]{2,}", "\n", s)
    s = re.sub(r"/{2,}", "\n", s)
    s = re.sub(r"\*{1,}", "", s)
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", s)
    s = re.sub(r"([A-Za-z])(\d)", r"\1 \2", s)
    s = re.sub(r"(\d)\s*(mg|ml|cc|g|dr)\b", r"\1 \2", s, flags=re.I)
    s = re.sub(r"\b(mg|ml|cc|g|dr)\s*(\d)", r"\1 \2", s, flags=re.I)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{2,}", "\n", s).strip()
    s = re.sub(r"[ ]{2,}", " ", s).strip()
    return s.lower()


def _gs_is_diag(line):
    k = keyify(line)
    return k in _GS_DIAG_KEYS or bool(ARABIC_RE.search(line))


_GS_STRENGTH_RE = re.compile(
    r"^(?P<drug>[a-z][a-z0-9]*)(?:\s+(?P<num>\d+(?:\.\d+)?)(?:\s*(?P<unit>mg|ml|cc|g))?)?$",
    re.I,
)


def _gs_is_med(line):
    line = (line or "").strip()
    if not line or line in {"\\","/","."}:
        return False
    tok = first_token_key(line)
    full = keyify(line)
    if tok in EMPTY_KEYS or full in EMPTY_KEYS:
        return False
    if _gs_is_diag(line) and not (SCHED_RE.search(line) or UNIT_OR_FORM_RE.search(line)):
        return False
    if SCHED_RE.search(line) or SCHED_WORDS_RE.search(line) or UNIT_OR_FORM_RE.search(line):
        return True
    m = _GS_STRENGTH_RE.match(line)
    if m and keyify(m.group("drug")) in _GS_DRUGS_KEYS:
        return True
    if tok in _GS_DRUGS_KEYS or full in _GS_DRUGS_KEYS:
        return True
    if tok in _GS_ALIASES or full in _GS_ALIASES:
        return True
    return False


def _gs_split(line):
    line = (line or "").strip()
    if not line:
        return None, None
    line = re.sub(r"[\\/ ]+$", "", line).strip()
    if not SCHED_RE.search(line) and not SCHED_WORDS_RE.search(line):
        m = _GS_STRENGTH_RE.match(line.strip())
        if m and keyify(m.group("drug")) in _GS_DRUGS_KEYS:
            drug = m.group("drug")
            strength = m.group("num") or ""
            unit = m.group("unit") or ""
            sch = f"{strength} {unit}".strip() if strength else ""
            alias = _GS_ALIASES.get(keyify(drug), drug.lower())
            return [alias], [sch]
    starts = []
    m1 = SCHED_RE.search(line)
    if m1: starts.append(m1.start())
    m2 = SCHED_WORDS_RE.search(line)
    if m2: starts.append(m2.start())
    if not starts:
        return [line], [""]
    start = min(starts)
    med_part = line[:start].strip()
    sch_part = re.sub(r"\s{2,}", " ", line[start:]).strip()
    return [med_part] if med_part else [line], [sch_part]


def parse_general_surgery(x):
    if pd.isna(x):
        return pd.Series([np.nan]*4, index=["diagnosis","plan_text","medication","dose_schedule"])
    s = _gs_norm(x)
    if pd.isna(s) or s == "":
        return pd.Series([np.nan, s, np.nan, np.nan], index=["diagnosis","plan_text","medication","dose_schedule"])
    if AR_PLACEHOLDER_RE.search(s):
        return pd.Series([np.nan, s, np.nan, np.nan], index=["diagnosis","plan_text","medication","dose_schedule"])
    if re.search(r"\bfollow\s*up\b", s, flags=re.I):
        return pd.Series([np.nan, s, np.nan, np.nan], index=["diagnosis","plan_text","medication","dose_schedule"])
    cell_key = keyify(s)
    if cell_key in EMPTY_KEYS or re.fullmatch(r"\s*no+\s*need\s*", s, flags=re.I):
        return pd.Series([np.nan, s, np.nan, np.nan], index=["diagnosis","plan_text","medication","dose_schedule"])
    lines = [ln.strip() for ln in s.split("\n") if ln.strip() and ln.strip() not in {"\\","/","."}]
    if not lines:
        return pd.Series([np.nan, s, np.nan, np.nan], index=["diagnosis","plan_text","medication","dose_schedule"])
    first_med = next((i for i, ln in enumerate(lines) if _gs_is_med(ln)), None)
    if first_med is None:
        return pd.Series([s, s, np.nan, np.nan], index=["diagnosis","plan_text","medication","dose_schedule"])
    diagnosis = "\n".join(lines[:first_med]).strip() or np.nan
    meds_out, sch_out = [], []
    for ln in lines[first_med:]:
        ms, ss = _gs_split(ln)
        for m, sch in zip(ms, ss):
            k = keyify(m or "")
            m = _GS_ALIASES.get(k, (m or "").lower()).strip()
            if m:
                meds_out.append(m)
                sch_out.append((sch or "").strip())
    meds2, sch2 = dedup_meds(meds_out, sch_out)
    return build_output_series(diagnosis, s, meds2, sch2)


# ---------------------------------------------------------------------------
# GYN / OBGYN  — reuses norm_text_base with extra dot-run cleanup
# ---------------------------------------------------------------------------
_GYN_DRUGS = {
    "paracetamol","acamol","trufen","naproxen","naprex","aspirin","aspirinbaby",
    "folic","folicacid","iron","ferrous","ferrousfolic","ferrousfolicacid",
    "jeferol","jeferal","jefarol","jeferalr","jeferolr","ferrogol","fergol",
    "multivitamin","multivitamins","mv","omega","omega3","vit","vitamin",
    "microgynon","microgyno","microgyne","microgenon","microgx",
    "canesten","clotrimazole","clotrimazol","clitramazol","miconazole","miconazol",
    "mycoten","mycotine","nizoral","fluconazole",
    "augmentin","augmanten","amoxicillin","amoxi","amoxcillin",
    "cefuroxime","cefuroxim","zinaxime","zinexem","zinaxem","zinnat","levofloxacin","levox",
    "ciprofloxacin","cipro","ciprocare",
    "famotidine","famodine","famodtine","omeprazole","omperazole","esomeprazole","esmoprazole",
    "spasmin","scobutyl","scubutyl","scubutylco","scobutylco","scobtyle",
    "betaderm","zinc","zincoxide","povidex","uroclean","reolin","ladiez",
    "doxycycline","doxycyclin","doxypharm","flagyl","metronidazole","metrindazol","denacin",
}
_GYN_DRUGS_KEYS = {keyify(x) for x in _GYN_DRUGS}

_GYN_ALIASES = {
    "acamol":"paracetamol","trufeen":"trufen","trufen":"trufen","naprex":"naproxen",
    "aspirinbaby":"aspirin baby","folicacid":"folic acid","folic":"folic acid",
    "ferrousfolicacid":"ferrous & folic acid","ferrousfolic":"ferrous & folic acid",
    "jeferal":"jeferol","jefarol":"jeferol","jeferolr":"jeferol","jeferalr":"jeferol",
    "ferrogol":"ferrogol","fergol":"ferrogol","mv":"multivitamin",
    "multivitamins":"multivitamin","omega3":"omega 3","omega":"omega 3","vit":"vitamin",
    "microgyno":"microgynon","microgyne":"microgynon","microgenon":"microgynon",
    "microgx":"microgynon","microgynon":"microgynon",
    "clotrimazol":"clotrimazole","clitramazol":"clotrimazole","miconazol":"miconazole",
    "mycotine":"mycoten","metrindazol":"metronidazole","metrindazole":"metronidazole",
    "flagyl":"metronidazole","augmanten":"augmentin","augmnten":"augmentin",
    "amoxi":"amoxicillin","amoxcillin":"amoxicillin","amoxcilline":"amoxicillin",
    "cefuroxim":"cefuroxime","zinaxime":"cefuroxime","zinaxem":"cefuroxime",
    "zinexem":"cefuroxime","zinnat":"cefuroxime","levox":"levofloxacin",
    "cipro":"ciprofloxacin","ciprocare":"ciprofloxacin","ciprocar":"ciprofloxacin",
    "famodine":"famotidine","famodtine":"famotidine","omperazole":"omeprazole",
    "omperazol":"omeprazole","omprazol":"omeprazole","esmoprazole":"esomeprazole",
    "scubutylco":"scobutyl co","scobutylco":"scobutyl co","scubutyl":"scobutyl",
    "uroclean":"uroclean","zincoxide":"zinc oxide","doxycyclin":"doxycycline",
    "doxypharm":"doxycycline","ladiez":"ladiez douche",
}
_GYN_PLAN_ONLY = {
    "out clinic care","outclinic care","outclinic","out clinic",
    "for out clinic","for outclinic","follow up","followup","reassurance","counselling",
}
_GYN_EMPTY = {"0","n","nan","none","null","na","nil",".","\\/"}


def _gyn_norm(x):
    if pd.isna(x):
        return np.nan
    s = str(x).replace("\r", "\n")
    s = re.sub(r"\bm\s*/\s*v\b", "mv", s, flags=re.I)
    s = re.sub(r"\bm\.?\s*v\.?\b", "mv", s, flags=re.I)
    s = re.sub(r"[\\]{2,}", "\n", s)
    s = re.sub(r"/{2,}", "\n", s)
    s = re.sub(r"(?<=\w)\\(?=\w)", "\n", s)
    s = re.sub(r"\*{3,}", "", s)
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", s)
    s = re.sub(r"([A-Za-z])(\d)", r"\1 \2", s)
    s = re.sub(r"(?<!\d)\.{2,}(?!\d)", "", s)
    s = re.sub(r"\b(tab|cap|tabs|caps|syp|syr|susp|cream|oveles|ovules|ovule|douch|douche|supp)\.\b", r"\1", s, flags=re.I)
    s = re.sub(r"\b(tab|cap|tabs|caps|syp|syr|susp|cream|oveles|ovules|ovule|douch|douche|supp)\.\s*", r"\1 ", s, flags=re.I)
    s = re.sub(r"(\d)\s*(mg|ml|cc|g|dr)\b", r"\1 \2", s, flags=re.I)
    s = re.sub(r"\b(mg|ml|cc|g|dr)\s*(\d)", r"\1 \2", s, flags=re.I)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{2,}", "\n", s).strip()
    s = re.sub(r"[ ]{2,}", " ", s).strip()
    return s.lower()


def _gyn_is_med(line):
    line = (line or "").strip()
    if not line:
        return False
    tok = first_token_key(line)
    full = keyify(line)
    if full in _GYN_EMPTY or tok in _GYN_EMPTY:
        return False
    if ARABIC_RE.search(line) and not re.search(r"[a-z0-9]", line) and not (SCHED_RE.search(line) or UNIT_OR_FORM_RE.search(line)):
        return False
    if SCHED_RE.search(line) or SCHED_WORDS_RE.search(line) or UNIT_OR_FORM_RE.search(line):
        return True
    m = STRENGTH_RE.match(line)
    if m and keyify(m.group("drug")) in _GYN_DRUGS_KEYS:
        return True
    if tok in _GYN_DRUGS_KEYS or full in _GYN_DRUGS_KEYS:
        return True
    if tok in _GYN_ALIASES or full in _GYN_ALIASES:
        return True
    return False


def _gyn_split(line):
    line = (line or "").strip()
    if not line:
        return None, None
    starts = []
    m1 = SCHED_RE.search(line)
    if m1: starts.append(m1.start())
    m2 = SCHED_WORDS_RE.search(line)
    if m2: starts.append(m2.start())
    if starts:
        start = min(starts)
        med_part = line[:start].strip()
        sch_part = line[start:].strip()
    else:
        med_part, sch_part = line.strip(), ""
    if not sch_part:
        m = STRENGTH_RE.match(med_part)
        if m:
            drug = m.group("drug").strip()
            num = m.group("num")
            unit = m.group("unit") or ""
            alias = _GYN_ALIASES.get(keyify(drug), drug.lower())
            return [alias], [f"{num} {unit}".strip()]
    ms = STRENGTH_IN_NAME_RE.search(med_part)
    strength = ""
    if ms:
        strength = f"{ms.group('num')}{ms.group('unit')}".lower()
        med_part = STRENGTH_IN_NAME_RE.sub("", med_part).strip()
    med_part = strip_trailing_form_words(med_part).strip(" .-")
    sch_part = re.sub(r"\s{2,}", " ", sch_part).strip()
    if strength:
        sch_part = f"{strength} {sch_part}".strip() if sch_part else strength
    return [med_part] if med_part else [line], [sch_part]


def _gyn_explode(line):
    line = (line or "").strip()
    if not line:
        return []
    if re.search(r"\bor\b", line, flags=re.I):
        parts = [p.strip() for p in re.split(r"\bor\b", line, flags=re.I) if p.strip()]
        out = []
        for p in parts:
            out.extend(explode_multi_med_line(p))
        return out
    return explode_multi_med_line(line)


def parse_gyn_obstit(x):
    if pd.isna(x):
        return pd.Series([np.nan]*4, index=["diagnosis","plan_text","medication","dose_schedule"])
    s = _gyn_norm(x)
    if pd.isna(s) or s == "":
        return pd.Series([np.nan, s, np.nan, np.nan], index=["diagnosis","plan_text","medication","dose_schedule"])
    if AR_PLACEHOLDER_RE.search(s):
        return pd.Series([np.nan, s, np.nan, np.nan], index=["diagnosis","plan_text","medication","dose_schedule"])
    if ARABIC_RE.search(s) and not re.search(r"[a-z0-9]", s) and not (SCHED_RE.search(s) or UNIT_OR_FORM_RE.search(s)):
        return pd.Series([np.nan, s, np.nan, np.nan], index=["diagnosis","plan_text","medication","dose_schedule"])
    s2 = " ".join(s.lower().split())
    if s2 in _GYN_PLAN_ONLY or s2.startswith("for "):
        return pd.Series([np.nan, s, np.nan, np.nan], index=["diagnosis","plan_text","medication","dose_schedule"])
    if keyify(s) in _GYN_EMPTY or re.fullmatch(r"\s*no+\s*need\s*", s, flags=re.I):
        return pd.Series([np.nan, s, np.nan, np.nan], index=["diagnosis","plan_text","medication","dose_schedule"])
    lines = [ln.strip() for ln in s.split("\n") if ln.strip() and ln.strip() not in {".","\\/","/"}]
    if not lines:
        return pd.Series([np.nan, s, np.nan, np.nan], index=["diagnosis","plan_text","medication","dose_schedule"])
    first_med = next((i for i, ln in enumerate(lines) if _gyn_is_med(ln)), None)
    if first_med is None:
        return pd.Series([s, s, np.nan, np.nan], index=["diagnosis","plan_text","medication","dose_schedule"])
    diag_lines = lines[:first_med]
    med_lines_raw = lines[first_med:]
    med_lines = []
    for ln in med_lines_raw:
        if _gyn_is_med(ln):
            med_lines.extend(_gyn_explode(ln))
        else:
            diag_lines.append(ln)
    diagnosis = "\n".join(diag_lines).strip() or np.nan
    meds_out, sch_out = [], []
    for ln in med_lines:
        ms, ss = _gyn_split(ln)
        for m, sch in zip(ms, ss):
            k = keyify(m or "")
            m = _GYN_ALIASES.get(k, (m or "").lower()).strip()
            if m:
                meds_out.append(m)
                sch_out.append((sch or "").strip())
    meds2, sch2 = dedup_meds(meds_out, sch_out)
    return build_output_series(diagnosis, s, meds2, sch2)


# ---------------------------------------------------------------------------
# Remaining clinics reuse the generic factory with their own lexicons.
# Physiotherapy, Internal Medicine, Psychiatric, Deworming
# ---------------------------------------------------------------------------

# --- PHYSIOTHERAPY ---
_PHYSIO_DRUGS = {
    "paracetamol","acamol","trufen","lorax","loratadine","vit","vitamin",
    "augmentin","ogmin","ogmain","ogmentin","zinnat","cefuroxime","cefuroxim",
    "amoxicillin","amoxcillin","amoxi","moxi","otrivin","omnic",
    "famodin","famodine","famotidine","miconazole","micanaz","fluconazole","fllucan","kirk",
}
_PHYSIO_DRUGS_KEYS = {keyify(x) for x in _PHYSIO_DRUGS}
_PHYSIO_ALIASES = {
    "acamol":"paracetamol","lorax":"loratadine","loratadine":"loratadine","vit":"vitamin",
    "ogmin":"augmentin","ogmain":"augmentin","ogmentin":"augmentin","augmentin":"augmentin",
    "zinnat":"cefuroxime","cefuroxim":"cefuroxime","amoxi":"amoxicillin","amoxcillin":"amoxicillin",
    "moxi":"amoxicillin","famodin":"famotidine","famodine":"famotidine","micanaz":"miconazole",
    "fllucan":"fluconazole",
}


def _physio_norm(x):
    if pd.isna(x):
        return np.nan
    s = str(x).replace("\r", "\n")
    s = re.sub(r"\bm\s*/\s*v\b", "mv", s, flags=re.I)
    s = re.sub(r"\bm\.?\s*v\.?\b", "mv", s, flags=re.I)
    s = re.sub(r"[\\]{2,}", "\n", s)
    s = re.sub(r"/{2,}", "\n", s)
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", s)
    s = re.sub(r"([A-Za-z])(\d)", r"\1 \2", s)
    s = re.sub(r"(?<!\d)\.{2,}(?!\d)", "", s)
    s = re.sub(r"(\d)\s*(mg|ml|cc|g|dr)\b", r"\1 \2", s, flags=re.I)
    s = re.sub(r"\b(mg|ml|cc|g|dr)\s*(\d)", r"\1 \2", s, flags=re.I)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{2,}", "\n", s).strip()
    s = re.sub(r"[ ]{2,}", " ", s).strip()
    # strip Arabic placeholder
    ar_ph = re.compile(r"يرجى\s+إدخال\s+التشخيص\s+الطبي\s+هنا", re.IGNORECASE)
    if ar_ph.search(s):
        s2 = ar_ph.sub("", s).strip().lstrip(".-").strip()
        if len(keyify(s2)) <= 12 and re.fullmatch(r"[a-z0-9]+", keyify(s2) or "x"):
            s = ""
        else:
            s = s2
    return s.lower()


parse_physiotherapy = _make_scored_parser(
    drugs_keys=_PHYSIO_DRUGS_KEYS,
    aliases=_PHYSIO_ALIASES,
    diag_abbrev_keys=set(),
    plan_only_keys=set(),
    extra_norm_fn=_physio_norm,
    use_per_line_classify=False,
)


# --- INTERNAL MEDICINE ---
_IM_DRUGS = {
    "paracetamol","acamol","trufen","paraflu","voltamol","diclofen","diclofenac","lornoxicam","tericox","norgesic",
    "histazine","histazin","histasazine","histastazine","histafed","fexodine","ketofen","avilac",
    "famodine","famodin","famotidine","ezomax","omeprazole","pantoprazol","pantoprazole",
    "ogmin","moclav","amoxicare","wemox","vemox","vermox","zinaxim","zinaxime","zinaxamin","zinnat","zinnate",
    "clindamycin","flagyl","metronidazole","azicare","azicre","azithromycin","doxycyclin",
    "spasmin","spamin","spsmin","spasmodigestin","scobutyl","scubtyle","scobutylco",
    "multivit","multivitamin","multivitanmin","fergole","ferrgole","zinc",
    "simvastatin","atorvastatin","atorvastin","crestor","concor","concorplus","warfin","warfarin","warafarin",
    "aspirin","babyaspirin","baspirin","isosorbid","isosobide","lasix","aldactone","aldacton","spironolactone",
    "metformin","metfoemin","colchicne","amicor",
    "beclotid","ventolin","vetolin","fucidine","silver","nystatin","nustatin","acyclovir","uroclean",
}
_IM_DRUGS_KEYS = {keyify(x) for x in _IM_DRUGS}
_IM_ALIASES = {
    "acamol":"paracetamol","voltamol":"paracetamol","trufn":"trufen","tufen":"trufen",
    "histazin":"histazine","histasazine":"histazine","histastazine":"histazine",
    "famodin":"famotidine","famodine":"famotidine","ezomax":"esomeprazole",
    "omeprazol":"omeprazole","pantoprazol":"pantoprazole","azicare":"azithromycin",
    "azicre":"azithromycin","flagyl":"metronidazole","ogmin":"amoxicillin/clavulanate",
    "moclav":"amoxicillin/clavulanate","amoxicare":"amoxicillin","wemox":"amoxicillin",
    "zinaxim":"cefuroxime","zinaxime":"cefuroxime","zinaxamin":"cefuroxime","zinnat":"cefuroxime","zinnate":"cefuroxime",
    "doxycyclin":"doxycycline","spamin":"spasmin","spsmin":"spasmin","sapasmin":"spasmin",
    "scobutyl":"scobutyl","scubtyle":"scobutyl","scobutylco":"scobutyl co",
    "multivit":"multivitamin","multivitanmin":"multivitamin","fergole":"fergole","ferrgole":"fergole",
    "crestor":"rosuvastatin","atorvastin":"atorvastatin","concor":"bisoprolol",
    "concorplus":"bisoprolol (combo)","warfin":"warfarin","warafarin":"warfarin",
    "baspirin":"aspirin","babyaspirin":"aspirin","isosobide":"isosorbide","isosorbid":"isosorbide",
    "lasix":"furosemide","aldacton":"spironolactone","aldactone":"spironolactone",
    "metfoemin":"metformin","colchicne":"colchicine",
    "beclotid":"beclotid inhaler","ventolin":"ventolin inhaler","vetolin":"ventolin inhaler",
    "fucidine":"fusidic acid","silver":"silver cream","nustatin":"nystatin",
}


def _im_norm(x):
    if pd.isna(x):
        return np.nan
    s = str(x).replace("\r", "\n")
    s = re.sub(r"[\\]{2,}", "\n", s)
    s = re.sub(r"/{2,}", "\n", s)
    s = re.sub(r"\n?\+\n?", "\n", s)
    s = re.sub(r"(?<!\d)\.{2,}(?!\d)", "", s)
    s = re.sub(r"(\d)\s*m\s*g\b", r"\1 mg", s, flags=re.I)
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", s)
    s = re.sub(r"([A-Za-z])(\d)", r"\1 \2", s)
    s = re.sub(r"(\d)\s*(mg|ml|cc|g|gm|mcg|iu)\b", r"\1 \2", s, flags=re.I)
    s = re.sub(r"\b(mg|ml|cc|g|gm|mcg|iu)\s*(\d)", r"\1 \2", s, flags=re.I)
    s = re.sub(r"\b(\d+)\s*x\s*(\d+)\b", r"\1*\2", s, flags=re.I)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{2,}", "\n", s).strip()
    s = re.sub(r"[ ]{2,}", " ", s).strip()
    return s.lower()


parse_internal_medicine = _make_scored_parser(
    drugs_keys=_IM_DRUGS_KEYS,
    aliases=_IM_ALIASES,
    diag_abbrev_keys=set(),
    plan_only_keys=set(),
    extra_norm_fn=_im_norm,
    use_per_line_classify=False,
)


# --- PSYCHIATRIC ---
_PSY_DRUGS = {
    "levox","trufen","cipro","ciprofloxacin","amoxi","amoxicillin","acamol","paracetamol",
    "vit","vitamin","scobutyl","famodin","famodine","famotidine",
    "doxal","fergol","fergole","zinnat","zinaxime","zinexem",
    "laxidin","laxadine","laxadin","clotrimazole","clotrimazol","clitramazol",
    "denacin","azicare","azithromycin","ogmin","augmentin","mebendazol","mebendazole",
}
_PSY_DRUGS_KEYS = {keyify(x) for x in _PSY_DRUGS}
_PSY_ALIASES = {
    "acamol":"paracetamol","levox":"levofloxacin","cipro":"ciprofloxacin","amoxi":"amoxicillin",
    "vit":"vitamin","scobutyl":"scobutyl","famodin":"famotidine","famodine":"famotidine",
    "fergol":"fergole","zinnat":"cefuroxime","zinaxime":"cefuroxime","zinexem":"cefuroxime",
    "laxidin":"laxadine","laxadin":"laxadine","clitramazol":"clotrimazole","clotrimazol":"clotrimazole",
    "azicare":"azithromycin","ogmin":"amoxicillin/clavulanate","augmentin":"amoxicillin/clavulanate",
    "mebendazol":"mebendazole",
}


def _psy_norm(x):
    if pd.isna(x):
        return np.nan
    s = str(x).replace("\r", "\n")
    s = re.sub(r"[\\]{2,}", "\n", s)
    s = re.sub(r"/{2,}", "\n", s)
    s = s.strip()
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", s)
    s = re.sub(r"([A-Za-z])(\d)", r"\1 \2", s)
    s = re.sub(r"(\d)\s*(mg|ml|cc|g|gm|mcg|iu)\b", r"\1 \2", s, flags=re.I)
    s = re.sub(r"\b(mg|ml|cc|g|gm|mcg|iu)\s*(\d)", r"\1 \2", s, flags=re.I)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{2,}", "\n", s).strip()
    s = re.sub(r"[ ]{2,}", " ", s).strip()
    return s.lower()


parse_psychiatric = _make_scored_parser(
    drugs_keys=_PSY_DRUGS_KEYS,
    aliases=_PSY_ALIASES,
    diag_abbrev_keys=set(),
    plan_only_keys=set(),
    extra_norm_fn=_psy_norm,
    use_per_line_classify=False,
)


# --- DEWORMING ---
_DEW_DRUGS = {
    "albendazol","albendazole","albendezol","albendozol","albenazol","albenzol","albenzole","albenda","albend","alben",
    "vermox","vermazol","mebendazol","mebendazole","vermazole",
    "ors","zinc","flagyl","flagyle","flagel","falgyle","flafyl",
    "scobutyl","scobutyl co","scobtyle","scopytel","scopo","scopotel","scopoten",
    "acamol","adool","trufen","sulprim","sulprime",
    "daktazol","dektazol","dectazol","duktazol","dexazol","dekta",
    "miconazol","nystatin","polycutan","cool","cool cream","silver cream",
    "betacare","predn","azicare","azytro","zinaxim","zinnat","famodin","mv","multivit",
    "salyin","n.s",
}
_DEW_DRUGS_KEYS = {keyify(x) for x in _DEW_DRUGS}
_DEW_ALIASES = {
    "albendazol":"albendazole","albendezol":"albendazole","albendozol":"albendazole",
    "albenazol":"albendazole","albenzol":"albendazole","albenzole":"albendazole",
    "albenda":"albendazole","albend":"albendazole","alben":"albendazole","albrndazol":"albendazole",
    "vermox":"mebendazole","mebendazol":"mebendazole","vermazol":"mebendazole",
    "vermazolsusp":"mebendazole","vermazolsyr":"mebendazole",
    "ors":"ors","zinc":"zinc","flagyl":"metronidazole","flagyle":"metronidazole",
    "flagel":"metronidazole","falgyle":"metronidazole","flafyl":"metronidazole",
    "scopytel":"scopytel","scopo":"scopytel","scopotel":"scopytel","scopoten":"scopytel",
    "scobutyl":"scobutyl","scobutylco":"scobutyl co","scobtyle":"scobutyl",
    "acamol":"paracetamol","adool":"paracetamol","azicare":"azithromycin","azi":"azithromycin",
    "azytro":"azithromycin","famodin":"famotidine","mv":"multivitamin","m.v":"multivitamin",
    "multivit":"multivitamin","zinnat":"cefuroxime","zinaxim":"cefuroxime",
    "miconazol":"miconazole","polycutan":"polycutan cream","cool":"cool cream","coolcream":"cool cream",
    "silvercream":"silver cream","daktazol":"daktazol","dektazol":"daktazol",
    "dectazol":"daktazol","duktazol":"daktazol","dexazol":"daktazol","dekta":"daktazol",
    "ns":"normal saline","n.s":"normal saline","salyin":"normal saline",
    "sulprim":"sulprim","sulprime":"sulprim","betacare":"betacare","predn":"predn",
}


def _dew_norm(x):
    if pd.isna(x):
        return np.nan
    s = str(x).replace("\r", "\n")
    s = re.sub(r"[\\]{2,}", "\n", s)
    s = re.sub(r"/{2,}", "\n", s)
    s = re.sub(r"\bo\s*[.\s]*r\s*[.\s]*s\b", "ors", s, flags=re.I)
    s = re.sub(r"\bo\s*[.\s]*p\s*[.\s]*s\b", "ors", s, flags=re.I)
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", s)
    s = re.sub(r"([A-Za-z])(\d)", r"\1 \2", s)
    s = re.sub(r"(\d)\s*(mg|ml|cc|g|gm|mcg|iu)\b", r"\1 \2", s, flags=re.I)
    s = re.sub(r"\b(mg|ml|cc|g|gm|mcg|iu)\s*(\d)", r"\1 \2", s, flags=re.I)
    s = re.sub(r"\b(\d+)\s*x\s*(\d+)\b", r"\1*\2", s, flags=re.I)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{2,}", "\n", s).strip()
    s = re.sub(r"[ ]{2,}", " ", s).strip()
    return s.lower()


def _dew_is_med(line):
    line = (line or "").strip()
    if not line or is_empty_like(line):
        return False
    tok = first_token_key(line)
    full = keyify(line)
    if SCHED_RE.search(line) or UNIT_OR_FORM_RE.search(line):
        return True
    m = STRENGTH_RE.match(line)
    if m and keyify(m.group("drug")) in _DEW_DRUGS_KEYS:
        return True
    if tok in _DEW_DRUGS_KEYS or full in _DEW_DRUGS_KEYS:
        return True
    if tok in _DEW_ALIASES or full in _DEW_ALIASES:
        return True
    if len(tok) >= 3:
        return True
    return False


def _dew_explode(line):
    toks = [t.strip() for t in re.split(r"[,\s]+", line) if t.strip()]
    if len(toks) >= 2:
        known = sum(1 for t in toks if keyify(t) in _DEW_DRUGS_KEYS or keyify(t) in _DEW_ALIASES)
        if known >= 2 and known == len(toks):
            return toks
    return [line]


def parse_deworming_clinic(x):
    if pd.isna(x):
        return pd.Series([np.nan]*4, index=["diagnosis","plan_text","medication","dose_schedule"])
    s = _dew_norm(x)
    if pd.isna(s) or s == "" or is_empty_like(s):
        return pd.Series([np.nan]*4, index=["diagnosis","plan_text","medication","dose_schedule"])
    lines = [ln.strip() for ln in s.split("\n") if ln.strip() and not is_empty_like(ln)]
    if not lines:
        return pd.Series([np.nan, s, np.nan, np.nan], index=["diagnosis","plan_text","medication","dose_schedule"])
    meds_out, sch_out, diag_lines = [], [], []
    for ln in lines:
        if _dew_is_med(ln):
            for part in _dew_explode(ln):
                msch = SCHED_RE.search(part)
                if msch:
                    med_part = part[:msch.start()].strip()
                    sch_part = part[msch.start():].strip()
                else:
                    m = STRENGTH_RE.match(part)
                    if m and keyify(m.group("drug")) in _DEW_DRUGS_KEYS:
                        drug = m.group("drug").strip()
                        num = m.group("num")
                        unit = m.group("unit") or ""
                        med_part = _DEW_ALIASES.get(keyify(drug), drug.lower())
                        sch_part = f"{num} {unit}".strip()
                    else:
                        med_part, sch_part = part, ""
                k = keyify(med_part)
                m_name = _DEW_ALIASES.get(k, med_part.lower()).strip(" .-")
                if m_name:
                    meds_out.append(m_name)
                    sch_out.append(sch_part.strip())
        else:
            diag_lines.append(ln)
    diagnosis = "\n".join(diag_lines).strip() or np.nan
    meds2, sch2 = dedup_meds(meds_out, sch_out, key_fn=lambda m, s: (m.lower(), s.lower()))
    return build_output_series(diagnosis, s, meds2, sch2)


# ---------------------------------------------------------------------------
# Dispatcher: maps clinic name -> parse function
# ---------------------------------------------------------------------------
CLINIC_PARSERS = {
    "dermatology":       parse_dermatology,
    "orthopedic":        parse_orthopedic,
    "urology":           parse_urology,
    "nutrition":         parse_nutrition,
    "pediatrics":        parse_pediatrics,
    "ent":               parse_ent,
    "general surgery":   parse_general_surgery,
    "gyn.&obstit":       parse_gyn_obstit,
    "physiotherapy":     parse_physiotherapy,
    "internal medicine": parse_internal_medicine,
    "psychiatric":       parse_psychiatric,
    "deworming clinic":  parse_deworming_clinic,
}


def parse_medical_cell(row):
    """Route a row to the correct clinic parser based on the 'Clinics' column."""
    clinic = str(row.get("Clinics", "")).strip().lower()
    parser = CLINIC_PARSERS.get(clinic)
    if parser:
        return parser(row["MEDICAL"])
    # Fallback: return NaNs for unknown clinics
    return pd.Series([np.nan]*4, index=["diagnosis","plan_text","medication","dose_schedule"])
