"""
Shared text normalization utilities used across all clinic parsers.
"""
import re
import numpy as np
import pandas as pd
from difflib import get_close_matches

ARABIC_RE = re.compile(r"[\u0600-\u06FF]")
AR_PLACEHOLDER_RE = re.compile(
    r"(يرجى\s+إدخال\s+التشخيص\s+الطبي|يرجى\s+ادخال\s+التشخيص\s+الطبي|ادخل\s+التشخيص\s+الطبي)",
    re.IGNORECASE,
)

NUM = r"(?:\d+(?:\.\d+)?|\d+/\d+)"

SCHED_RE = re.compile(
    rf"{NUM}(?:\s*(?:cc|ml|mg|g|gm|dr))?(?:\s*[*xX&]\s*{NUM}){{1,4}}(?=\D|$)",
    re.IGNORECASE,
)
SCHED_WORDS_RE = re.compile(
    r"\b(once|once\s+daily|twice\s+daily|daily|bid|tid|stat|q\d+h|every\s+\d+\s*h)\b",
    re.IGNORECASE,
)
UNIT_OR_FORM_RE = re.compile(
    r"(?i)\b(mg|ml|cc|g|gm|mcg|iu|tab|tabs|cap|caps|gel|drops|drop|cream|ointment|"
    r"syrup|syr|syp|supp|susp|lotion|spray|sachet|ovule|ovules|douch|douche)\b"
)
STRENGTH_RE = re.compile(
    r"^(?P<drug>[a-z][a-z0-9\s&\.\-]+?)\s+(?P<num>\d+(?:\.\d+)?)(?:\s*(?P<unit>mg|ml|cc|g|gm|mcg|iu))?\s*$",
    re.I,
)
STRENGTH_IN_NAME_RE = re.compile(
    r"\b(?P<num>\d+(?:\.\d+)?)\s*(?P<unit>mg|ml|cc|g|gm|mcg|iu)\b", re.I
)

EMPTY_KEYS = {"0", "n", "nan", "none", "null", "na", "nil", ".", "\\", "/"}


def keyify(s: str) -> str:
    """Normalize for matching: lowercase, keep only letters/digits/+."""
    if s is None:
        return ""
    s = str(s).strip().lower()
    return re.sub(r"[^a-z0-9+]+", "", s)


def fuzzy_key_match(token_key: str, known_keys: set, cutoff: float = 0.88):
    if not token_key:
        return None
    if token_key in known_keys:
        return token_key
    m = get_close_matches(token_key, list(known_keys), n=1, cutoff=cutoff)
    return m[0] if m else None


def first_token_key(line: str) -> str:
    line = (line or "").strip().lower()
    if not line:
        return ""
    return keyify(line.split()[0])


def is_empty_like(s: str) -> bool:
    return keyify(s) in EMPTY_KEYS or not str(s).strip()


def strip_trailing_form_words(name: str) -> str:
    name = (name or "").strip()
    if not name:
        return name
    toks = name.split()
    while toks and UNIT_OR_FORM_RE.fullmatch(toks[-1]):
        toks.pop()
    return " ".join(toks).strip()


def slash_to_pipe(s: str) -> str:
    """Convert internal '/' in medication text to ' | '."""
    if not s:
        return s
    s = re.sub(r"\s*/\s*", " | ", s)
    s = re.sub(r"\s*\|\s*\|\s*", " | ", s)
    return re.sub(r"[ ]{2,}", " ", s).strip()


def norm_text_base(x, *, lower: bool = True) -> str:
    """
    Core normalization shared by most clinic parsers:
    - preserve m/v
    - \\\\ and // -> newline
    - CamelCase split
    - word+number split
    - number+unit split
    - whitespace cleanup
    """
    if pd.isna(x):
        return np.nan

    s = str(x).replace("\r", "\n")
    s = re.sub(r"\bm\s*/\s*v\b", "mv", s, flags=re.I)
    s = re.sub(r"\bm\.?\s*v\.?\b", "mv", s, flags=re.I)
    s = re.sub(r"[\\]{2,}", "\n", s)
    s = re.sub(r"/{2,}", "\n", s)
    s = re.sub(r"(?<=\w)\\(?=\w)", "\n", s)
    s = re.sub(r"\*{1,}", "", s)
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", s)
    s = re.sub(r"([A-Za-z])(\d)", r"\1 \2", s)
    s = re.sub(r"(\d)\s*(mg|ml|cc|g|gm|mcg|iu)\b", r"\1 \2", s, flags=re.I)
    s = re.sub(r"\b(mg|ml|cc|g|gm|mcg|iu)\s*(\d)", r"\1 \2", s, flags=re.I)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{2,}", "\n", s).strip()
    s = re.sub(r"[ ]{2,}", " ", s).strip()
    return s.lower() if lower else s


def explode_multi_med_line(line: str) -> list:
    """Split a line that contains multiple schedule tokens into separate med segments."""
    line = (line or "").strip()
    if not line:
        return []
    segs = []
    cur = line
    while True:
        matches = list(SCHED_RE.finditer(cur))
        if len(matches) <= 1:
            segs.append(cur.strip())
            break
        m0 = matches[0]
        left = cur[: m0.end()].strip()
        right = cur[m0.end() :].strip()
        if left:
            segs.append(left)
        if not right:
            break
        cur = right
    return [s for s in segs if s and keyify(s) not in EMPTY_KEYS]


def build_output_series(diagnosis, plan_text, medication_list, schedule_list):
    """
    Given lists of meds and schedules (already de-duped), return a pd.Series
    with columns: diagnosis, plan_text, medication, dose_schedule.
    """
    medication = " | ".join(medication_list) if medication_list else np.nan
    dose_schedule = " | ".join(schedule_list) if medication_list else np.nan
    if isinstance(dose_schedule, str) and not dose_schedule.strip():
        dose_schedule = np.nan
    return pd.Series(
        [
            diagnosis if diagnosis not in ("", None) else np.nan,
            plan_text,
            medication,
            dose_schedule,
        ],
        index=["diagnosis", "plan_text", "medication", "dose_schedule"],
    )


def dedup_meds(meds_out: list, sch_out: list, key_fn=None):
    """De-duplicate (med, schedule) pairs preserving insertion order."""
    if key_fn is None:
        key_fn = lambda m, s: m.lower()
    seen = set()
    meds2, sch2 = [], []
    for m, sch in zip(meds_out, sch_out):
        k = key_fn(m, sch)
        if k in seen:
            continue
        seen.add(k)
        meds2.append(m)
        sch2.append(sch)
    return meds2, sch2
