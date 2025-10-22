# api/utils_slug.py
import re
import unicodedata

_slug_re = re.compile(r"[^a-z0-9]+")

def slugify(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = s.lower().strip()
    s = _slug_re.sub("-", s).strip("-")
    return s or "tenant"