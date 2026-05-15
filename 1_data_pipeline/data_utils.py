"""Utilities pour la manipulation des données (regex, helpers)."""
import re

TIMESTAMP_RE = re.compile(r"\d{4}-\d{2}-\d{2}")

def looks_like_timestamp(s):
    return bool(TIMESTAMP_RE.search(str(s)))
