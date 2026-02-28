from __future__ import annotations
import json
import re
from typing import Any, Tuple, Optional

def safe_json_loads(text: str) -> Tuple[Optional[Any], str]:
    try:
        return json.loads(text), ""
    except Exception as e:
        m = re.search(r'\{.*\}', text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0)), ""
            except Exception as e2:
                return None, f"json_parse_failed: {e2}"
        return None, f"json_parse_failed: {e}"

def ensure_dir(path: str) -> None:
    import os
    os.makedirs(path, exist_ok=True)

def write_text(path: str, content: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
