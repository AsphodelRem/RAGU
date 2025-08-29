# Default structured data extractors

import json
import logging
import re


def dummy_extractor(input_str: str) -> dict:
    """
    Just do nothing
    """
    return {"text": input_str}


def json_extractor(input_str: str) -> dict:
    match = re.search(r'\{.*\}', input_str, re.DOTALL)
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        logging.warning(f"Bad JSON: {input_str}")
        return None
