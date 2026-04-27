# Copyright © 2026 Apple Inc.

import json
import re
from typing import Any

tool_call_start = "<｜DSML｜function_calls>"
tool_call_end = "</｜DSML｜function_calls>"

_invoke_regex = re.compile(
    r'<｜DSML｜invoke\s+name="([^"]+)">(.*?)</｜DSML｜invoke>',
    re.DOTALL,
)
_parameter_regex = re.compile(
    r'<｜DSML｜parameter\s+name="([^"]+)"\s+string="(true|false)">(.*?)</｜DSML｜parameter>',
    re.DOTALL,
)


def _parse_invoke(match: re.Match):
    name, body = match.groups()
    arguments = {}
    for parameter in _parameter_regex.finditer(body):
        param_name, is_string, value = parameter.groups()
        if is_string != "true":
            value = json.loads(value)
        arguments[param_name] = value
    return {"name": name, "arguments": arguments}


def parse_tool_call(text: str, _: Any | None = None):
    calls = [_parse_invoke(invoke) for invoke in _invoke_regex.finditer(text)]
    if not calls:
        raise ValueError("No function provided.")
    return calls[0] if len(calls) == 1 else calls
