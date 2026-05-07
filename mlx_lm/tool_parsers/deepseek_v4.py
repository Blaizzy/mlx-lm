import json
from typing import Any

import regex as re

_dsml = "｜DSML｜"

# Match from the DSML tag, not the official text sentinel's leading newlines.
# The server detects tool calls with token sequences, and DS4 can tokenize the
# same text as either "\n\n" or as part of the previous token, e.g. ".\n\n".
# Stop before ">" as well because the closing bracket can merge with "\n".
tool_call_start: str = f"<{_dsml}tool_calls"
tool_call_end: str = f"</{_dsml}tool_calls>"

_invoke_re = re.compile(
    rf'<{re.escape(_dsml)}invoke name="([^"]+)">(.*?)</{re.escape(_dsml)}invoke>',
    re.DOTALL,
)
_param_re = re.compile(
    rf'<{re.escape(_dsml)}parameter name="([^"]+)" string="(true|false)">(.*?)</{re.escape(_dsml)}parameter>',
    re.DOTALL,
)


def parse_tool_call(text: str, tools: Any = None):
    """Parse one or more DSML tool calls from the text between tool_call_start/end."""
    calls = []
    for invoke in _invoke_re.finditer(text):
        name = invoke.group(1)
        body = invoke.group(2)
        arguments = {}
        for param in _param_re.finditer(body):
            pname = param.group(1)
            is_string = param.group(2) == "true"
            value = param.group(3)
            arguments[pname] = value if is_string else json.loads(value)
        calls.append({"name": name, "arguments": arguments})

    if not calls:
        raise ValueError("No tool calls found in DSML block")
    return calls[0] if len(calls) == 1 else calls
