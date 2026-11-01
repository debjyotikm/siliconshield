"""Conservative lexical RTL hints, never elaboration or a completeness proof."""

import re
from typing import TypedDict

_SYMBOL = re.compile(r"\b[A-Za-z_][\w$]*\b")
_KEYWORDS = frozenset(
    "module endmodule input output inout wire reg logic bit signed unsigned integer int always always_ff always_comb always_latch assign if else case casex casez endcase for while function endfunction task endtask begin end posedge negedge parameter localparam generate endgenerate automatic static default typedef struct return genvar initial unique priority or".split()
)
_DECL = re.compile(
    r"\b(input|output|inout|wire|reg|logic|bit|parameter|localparam|integer|int|genvar)\b([^;]*)(?:;|$)"
)
_ASSIGN = re.compile(
    r"\b([A-Za-z_][\w$]*)\s*(?:\[[^\]\n]+\]\s*)*(?:<=|=(?!=|>))"
)
_NUMBER = re.compile(r"(?:\b\d+)?'[sS]?[bBdDhHoO][0-9a-fA-F_xXzZ?]+|'[01xXzZ]")


class StructureMap(TypedDict):
    """Lexical hints with explicit limits on completeness."""

    writers: dict[str, list[int]]
    guards: list[str]
    state: list[str]
    helper_calls: list[str]
    unknown_symbols: list[str]
    complete: bool
    limitations: list[str]


def _mask(text: str) -> str:
    """Mask comments and strings while preserving positions and line numbers."""
    return re.sub(
        r'//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"',
        lambda match: "".join(
            "\n" if char == "\n" else " " for char in match[0]
        ),
        text,
    )


def _declarations(code: str) -> tuple[set[str], set[str]]:
    """Extract declaration names, excluding widths and initializer operands."""
    declared: set[str] = set()
    state: set[str] = set()
    for match in _DECL.finditer(code):
        body = match[1] + " " + match[2]
        # Commas in expressions/dimensions are not declaration separators.
        depth = 0
        pieces = []
        start = 0
        for index, char in enumerate(body):
            if char in "([{":
                depth += 1
            elif char in ")]}":
                if depth == 0:  # End of an ANSI module port list.
                    body = body[:index]
                    break
                depth -= 1
            elif char == "," and depth == 0:
                pieces.append(body[start:index])
                start = index + 1
        pieces.append(body[start:])
        storage = False
        for piece in pieces:
            lhs = piece.split("=", 1)[0]
            lhs = re.sub(r"\[[^\]]*\]", " ", lhs)
            names = _SYMBOL.findall(lhs)
            if any(name in ("input", "output", "inout") for name in names):
                storage = False
            storage = storage or any(name in ("logic", "reg") for name in names)
            names = [name for name in names if name not in _KEYWORDS]
            if names:
                declared.add(names[0])
                if storage:
                    state.add(names[0])
    declared.update(re.findall(r"\bmodule\s+([A-Za-z_][\w$]*)", code))
    return declared, state


def inspect_structure(rtl: str, known_symbols: list[str]) -> StructureMap:
    """Extract comment-free lexical hints with one-based source lines.

    Balanced if conditions are masked before searching assignments so a
    comparison does not become a writer. Macro expansion, hierarchy and
    procedural dataflow are deliberately left to a real SystemVerilog tool.
    """
    code = _mask(rtl)
    declared, state = _declarations(code)
    guards = []
    assignment_text = list(code)
    for match in re.finditer(r"\bif\s*\(", code):
        start = match.end()
        depth = 1
        end = start
        while end < len(code) and depth:
            depth += (code[end] == "(") - (code[end] == ")")
            end += 1
        if depth == 0:
            guard = code[start : end - 1].strip()
            if guard not in guards:
                guards.append(guard)
            for index in range(match.start(), end):
                if assignment_text[index] != "\n":
                    assignment_text[index] = " "
    writers: dict[str, list[int]] = {}
    for match in _ASSIGN.finditer("".join(assignment_text)):
        lineno = code.count("\n", 0, match.start()) + 1
        lines = writers.setdefault(match[1], [])
        if lineno not in lines:
            lines.append(lineno)
    module_names = set(re.findall(r"\bmodule\s+([A-Za-z_][\w$]*)", code))
    calls = list(
        dict.fromkeys(
            name
            for name in re.findall(r"\b([A-Za-z_][\w$]*)\s*\(", code)
            if name not in _KEYWORDS and name not in module_names
        )
    )
    unknown = sorted(
        set(_SYMBOL.findall(_NUMBER.sub(" ", code)))
        - _KEYWORDS
        - declared
        - set(known_symbols)
    )
    limitations = [
        "Lexical hints only: no elaboration, reachability, driver resolution or proof of completeness."
    ]
    if re.search(r"`|\b(?:case|generate|task|function|typedef)\b", code):
        limitations.append(
            "Macros or complex declarations/control flow require parser review."
        )
    return dict(
        writers=writers,
        guards=guards,
        state=sorted(state),
        helper_calls=calls,
        unknown_symbols=unknown,
        complete=False,
        limitations=limitations,
    )
