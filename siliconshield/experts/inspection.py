"""Conservative lexical RTL hints, never elaboration or a completeness proof."""

import re
from typing import TypedDict

_SYMBOL = re.compile(r"(?<!\$)\b[A-Za-z_][\w$]*\b")
_KEYWORDS = frozenset(
    "module endmodule input output inout wire reg logic bit signed unsigned integer int always always_ff always_comb always_latch assign if else case casex casez endcase for while function endfunction task endtask begin end posedge negedge parameter localparam generate endgenerate automatic static default typedef struct return genvar initial unique priority or".split()
    + "byte shortint longint shortreal real realtime time const ref extern pure global enum union packed interface modport package import export class extends new super this null virtual protected local rand randc constraint solve before dist inside with matches throughout within first_match iff illegal_bins ignore_bins wildcard bins binsof covergroup endgroup coverpoint cross option type_option sample cover assert assume expect property sequence clocking disable fork join join_any join_none wait repeat forever break continue do void chandle string event checker endchecker let bind alias tri wand wor triand trior supply0 supply1 highz0 highz1 small medium large pull0 pull1 weak0 weak1 scalared vectored macromodule primitive specify endspecify table endtable edge ifnone unique0 cell config endconfig design instance liblist use timeunit timeprecision".split()
)
_DECL = re.compile(
    r"\b(input|output|inout|wire|reg|logic|bit|parameter|localparam|integer|int|genvar)\b([^;]*)(?:;|$)"
)
_ASSIGN = re.compile(
    r"\b([A-Za-z_][\w$]*)\s*(?:\[[^\]\n]+\]\s*)*"
    r"(<<<=|>>>=|<<=|>>=|<=|\+=|-=|\*=|/=|%=|&=|\|=|\^=|=(?!=|>)|\+\+|--)"
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
    """Mask comments and strings while preserving positions and line numbers.

    An unterminated block comment masks the rest of the file: there
    is no code after it worth misreading.
    """
    return re.sub(
        r'//[^\n]*|/\*[\s\S]*?(?:\*/|$)|"(?:\\.|[^"\\])*"',
        lambda match: "".join(
            "\n" if char == "\n" else " " for char in match[0]
        ),
        text,
    )


def _declarations(code: str) -> tuple[set[str], set[str]]:
    """Extract declaration names, excluding widths and initializer operands."""
    declared: set[str] = set()
    state: set[str] = set()
    # Parameter-value lists `#(...)` hold `=` initializers, not port
    # declarations; blank them so the port scan below starts clean.
    chars = list(code)
    for header in re.finditer(r"#\s*\(", code):
        start, depth, end = header.end(), 1, header.end()
        while end < len(code) and depth:
            depth += (code[end] == "(") - (code[end] == ")")
            end += 1
        if depth == 0:
            for index in range(header.start(), end):
                if chars[index] != "\n":
                    chars[index] = " "
    joined = "".join(chars)
    # Brace-delimited bodies (struct/union/enum members, aggregate
    # initializers) declare scoped names, not top-level signals;
    # blank them so member names never read as design state.
    depth = 0
    start = -1
    scoped = list(joined)
    for index, char in enumerate(joined):
        if char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}":
            if depth:
                depth -= 1
                if depth == 0 and start >= 0:
                    for blank in range(start, index + 1):
                        if scoped[blank] != "\n":
                            scoped[blank] = " "
                    start = -1
    joined = "".join(scoped)
    # A typedef names a type, not storage: blank the whole statement
    # so the type name neither becomes state nor hides as declared.
    # The parser-review limitation still fires for the file.
    for match in re.finditer(r"\btypedef\b", joined):
        end = match.end()
        while end < len(joined) and joined[end] != ";":
            end += 1
        scoped = list(joined)
        for index in range(match.start(), min(end + 1, len(joined))):
            if scoped[index] != "\n":
                scoped[index] = " "
        joined = "".join(scoped)
    for match in _DECL.finditer(joined):
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
            if "input" in names:
                storage = False
            else:
                # Output and inout regs hold state across cycles;
                # only driven inputs are purely combinational ports.
                storage = storage or any(
                    name in ("logic", "reg") for name in names
                )
            names = [name for name in names if name not in _KEYWORDS]
            if names:
                declared.add(names[0])
                if storage:
                    state.add(names[0])
    declared.update(re.findall(r"\bmodule\s+([A-Za-z_][\w$]*)", code))
    return declared, state


def _is_comparison(searchable: str, at: int) -> bool:
    """Whether a `<=` at `at` compares instead of assigning.

    A `<=` after a plain `=` in the same statement (as in
    `assign out = a <= b;`) is a less-or-equal comparison, not a
    nonblocking write of its left operand. Assignments inside
    parentheses (such as for-loop headers) do not count: they
    belong to an enclosed statement, not this one.
    """
    start = max(
        searchable.rfind(";", 0, at),
        searchable.rfind("{", 0, at),
        searchable.rfind("}", 0, at),
    )
    segment = searchable[start + 1 : at]
    segment = re.sub(r"<=|==|!=|>=|=<", " ", segment)
    stripped: list[str] = []
    # The segment may open inside a for-loop header; inherit the
    # enclosing paren depth so header assignments stay hidden.
    depth = searchable.count("(", 0, start + 1) - searchable.count(
        ")", 0, start + 1
    )
    for char in segment:
        if char == "(":
            depth += 1
        elif char == ")":
            depth = max(depth - 1, 0)
        elif depth == 0:
            stripped.append(char)
    return "=" in "".join(stripped)


def inspect_structure(rtl: str, known_symbols: list[str]) -> StructureMap:
    """Extract comment-free lexical hints with one-based source lines.

    Balanced if conditions are masked before searching assignments,
    and a nonblocking operator inside parentheses, or after a
    plain assignment in the same statement, reads as a
    comparison, so neither becomes a writer. Macro expansion,
    hierarchy and procedural dataflow are deliberately left to a
    real SystemVerilog tool.
    """
    code = _mask(rtl)
    declared, state = _declarations(code)
    guards = []
    assignment_text = list(code)

    def _blank(start: int, end: int) -> None:
        for index in range(start, end):
            if assignment_text[index] != "\n":
                assignment_text[index] = " "

    def _balanced_header(match: "re.Match[str]") -> None:
        start = match.end()
        depth = 1
        end = start
        while end < len(code) and depth:
            depth += (code[end] == "(") - (code[end] == ")")
            end += 1
        if depth == 0:
            _blank(match.start(), end)

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
            _blank(match.start(), end)
    # Parameter initializers and loop headers assign names that are
    # not signal writes: a parameter is elaboration-time, and a loop
    # index is iteration bookkeeping, not design state.
    for match in re.finditer(r"\b(?:parameter|localparam)\b[^;)]*", code):
        _blank(match.start(), match.end())
    for match in re.finditer(r"\bfor\s*\(", code):
        _balanced_header(match)
    searchable = "".join(assignment_text)
    paren_depth = [0] * (len(searchable) + 1)
    level = 0
    for index, char in enumerate(searchable):
        paren_depth[index] = level
        if char == "(":
            level += 1
        elif char == ")":
            level = max(level - 1, 0)
    writers: dict[str, list[int]] = {}
    for match in _ASSIGN.finditer(searchable):
        if match[2] == "<=" and (
            paren_depth[match.start()] > 0
            or _is_comparison(searchable, match.start())
        ):
            continue
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
        "Lexical hints only: no macro expansion, elaboration, reachability, driver resolution or proof of completeness."
    ]
    if re.search(
        r"`|\b(?:case|casex|casez|generate|task|function|typedef"
        r"|struct|union|enum|interface|package|class)\b",
        code,
    ):
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
