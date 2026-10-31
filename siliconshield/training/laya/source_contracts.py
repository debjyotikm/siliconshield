"""Extract candidate transition contracts from small source RTL modules.

Candidates are explicit local behavioral contracts, not certifications of the
source project. A solver must establish each label before a row is admitted.
Pyverilog is an optional data-preparation dependency, not a runtime dependency.
"""

import copy
import hashlib
import re
import tempfile


def candidates(code, source, max_conditions=4):
    from pyverilog.ast_code_generator.codegen import ASTCodeGenerator
    from pyverilog.vparser import ast as ast
    from pyverilog.vparser.parser import VerilogParser

    from siliconshield.training.laya.build_security_data import (
        clean,
        hold_check,
        requirement_for,
    )

    code = clean(code)
    if re.search(r"\b(initial|task|function|generate)\b", code):
        return [], "unsupported_initial_task_function_or_generate"
    with tempfile.TemporaryDirectory(prefix="rtl-parser-") as directory:
        tree = VerilogParser(outputdir=directory, debug=False).parse(code)
    if len(tree.description.definitions) != 1:
        return [], "requires_one_complete_module"
    module = tree.description.definitions[0]
    if module.name != "topmodule":
        return [], "requires_topmodule"
    cg = ASTCodeGenerator()
    reset = (
        "rst_n"
        if re.search(r"\brst_n\b", code)
        else ("!rst" if re.search(r"\brst\b", code) else "")
    )
    if not reset or not re.search(r"\bposedge\s+clk\b", code):
        return [], "requires_supported_clock_and_reset"
    reset_name = reset.lstrip("!")
    edges = re.findall(r"\b(posedge|negedge)\s+(\w+)", code)
    if any(
        not ((edge == "posedge" and signal == "clk") or signal == reset_name)
        for edge, signal in edges
    ):
        return [], "unsupported_additional_clock_or_async_reset"
    conditions = []

    def expr(node):
        return cg.visit(node)

    def reset_truth(node):
        if isinstance(node, ast.Identifier) and node.name == reset_name:
            return reset.startswith("!")
        if (
            isinstance(node, (ast.Unot, ast.Ulnot))
            and isinstance(node.right, ast.Identifier)
            and node.right.name == reset_name
        ):
            return not reset.startswith("!")
        return None

    def writes():
        found = []

        def walk(node, terms, parents):
            if node is None:
                return
            if isinstance(node, ast.IfStatement):
                active = reset_truth(node.cond)
                if active is not None:
                    walk(
                        node.false_statement if active else node.true_statement,
                        terms,
                        parents,
                    )
                    return
                condition = expr(node.cond)
                if reset_name in condition:
                    return
                walk(
                    node.true_statement,
                    terms + ["(" + condition + ")"],
                    parents + [node],
                )
                walk(
                    node.false_statement,
                    terms + ["!(" + condition + ")"],
                    parents + [node],
                )
            elif isinstance(
                node,
                (ast.CaseStatement, ast.CasexStatement, ast.CasezStatement),
            ):
                # Wildcard case semantics need a separate extractor.
                if type(node) is not ast.CaseStatement:
                    return
                selector = expr(node.comp)
                branches = [
                    "(" + selector + " == " + expr(value) + ")"
                    for case in node.caselist
                    if case.cond
                    for value in case.cond
                ]
                for case in node.caselist:
                    guard = (
                        " || ".join(
                            "(" + selector + " == " + expr(v) + ")"
                            for v in case.cond
                        )
                        if case.cond
                        else "!(" + " || ".join(branches) + ")"
                    )
                    walk(case.statement, terms + ["(" + guard + ")"], parents)
            elif isinstance(node, ast.NonblockingSubstitution) and isinstance(
                node.left.var, ast.Identifier
            ):
                if expr(node.left) != expr(node.right):
                    found.append(
                        {
                            "signal": expr(node.left),
                            "rhs": expr(node.right),
                            "condition": " && ".join(terms) or "1'b1",
                            "parents": parents,
                        }
                    )
            elif isinstance(node, ast.Block):
                for child in node.statements:
                    walk(child, terms, parents)

        for item in module.items:
            if isinstance(item, ast.Always) and any(
                s.type == "posedge" and expr(s.sig) == "clk"
                for s in item.sens_list.list
            ):
                walk(item.statement, [], [])
        return found

    original_writes = writes()
    for write in original_writes:
        for node in write["parents"]:
            if all(node is not seen for seen in conditions):
                conditions.append(node)
    conditions.sort(
        key=lambda n: (
            -sum(
                word in expr(n.cond).lower()
                for word in (
                    "lock",
                    "priv",
                    "auth",
                    "permit",
                    "protect",
                    "debug",
                    "valid",
                    "access",
                )
            ),
            n.lineno,
        )
    )
    output = []
    for node_index, node in enumerate(conditions[:max_conditions]):
        original_cond = copy.deepcopy(node.cond)
        original_expr = expr(original_cond)
        changed_expr = "!(" + original_expr + ")"
        affected = sorted(
            {
                w["signal"]
                for w in original_writes
                if any(p is node for p in w["parents"])
            }
        )
        node.cond = ast.Identifier("__CONTROL_TARGET__")
        template = clean(cg.visit(tree))
        node.cond = original_cond
        if template.count("__CONTROL_TARGET__") != 1:
            continue
        good_fill, bad_fill = (
            "(" + original_expr + ")",
            "(" + changed_expr + ")",
        )
        reference = template.replace("__CONTROL_TARGET__", good_fill)
        guard = "if(" + good_fill + ")"
        bad_guard = "if(" + bad_fill + ")"
        if reference.count(guard) != 1:
            continue
        node.cond = ast.Ulnot(copy.deepcopy(original_cond))
        changed_writes = writes()
        node.cond = original_cond
        for signal in affected[:2]:
            for kind in ("hold", "update"):
                for policy, items in (
                    ("original", original_writes),
                    ("counterfactual", changed_writes),
                ):
                    relevant = [w for w in items if w["signal"] == signal]
                    if kind == "hold":
                        condition = (
                            "!("
                            + " || ".join(
                                "(" + w["condition"] + ")" for w in relevant
                            )
                            + ")"
                        )
                        requirement = requirement_for(signal, condition, "hold")
                        assertion = hold_check(signal, condition, reset)
                    else:
                        related = [
                            w
                            for w in relevant
                            if any(p is node for p in w["parents"])
                        ]
                        if not related:
                            continue
                        write = related[0]
                        condition, value = write["condition"], write["rhs"]
                        requirement = (
                            "Except during reset, when (%s) holds immediately before a rising clk edge, "
                            "%s immediately after that edge must equal the pre-edge value of (%s)."
                        ) % (condition, signal, value)
                        assertion = (
                            "reg __past_valid = 0;\nalways @(posedge clk) begin\n"
                            "__past_valid <= 1;\n  if (__past_valid && (%s) && $past(%s) && $past(%s)) "
                            "assert(%s == $past(%s));\nend"
                        ) % (
                            reset_name
                            if not reset.startswith("!")
                            else "!" + reset_name,
                            reset_name
                            if not reset.startswith("!")
                            else "!" + reset_name,
                            condition,
                            signal,
                            value,
                        )
                    reverse = policy == "counterfactual"
                    identity = "%s:%d:%s:%s:%s" % (
                        source["case"],
                        node_index,
                        signal,
                        kind,
                        policy,
                    )
                    output.append(
                        {
                            "uid": hashlib.sha256(
                                identity.encode()
                            ).hexdigest()[:24],
                            "base_uid": source["case"],
                            "family": source["family"],
                            "topic": source["topic"],
                            "code": reference.replace(guard, bad_guard)
                            if reverse
                            else reference,
                            "guard": bad_guard if reverse else guard,
                            "bad_guard": guard if reverse else bad_guard,
                            "occurrences": (0,),
                            "requirement": requirement,
                            "assertion": assertion,
                            "reset": reset,
                            "kind": kind,
                            "split": "unassigned",
                            "source": source,
                            "hole_width": 1,
                            "good_fill": bad_fill if reverse else good_fill,
                            "bad_fill": good_fill if reverse else bad_fill,
                            "opaque_guard": "if(%s)",
                            "mutation": "condition_negation",
                            "policy": policy,
                            "property_origin": "source_transition_contract",
                            "not_a_full_source_security_certification": True,
                        }
                    )
    return output, None
