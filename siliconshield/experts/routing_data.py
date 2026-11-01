"""Measured routing evidence from the public RTL benchmark (Qt-free).

Tiers derive from benchmark accuracy with a one-tier demotion when
the false-safe rate on violations exceeds 15%: a model that misses
many violations is never strong, however high its accuracy. Counts
come from the public benchmark revision below, so this table is
public evidence, not private data. Re-measure (never hand-edit)
when the benchmark or a checkpoint changes.
"""

BENCHMARK_SOURCE = (
    "hf://danieltombay/siliconshield-rtl-benchmark"
    "@ab2b7dbee3158a292edc4f7a0127379aab996636:data/benchmark.jsonl"
)

MODEL_WEIGHTS = {
    "contract-1": (
        "f4130223ed19be5cc7e21b3da70cb7aebf944ad6ed4a8a80f11d8f59bcd05ee7"
    ),
    "contract-2": (
        "f916444ff45cfc67bc5a67c81a36a770a20709fcd7a85fbfc0258fb411144ed8"
    ),
}
"""Weights each reliability row was measured on; other weights unrate."""

# Accuracy bands: strong >= 0.80, partial >= 0.60, weak >= 0.40,
# none below; demote one tier when false-safes/violations > 0.15.
AREA_RELIABILITY: dict[tuple[str, str], dict[str, object]] = {
    ("contract-1", "arbitration"): {
        "tier": "partial",
        "correct": 14,
        "total": 18,
        "false_safe": 0,
    },
    ("contract-1", "counter"): {
        "tier": "none",
        "correct": 41,
        "total": 108,
        "false_safe": 1,
    },
    ("contract-1", "fifo"): {
        "tier": "none",
        "correct": 36,
        "total": 108,
        "false_safe": 0,
    },
    ("contract-1", "guard-semantics"): {
        "tier": "none",
        "correct": 171,
        "total": 432,
        "false_safe": 6,
    },
    ("contract-1", "input-filter"): {
        "tier": "weak",
        "correct": 91,
        "total": 162,
        "false_safe": 3,
    },
    ("contract-1", "interrupt-wake"): {
        "tier": "partial",
        "correct": 158,
        "total": 234,
        "false_safe": 0,
    },
    ("contract-1", "local-safety-cwe"): {
        "tier": "strong",
        "correct": 2601,
        "total": 2610,
        "false_safe": 7,
    },
    ("contract-1", "ot-access-control"): {
        "tier": "strong",
        "correct": 108,
        "total": 108,
        "false_safe": 0,
    },
    ("contract-1", "requirement-clarity"): {
        "tier": "weak",
        "correct": 346,
        "total": 756,
        "false_safe": 22,
    },
    ("contract-1", "transfer-pairs"): {
        "tier": "strong",
        "correct": 606,
        "total": 621,
        "false_safe": 6,
    },
    ("contract-1", "write-paths"): {
        "tier": "weak",
        "correct": 179,
        "total": 432,
        "false_safe": 12,
    },
    ("contract-2", "arbitration"): {
        "tier": "weak",
        "correct": 14,
        "total": 18,
        "false_safe": 1,
    },
    ("contract-2", "counter"): {
        "tier": "partial",
        "correct": 92,
        "total": 108,
        "false_safe": 9,
    },
    ("contract-2", "fifo"): {
        "tier": "partial",
        "correct": 97,
        "total": 108,
        "false_safe": 7,
    },
    ("contract-2", "guard-semantics"): {
        "tier": "weak",
        "correct": 268,
        "total": 432,
        "false_safe": 27,
    },
    ("contract-2", "input-filter"): {
        "tier": "weak",
        "correct": 94,
        "total": 162,
        "false_safe": 2,
    },
    ("contract-2", "interrupt-wake"): {
        "tier": "weak",
        "correct": 145,
        "total": 234,
        "false_safe": 12,
    },
    ("contract-2", "local-safety-cwe"): {
        "tier": "none",
        "correct": 933,
        "total": 2610,
        "false_safe": 281,
    },
    ("contract-2", "ot-access-control"): {
        "tier": "weak",
        "correct": 77,
        "total": 108,
        "false_safe": 31,
    },
    ("contract-2", "requirement-clarity"): {
        "tier": "none",
        "correct": 271,
        "total": 756,
        "false_safe": 32,
    },
    ("contract-2", "transfer-pairs"): {
        "tier": "weak",
        "correct": 275,
        "total": 621,
        "false_safe": 31,
    },
    ("contract-2", "write-paths"): {
        "tier": "weak",
        "correct": 244,
        "total": 432,
        "false_safe": 17,
    },
}

MECHANISM_TO_AREA = {
    "arbiter-grant-handshake": "arbitration",
    "arbiter-tree-acceptance": "arbitration",
    "clear-before-load:w3:reset0:style1": "counter",
    "commit-hold:w3:reset0:style1": "counter",
    "commit-hold:w5:reset1:style1": "counter",
    "decrement-by-sampled-step:w3:reset1:style0": "counter",
    "increment-by-sampled-step:w8:reset1:style1": "counter",
    "overflow-saturates:w8:reset1:style0": "counter",
    "underflow-saturates:w3:reset0:style0": "counter",
    "underflow-saturates:w3:reset0:style1": "counter",
    "underflow-saturates:w5:reset0:style1": "counter",
    "underflow-saturates:w5:reset1:style1": "counter",
    "accepted-word-capture:pass0": "fifo",
    "backpressure-keeps-occupancy:pass0": "fifo",
    "bypass-capture-under-stall:pass1": "fifo",
    "clear-drops-occupancy:pass1": "fifo",
    "empty-data-blanking:pass0": "fifo",
    "empty-data-blanking:pass1": "fifo",
    "stored-word-hold:pass0": "fifo",
    "stored-word-hold:pass1": "fifo",
    "authorization-policy0": "guard-semantics",
    "authorization-policy1": "guard-semantics",
    "authorization-policy2": "guard-semantics",
    "authorization-policy3": "guard-semantics",
    "authorization-policy4": "guard-semantics",
    "authorization-policy5": "guard-semantics",
    "filter-disabled-bypass": "input-filter",
    "filter-no-premature-update": "input-filter",
    "filter-restart-on-change": "input-filter",
    "filter-saturates": "input-filter",
    "timed-low-breaks-streak": "input-filter",
    "timed-threshold-rearms": "input-filter",
    "adc-interrupt-event-binding": "interrupt-wake",
    "interrupt-edge-not-level": "interrupt-wake",
    "interrupt-enable-mask": "interrupt-wake",
    "interrupt-no-overlap": "interrupt-wake",
    "interrupt-pending-not-prematurely-complete": "interrupt-wake",
    "interrupt-sticky-events": "interrupt-wake",
    "interrupt-sticky-pending": "interrupt-wake",
    "wake-edge-rearm": "interrupt-wake",
    "wake-enable-gating": "interrupt-wake",
    "i2c-target-address-mask": "ot-access-control",
    "address": "requirement-clarity",
    "lock": "requirement-clarity",
    "lock_privilege": "requirement-clarity",
    "permission": "requirement-clarity",
    # "privilege" is ambiguous: it names mechanisms in both
    # requirement-clarity and transfer-pairs, so it never resolves
    # to an area. Scope those rows by area name instead.
    "role": "requirement-clarity",
    "address-window": "transfer-pairs",
    "alternative-authority": "transfer-pairs",
    "burst-overflow": "transfer-pairs",
    "captured-authorization": "transfer-pairs",
    "complete-clear": "transfer-pairs",
    "default-deny": "transfer-pairs",
    "lane-strobe": "transfer-pairs",
    "locked-multiport": "transfer-pairs",
    "owner-width": "transfer-pairs",
    "sticky-state": "transfer-pairs",
    "wipe-priority": "transfer-pairs",
    "implementation-policy0": "write-paths",
    "implementation-policy1": "write-paths",
    "implementation-policy2": "write-paths",
    "policy-policy0": "write-paths",
    "policy-policy1": "write-paths",
    "policy-policy2": "write-paths",
}

_CWE_MECHANISMS = (
    "CWE-1189:hold",
    "CWE-1189:update",
    "CWE-1191:hold",
    "CWE-1191:update",
    "CWE-1220:hold",
    "CWE-1220:update",
    "CWE-1222:hold",
    "CWE-1222:update",
    "CWE-1223:hold",
    "CWE-1223:update",
    "CWE-1224:hold",
    "CWE-1224:update",
    "CWE-1231:hold",
    "CWE-1231:update",
    "CWE-1232:hold",
    "CWE-1232:update",
    "CWE-1233:hold",
    "CWE-1233:update",
    "CWE-1243:update",
    "CWE-1251:update",
    "CWE-1254:hold",
    "CWE-1254:update",
    "CWE-1256:hold",
    "CWE-1256:update",
    "CWE-1257:update",
    "CWE-1260:update",
    "CWE-1262:hold",
    "CWE-1262:update",
    "CWE-1264:update",
    "CWE-1267:update",
    "CWE-1268:hold",
    "CWE-1268:update",
    "CWE-1271:hold",
    "CWE-1271:update",
    "CWE-1272:update",
    "CWE-1274:update",
    "CWE-1280:hold",
    "CWE-1280:update",
    "CWE-1282:hold",
    "CWE-1282:update",
    "CWE-1294:hold",
    "CWE-1294:update",
    "CWE-1299:hold",
    "CWE-1299:update",
    "CWE-1302:hold",
    "CWE-1302:update",
    "CWE-1312:update",
    "CWE-1314:hold",
    "CWE-1314:update",
    "CWE-1317:update",
    "CWE-1318:update",
    "CWE-276:hold",
    "CWE-276:update",
)

for _mech in _CWE_MECHANISMS:
    MECHANISM_TO_AREA[_mech] = "local-safety-cwe"


def area_for_tag(tag: str) -> str | None:
    """Resolve a scope tag to a benchmark area, if known."""
    if tag in {
        "arbitration",
        "counter",
        "fifo",
        "guard-semantics",
        "input-filter",
        "interrupt-wake",
        "local-safety-cwe",
        "ot-access-control",
        "requirement-clarity",
        "transfer-pairs",
        "write-paths",
    }:
        return tag
    return MECHANISM_TO_AREA.get(tag)
