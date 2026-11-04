"""Benchmark areas and mechanism tags for routing (Qt-free).

Tiers derive from per-profile routing evidence at route time; this
module only maps mechanism tags to their benchmark areas. Extend
the mapping when the benchmark gains mechanisms.
"""

BENCHMARK_AREAS = frozenset(
    {
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
    }
)
"""Benchmark areas scopes and routing evidence may name."""

# Accuracy bands: strong >= 0.80, partial >= 0.60, weak >= 0.40,
# none below; demote one tier when false-safes/violations > 0.15.
# Tiers now derive from per-profile routing evidence at route time.
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
    tag = tag.strip()
    if tag in BENCHMARK_AREAS:
        return tag
    return MECHANISM_TO_AREA.get(tag)
