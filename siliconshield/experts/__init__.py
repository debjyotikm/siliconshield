"""Multi-expert routing and assessment (CLI/Muse-CLI territory).

This package owns capability profiles, expert routing, response
envelopes, the shared inference service, structural inspection,
arbitration, and traceable findings. It is Qt-free and imports
nothing from the GUI demo: the desktop window is one consumer
among others (CLI, Muse), never the owner. Persistent model
execution stays in worker subprocesses reached over local pipes.
"""

from siliconshield.experts import (
    arbitration,
    capabilities,
    contracts,
    findings,
    inspection,
    registry,
    responses,
    router,
    service,
    worker,
)

__all__ = [
    "arbitration",
    "capabilities",
    "contracts",
    "findings",
    "inspection",
    "registry",
    "responses",
    "router",
    "service",
    "worker",
]
