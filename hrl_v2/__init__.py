"""DyIM-CHRL: Dynamic Influence-Map Abstracted Cluster-Guided HRL (v2).

Completely isolated from the frozen ``bch/`` package and running v11 jobs.
Supports both SC2BA scenarios (e.g., 3m, 2s3z with multi-opponent pools)
and legacy HRL-IMCBS maps (e.g., MM4m, MM8) under a unified atomic-action
and evaluation interface.
"""

from .config import HRLv2Config

__all__ = ["HRLv2Config"]
