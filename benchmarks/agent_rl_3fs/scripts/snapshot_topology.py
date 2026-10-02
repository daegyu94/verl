# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Daegyu Han
"""Record service identity and storage mounts without credential-bearing env."""

import json
import os
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[1]
services = [
    "3fs-client",
    "3fs-storage",
    "3fs-meta",
    "3fs-mgmtd",
    "3fs-monitor",
    "3fs-clickhouse",
    "3fs-fdb",
    "3fs-postgres",
]
raw = json.loads(subprocess.check_output(["docker", "inspect", *services], text=True))
selected = [
    {
        "name": x["Name"],
        "image": x["Config"]["Image"],
        "image_id": x["Image"],
        "pid": x["State"]["Pid"],
        "status": x["State"]["Status"],
        "started_at": x["State"]["StartedAt"],
        "mounts": [{k: m[k] for k in ["Source", "Destination", "Type", "RW"]} for m in x["Mounts"]],
    }
    for x in raw
]
backing = next(
    m["Source"]
    for x in selected
    if x["name"] == "/3fs-storage"
    for m in x["mounts"]
    if m["Destination"] == "/mnt/3fsdata"
)
fs = json.loads(subprocess.check_output(["findmnt", "-J", "-T", backing, "-o", "TARGET,SOURCE,FSTYPE"], text=True))
config = Path(
    os.environ.get("THREEFS_CLUSTER_CONFIG", str(Path.home() / "workspace/dfs-evaluation/3fs/poc-config.yml"))
)
rf = [line.strip() for line in config.read_text().splitlines() if "replicationFactor:" in line]
(root / "raw/topology-final.json").write_text(
    json.dumps({"services": selected, "storage_backing": fs, "replication_factor_config": rf}, indent=2) + "\n"
)
