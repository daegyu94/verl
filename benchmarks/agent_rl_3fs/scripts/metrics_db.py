# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Daegyu Han
"""Read 3FS counters using the existing collector credentials without saving them."""

import base64
import subprocess
import sys
import urllib.parse
import urllib.request

import tomllib

cfg = tomllib.loads(
    subprocess.check_output(
        ["docker", "exec", "3fs-monitor", "cat", "/opt/3fs/etc/monitor_collector_main.toml"], text=True
    )
)["server"]["monitor_collector"]["reporter"]["clickhouse"]
query = sys.argv[1]
req = urllib.request.Request("http://" + cfg["host"] + ":8123/?" + urllib.parse.urlencode({"query": query}))
req.add_header("Authorization", "Basic " + base64.b64encode((cfg["user"] + ":" + cfg["passwd"]).encode()).decode())
print(urllib.request.urlopen(req, timeout=30).read().decode(), end="")
