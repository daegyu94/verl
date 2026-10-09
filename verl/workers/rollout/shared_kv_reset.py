# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
import json


def uses_shared_mooncake_store(engine_kwargs):
    transfer = engine_kwargs.get("kv_transfer_config")
    if isinstance(transfer, str):
        transfer = json.loads(transfer)

    def contains_store(config):
        if not config:
            return False
        if config.get("kv_connector") == "MooncakeStoreConnector":
            return True
        if config.get("kv_connector") == "MultiConnector":
            return any(contains_store(c) for c in config.get("kv_connector_extra_config", {}).get("connectors", []))
        return False

    return contains_store(transfer)
