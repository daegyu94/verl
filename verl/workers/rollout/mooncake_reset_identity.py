# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
import json


def mooncake_reset_key(engine_kwargs):
    transfer = engine_kwargs.get("kv_transfer_config")
    if isinstance(transfer, str):
        transfer = json.loads(transfer)
    if not transfer or transfer.get("kv_connector") != "MooncakeStoreConnector":
        return None
    from vllm.distributed.mooncake_store import MooncakeStoreConfig

    config = MooncakeStoreConfig.load_from_config()
    prefix = transfer.get("kv_connector_extra_config", {}).get("cache_prefix")
    if not config.master_server_address or not isinstance(prefix, str) or not prefix or "@" in prefix:
        return None
    return config.master_server_address, config.tenant_id, prefix
