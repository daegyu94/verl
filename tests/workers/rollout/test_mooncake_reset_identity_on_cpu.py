# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
import json

import pytest

from verl.workers.rollout.mooncake_reset_identity import mooncake_reset_key


@pytest.mark.parametrize("as_json", [False, True])
def test_reset_key_tracks_resolved_master_tenant_and_prefix(monkeypatch, tmp_path, as_json):
    path = tmp_path / "store.json"
    path.write_text(json.dumps({"master_server_address": "master-a:50051", "tenant_id": "tenant-a"}))
    monkeypatch.setenv("MOONCAKE_CONFIG_PATH", str(path))
    transfer = {"kv_connector": "MooncakeStoreConnector", "kv_connector_extra_config": {"cache_prefix": "policy"}}
    kwargs = {"kv_transfer_config": json.dumps(transfer) if as_json else transfer}
    assert mooncake_reset_key(kwargs) == ("master-a:50051", "tenant-a", "policy")
    path.write_text(json.dumps({"master_server_address": "master-b:50051", "tenant_id": "tenant-b"}))
    assert mooncake_reset_key(kwargs) == ("master-b:50051", "tenant-b", "policy")
    assert mooncake_reset_key({"kv_transfer_config": {"kv_connector": "MultiConnector"}}) is None
