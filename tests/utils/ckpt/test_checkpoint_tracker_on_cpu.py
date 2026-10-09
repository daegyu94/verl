# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
import errno
import os
import stat

import pytest

from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, write_checkpoint_tracker


def test_tracker_replaces_complete_value_and_preserves_permissions(tmp_path, monkeypatch):
    tracker = tmp_path / "latest_checkpointed_iteration.txt"
    tracker.write_text("1")
    tracker.chmod(0o640)
    (tmp_path / "global_step_12").mkdir()
    original_fsync, original_replace = os.fsync, os.replace
    events = []

    def fsync(fd):
        events.append("directory" if stat.S_ISDIR(os.fstat(fd).st_mode) else "file")
        original_fsync(fd)

    def replace(source, target):
        assert tracker.read_text() == "1"
        assert open(source).read() == "12"
        assert os.path.dirname(source) == str(tmp_path)
        events.append("replace")
        original_replace(source, target)

    monkeypatch.setattr(os, "fsync", fsync)
    monkeypatch.setattr(os, "replace", replace)
    write_checkpoint_tracker(str(tmp_path), 12)
    assert events == ["file", "replace", "directory"]
    assert tracker.read_text() == "12"
    assert stat.S_IMODE(tracker.stat().st_mode) == 0o640
    assert find_latest_ckpt_path(str(tmp_path)) == str(tmp_path / "global_step_12")
    assert not list(tmp_path.glob(".latest_checkpointed_iteration.*"))


def test_directory_fsync_failure_reports_error_with_a_complete_visible_tracker(tmp_path, monkeypatch):
    tracker = tmp_path / "latest_checkpointed_iteration.txt"
    tracker.write_text("1")
    original_fsync = os.fsync

    def fsync(fd):
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            raise OSError(errno.EIO, "directory sync failed")
        original_fsync(fd)

    monkeypatch.setattr(os, "fsync", fsync)
    with pytest.raises(OSError, match="directory sync"):
        write_checkpoint_tracker(str(tmp_path), 12)
    assert tracker.read_text() == "12"
    assert not list(tmp_path.glob(".latest_checkpointed_iteration.*"))
