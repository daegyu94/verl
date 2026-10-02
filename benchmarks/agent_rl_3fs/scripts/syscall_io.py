# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Daegyu Han
"""Optional read-only BPF syscall-byte accounting for the existing storage PID."""

import subprocess
from pathlib import Path


class StorageIO:
    def __init__(self, output):
        self.output = Path(output)
        self.closed = False
        self.pid = int(
            subprocess.check_output(["docker", "inspect", "3fs-storage", "--format", "{{.State.Pid}}"], text=True)
        )
        assert self.pid > 1
        program = f"""
tracepoint:syscalls:sys_enter_pwrite64 /pid == {self.pid}/ {{ @wfd[tid]=args.fd; }}
tracepoint:syscalls:sys_exit_pwrite64 /pid == {self.pid}/ {{
 if(args.ret>0){{ @pwrite_bytes[@wfd[tid]]=sum(args.ret); @pwrite_calls[@wfd[tid]]=count(); }}
 delete(@wfd[tid]);
}}
tracepoint:syscalls:sys_enter_pread64 /pid == {self.pid}/ {{ @rfd[tid]=args.fd; }}
tracepoint:syscalls:sys_exit_pread64 /pid == {self.pid}/ {{
 if(args.ret>0){{ @pread_bytes[@rfd[tid]]=sum(args.ret); @pread_calls[@rfd[tid]]=count(); }}
 delete(@rfd[tid]);
}}
"""
        (self.output / "storage-io.bt").write_text(program)
        self.console = (self.output / "bpf-console.txt").open("w")
        self.process = subprocess.Popen(
            [
                "sudo",
                "-n",
                "bpftrace",
                "-B",
                "line",
                "-f",
                "json",
                "-o",
                str(self.output / "syscall-io.jsonl"),
                str(self.output / "storage-io.bt"),
            ],
            stdout=self.console,
            stderr=subprocess.STDOUT,
        )
        for _ in range(100):
            if self.process.poll() is not None:
                self.console.close()
                raise RuntimeError("BPF probe failed; inspect bpf-console.txt")
            path = self.output / "syscall-io.jsonl"
            if path.exists() and "attached_probes" in path.read_text():
                break
            try:
                self.process.wait(timeout=0.1)
            except subprocess.TimeoutExpired:
                pass
        else:
            self.close()
            raise RuntimeError("BPF probe readiness timeout")
        self.snapshot("before")

    def snapshot(self, name):
        # Record FD paths, not data or environment. sudo is required for this root-owned PID.
        data = subprocess.check_output(
            [
                "sudo",
                "-n",
                "python3",
                "-c",
                'import os,json,sys; p="/proc/"+sys.argv[1]+"/fd"; d={};\n'
                "for f in os.listdir(p):\n"
                ' try:d[f]=os.readlink(p+"/"+f)\n'
                " except FileNotFoundError:pass\n"
                "print(json.dumps(d))",
                str(self.pid),
            ],
            text=True,
        )
        (self.output / f"storage-fds-{name}.json").write_text(data)

    def close(self):
        if self.closed:
            return
        if self.process.poll() is None:
            subprocess.run(["sudo", "-n", "kill", "-INT", str(self.process.pid)], check=True)
            self.process.wait(timeout=15)
        self.snapshot("after")
        self.console.close()
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
