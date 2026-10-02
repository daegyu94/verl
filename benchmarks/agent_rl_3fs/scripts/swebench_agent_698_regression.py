# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Daegyu Han
"""Behavioral regression for SWE-Bench train instance DataDog__integrations-core-698.

Run this only inside the no-network Docker grader. It extracts the affected
method so old Datadog dependencies are unnecessary for this small POC.
"""

import ast
import sys


def main(path: str) -> None:
    module = ast.parse(open(path, encoding="utf-8").read(), filename=path)
    network = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "Network")
    method = next(node for node in network.body if isinstance(node, ast.FunctionDef) and node.name == "_check_bsd")
    isolated = ast.Module(body=[method], type_ignores=[])
    ast.fix_missing_locations(isolated)

    class Platform:
        @staticmethod
        def is_freebsd():
            return False

    class Log:
        def __init__(self):
            self.messages = []

        def error(self, message):
            self.messages.append(message)

    class Check:
        def __init__(self):
            self.log = Log()

    namespace = {
        "Platform": Platform,
        "get_subprocess_output": lambda args, log: ("Name Mtu Network Address\n", "", 0),
        "SubprocessOutputEmptyError": type("SubprocessOutputEmptyError", (Exception,), {}),
    }
    exec(compile(isolated, path, "exec"), namespace)
    check = Check()
    assert namespace["_check_bsd"](check, {}) is False
    assert check.log.messages and "Ipkts not found" in check.log.messages[0]
    print("PASS: missing header is logged through self.log without AttributeError")


if __name__ == "__main__":
    main(sys.argv[1])
