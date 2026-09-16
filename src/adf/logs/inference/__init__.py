# SPDX-FileCopyrightText: 2026 Deutsche Telekom AG
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any


def __getattr__(name: str) -> Any:
    if name == "DSAInferencePipeline":
        from adf.logs.inference.inference_module import DSAInferencePipeline

        return DSAInferencePipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
