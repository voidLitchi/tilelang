"""Pytest suite for DeepSeek-V4.1-Flash Lightning Indexer Sparse Backward on Ascend 950."""

from __future__ import annotations

import pytest
import tilelang.testing

from examples.ascend.lightning_indexer.verify_lightning_indexer_bwd import (
    run_compile_checks,
    run_cpu_reference_checks,
    run_npu_verification,
)


def test_cpu_reference_and_gradcheck():
    run_cpu_reference_checks()


def test_ascend_compile_and_codegen_audit():
    run_compile_checks(num_cores_list=(28, 32, 36), staged_pipeline=True)


@tilelang.testing.requires_ascend
@pytest.mark.parametrize("num_cores", [28, 32])
@pytest.mark.parametrize("bk", [64, 128])
def test_npu_accuracy(num_cores, bk):
    run_npu_verification(bk=bk, num_cores=num_cores, include_large=False, bench=False)
