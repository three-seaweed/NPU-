#!/bin/bash
# bf16selftest_run.sh -- wrapper for sudo -n npurun (env_reset scrubs XNPU_*)
export XNPU_BASE=0x400000000
export XNPU_RESMEM_PHYS=0x60000000
export XNPU_RESMEM_SIZE=0x10000000
exec ./bf16selftest
