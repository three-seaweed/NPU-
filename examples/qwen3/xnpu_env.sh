#!/bin/sh
# xnpu_env.sh -- generic npurun wrapper: re-export XNPU_* (sudo env_reset
# scrubs them), then exec the given command.  usage: ./xnpu_env.sh ./prog ...
export XNPU_BASE=0x400000000
export XNPU_RESMEM_PHYS=0x60000000
export XNPU_RESMEM_SIZE=0x10000000
exec "$@"
