#!/bin/sh
# osws_run.sh -- board-side one-shot OS/WS bench for the webui bridge.
# usage (from host, via ssh exec): ./ossws_run.sh os|ws|wskeep|osback|all
# Self-contained: claim -> npurun wsbench <mode> -> release. The pair must
# stay here: server.js execs this as one command and keeps no lease state.
# env goes through wsbench_run.sh (plain sh exports, then exec) -- a
# `VAR=x sudo -n npurun` prefix would be stripped by sudo env_reset.
cd /home/liangyu/deploy_qwen3 || exit 2
sudo -n plctl claim > /dev/null 2>&1 || { echo "OSWS_CLAIM_FAIL"; exit 2; }
sudo -n npurun ./wsbench_run.sh "$1"
rc=$?
sudo -n plctl release
echo "OSWS_RC=$rc"
exit $rc
