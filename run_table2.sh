#!/bin/bash
# Reproduce Table 2 (ImageNet-to-ImageNet-C, ResNet-50, severity 5, continual):
# the three baselines and their +DO-ALL variants (same config + --do_all).
#
# Override any of the variables below, e.g.:
#   SYN_ROOT=./WMDD_imagenet IPC=10 GPU=0 bash run_table2.sh
# DD anchors must live at  ${SYN_ROOT}/IPC_${IPC}  (an ImageFolder of distilled images).

set -u
cd "$(dirname "$0")"

GPU="${GPU:-0}"
IPC="${IPC:-10}"
STRIDE="${STRIDE:-1}"
SYN_ROOT="${SYN_ROOT:-./DD_anchor/imagenet_WMDD_resnet50}"
SYNPATH="${SYN_ROOT}/IPC_${IPC}"

mkdir -p run_logs

run () {  # $1 = config name, $2... = extra args
  local cfg="$1"; shift
  echo ">>> ${cfg} $*"
  CUDA_VISIBLE_DEVICES="${GPU}" python test_time.py --cfg "cfgs/imagenet_c/${cfg}.yaml" "$@" \
    2>&1 | tee "run_logs/${cfg}$( [ "$*" ] && echo _doall ).log"
}

# baselines
run eata
run rmt
run roid
# + DO-ALL  (same base config, just add the flag)
run eata --do_all --synpath "${SYNPATH}" --stride "${STRIDE}"
run rmt  --do_all --synpath "${SYNPATH}" --stride "${STRIDE}"
run roid --do_all --synpath "${SYNPATH}" --stride "${STRIDE}"

