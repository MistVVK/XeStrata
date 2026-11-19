#!/bin/bash
# check_mac.sh (owner PC): syncs the .metal file to m1002 and runs the compile check under the GPU lock
export MSYS_NO_PATHCONV=1
M=/c/Users/AI-Server/Desktop/Strata/GitHub/wt-hy4-metal/engine/src/hy4flash/metal
scp -q $M/hy4.metal m1002:strata-mac/hy4/wt/engine/src/hy4flash/metal/hy4.metal 2>&1 | tr -d '\0'
ssh m1002 '~/strata-mac/locks/run_locked.sh hy4-metal-check ~/strata-mac/hy4/bin/hy4_compile_check ~/strata-mac/hy4/wt/engine/src/hy4flash/metal/hy4.metal 2>&1 | grep -v keep-alive | head -${1:-60}' 2>&1 | tr -d '\0'
