#!/bin/sh
# 两个 Rigid-IPC 场景、两边、几种设置，依次跑（不并行，免得互相抢 CPU 影响计时）。在仓库根目录运行。
# 用法：sh compare/rigid_ipc/run_all.sh [gears|chain|all]
set -e
R=${RIGID_IPC:-D:/Project/rigid-ipc}
P="python -I"
ABS='{"newton_solver": {"velocity_conv_tol": 1e-6, "is_velocity_conv_tol_abs": true}}'
what=${1:-all}

if [ "$what" = gears ] || [ "$what" = all ]; then
  out=compare/rigid_ipc/results/gears_single
  $P compare/rigid_ipc/run_rigid_ipc.py $R/fixtures/3D/mechanisms/gears/single-kinematic.json --out $out
  $P compare/rigid_ipc/run_rigid_ipc.py $R/fixtures/3D/mechanisms/gears/single-kinematic.json --out $out --name sim_abs.json --patch "$ABS"
  $P compare/rigid_ipc/measure_rcipc.py compare/rigid_ipc/gears_single.xml --out $out --T 10 --h 0.01 --dhat 0.001 --mu 0 --gpu 1
  $P compare/rigid_ipc/measure_rcipc.py compare/rigid_ipc/gears_single.xml --out $out --T 10 --h 0.01 --dhat 0.001 --mu 0 --gpu 0
fi

if [ "$what" = chain ] || [ "$what" = all ]; then
  out=compare/rigid_ipc/results/chain_10
  $P compare/rigid_ipc/run_rigid_ipc.py $R/fixtures/3D/chain/10-links.json --out $out
  $P compare/rigid_ipc/run_rigid_ipc.py $R/fixtures/3D/chain/10-links.json --out $out --name sim_abs.json --patch "$ABS"
  $P compare/rigid_ipc/measure_rcipc.py compare/rigid_ipc/chain_10.xml --out $out --T 10 --h 0.01 --dhat 0.001 --mu 0 --gpu 1
  $P compare/rigid_ipc/measure_rcipc.py compare/rigid_ipc/chain_10.xml --out $out --T 10 --h 0.01 --dhat 0.001 --mu 0 --gpu 0
fi
