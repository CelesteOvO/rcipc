#!/bin/sh
# 第二批 Rigid-IPC 场景（笔记 notes/compare_rigid_ipc.md 第 5 节），依次跑（不并行，免得互相抢 CPU 影响计时）。在仓库根目录运行。
# 用法：sh compare/rigid_ipc/run_more.sh [theirs|ours|wrecking]
#   theirs：Rigid-IPC 跑五个小场景（各自的默认容差 -> sim.json；钟摆、螺栓另加绝对容差 1e-6 -> sim_abs.json）
#   ours：我们 GPU 跑同样五个（参数照各场景 JSON：T、h、dhat、mu 见每个 xml 开头的注释）
#   wrecking：落锤球，两边各一遍（最慢，放最后）
#   relax：我们六个场景再跑一遍 relax=0（kappa 只翻倍不减半，笔记 4.3 节）
#   tol：我们八个场景（含第一批两个）relax=0 加上 Rigid-IPC 实际用的绝对容差（它的 run.log 里 tol=，0.01 x 包围盒对角线）
set -e
R=${RIGID_IPC:-D:/Project/rigid-ipc}
F=$R/fixtures/3D
P="python -I"
ABS='{"newton_solver": {"velocity_conv_tol": 1e-6, "is_velocity_conv_tol_abs": true}}'
res=compare/rigid_ipc/results
what=${1:-theirs}

if [ "$what" = theirs ]; then
  $P compare/rigid_ipc/run_rigid_ipc.py $F/piles/cone.json --out $res/cone_pile
  $P compare/rigid_ipc/run_rigid_ipc.py $F/pendulum/pendulum.json --out $res/pendulum
  $P compare/rigid_ipc/run_rigid_ipc.py $F/pendulum/pendulum.json --out $res/pendulum --name sim_abs.json --patch "$ABS"
  $P compare/rigid_ipc/run_rigid_ipc.py $F/mechanisms/bolt.json --out $res/bolt
  $P compare/rigid_ipc/run_rigid_ipc.py $F/mechanisms/bolt.json --out $res/bolt --name sim_abs.json --patch "$ABS"
  $P compare/rigid_ipc/run_rigid_ipc.py $F/friction/card-house/card-house-2-levels.json --out $res/card_house_2
  $P compare/rigid_ipc/run_rigid_ipc.py $F/blender/blender-25.json --out $res/blender_25
fi

if [ "$what" = ours ]; then
  $P compare/rigid_ipc/measure_rcipc.py compare/rigid_ipc/cone_pile.xml --out $res/cone_pile --T 4 --h 0.01 --dhat 0.001 --mu 0 --gpu 1
  $P compare/rigid_ipc/measure_rcipc.py compare/rigid_ipc/pendulum.xml --out $res/pendulum --T 10 --h 0.01 --dhat 0.0001 --mu 0 --gpu 1
  $P compare/rigid_ipc/measure_rcipc.py compare/rigid_ipc/bolt.xml --out $res/bolt --T 5 --h 0.01 --dhat 0.0001 --mu 0 --gpu 1
  $P compare/rigid_ipc/measure_rcipc.py compare/rigid_ipc/card_house_2.xml --out $res/card_house_2 --T 2 --h 0.01 --dhat 0.001 --mu 0.5 --gpu 1
  $P compare/rigid_ipc/measure_rcipc.py compare/rigid_ipc/blender_25.xml --out $res/blender_25 --T 5 --h 0.005 --dhat 0.001 --mu 0 --gpu 1
fi

if [ "$what" = wrecking ]; then
  $P compare/rigid_ipc/measure_rcipc.py compare/rigid_ipc/wrecking_ball.xml --out $res/wrecking_ball --T 10 --h 0.01 --dhat 0.001 --mu 0 --gpu 1
  $P compare/rigid_ipc/run_rigid_ipc.py $F/chain/wrecking-ball.json --out $res/wrecking_ball
fi

if [ "$what" = relax ]; then
  for sc in "cone_pile 4 0.01 0.001 0" "pendulum 10 0.01 0.0001 0" "bolt 5 0.01 0.0001 0" "card_house_2 2 0.01 0.001 0.5"             "blender_25 5 0.005 0.001 0" "wrecking_ball 10 0.01 0.001 0"; do
    set -- $sc
    $P compare/rigid_ipc/measure_rcipc.py compare/rigid_ipc/$1.xml --out $res/$1 --T $2 --h $3 --dhat $4 --mu $5 --gpu 1 --tag gpu_relax_0 --extra relax=0
  done
fi

if [ "$what" = tol ]; then
  for sc in "cone_pile 4 0.01 0.001 0 0.28" "pendulum 10 0.01 0.0001 0 0.0051" "bolt 5 0.01 0.0001 0 0.0019" "card_house_2 2 0.01 0.001 0.5 0.015"             "blender_25 5 0.005 0.001 0 0.065" "wrecking_ball 10 0.01 0.001 0 0.37" "gears_single 10 0.01 0.001 0 0.048" "chain_10 10 0.01 0.001 0 0.061"; do
    set -- $sc
    $P compare/rigid_ipc/measure_rcipc.py compare/rigid_ipc/$1.xml --out $res/$1 --T $2 --h $3 --dhat $4 --mu $5 --gpu 1 --tag gpu_relax_0_toltheirs --extra relax=0 tol=$6
  done
fi
