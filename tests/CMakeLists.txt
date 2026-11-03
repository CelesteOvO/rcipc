include(GoogleTest)

# rcipc_test(名字 依赖...)：把 tests/名字.cc 编译成一个测试程序，链接给定的库和 gtest_main
# （gtest_main 提供 main 函数，所以测试文件里只需要写 TEST）。
# gtest_discover_tests 让 ctest 认识程序里的每一个 TEST，可以单独跑、单独报告。
# 也可以直接运行测试程序，例如 build/tests/Debug/test_ipc1d.exe --gtest_filter=Ccd.*
function(rcipc_test name)
  add_executable(${name} ${name}.cc)
  target_link_libraries(${name} PRIVATE ${ARGN} GTest::gtest_main)
  rcipc_copy_dlls(${name})
  gtest_discover_tests(${name} DISCOVERY_MODE PRE_TEST)
endfunction()

rcipc_test(test_smoke rcipc)       # 阶段 0：编译和依赖都能用
rcipc_test(test_barrier rcipc)     # 阶段 1：障碍函数
rcipc_test(test_ipc1d ipc1d)       # 阶段 1：一维 IPC 的时间步
rcipc_test(test_quat rcipc)        # 阶段 2：四元数
rcipc_test(test_kinematics rcipc)  # 阶段 2：正运动学、积分、Jacobian，和 MuJoCo 对比
rcipc_test(test_inertia rcipc)     # 阶段 3：body 形式的惯性项，和 mj_fullM 对比
rcipc_test(test_step rcipc)        # 阶段 3：关节坐标里的时间步
rcipc_test(test_contact rcipc)     # 阶段 4：地面接触、kappa、落地
rcipc_test(test_ccd rcipc)         # 阶段 4：直弦 CCD 和它漏掉的反例；阶段 5：曲线 CCD
rcipc_test(test_curvature rcipc)   # 阶段 6：曲率上界
rcipc_test(test_distance rcipc)    # 阶段 7：最近点
rcipc_test(test_pairs rcipc)       # 阶段 7：刚体之间的接触、CCD、堆叠
rcipc_test(test_friction rcipc)    # 阶段 8：摩擦
rcipc_test(test_solver rcipc)      # 阶段 9：PCG、预条件器、停止规则、kappa 调整
rcipc_test(test_scenes rcipc)      # 阶段 10：关节限位、布娃娃、链条、布娃娃堆
if(TARGET rcipc_gpu)
  rcipc_test(test_gpu rcipc_gpu)    # 阶段 11：GPU 版和 CPU 版逐项对比
endif()
