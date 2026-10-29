include(GoogleTest)

function(rcipc_test name)
  add_executable(${name} ${name}.cc)
  target_link_libraries(${name} PRIVATE ${ARGN} GTest::gtest_main)
  gtest_discover_tests(${name} DISCOVERY_MODE PRE_TEST)
endfunction()

rcipc_test(test_smoke rcipc)
rcipc_test(test_barrier rcipc)
rcipc_test(test_ipc1d ipc1d)
