#enable_testing()

add_executable(test_jdqz_1 test1.C)
add_executable(test_jdqz_2 test2.C)

add_test(test1 test_jdqz_1)
add_test(test2 test_jdqz_2)

# Use make check to both build tests and test
add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND}
  DEPENDS test_jdqz_1 test_jdqz_2)

target_link_libraries (test_jdqz_1 PUBLIC jdqz_tools)
target_link_libraries (test_jdqz_2 PUBLIC jdqz_tools)

# Google test
find_package(GTest)

if (NOT GTEST_FOUND)
  set(GTEST_ROOT "$ENV{SHARED_DIR}")
  unset(GTEST_INCLUDE_DIR CACHE) #FindGTEST quirk
  find_package(GTest)
endif (NOT GTEST_FOUND)

include_directories(${GTEST_INCLUDE_DIRS})
target_link_libraries (test_jdqz_1 PUBLIC ${GTEST_LIBRARIES})
target_link_libraries (test_jdqz_2 PUBLIC ${GTEST_LIBRARIES})

# Pthread
find_package(Threads)
target_link_libraries (test_jdqz_1 PUBLIC ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries (test_jdqz_2 PUBLIC ${CMAKE_THREAD_LIBS_INIT})

# Lapack
find_package(LAPACK)
target_link_libraries(test_jdqz_1 PUBLIC ${LAPACK_LIBRARIES})
target_link_libraries(test_jdqz_2 PUBLIC ${LAPACK_LIBRARIES})

set(TEST_INCL_DIRS ../jdqz)

target_include_directories(test_jdqz_1 PUBLIC ${TEST_INCL_DIRS})
target_include_directories(test_jdqz_2 PUBLIC ${TEST_INCL_DIRS})

install(TARGETS test_jdqz_1 test_jdqz_2 DESTINATION test)