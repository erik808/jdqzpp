#enable_testing()

add_executable(test_jdqz test.C)
add_test(test1 test_jdqz)

# Use make check to both build tests and test
add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND}
  DEPENDS test_jdqz)

target_link_libraries (test_jdqz PUBLIC jdqz_tools)

# Google test
find_package(GTest)

if (NOT GTEST_FOUND)
  set(GTEST_ROOT "$ENV{SHARED_DIR}")
  unset(GTEST_INCLUDE_DIR CACHE) #FindGTEST quirk
  find_package(GTest)
endif (NOT GTEST_FOUND)

include_directories(${GTEST_INCLUDE_DIRS})
target_link_libraries (test_jdqz PUBLIC ${GTEST_LIBRARIES})

# Pthread
find_package(Threads)
target_link_libraries (test_jdqz PUBLIC ${CMAKE_THREAD_LIBS_INIT})

# Lapack
find_package(LAPACK)
target_link_libraries(test_jdqz PUBLIC ${LAPACK_LIBRARIES})
