cmake_minimum_required(VERSION 2.8.9)
project (jdqz)

enable_testing()

add_executable(test_jdqz test.C)

add_test(test1 test_jdqz)

# Use make check to both build tests and test
add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND}
                  DEPENDS test_jdqz)

set(CMAKE_CXX_FLAGS "-Wall -std=c++11")
set(CMAKE_CXX_FLAGS_DEBUG "-g")

#CCACHE
find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
  set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
  set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
endif(CCACHE_FOUND)
  
# Google test
find_package(GTest)

if (NOT GTEST_FOUND)
  set(GTEST_ROOT "$ENV{SHARED_DIR}")
  unset(GTEST_INCLUDE_DIR CACHE) #FindGTEST quirk
  find_package(GTest)
endif (NOT GTEST_FOUND)

include_directories(${GTEST_INCLUDE_DIRS})
target_link_libraries (test_jdqz ${GTEST_LIBRARIES})

# Pthread
find_package(Threads)
target_link_libraries (test_jdqz ${CMAKE_THREAD_LIBS_INIT})