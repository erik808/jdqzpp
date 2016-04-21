cmake_minimum_required(VERSION 2.8.9)
project (jdqz)

enable_language(Fortran)

# FFLAGS depend on the compiler
get_filename_component(Fortran_COMPILER_NAME ${CMAKE_Fortran_COMPILER} NAME)

if (Fortran_COMPILER_NAME MATCHES "gfortran.*") #gfortran
  set (CMAKE_Fortran_FLAGS "-g -u -O0")
elseif (Fortran_COMPILER_NAME MATCHES "ifort.*") #ifort
  set (CMAKE_Fortran_FLAGS "")
endif (Fortran_COMPILER_NAME MATCHES "gfortran.*")

add_subdirectory(jdqz_tools)

enable_testing()
add_executable(test_jdqz test.C)
add_test(test1 test_jdqz)

# Use make check to both build tests and test
add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND}
                  DEPENDS test_jdqz)

set(CMAKE_CXX_FLAGS "-g -Wall -std=c++11 -O0")

target_link_libraries (test_jdqz PUBLIC jdqz_tools)

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
target_link_libraries (test_jdqz PUBLIC ${GTEST_LIBRARIES})

# Pthread
find_package(Threads)
target_link_libraries (test_jdqz PUBLIC ${CMAKE_THREAD_LIBS_INIT})

# Lapack
find_package(LAPACK)
target_link_libraries(test_jdqz PUBLIC ${LAPACK_LIBRARIES})