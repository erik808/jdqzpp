cmake_minimum_required(VERSION 2.8.9)
project (jdqz)

enable_language(Fortran)

enable_testing()

# FFLAGS depend on the compiler
get_filename_component(Fortran_COMPILER_NAME ${CMAKE_Fortran_COMPILER} NAME)

if (Fortran_COMPILER_NAME MATCHES "gfortran.*") #gfortran
  set (CMAKE_Fortran_FLAGS "-g -u -O0")
elseif (Fortran_COMPILER_NAME MATCHES "ifort.*") #ifort
  set (CMAKE_Fortran_FLAGS "")
endif (Fortran_COMPILER_NAME MATCHES "gfortran.*")

set(CMAKE_CXX_FLAGS "-g -Wall -std=c++11 -O0")

include_directories(${PROJECT_SOURCE_DIR})

#CCACHE
find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
  set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
  set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
endif(CCACHE_FOUND)

add_subdirectory(jdqz_tools)
add_subdirectory(tests)

