cmake_minimum_required(VERSION 3.3)

find_package(LAPACK)
if (NOT LAPACK_FOUND)
  message(FATAL_ERROR "Lapack not found")
  else ()
  message("-- LAPACK libs: ${LAPACK_LIBRARIES}")
endif ()

# Set CXX flags
message("-- CXX compiler ID: ${CMAKE_CXX_COMPILER_ID}")
if (NOT DEFINED ${CMAKE_CXX_FLAGS})
  if (CMAKE_CXX_COMPILER_ID MATCHES "GNU") #gcc, g++
    set (CMAKE_CXX_FLAGS "-g -std=c++11 -O3 -Wall")
  elseif (CMAKE_CXX_COMPILER_ID MATCHES "Clang") #clang, clang++
    set (CMAKE_CXX_FLAGS "-g -std=c++11 -O3 -Wall")
  elseif (CMAKE_CXX_COMPILER_ID MATCHES "Intel") #icc, icpp
    set (CMAKE_CXX_FLAGS "-g -std=c++11 -O3")
  endif ()
endif ()

message("-- JDQZPP CXX flags: " ${CMAKE_CXX_FLAGS})

#CCACHE
find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
  set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
  set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
endif(CCACHE_FOUND)

add_subdirectory(jdqz)
add_subdirectory(interfaces)
add_subdirectory(tests)

set(library_list ${library_list} PARENT_SCOPE)
