cmake_minimum_required(VERSION 2.8.9)

find_package(LAPACK)
if (NOT LAPACK_FOUND)
  message(FATAL_ERROR "Lapack not found")
  else ()
  message("-- LAPACK libs: ${LAPACK_LIBRARIES}")
endif ()

# Enable fortran
get_property(languages GLOBAL PROPERTY ENABLED_LANGUAGES)
if (NOT languages MATCHES "Fortran")
  enable_language(Fortran)
  message("-- Fortran enabled")
endif ()
get_property(languages GLOBAL PROPERTY ENABLED_LANGUAGES)
message("-- Languages: ${languages}")

# Set Fortran flags
message("-- Fortran compiler ID: ${CMAKE_Fortran_COMPILER_ID}")
if (NOT DEFINED ${CMAKE_Fortran_FLAGS})
  if (CMAKE_Fortran_COMPILER_ID MATCHES "GNU") #gfortran, gcc
    set (CMAKE_Fortran_FLAGS 
      "-g -O3 -ffixed-line-length-132 -ffree-line-length-none")
  elseif (CMAKE_Fortran_COMPILER_ID MATCHES "Intel") #ifort, icc
    set (CMAKE_Fortran_FLAGS "-g -O3 -heap-arrays 1 -extend-source 132")
  endif ()
endif ()

# Set CXX flags
message("-- CXX compiler ID: ${CMAKE_CXX_COMPILER_ID}")
if (NOT DEFINED ${CMAKE_CXX_FLAGS})
  if (CMAKE_CXX_COMPILER_ID MATCHES "GNU") #gfortran, gcc
    set (CMAKE_CXX_FLAGS "-g -std=c++11 -O3 -Wall")
  elseif (CMAKE_CXX_COMPILER_ID MATCHES "Intel") #ifort, icc
    set (CMAKE_CXX_FLAGS "-std=c++11 -O3")
  endif ()
endif ()

message("-- JDQZPP Fortran flags: " ${CMAKE_Fortran_FLAGS})
message("-- JDQZPP CXX flags: " ${CMAKE_CXX_FLAGS})

#CCACHE
find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
  set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
  set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
endif(CCACHE_FOUND)

add_subdirectory(jdqz_tools)
add_subdirectory(jdqz)
add_subdirectory(interfaces)
add_subdirectory(tests)

set(library_list ${library_list} PARENT_SCOPE)
