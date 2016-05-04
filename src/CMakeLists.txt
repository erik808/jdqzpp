cmake_minimum_required(VERSION 2.8.9)

enable_testing()

# Enable fortran
get_property(languages GLOBAL PROPERTY ENABLED_LANGUAGES)
if (NOT languages MATCHES "Fortran")
  enable_language(Fortran)
  message("-- Fortran enabled")
endif ()
get_property(languages GLOBAL PROPERTY ENABLED_LANGUAGES)
message("-- Languages: ${languages}")

# Set CXX flags
if (NOT DEFINED ${CMAKE_CXX_FLAGS})
  set(CMAKE_CXX_FLAGS "-g -Wall -std=c++11 -O0")  
endif ()

# Set Fortran flags
# Depend on the compiler
if (NOT DEFINED ${CMAKE_Fortran_FLAGS})
  get_filename_component(Fortran_COMPILER_NAME ${CMAKE_Fortran_COMPILER} NAME)
  if (Fortran_COMPILER_NAME MATCHES "gfortran.*") #gfortran
	set (CMAKE_Fortran_FLAGS "-g -u -O0")
  elseif (Fortran_COMPILER_NAME MATCHES "ifort.*") #ifort
	set (CMAKE_Fortran_FLAGS "")
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
add_subdirectory(tests)