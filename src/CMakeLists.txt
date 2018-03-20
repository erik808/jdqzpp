cmake_minimum_required(VERSION 2.8.9)

# Lapack
# --> For now I hardcode a few things I need on cartesius
# --> When I find out how to detect these I will do that
if ($ENV{PLAT} MATCHES "cartesius") 
  message("-- Get info from Trilinos")

  find_package(Anasazi)

  if (NOT Anasazi_FOUND)
	message(FATAL_ERROR " PATH should contain Trilinos_DIR/bin ")
  endif ()
  
  if (NOT DEFINED ${CMAKE_CXX_COMPILER})
	set(CMAKE_CXX_COMPILER ${Anasazi_CXX_COMPILER})
  endif ()

  message("-- CXX compiler: ${CMAKE_CXX_COMPILER}")
  
  if (NOT DEFINED ${CMAKE_Fortran_COMPILER})
	set(CMAKE_Fortran_COMPILER ${Anasazi_Fortran_COMPILER})
  endif ()

  message("-- Fortran compiler: ${CMAKE_Fortran_COMPILER}")
  
  set(LAPACK_LIBRARIES
	mkl_intel_lp64
	mkl_intel_thread
	mkl_core
	iomp5
	pthread )
  
else ()
  
  find_package(LAPACK)
  if (NOT LAPACK_FOUND)
	message(FATAL_ERROR "Lapack not found")
  endif ()  
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
if (NOT DEFINED ${CMAKE_Fortran_FLAGS})
  if (CMAKE_Fortran_COMPILER_ID MATCHES "GNU") #gfortran, gcc
	set (CMAKE_Fortran_FLAGS
	  "-g -u -O3 -ffixed-line-length-132 -ffree-line-length-none")
  elseif (CMAKE_Fortran_COMPILER_ID MATCHES "Intel") #ifort, icc
	set (CMAKE_Fortran_FLAGS "-g -O3 -heap-arrays 1 -extend-source 132")
  endif ()
endif ()

# Set CXX flags
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
