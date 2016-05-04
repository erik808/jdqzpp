cmake_minimum_required (VERSION 2.8.9)

project (jdqz)

add_subdirectory (src)

# If no CMAKE_INSTALl_PREFIX is available we set it to the current source dir
# .i.e. project root
if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set (CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}" CACHE PATH "default install path" FORCE )
endif ()

message ("-- Install prefix = ${CMAKE_INSTALL_PREFIX}")