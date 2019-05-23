cmake_minimum_required (VERSION 3.3)

project (jdqz)

option(BUILD_SHARED_LIBS "Build shared libraries" ON)

# Installation variables
set(library_list)
set(INCLUDE_INSTALL_DIR include)
set(LIB_INSTALL_DIR lib)
set(BIN_INSTALL_DIR bin)

enable_testing()

add_subdirectory (src)

# Install CMake configuration files
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/JDQZPP/JDQZPPConfigVersion.cmake"
  VERSION 0 COMPATIBILITY SameMajorVersion
  )

set(ConfigPackageLocation lib/cmake/JDQZPP)
configure_package_config_file("${PROJECT_SOURCE_DIR}/cmake/JDQZPPConfig.cmake.in"
  "${PROJECT_BINARY_DIR}/JDQZPPConfig.cmake" INSTALL_DESTINATION ${ConfigPackageLocation}
  PATH_VARS INCLUDE_INSTALL_DIR)

install(EXPORT JDQZPPTargets FILE JDQZPPTargets.cmake DESTINATION ${ConfigPackageLocation})
install(FILES "${PROJECT_BINARY_DIR}/JDQZPPConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/JDQZPP/JDQZPPConfigVersion.cmake"
  DESTINATION ${ConfigPackageLocation}
  )

set(CTEST_OPTIONS --force-new-ctest-process --verbose CACHE STRING "ctest options")
# Adding custom test target "check" because it is the only way to show the full output of test errors on console
add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND} ${CTEST_OPTIONS})
