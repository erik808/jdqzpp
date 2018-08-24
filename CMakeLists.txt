cmake_minimum_required (VERSION 2.8.9)

project (jdqz)

option(BUILD_SHARED_LIBS "Build shared libraries" ON)

# Installation variables
set(library_list)
set(INCLUDE_INSTALL_DIR ${PROJECT_SOURCE_DIR}/include)
set(LIB_INSTALL_DIR ${PROJECT_SOURCE_DIR}/lib)
set(BIN_INSTALL_DIR ${PROJECT_SOURCE_DIR}/bin)

enable_testing()

add_subdirectory (src)

# Install CMake configuration files
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/JDQZPP/JDQZPPConfigVersion.cmake"
  VERSION 0 COMPATIBILITY SameMajorVersion
  )

set(ConfigPackageLocation ${LIB_INSTALL_DIR}/cmake/JDQZPP)
configure_package_config_file("${PROJECT_SOURCE_DIR}/cmake/JDQZPPConfig.cmake.in"
  "${PROJECT_BINARY_DIR}/JDQZPPConfig.cmake" INSTALL_DESTINATION ${ConfigPackageLocation}
  PATH_VARS INCLUDE_INSTALL_DIR)

install(EXPORT JDQZPPTargets FILE JDQZPPTargets.cmake DESTINATION ${ConfigPackageLocation})
install(FILES "${PROJECT_BINARY_DIR}/JDQZPPConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/JDQZPP/JDQZPPConfigVersion.cmake"
  DESTINATION ${ConfigPackageLocation}
  )
