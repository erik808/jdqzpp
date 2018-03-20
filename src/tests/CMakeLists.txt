# Google test
INCLUDE(ExternalProject)

SET_DIRECTORY_PROPERTIES(PROPERTIES EP_PREFIX ${CMAKE_BINARY_DIR}/third_party)

ExternalProject_Add(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG release-1.8.0
  INSTALL_COMMAND ""
  LOG_DOWNLOAD ON
  LOG_CONFIGURE ON
  LOG_BUILD ON)

ExternalProject_Get_Property(googletest source_dir)
set(GTEST_INCLUDE_DIRS ${source_dir}/googletest/include)

ExternalProject_Get_Property(googletest binary_dir)
set(GTEST_LIBRARY_PATH ${binary_dir}/googlemock/gtest/${CMAKE_FIND_LIBRARY_PREFIXES}gtest.a)
set(GTEST_MAIN_LIBRARY_PATH ${binary_dir}/googlemock/gtest/${CMAKE_FIND_LIBRARY_PREFIXES}gtest_main.a)
set(GTEST_LIBRARY gtest)
set(GTEST_MAIN_LIBRARY gtest_main)
set(GTEST_BOTH_LIBRARIES gtest gtest_main)

add_library(${GTEST_LIBRARY} UNKNOWN IMPORTED)
set_property(TARGET ${GTEST_LIBRARY} PROPERTY IMPORTED_LOCATION
  ${GTEST_LIBRARY_PATH})
add_dependencies(${GTEST_LIBRARY} googletest)

add_library(${GTEST_MAIN_LIBRARY} UNKNOWN IMPORTED)
set_property(TARGET ${GTEST_MAIN_LIBRARY} PROPERTY IMPORTED_LOCATION
  ${GTEST_MAIN_LIBRARY_PATH})
add_dependencies(${GTEST_MAIN_LIBRARY} googletest)

include_directories(${GTEST_INCLUDE_DIRS})

set(TEST_SOURCES
  test1
  test2
  )

if (${Anasazi_FOUND})
  list(APPEND TEST_SOURCES test_interfaces)

  include_directories(${Anasazi_INCLUDE_DIRS})
  include_directories(${Anasazi_TPL_INCLUDE_DIRS})

  link_directories(${Anasazi_LIBRARY_DIRS})
  link_directories(${Anasazi_TPL_LIBRARY_DIRS})
endif()

set(test_list)

include_directories(${PROJECT_SOURCE_DIR}/src/jdqz)
include_directories(${PROJECT_SOURCE_DIR}/src/interfaces)

foreach(test_source ${TEST_SOURCES})
  get_filename_component(test_name ${test_source} NAME)
  add_executable(${test_name} ${test_source})
  target_link_libraries(${test_name} jdqz_tools)
  target_link_libraries(${test_name} ${LAPACK_LIBRARIES})
  target_link_libraries(${test_name} ${GTEST_BOTH_LIBRARIES})
  target_link_libraries(${test_name} ${CMAKE_THREAD_LIBS_INIT})

  if (Anasazi_FOUND)
    target_link_libraries(${test_name} ${Anasazi_LIBRARIES})
    target_link_libraries(${test_name} ${Anasazi_TPL_LIBRARIES})
  endif()

  add_test(${test_name} ${test_name})
  list(APPEND test_list ${test_name})
endforeach()

# Use make check to both build tests and test
add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND} --force-new-ctest-process --verbose
  DEPENDS ${test_targets})
