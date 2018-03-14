#enable_testing()

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

add_executable(test_jdqz_1 test1.C)
add_executable(test_jdqz_2 test2.C)

add_test(test1 test_jdqz_1)
add_test(test2 test_jdqz_2)

# Use make check to both build tests and test
add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND}
  DEPENDS test_jdqz_1 test_jdqz_2)

target_link_libraries (test_jdqz_1 PUBLIC jdqz_tools)
target_link_libraries (test_jdqz_2 PUBLIC jdqz_tools)

target_link_libraries (test_jdqz_1 PUBLIC ${GTEST_BOTH_LIBRARIES})
target_link_libraries (test_jdqz_2 PUBLIC ${GTEST_BOTH_LIBRARIES})

target_link_libraries(test_jdqz_1 PUBLIC ${LAPACK_LIBRARIES})
target_link_libraries(test_jdqz_2 PUBLIC ${LAPACK_LIBRARIES})

# Pthread
find_package(Threads)
target_link_libraries (test_jdqz_1 PUBLIC ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries (test_jdqz_2 PUBLIC ${CMAKE_THREAD_LIBS_INIT})

set(TEST_INCL_DIRS ../jdqz)

target_include_directories(test_jdqz_1 PUBLIC ${TEST_INCL_DIRS})
target_include_directories(test_jdqz_2 PUBLIC ${TEST_INCL_DIRS})

install(TARGETS test_jdqz_1 test_jdqz_2 DESTINATION test)
