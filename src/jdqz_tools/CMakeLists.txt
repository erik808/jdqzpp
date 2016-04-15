set(FORTRAN_SOURCES
  select.f
  myexc.f
  )

add_library(jdqz_tools ${FORTRAN_SOURCES})
target_include_directories (jdqz_tools PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})