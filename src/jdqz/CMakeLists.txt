include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set(HEADERS jdqz.hpp jdqz_decl.hpp jdqz_macros.hpp)

# Install headers
target_include_directories(jdqz_tools INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/>
  $<INSTALL_INTERFACE:include/>
)
install(FILES ${HEADERS} DESTINATION ${INCLUDE_INSTALL_DIR})
