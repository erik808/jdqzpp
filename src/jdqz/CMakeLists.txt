include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set(HEADERS jdqz.hpp jdqz_decl.hpp jdqz_macros.hpp)

# Install headers
install(FILES ${HEADERS} DESTINATION ${INCLUDE_INSTALL_DIR})
