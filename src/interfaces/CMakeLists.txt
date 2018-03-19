include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set(HEADERS AnasaziJdqzppSolMgr.hpp EpetraComplexVector.hpp)

# Install headers
install(FILES ${HEADERS} DESTINATION ${INCLUDE_INSTALL_DIR})
