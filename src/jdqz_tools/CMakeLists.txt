set (FORTRAN_SOURCES select.f myexc.f)

add_library (jdqz_tools SHARED ${FORTRAN_SOURCES})

install(TARGETS jdqz_tools EXPORT JDQZPPTargets LIBRARY DESTINATION ${LIB_INSTALL_DIR})

set(library_list ${library_list} jdqz_tools PARENT_SCOPE)
