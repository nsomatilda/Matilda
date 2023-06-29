
include_directories(${CMAKE_BINARY_DIR})
add_library( matilda STATIC matilda.cc )
target_include_directories(matilda PUBLIC ../include)
set_target_properties( matilda PROPERTIES PUBLIC_HEADER ../include/matilda.h )
install( TARGETS matilda ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}" )
install( TARGETS matilda PUBLIC_HEADER DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
