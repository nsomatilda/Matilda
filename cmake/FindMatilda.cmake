# - Find Matilda library
#
#  MATILDA_INCLUDES    - where to find phx_api.h
#  MATILDA_LIBRARIES   - List of libraries when using MATILDA
#  MATILDA_FOUND       - True if MATILDA found.

SET(MATILDA_SEARCH_PATHS /opt/matilda/include /opt/matilda/lib )


if (MATILDA_INCLUDES)
  # Already in cache, be silent
  set (MATILDA_FIND_QUIETLY TRUE)
endif (MATILDA_INCLUDES)

find_path (MATILDA_INCLUDES matilda.h PATHS ${MATILDA_SEARCH_PATHS})

find_library (MATILDA_LIBRARIES NAMES matilda PATHS ${MATILDA_SEARCH_PATHS})

# handle the QUIETLY and REQUIRED arguments and set EURESYS_FOUND to TRUE if
# all listed variables are TRUE
include (FindPackageHandleStandardArgs)
find_package_handle_standard_args (MATILDA DEFAULT_MSG MATILDA_INCLUDES)

mark_as_advanced( MATILDA_INCLUDES)

