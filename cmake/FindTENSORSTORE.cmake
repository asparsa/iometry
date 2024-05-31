# Try to find TensorStore
# Once done this will define
#  TENSORSTORE_FOUND - System has TensorStore
#  TENSORSTORE_INCLUDE_DIRS - The TensorStore include directories
#  TENSORSTORE_LIBRARIES - The libraries needed to use TensorStore

FIND_PATH(WITH_TENSORSTORE_PREFIX
    NAMES tensorstore/tensorstore.h
)

FIND_LIBRARY(TENSORSTORE_LIBRARIES
    NAMES tensorstore
    HINTS ${WITH_TENSORSTORE_PREFIX}/lib
)

FIND_PATH(TENSORSTORE_INCLUDE_DIRS
    NAMES tensorstore/tensorstore.h
    HINTS ${WITH_TENSORSTORE_PREFIX}/include
)

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(TENSORSTORE DEFAULT_MSG
    TENSORSTORE_LIBRARIES
    TENSORSTORE_INCLUDE_DIRS
)

# Hide these vars from ccmake GUI
MARK_AS_ADVANCED(
    TENSORSTORE_LIBRARIES
    TENSORSTORE_INCLUDE_DIRS
)
