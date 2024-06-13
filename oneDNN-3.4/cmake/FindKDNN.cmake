# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

# ----------
# FindKPL
# ----------
#
# This module defines the following variables:
#
#   KDNN_FOUND          - True if KPL was found
#   KPL_INCLUDE_DIRS   - include directories for KPL
#   KPL_LIBRARIES      - link against this library to use KPL
#

# Use KDNN_ROOT_DIR environment variable to find the library and headers
find_path(KDNN_INCLUDE_DIR
  NAMES kdnn.hpp
  PATHS ENV KDNN_ROOT_DIR
  PATH_SUFFIXES include/kdnn
  NO_DEFAULT_PATH
)

find_path(KPL_BLAS_INCLUDE_DIR
  NAMES kblas.h
  PATHS ENV KML_ROOT_DIR
  PATH_SUFFIXES include
  NO_DEFAULT_PATH
)

find_path(KPL_FFT_INCLUDE_DIR
  NAMES kfft.h
  PATHS ENV KML_ROOT_DIR
  PATH_SUFFIXES include
  NO_DEFAULT_PATH
)

find_path(KPL_VML_INCLUDE_DIR
  NAMES kvml.h
  PATHS ENV KML_ROOT_DIR
  PATH_SUFFIXES include
  NO_DEFAULT_PATH
)

find_path(KPL_LIBM_INCLUDE_DIR
  NAMES km.h
  PATHS ENV KML_ROOT_DIR
  PATH_SUFFIXES include
  NO_DEFAULT_PATH
)

find_library(KDNN_LIBRARY
  NAMES kdnn
  PATHS ENV KDNN_ROOT_DIR
  PATH_SUFFIXES lib
  NO_DEFAULT_PATH
  )

find_library(KPL_BLAS_LIBRARY
  NAMES kblas
  PATHS ENV KML_ROOT_DIR
  PATH_SUFFIXES lib/kblas/omp
  NO_DEFAULT_PATH
)

find_library(KPL_FFT_LIBRARY
  NAMES kfft_omp
  PATHS ENV KML_ROOT_DIR
  PATH_SUFFIXES lib
  NO_DEFAULT_PATH
)

find_library(KPL_FFTF_LIBRARY
  NAMES kfftf_omp
  PATHS ENV KML_ROOT_DIR
  PATH_SUFFIXES lib
  NO_DEFAULT_PATH
)

find_library(KPL_FFTH_LIBRARY
  NAMES kffth_omp
  PATHS ENV KML_ROOT_DIR
  PATH_SUFFIXES lib
  NO_DEFAULT_PATH
)

find_library(KPL_VML_LIBRARY
  NAMES kvml
  PATHS ENV KML_ROOT_DIR
  PATH_SUFFIXES lib/kvml/multi
  NO_DEFAULT_PATH
)

find_library(KPL_LIBM_LIBRARY
  NAMES km
  PATHS ENV KML_ROOT_DIR
  PATH_SUFFIXES lib
  NO_DEFAULT_PATH
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(KDNN DEFAULT_MSG
  KDNN_INCLUDE_DIR
  KPL_BLAS_INCLUDE_DIR
  KPL_FFT_INCLUDE_DIR
  KPL_VML_INCLUDE_DIR
  KPL_LIBM_INCLUDE_DIR
  KDNN_LIBRARY
  KPL_BLAS_LIBRARY
  KPL_FFT_LIBRARY
  KPL_FFTF_LIBRARY
  KPL_FFTH_LIBRARY
  KPL_VML_LIBRARY
  KPL_LIBM_LIBRARY
)

mark_as_advanced(
  KDNN_LIBRARY
  KPL_BLAS_LIBRARY
  KPL_FFT_LIBRARY
  KPL_FFTF_LIBRARY
  KPL_FFTH_LIBRARY
  KPL_VML_LIBRARY
  KPL_LIBM_LIBRARY
  KDNN_INCLUDE_DIR
  KPL_BLAS_INCLUDE_DIR
  KPL_FFT_INCLUDE_DIR
  KPL_VML_INCLUDE_DIR
  KPL_LIBM_INCLUDE_DIR
)

# Find the extra libraries and include dirs
if(KDNN_FOUND)
  list(APPEND KDNN_INCLUDE_DIRS
    ${KDNN_INCLUDE_DIR} ${KPL_BLAS_INCLUDE_DIR} ${KPL_FFT_INCLUDE_DIR} ${KPL_VML_INCLUDE_DIR} ${KPL_LIBM_INCLUDE_DIR})
  list(APPEND KDNN_LIBRARIES
    ${KDNN_LIBRARY} ${KPL_BLAS_LIBRARY} ${KPL_FFT_LIBRARY} ${KPL_FFTF_LIBRARY} ${KPL_FFTH_LIBRARY} ${KPL_VML_LIBRARY} ${KPL_LIBM_LIBRARY})
endif()
