# ******************************************************************************
# Copyright 2020-2023 Arm Limited and affiliates.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************

if(kdnn_cmake_included)
    return()
endif()
set(kdnn_cmake_included true)
include("cmake/options.cmake")

if(NOT DNNL_TARGET_ARCH STREQUAL "AARCH64")
    return()
endif()

if(NOT DNNL_AARCH64_USE_KDNN)
    return()
endif()

find_package(KDNN REQUIRED)

if(KDNN_FOUND)
    list(APPEND EXTRA_SHARED_LIBS ${KDNN_LIBRARIES})

    include_directories(${KDNN_INCLUDE_DIRS})

    message(STATUS "KPL Library: ${KDNN_LIBRARIES}")
    message(STATUS "KPL Library headers: ${KDNN_INCLUDE_DIRS}")

    add_definitions(-DDNNL_AARCH64_USE_KDNN)
    set(CMAKE_CXX_STANDARD 14)
    set(CMAKE_CXX_EXTENSIONS "OFF")
endif()
