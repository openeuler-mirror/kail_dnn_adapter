#!/bin/bash
# ------------------------------------------------------------------------------
# Copyright (c) 2024-2024 Huawei Technologies Co., Ltd. All rights reserved.
# ------------------------------------------------------------------------------
set -Eeuo pipefail

# manual for the build script
function print_help(){
    echo "[Usage] sh build.sh --xxx=xxx"
    echo "Parameters:"
    echo "--output_dir=absolute_path_to_output_lib_directory  "
    echo "--compiler=[gnu(default)|clang]                      |  compiler type"
    return
}

#compiler to use
COMP_TOOLCHAIN=gnu
C_COMP_TO_USE=gcc
CXX_COMP_TO_USE=g++

# 依赖库路径设置
KAIL_INSTALL_DIR=/usr/local/kail
KML_INSTALL_DIR=/usr/local/kml

# project root path
BUILD_SCRIPT_DIR=$(readlink -e "$(dirname "${0}")")
PROJECT_DIR=${BUILD_SCRIPT_DIR}
ONEDNN_SRC_DIR=${PROJECT_DIR}/oneDNN-3.4
LLT_SRC_DIR=${PROJECT_DIR}/llt

# default parameter
OUTPUT_DIR=${PROJECT_DIR}/out
DNNL_CODE_COVERAGE=OFF

# read input parameter
for i in "$@"; do
  case $i in
    --output_dir=*)
      OUTPUT_DIR="${i#*=}"
    ;;
    --compiler=*)
      COMP_TOOLCHAIN="${i#*=}"
    ;;
    --help|-h)
      print_help
      exit
    ;;
  esac
done

ACL_ROOT_DIR=${OUTPUT_DIR}/ComputeLibrary-23.11
ACL_SRC_DIR=${ACL_ROOT_DIR}
ACL_BUILD_DIR=${ACL_ROOT_DIR}/build
ONEDNN_BUILD_DIR=${OUTPUT_DIR}/oneDNN-3.4/build
LLT_OUT_DIR=${OUTPUT_DIR}/llt

# compiler tool chain
if [ ${COMP_TOOLCHAIN} = "clang" ]; then
    C_COMP_TO_USE=clang
    CXX_COMP_TO_USE=clang++
fi

function check_env()
{
    if ! rpm -qa | grep kml &>/dev/null; then
        echo "Please install kml."
        exit 1
    fi

    if ! rpm -qa | grep boostkit-kail &>/dev/null; then
        echo "Please install boostkit-kail."
        exit 1
    fi
}

# copy llt scripts and inputs
function copy_llt_files()
{
  mkdir -p ${LLT_OUT_DIR}
  cp -r ${LLT_SRC_DIR}/inputs ${LLT_OUT_DIR}
  cp -r ${LLT_SRC_DIR}/scripts ${LLT_OUT_DIR}
}

# build ACL
function build_acl()
{
    tar -zxvf dependencies/ComputeLibrary-23.11.tar.gz -C ${OUTPUT_DIR}/

    mkdir -p ${ACL_BUILD_DIR} &&  cd ${ACL_BUILD_DIR}
    echo -e "\033[32m=========Start To Build ARM Compute Library==============\033[0m"
    CC=${C_COMP_TO_USE} CXX=${CXX_COMP_TO_USE} cmake ${ACL_SRC_DIR} -DCMAKE_BUILD_TYPE=Release -DARM_COMPUTE_OPENMP=1 -DARM_COMPUTE_WERROR=0 -DARM_COMPUTE_BUILD_EXAMPLES=1 -DARM_COMPUTE_BUILD_TESTING=1 -DCMAKE_INSTALL_LIBDIR=${ACL_BUILD_DIR} 2>&1 | tee -a ${ACL_BUILD_DIR}/build.log
    CC=${C_COMP_TO_USE} CXX=${CXX_COMP_TO_USE} cmake --build . --parallel $(nproc --all) 2>&1 | tee -a ${ACL_BUILD_DIR}/build.log
    echo -e "\033[32m=========Build ARM Compute Library Successfully==============\033[0m"
}

# build oneDNN
function build_onednn()
{
    mkdir -p ${ONEDNN_BUILD_DIR} &&  cd ${ONEDNN_BUILD_DIR}
    echo -e "\033[32m=========Start To Build oneDNN==============\033[0m"
    KDNN_ROOT_DIR=${KAIL_INSTALL_DIR} KML_ROOT_DIR=${KML_INSTALL_DIR}  ACL_ROOT_DIR=${ACL_BUILD_DIR}/..  CC=${C_COMP_TO_USE} CXX=${CXX_COMP_TO_USE} cmake ${ONEDNN_SRC_DIR} -DDNNL_AARCH64_USE_ACL=ON -DDNNL_AARCH64_USE_KDNN=ON -DCMAKE_BUILD_TYPE=Release -DDNNL_CODE_COVERAGE=${DNNL_CODE_COVERAGE} 2>&1 | tee -a ${ONEDNN_BUILD_DIR}/build.log
    KDNN_ROOT_DIR=${KAIL_INSTALL_DIR} KML_ROOT_DIR=${KML_INSTALL_DIR}  ACL_ROOT_DIR=${ACL_BUILD_DIR}/.. CC=${C_COMP_TO_USE} CXX=${CXX_COMP_TO_USE} cmake --build . --parallel $(nproc --all) 2>&1 | tee -a ${ONEDNN_BUILD_DIR}/build.log
    echo -e "\033[32m=========Build oneDNN Successfully==============\033[0m"
}

function main()
{
    check_env

    if [ -d ${OUTPUT_DIR} ]; then
        rm -r ${OUTPUT_DIR}
    fi

    mkdir -p ${OUTPUT_DIR}

    build_acl

    build_onednn

    copy_llt_files
}

main $@

set +Eeuo pipefail
