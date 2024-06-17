#!/bin/bash
# help function
function print_help()
{
    echo "test_dnn.sh"
    echo "Parameters:"
    echo "--output_dir=path_to_output_dir            | output directory after build the library"
    echo "--qemu_dir=absoulte_path_to_qemu_aarch64   | path to qemu-aarch64 (empty path indicates using real hardware platform)"
    return
}

# default output path
CUR_PATH=$(cd `dirname $0` ; pwd)
ROOT_DIR=${CUR_PATH%llt*}
OUTPUT_PATH=${ROOT_DIR}/out
TEST_TYPE=ut
QEMU_PATH=""

# parse input parameters
for i in "$@"; do
    case $i in
        --output_dir=*)
        OUTPUT_PATH="${i#*=}"
        ;;
        --qemu_dir=*)
        QEMU_PATH="${i#*=}"
        ;;
        --help|-h)
        print_help
        exit
        ;;
    esac
done

# test output directory
OUTPUT_TEST_DIR="${OUTPUT_PATH}"
BENCHDNN_DIR="${OUTPUT_TEST_DIR}/oneDNN-3.4/build/tests/benchdnn"
BENCHDNN_BIN="${BENCHDNN_DIR}/benchdnn"
BENCHDNN_INPUTS_DIR="${BENCHDNN_DIR}/inputs"
LLT_OUTPUT_DIR="${OUTPUT_TEST_DIR}/llt"
LLT_INPUTS_DIR="${LLT_OUTPUT_DIR}/inputs"
TEST_OUT_DIR="${OUTPUT_PATH}/${TEST_TYPE}_out"
TEST_LOG_DIR="${TEST_OUT_DIR}/log"
TEST_REPORT_DIR="${TEST_OUT_DIR}/report"
TEST_COVERAGE_RESULT_DIR="${TEST_REPORT_DIR}/coverage_result"
TEST_COVERAGE_INFO_DIR="${TEST_REPORT_DIR}/coverage_info"

# test counts
all_tsts_amount=0
passed_tsts_amount=0
skip_tsts_amount=0
mistrust_tsts_amount=0
unimpl_tsts_amount=0
invalid_tsts_amount=0
failed_tsts_amount=0
listed_tsts_amount=0

# qemu configuration
echo "----------- $QEMU_PATH ---------"
if [[ "${QEMU_PATH}" != "" ]]; then
    QEMU="${QEMU_PATH} -cpu max,sve256=on"
    echo "qemu configuration: ${QEMU}"
    BENCHDNN_BIN="${QEMU} ${BENCHDNN_BIN}"
fi

function gen_test_summary()
{
    echo '<?xml version="1.0" encoding="UTF-8"?>' > ${TEST_REPORT_DIR}/${TEST_TYPE}_report.xml
    echo "<testsuites tests=\"${all_tsts_amount}\" pass=\"${passed_tsts_amount}\" failures=\"${failed_tsts_amount}\"
        skips=\"${skip_tsts_amount}\" mistrust=\"${mistrust_tsts_amount}\" unimplemented=\"${unimpl_tsts_amount}\"
        invalid=\"${invalid_tsts_amount}\" listed=\"${listed_tsts_amount}\"
        name=\"${TEST_TYPE}_test\">" >> ${TEST_REPORT_DIR}/${TEST_TYPE}_report.xml
    echo '</testsuites>' >> ${TEST_REPORT_DIR}/${TEST_TYPE}_report.xml
}

function prepare_test()
{
    [[ -d "${TEST_OUT_DIR}" ]] && rm -rf ${TEST_OUT_DIR}
    mkdir -p ${TEST_OUT_DIR}
    [[ -d "${TEST_LOG_DIR}" ]] && rm -rf ${TEST_LOG_DIR}
    mkdir -p ${TEST_LOG_DIR}

    [[ -d "${TEST_REPORT_DIR}" ]] && rm -rf ${TEST_REPORT_DIR}
    mkdir -p ${TEST_REPORT_DIR}
    if [[ "${GEN_REPORT}" == "on" ]] ;then
        [[ -d "${TEST_COVERAGE_RESULT_DIR}" ]] && rm -rf ${TEST_COVERAGE_RESULT_DIR}
        mkdir -p ${TEST_COVERAGE_RESULT_DIR}
        [[ -d "${TEST_COVERAGE_INFO_DIR}" ]] && rm -rf ${TEST_COVERAGE_INFO_DIR}
        mkdir -p ${TEST_COVERAGE_INFO_DIR}
    fi
}

function count_tsts_status()
{
    local file_name=$1

    local tst_cnt=`cat ${TEST_LOG_DIR}/${file_name} | grep -Eo "tests:[0-9]*" | grep -Eo "[0-9]*"`
    local passed_tst_cnt=`cat ${TEST_LOG_DIR}/${file_name} | grep -Eo "passed:[0-9]*" | grep -Eo "[0-9]*"`
    local skipped_tst_cnt=`cat ${TEST_LOG_DIR}/${file_name} | grep -Eo "skipped:[0-9]*" | grep -Eo "[0-9]*"`
    local mistrust_tst_cnt=`cat ${TEST_LOG_DIR}/${file_name} | grep -Eo "mistrusted:[0-9]*" | grep -Eo "[0-9]*"`
    local unimpl_tst_cnt=`cat ${TEST_LOG_DIR}/${file_name} | grep -Eo "unimplemented:[0-9]*" | grep -Eo "[0-9]*"`
    local invalid_tst_cnt=`cat ${TEST_LOG_DIR}/${file_name} | grep -Eo "invalid_arguments:[0-9]*" | grep -Eo "[0-9]*"`
    local failed_tst_cnt=`cat ${TEST_LOG_DIR}/${file_name} | grep -Eo "failed:[0-9]*" | grep -Eo "[0-9]*"`
    local listed_tst_cnt=`cat ${TEST_LOG_DIR}/${file_name} | grep -Eo "listed:[0-9]*" | grep -Eo "[0-9]*"`

    all_tsts_amount=`expr ${all_tsts_amount} + ${tst_cnt}`
    passed_tsts_amount=`expr ${passed_tsts_amount} + ${passed_tst_cnt}`
    skip_tsts_amount=`expr ${skip_tsts_amount} + ${skipped_tst_cnt}`
    mistrust_tsts_amount=`expr ${mistrust_tsts_amount} + ${mistrust_tst_cnt}`
    unimpl_tsts_amount=`expr ${unimpl_tsts_amount} + ${unimpl_tst_cnt}`
    invalid_tsts_amount=`expr ${invalid_tsts_amount} + ${invalid_tst_cnt}`
    failed_tsts_amount=`expr ${failed_tsts_amount} + ${failed_tst_cnt}`
    listed_tsts_amount=`expr ${listed_tsts_amount} + ${listed_tst_cnt}`

    # if [ ${failed_tst_cnt} -ge 1 ];then
    #    echo "Detect ${failed_tst_cnt} test cases fail in ${file_name}"
    #    exit 1
    # fi
}

function run_ut_test()
{
    echo "${TEST_TYPE} Test Begin"
    local num_threads=$1

    # elementwise
    OMP_NUM_THREADS=${num_threads} ${BENCHDNN_BIN} --eltwise --mode=C --batch=${BENCHDNN_INPUTS_DIR}/eltwise/test_eltwise_kdnn_ci >> ${TEST_LOG_DIR}/eltwise_log1.log&
    OMP_NUM_THREADS=${num_threads} ${BENCHDNN_BIN} --eltwise --mode=C --batch=${LLT_INPUTS_DIR}/eltwise/eltwise_func >> ${TEST_LOG_DIR}/eltwise_log2.log&

    # softmax
    OMP_NUM_THREADS=${num_threads} ${BENCHDNN_BIN} --softmax --mode=C --batch=${LLT_INPUTS_DIR}/softmax/softmax_func >> ${TEST_LOG_DIR}/sofmax_log.log&

    # prelu
    OMP_NUM_THREADS=${num_threads} ${BENCHDNN_BIN} --prelu --mode=C --batch=${LLT_INPUTS_DIR}/prelu/prelu_func >> ${TEST_LOG_DIR}/prelu_log.log&

    wait
    count_tsts_status eltwise_log1.log
    count_tsts_status eltwise_log2.log
    count_tsts_status sofmax_log.log
    count_tsts_status prelu_log.log

    # matmul
    OMP_NUM_THREADS=${num_threads} ${BENCHDNN_BIN} --matmul --mode=C --batch=${LLT_INPUTS_DIR}/matmul/matmul_func > ${TEST_LOG_DIR}/matmul_log.log&

    #inner product
    OMP_NUM_THREADS=${num_threads} ${BENCHDNN_BIN} --ip --mode=C --batch=${LLT_INPUTS_DIR}/ip/ip_func > ${TEST_LOG_DIR}/ip_log.log&
    wait

    count_tsts_status matmul_log.log
    count_tsts_status ip_log.log

    # reduction
    OMP_NUM_THREADS=${num_threads} ${BENCHDNN_BIN} --reduction --skip-impl=jit,acl,ref --mode=C --alg=min,max,mean,sum,mul --batch=${BENCHDNN_INPUTS_DIR}/reduction/shapes_ci >> ${TEST_LOG_DIR}/reduction_log_f32.log&
    OMP_NUM_THREADS=${num_threads} ${BENCHDNN_BIN} --reduction --skip-impl=jit,acl,ref --sdt=f16 --ddt=f16 --mode=C --alg=min,max,mean,sum,mul --batch=${BENCHDNN_INPUTS_DIR}/reduction/shapes_ci >> ${TEST_LOG_DIR}/reduction_log_f16.log&
    OMP_NUM_THREADS=${num_threads} ${BENCHDNN_BIN} --reduction --skip-impl=jit,acl,ref --sdt=bf16 --ddt=bf16 --mode=C --alg=min,max,mean,sum,mul --batch=${BENCHDNN_INPUTS_DIR}/reduction/shapes_ci >> ${TEST_LOG_DIR}/reduction_log_bf16.log&

    # layer normalization
    OMP_NUM_THREADS=${num_threads} ${BENCHDNN_BIN} --lnorm  --mode=C --batch=${LLT_INPUTS_DIR}/lnorm/lnorm_func >> ${TEST_LOG_DIR}/lnorm_log.log&

    # binary operations
    OMP_NUM_THREADS=${num_threads} ${BENCHDNN_BIN} --binary --mode=C --batch=${LLT_INPUTS_DIR}/binary/binary_func >> ${TEST_LOG_DIR}/binary_log.log&

    # sum operations
    OMP_NUM_THREADS=${num_threads} ${BENCHDNN_BIN} --sum --mode=C --batch=${LLT_INPUTS_DIR}/sum/sum_func >> ${TEST_LOG_DIR}/sum_log.log&
    wait

    count_tsts_status reduction_log_f32.log
    count_tsts_status reduction_log_f16.log
    count_tsts_status reduction_log_bf16.log
    count_tsts_status lnorm_log.log
    count_tsts_status binary_log.log
    count_tsts_status sum_log.log

    # convolution operations
    OMP_NUM_THREADS=${num_threads} ${BENCHDNN_BIN} --conv --mode=C --batch=${LLT_INPUTS_DIR}/convolution/convolution_func >> ${TEST_LOG_DIR}/conv_log.log&
    wait

    count_tsts_status conv_log.log

    # deconvolution operations
    OMP_NUM_THREADS=${num_threads} ${BENCHDNN_BIN} --deconv --mode=C --batch=${LLT_INPUTS_DIR}/deconvolution/deconvolution_func >> ${TEST_LOG_DIR}/deconv_log.log&
    wait

    count_tsts_status deconv_log.log

    echo "${TEST_TYPE} Test Finished"
    gen_test_summary
}

prepare_test

run_ut_test 4
