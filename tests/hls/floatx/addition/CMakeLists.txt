# Register three variants of the addition test, each compiling the HLS sources
# with a different numeric format. We pass extra compile flags via the
# HLS_EXTRA_CFLAGS environment variable which `run_vitis.py` appends to the
# Vitis HLS cflags. We also set NUMERIC_FORMAT so run.py can pick a distinct
# component/workspace name.

set(FLOATX_ADD_HLS_COMMAND
    "source ${VITIS_SETTINGS_SCRIPT} && NUMERIC_FORMAT=FloatX HLS_EXTRA_CFLAGS='-DFORMAT_FLOATX' ${VITIS_EXECUTABLE} -s ${CMAKE_CURRENT_SOURCE_DIR}/run.py"
)
add_test(
    NAME FloatX_Add_HLS_Test
    COMMAND bash -c "${FLOATX_ADD_HLS_COMMAND}"
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
set_tests_properties(FloatX_Add_HLS_Test PROPERTIES
    TIMEOUT 1800
    LABELS "HLS;FPGA;LinAlg;FloatX"
)

set(POSIT_ADD_HLS_COMMAND
    "source ${VITIS_SETTINGS_SCRIPT} && NUMERIC_FORMAT=Posit HLS_EXTRA_CFLAGS='-DFORMAT_POSIT' ${VITIS_EXECUTABLE} -s ${CMAKE_CURRENT_SOURCE_DIR}/run.py"
)
add_test(
    NAME Posit_Add_HLS_Test
    COMMAND bash -c "${POSIT_ADD_HLS_COMMAND}"
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
set_tests_properties(Posit_Add_HLS_Test PROPERTIES
    TIMEOUT 1800
    LABELS "HLS;FPGA;LinAlg;Posit"
)

set(FLOAT_ADD_HLS_COMMAND
    "source ${VITIS_SETTINGS_SCRIPT} && NUMERIC_FORMAT=Float HLS_EXTRA_CFLAGS='-DFORMAT_FLOAT' ${VITIS_EXECUTABLE} -s ${CMAKE_CURRENT_SOURCE_DIR}/run.py"
)
add_test(
    NAME Float_Add_HLS_Test
    COMMAND bash -c "${FLOAT_ADD_HLS_COMMAND}"
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
set_tests_properties(Float_Add_HLS_Test PROPERTIES
    TIMEOUT 1800
    LABELS "HLS;FPGA;LinAlg;Float"
)
