import vitis
import os
import sys

sys.path.append("../..")

from run_vitis import run_vitis

# --- Configuration ---
PART = 'xc7z020clg400-1'
CLOCK_PERIOD_NS = "10"
COMPONENT_NAME = "posit_multiplication_hls"
TOP_FUNCTION_NAME = "top"
SYNTHESIS_FILE = "top.cpp"
TESTBENCH_FILE = "test.cpp"


def main():
    """Main function to run the Vitis HLS flow for linalg ops."""
    # --- Paths ---
    # The script is expected to be run from the 'tests/hls' directory
    cwd = os.getcwd()
    project_root = os.path.abspath(os.path.join(cwd, '..', '..', '..', '..'))
    include_path = os.path.join(project_root, 'include')
    workspace_path = os.path.join(project_root, "build/" + COMPONENT_NAME)

    run_vitis(workspace_path, include_path, PART, CLOCK_PERIOD_NS, COMPONENT_NAME, TOP_FUNCTION_NAME, SYNTHESIS_FILE, TESTBENCH_FILE, 1, 83)

if __name__ == "__main__":
    main()
