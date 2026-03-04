import vitis
import os
import sys

sys.path.append("../..")

from run_vitis import run_vitis

# --- Configuration ---
PART = 'xc7z020clg400-1'
CLOCK_PERIOD_NS = "10"
COMPONENT_NAME = "linalg_determinant_hls"
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
    # Allow NUMERIC_FORMAT to be appended to component/workspace name so
    # multiple format runs don't collide.
    num_fmt = os.environ.get('NUMERIC_FORMAT', '')
    if num_fmt:
        component = COMPONENT_NAME + "_" + num_fmt
    else:
        component = COMPONENT_NAME
    workspace_path = os.path.join(project_root, "build/" + component)
    run_vitis(workspace_path, include_path, PART, CLOCK_PERIOD_NS, component, TOP_FUNCTION_NAME, SYNTHESIS_FILE, TESTBENCH_FILE, 9, 40)

if __name__ == "__main__":
    main()
