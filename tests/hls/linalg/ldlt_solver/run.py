import vitis
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from run_vitis import run_vitis

# --- Configuration ---
PART = 'xc7z020clg400-1'
CLOCK_PERIOD_NS = "10"
COMPONENT_NAME = "linalg_ldlt_hls"
TOP_FUNCTION_NAME = "top"
SYNTHESIS_FILE = "top.cpp"
TESTBENCH_FILE = "ldlt_solver_test.cpp"

def main():
    cwd = os.getcwd()
    project_root = os.path.abspath(os.path.join(cwd, '..', '..', '..', '..'))
    include_path = os.path.join(project_root, 'include')
    workspace_path = os.path.join(project_root, "build/vitis_workspace_linalg_ldlt")

    run_vitis(workspace_path, include_path, PART, CLOCK_PERIOD_NS, COMPONENT_NAME, TOP_FUNCTION_NAME, SYNTHESIS_FILE, TESTBENCH_FILE, 9, 64)

if __name__ == "__main__":
    main()
