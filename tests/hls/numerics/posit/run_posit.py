import vitis
import os
import sys

# --- Configuration ---
# TODO: Update TOP_FUNCTION_NAME to the actual top-level function for synthesis.
# TODO: Update SYNTHESIS_FILE if your top-level function is not in this file.
PART = 'xc7z020clg400-1'
CLOCK_PERIOD_NS = "10"
COMPONENT_NAME = "posit_hls"
TOP_FUNCTION_NAME = "posit_top"
SYNTHESIS_FILE = "posit_top.cpp"
TESTBENCH_FILE = "posit_test.cpp"


def main():
    """Main function to run the Vitis HLS flow."""
    # --- Paths ---
    # The script is expected to be run from the 'tests/hls' directory
    cwd = os.getcwd()
    project_root = os.path.abspath(os.path.join(cwd, '..', '..'))
    include_path = os.path.join(project_root, 'include')
    workspace_path = os.path.join(cwd, "vitis_workspace")
    component_path = os.path.join(workspace_path, COMPONENT_NAME)

    # --- Vitis Setup ---
    client = vitis.create_client()
    client.set_workspace(path=workspace_path)

    # Clean up previous runs
    if os.path.exists(component_path):
        print(f"--- Deleting existing component {COMPONENT_NAME} ---")
        client.delete_component(name=COMPONENT_NAME)

    # Create HLS component
    print(f"--- Creating HLS component {COMPONENT_NAME} ---")
    comp = client.create_hls_component(name=COMPONENT_NAME,
                                       template='empty_hls_component')

    # Configure the component
    print("--- Configuring component ---")
    cfg_file = client.get_config_file(path=component_path + "/hls_config.cfg")
    cfg_file.set_value(key='part', value=PART)
    cfg_file.set_value(section='hls', key='clock', value=CLOCK_PERIOD_NS)
    cfg_file.set_value(section='hls', key='flow_target', value='vivado')

    # Set source, testbench, and top function
    cfg_file.set_value(section='hls', key='syn.top', value=TOP_FUNCTION_NAME)
    cfg_file.add_values(section='hls', key='syn.file', values=[os.path.join(cwd, SYNTHESIS_FILE)])
    cfg_file.add_values(section='hls', key='tb.file', values=[os.path.join(cwd, TESTBENCH_FILE)])

    # Add include paths for headers (linalg, hls_numerics)
    cflags = f"-I{include_path} -D__SYNTHESIS__ -Xclang -fnative-half-type -Xclang -fallow-half-arguments-and-returns"
    cfg_file.set_value(section='hls', key='syn.cflags', value=cflags)
    cfg_file.set_value(section='hls', key='tb.cflags', value=cflags)

    # --- Run Simulation ---
    print("--- Running C-Simulation ---")
    try:
        # Using run() which is the standard for Vitis 2023.1+
        comp.run(operation='C_SIMULATION')
    except Exception as e:
        print(f"ERROR: C-Simulation failed: {e}", file=sys.stderr)
        client.close()
        sys.exit(1)

    print("--- C-Simulation successful ---")

    # --- Clean up ---
    client.close()
    print("--- Vitis client closed ---")

if __name__ == "__main__":
    main()