import vitis
import os
import sys
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
import subprocess, re
import shutil

from parse_hls_xml import parse_xml_reports

def vitis_version_pkg():
    try:
        return version("vitis")  # e.g. "2024.2" or "2025.1"
    except PackageNotFoundError:
        return None

def vitis_version_env():
    xv = os.environ.get("XILINX_VITIS")  # e.g. /tools/Xilinx/Vitis/2024.2
    return Path(xv).name if xv else None

def vitis_version_cli():
    out = subprocess.run(["vitis", "-v"], capture_output=True, text=True).stdout
    m = re.search(r"(\d{4}\.\d)", out)
    return m.group(1) if m else out.strip()  # fallback to full banner if pattern changes

def run_vitis(workspace_path, include_path, part, clock_period_ns, component_name, top_function_name, synthesis_file, testbench_file, ii_threshold=100, depth_threshold=100):

    vitis_version = vitis_version_env()

    # --- Vitis Setup ---
    client = vitis.create_client()
    if os.path.exists(workspace_path):
        print(f"--- Deleting existing workspace {workspace_path} ---")
        shutil.rmtree(workspace_path)
    
    client.set_workspace(path=workspace_path)

    cwd = os.getcwd()
    component_path = os.path.join(workspace_path, component_name)
    
    # Clean up previous runs
    if os.path.exists(component_path):
        print(f"--- Deleting existing component {component_name} ---")
        client.delete_component(name=component_name)

    # Create HLS component
    print(f"--- Creating HLS component {component_name} ---")
    comp = client.create_hls_component(name=component_name,
                                       cfg_file=['hls_config.cfg'],
                                       template='empty_hls_component')

    # Configure the component
    print("--- Configuring component ---")
    cfg_file = client.get_config_file(path=component_path + "/hls_config.cfg")
    cfg_file.set_value(key='part', value=part)
    cfg_file.set_value(section='hls', key='clock', value=clock_period_ns)
    cfg_file.set_value(section='hls', key='flow_target', value='vivado')
    cfg_file.set_value(section='hls', key='package.output.syn', value='false')
    cfg_file.set_value(section='hls', key='package.output.format', value='ip_catalog')

    # Set source, testbench, and top function
    cfg_file.set_value(section='hls', key='syn.top', value=top_function_name)
    cfg_file.add_values(section='hls', key='syn.file', values=[os.path.join(cwd, synthesis_file)])
    cfg_file.add_values(section='hls', key='tb.file', values=[os.path.join(cwd, testbench_file)])

    # Add include paths for headers (project includes, Eigen)
    # Note: Eigen is typically installed at /usr/include/eigen3 in Ubuntu images
    eigen_inc = "/usr/include/eigen3"
    cflags = f"-I{include_path} -I{eigen_inc} -DUSE_VITIS"
    # Allow tests to inject extra compilation flags via environment
    extra_cflags = os.environ.get('HLS_EXTRA_CFLAGS', '')
    if extra_cflags:
        print(f"--- Adding extra cflags from HLS_EXTRA_CFLAGS: {extra_cflags} ---")
        cflags += f" {extra_cflags}"
    #if vitis_version > "2023.2":
    #    cflags += "-Xclang -fnative-half-type -Xclang -fallow-half-arguments-and-returns"
        
    cfg_file.set_value(section='hls', key='syn.cflags', value=cflags)
    cfg_file.set_value(section='hls', key='tb.cflags', value=cflags)
        
    # --- Run Simulation ---
    print("--- Running C-Simulation ---")
    try:
        # Using run() which is the standard for Vitis 2023.1+
        if vitis_version > "2023.2":
            comp.run(operation='C_SIMULATION')
        else:
            comp.execute(operation='C_SIMULATION')
    except Exception as e:
        print(f"ERROR: C-Simulation failed: {e}")
        client.close()
        vitis.dispose()
        sys.exit(1)
    print("--- C-Simulation successful ---")
    
    print("--- Running Synthesis ---")
    try:
        # Using run() which is the standard for Vitis 2023.1+
        if vitis_version > "2023.2":
            comp.run(operation='SYNTHESIS')
        else:
            comp.execute(operation='SYNTHESIS')
    except Exception as e:
        print(f"ERROR: Synthesis failed: {e}")
        client.close()
        vitis.dispose()
        sys.exit(1)
    print("--- Synthesis successful ---")
    
    print("--- Check pipeline ---")
    ip_path = component_path + "/" + component_name
    csynth_report, impl_report = parse_xml_reports(ip_path)

    ii = csynth_report['II']
    depth = csynth_report['depth']
    
    print("ii: ", ii ," max: ", ii_threshold)
    print("depth: ", depth ," max: ", depth_threshold)
    if ii > ii_threshold or depth > depth_threshold:
        print(f"ERROR: Bad pipelining")
        client.close()
        vitis.dispose()
        sys.exit(1)
    print("--- Pipeline ok ---")
    """
    print("--- Running Implementation ---")
    try:
        # Using run() which is the standard for Vitis 2023.1+
        if vitis_version > "2023.2":
            comp.run(operation='IMPLEMENTATION')
        else:
            comp.execute(operation='IMPLEMENTATION')
    except Exception as e:
        print(f"ERROR: Implementation failed: {e}", file=sys.stderr)
        client.close()
        vitis.dispose()
        sys.exit(1)

    print("--- Implementation successful ---")
    
    print("--- Running Co-simulation ---")
    try:
        # Using run() which is the standard for Vitis 2023.1+
        if vitis_version > "2023.2":
            comp.run(operation='CO_SIMULATION')
        else:
            comp.execute(operation='CO_SIMULATION')
    except Exception as e:
        print(f"ERROR: Co-Simulation failed: {e}", file=sys.stderr)
        client.close()
        vitis.dispose()
        sys.exit(1)

    print("--- Co-simulation successful ---")
    """
    # --- Clean up ---
    client.close()
    vitis.dispose()
    print("--- Vitis client closed ---")
