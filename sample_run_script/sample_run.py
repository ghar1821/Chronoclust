from chronoclust.main import main

###############################################
# Using manual specification to pass parameters
###############################################
input_xml = 'sample_run_script/config/input.xml'
config_xml = 'sample_run_script/config/config.xml'
out = 'output'
gating = None
prog = None
###############################################

main.run(config_xml=config_xml, input_xml=input_xml, output_dir=out, log_dir=out, gating_file=gating, program_state_dir=prog,
    sort_output=True, normalise=True)

