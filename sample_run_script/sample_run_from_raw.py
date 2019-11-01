from main.main import run

input_xml = '/Users/givanna/input.xml'
config_xml = '/Users/givanna/config.xml'
out = '/Users/givanna/out'
gating = None
prog = None


run(config_xml=config_xml, input_xml=input_xml, output_dir=out, log_dir=out, gating_file=gating, program_state_dir=prog,
    sort_output=True, normalise=False)

