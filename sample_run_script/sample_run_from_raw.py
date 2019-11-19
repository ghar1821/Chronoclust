from main.main import run
from pathlib import Path

home_dir = Path("/Users/givanna/workdir")
input_xml = home_dir / Path("input.xml")
config_xml = home_dir / Path("config.xml")
out = home_dir / Path("output")
gating = None
prog = None

run(config_xml=config_xml, input_xml=input_xml, output_dir=out, log_dir=out, gating_file=gating, program_state_dir=prog,
    sort_output=True, normalise=True)

