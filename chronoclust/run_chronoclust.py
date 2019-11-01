import argparse
from main import run

#############################################
# Using argument parser to pass parameters
#############################################
parser = argparse.ArgumentParser(description="ChronoClust")
# Optional arguments
parser.add_argument("-g", "--g", type=str, help="Gating file location")
parser.add_argument("-p", "--programstate", type=str, help="Directory containing the program's state to resume from")

# Mandatory arguments
required_args = parser.add_argument_group('required named arguments')
required_args.add_argument('-i', '--i', type=str, help='Location of xml containing mapping of all input files', required=True)
required_args.add_argument("-c", "--c", type=str, help="Location of xml containing the parameter settings", required=True)
required_args.add_argument("-o", "--o", type=str, help="Directory for result output", required=True)
required_args.add_argument("-l", "--l", type=str, help="Directory for log file", required=True)
args = parser.parse_args()

input_xml = args.i
config_xml = args.c
out = args.o
gating = args.g
prog = args.programstate
#############################################

###############################################
# Using manual specification to pass parameters
###############################################
# input_xml = '/Users/givanna/Documents/phd/workdir/input.xml'
# config_xml = '/Users/givanna/Documents/phd/workdir/config.xml'
# out = '/Users/givanna/Documents/phd/workdir/out'
# gating = None
# prog = None
###############################################

run(config_xml=config_xml, input_xml=input_xml, output_dir=out, log_dir=out, gating_file=gating, program_state_dir=prog,
    sort_output=True, normalise=False)

