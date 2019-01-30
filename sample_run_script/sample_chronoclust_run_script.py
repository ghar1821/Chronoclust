"""
A sample script on how to run ChronoClust from another python script.

"""
import sys

from chronoclust import chronoclust


config_xml = 'sample_run_script/config/config.xml'
input_xml = 'sample_run_script/config/input.xml'
output = 'sample_run_script/output'
gating_centroid_file = 'synthetic_dataset/gating_coarse/gating_centroids.csv'
chronoclust.run(config_xml=config_xml, input_xml=input_xml, log_dir=output, output_dir=output,
                gating_file=gating_centroid_file)

