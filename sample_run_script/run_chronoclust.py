"""
A sample script on how to run ChronoClust from another python script.
Please note that the argument gating_file is OPTIONAL. Only provide this if you have a labelled/gated dataset and want to see how effective chronoclust is.
The file must contain the "centroid" of each of your cluster / population.
Have a look at the synthetic_dataset/gating_fine/gating_centroids.csv for example
"""

import chronoclust


config_xml = 'sample_run_script/config/config.xml'
input_xml = 'sample_run_script/config/input.xml'
output = 'sample_run_script/output'
gating_centroid_file = 'synthetic_dataset/gating_fine/gating_centroids.csv'
chronoclust.run(config_xml=config_xml, input_xml=input_xml, log_dir=output, output_dir=output,
                gating_file=gating_centroid_file)

