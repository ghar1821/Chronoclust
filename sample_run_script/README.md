# Sample run script

This folder contains some sample script on how to run ChronoClust.

sample_chronoclust_run_script.py shows an example on how to run ChronoClust and what is needed. Due to import difficulty of python, in order for this script to run, you need to copy it to the main directory containing chronoclust, cluster_evaluator, cluster_labelling, synthetic dataset, and sample_run_script folders.

In order to label and evaluate chronoclust's clustering, you need to run the scripts in cluster_labelling and cluster_evaluator folder.

## Cluster_labelling
Files:
1) label_cluster_points.py: this will map manual gating label to chronoclust's cluster. To run the file, you need to supply a json config written in manner similar to labelling_config.json file in config folder.
2) label_lineage.py: this will map chronoclust's cluster tracking label to manual gating label. To run the file, you need to supply a json config written in manner similar to lineage_labelling_config.json file in config folder.

## Cluster_evaluator
There are many scripts in here, but they are all run under evaluator.py.
evaluator.py requires a config json file written in similar manner as evaluator_config.json file in config folder.

## Config
This folder contains all the config files required to run ChronoClust and all the labelling and evaluator to produce the result in manuscript submitted to Knowledge Based Systems journal.

Files:
1) config.xml: config for ChronoClust
2) input.xml: input dataset for ChronoClust
3) labelling_config.json: labelling config to label ChronoClust's result. This allow ChronoClust's cluster to be compared against manual gating label.
4) lineage_labelling_config.json: labelling config to label ChronoClust's cluster lineage. This allow ChronoClust's cluster transition to be compared against those that are biologically sensible.
5) evaluator_config.json: config file to run evaluator for ChronoClust

