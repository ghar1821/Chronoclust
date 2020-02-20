# Code for reproducing manuscript result

Hereby contains all the code (in addition to chronoclust) needed to reproduce the result in our manuscript.

## Cluster labelling
The codes in the this folder are used to match each cluster with a ground truth label.
It works based on distance between cluster centroids.
Each cluster will be labelled with the label of the ground truth which centroid lies the closest to the cluster's centroid.

For this method to work, you first need to calculate the centroid of each of your ground truth clusters.
An example on how to do so is provided in ``calculate_gating_centroid.py`` script.

Thereafter, you can run ``label_cluster_points.py`` script to add the estimated label discussed above as ``PredictedLabel``.

## Cluster Evaluator
The folder contains all the codes used to evaluate how good the clustering produced by chronoclust is.
The script name describe the evaluation method therein.

``evaluator.py`` script is the main script to run all the methods.

## Config
Directory containing sample config file for cluster labelling and evaluator.