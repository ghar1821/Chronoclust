# ChronoClust

A clustering algorithm that will perform clustering on each of a time-series of discrete (not a data stream... yet) datasets, and explicitly track the evolution of clusters over time.

If you use the ChronoClust algorithm, please cite the associated publication:

>Putri, Givanna H., Mark N. Read, Irena Koprinska, Deeksha Singh, Uwe Röhm, Thomas M. Ashhurst, and Nicholas JC King. "ChronoClust: Density-based clustering and cluster tracking in high-dimensional time-series data." Knowledge-Based Systems 174 (2019): 9-26

DOI: https://doi.org/10.1016/j.knosys.2019.02.018

To run the project you will require the following packages for python 3:
1. pandas
2. numpy
3. scipy
4. scikit-learn
5. pickle
6. tqdm
7. deprecation

## How do I use chronoclust?
Chronoclust is available on:
1. Anaconda: https://anaconda.org/givanna/chronoclust
2. Pypi: https://pypi.org/project/chronoclust/

Download them or just use the stable source code here: https://github.com/ghar1821/Chronoclust/releases

Secondly, you need to have a bunch of data (doh!), and create a xml file containing the location of the data files. The files have to be in csv format compressed as gzip (extension csv.gz).
Each file need to be associated with a time point - day 1, 2, 3 etc.
Have a look at sample_run_script/config/input.xml for an example.

Thirdly, you need to define the config Chronoclust will be run with. Define the configs in a xml file.
Look at sample_run_script/config/config.xml for an example, and the published article linked above for information on what parameters are there and how are they used in Chronoclust.

### Running Chronoclust in Python
After setting everything up, all you need to is just import chronoclust, and run it like below.
```
import chronoclust

basedir = '/Users/example/Documents/workdir'
in_file = basedir + '/input.xml'
config_file = basedir + '/config.xml'
chronoclust.run(config_file, in_file, basedir, basedir)
```
In the above script, the input and config files are stored in **/Users/example/Documents/workdir** and the clustering result will also be written into the same directory. You can store the config/input files in different directories (doesn't matter).

If you have a _gated_ or _labelled_ data and want to see how effective Chronoclust is, you can pass a ``gating_file`` argument to the run command above and supply it with a csv file containing the centroid of each population or cluster.
Please look at ``sample_run_script/run_chronoclust.py`` for example.

### Running Chronoclust in R
Well, Chronoclust is not written in R but thanks to the fabulous people who write [reticulate](https://rstudio.github.io/reticulate/) package, it is now posisble to run Python code in R.

You still need to download Chronoclust API from Anaconda or Pypi and setup the xml files.
Thereafter, have a look at ``sample_run_script/run_chronoclust.R`` for example on how to run Chronoclust in R.

## Where to start with the parameters?
You can pretty much start with any value for any parameters, but to at least get some kind of clustering, I recommend you start with setting ``pi`` to be the dimensionality of your dataset (number of columns or markers in the dataset).
This gives Chronoclust the flexibility in creating the Microclusters.

Once you have some kind of clustering going, then you can start playing around with others.
I will generally start by looking at the number of clusters produced and tuning ``epsilon``.
If there are too many clusters (overclustering), I'll tune ``epsilon`` down (make it smaller).
Otherwise, make it a bit bigger.
**Do note that a small reduction/increment in ``epsilon`` can dramatically alter the clustering produced.**
After it looks sort of right, then you can move on to ``beta``, ``mu``, and/or ``upsilon``.

If you find that the clusters are too wide or big (has too big of a reach), then it could very well be that you have set the requirement for the MCs to be too _lenient_, i.e. the parameter combination allows small MCs to be formed and included in the final clustering.
What you can do here is make ``beta`` and ``mu`` bigger so small MCs are treated as outliers and not included in final clustering.
You can also make ``upsilon`` smaller, which will split your big wide cluster into few smaller ones.

## Data files
The synthetic dataset and corresponding gating is available under ``synthetic_dataset`` folder.

The WNV dataset and gating are available from FlowRepository [repository FR-FCM-Z285](https://flowrepository.org/id/FR-FCM-Z285).

## Providing gated files for ChronoClust
For the clustering result to be meaningful, there must be some sort of labelling attached to each cluster produced by ChronoClust.
You can do this by manually annotating either the result file or the file containing the points belonging to each cluster.
However, there are times (such as when we prepare the result for our paper) when a ``ground truth`` is already available.
In this case, you can automatically get ChronoClust to label the clusters based on the ground truth label.

To do this, you need to first find the centroid of each ``cell population`` or grouping in your ground truth.
This can easily be done by just taking the mean of the data points for each population/grouping.
You can either do this yourself or just use the script in ``cluster_labelling/calculate_gating_centroid.py`` (this is the script we used for our paper).
It shall produce the file similar to the gating centroid found in synthetic dataset (``synthetic_dataset/gating_fine/gating_centroids.csv``) or the WNV dataset.
Thereafter, you need to pass this file (just the location) to ChronoClust as ``gating_file`` parameter.
ChronoClust will then attempt to match each cluster to the nearest population/grouping.
For more information on how it does this, please download the paper.

If you do this, the result file will have an extra column call ``predicted_label``, the cluster label based on supplied ground truth.
Only after you have this then you can label each data points in ``cluster_points`` file (generated by ChronoClust) with their predicted label (based on ``predicted_label`` above) and true label (given by ground truth).
