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
5. tqdm
6. numba

## How do I use chronoclust?

You can install ChronoClust using Pip (pip install Chronoclust): https://pypi.org/project/Chronoclust/

OR using the setup.py script from this repo directly.

The following instructions assume you have python 3 installed. If you haven't, please install python 3.6 or 3.7.
Visit https://www.python.org/downloads/ for instruction.

1. Download this repo to a local folder. You can do this by clicking "clone or download" button and "Download ZIP".
2. Unzip the downloaded repo file in step (1).
3. Open the Terminal app.
4. Change your active directory to the directory containing the unzipped files in step (2). To do this type ``cd `` (don't miss the space!) then drag the unzipped directory to the terminal, then press enter.
5. Type ``python3 setup.py install`` and press enter. This shall install chronoclust into your computer.
6. Type ``cd sample_run_script`` and press enter. This shall change your active directory to where the sample script is stored.
7. Type ``python3 sample_run.py`` and press enter. Chronoclust will execute on a synthetic dataset.

It is highly recommended to learn how to use environment manager such as Miniconda (https://docs.conda.io/en/latest/miniconda.html) prior to installing Chronoclust. 
However, if this is too much, stick with instruction above.


### Running Chronoclust
``sample_run.py`` file stored within ``sample_run_script`` folder shows how to run Chronoclust.

To run Chronoclust in R using reticulate, please see Spectre package: https://github.com/sydneycytometry/Spectre

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
You can either do this yourself or just use the script in ``helper_code_for_manuscript/cluster_labelling/calculate_gating_centroid.py`` (this is the script we used for our paper).
It shall produce the file similar to the gating centroid found in synthetic dataset (``synthetic_dataset/gating_fine/gating_centroids.csv``) or the WNV dataset.
Please note that if you want to do this, make sure you format your ground truth data in similar format as ours (label is named ``PopName`` at least).
Consult ``synthetic_dataset/gating_fine/synthetic_d0.csv`` for example.
Thereafter, you need to pass this file (just the location) to ChronoClust as ``gating_file`` parameter.
ChronoClust will then attempt to match each cluster to the nearest population/grouping.
For more information on how it does this, please download the paper.

If you do this, the result file will have an extra column call ``predicted_label``, the cluster label based on supplied ground truth.
Only after you have this then you can label each data points in ``cluster_points`` file (generated by ChronoClust) with their predicted label (based on ``predicted_label`` above) and true label (given by ground truth).

## Issues/Bugs/Features request
We're all only humans and we do make mistakes. 
Hence please forgive me if you find some bugs/issues in the code.
I will greatly appreciate it if you please kindly inform me of them by either sending me an email (see the paper for my contact details) or opening an issue ticket.
I'll try my best to address it as soon as possible.

In addition, if you have a feature request please do the same. 

## Code for reproducing result in the paper
In addition to using chronoclust, there exists other codes used to generate our manuscript.
You can find all them under ``helper_code_for_manuscript``.
See separate README in the folder for more details.
