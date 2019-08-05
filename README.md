# ChronoClust

A clustering algorithm that will perform clustering on each of a time-series of discrete (not a data stream... yet) datasets, and explicitly track the evolution of clusters over time. 

If you use the ChronoClust algorithm, please cite the associated publication:

>ChronoClust: Density-based clustering and cluster tracking in high-dimensional time-series data. Givanna H. Putri, Mark N. Read, Irena Koprinska, Deeksha Singh, Uwe Rohm, Thomas M. Ashhurst, Nick J.C. King. Accepted to Knowledge Based Systems, 2019.

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

Afterwards, you need to just import chronoclust, and run it like below.
```
import chronoclust

basedir = '/Users/example/Documents/workdir'
in_file = basedir + '/input.xml'
config_file = basedir + '/config.xml'
chronoclust.run(config_file, in_file, basedir, basedir)
```
In the above script, the input and config files are stored in **/Users/example/Documents/workdir** and the clustering result will also be written into the same directory. You can store the config/input files in different directories (doesn't matter). 

## Who do I talk to?
* Givanna Putri ghar1821@uni.sydney.edu.au
