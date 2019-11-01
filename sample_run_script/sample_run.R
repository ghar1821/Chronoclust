# Load Reticulate to run Python code in R
install.packages("reticulate")
library(reticulate)

# If you decided to use Python virtual environment (highly recommended), then load the environment name.
# Make sure you have installed chronoclust API in your virtual environment.
# You can either setup the virtual environment via 
# 1. Anaconda: https://docs.anaconda.com/anaconda/navigator/getting-started/
# 2. PipEnv: https://docs.python-guide.org/dev/virtualenvs/
# Chronoclust packages are available from both PyPi. Refer to README for link
use_virtualenv("chronoclust")

# OPTIONAL: Use this to manually set your working directory
setwd("/Users/givanna")
PrimaryDirectory <- getwd()

# Setup Chronoclust
chronoclust <- import("chronoclust")  # Import Chronoclust API

# The following refers to the place where you place the config.xml and input.xml file
configfile <- paste(PrimaryDirectory, "config.xml", sep = "/")
inputfile <- paste(PrimaryDirectory, "input.xml", sep = "/")

# This is the directory (within your current working directory) where Chronoclust will output clustering result.
time_now <- Sys.time()
time_now <- gsub(":", "-", time_now)
time_now <- gsub(" ", "_", time_now)
outdir <- paste0("chronoclust", "_", time_now)

# run chronoclust
chronoclust$chronoclust$run(config_xml = configfile, input_xml = inputfile, log_dir = outdir, output_dir = outdir)

