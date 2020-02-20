# Load Reticulate to run Python code in R
install.packages("reticulate")
library(reticulate)
library(rstudioapi)

# If you decided to use Python virtual environment (highly recommended), then load the environment name.
# Make sure you have installed chronoclust API in your virtual environment.
# You can either setup the virtual environment via 
# 1. Anaconda: https://docs.anaconda.com/anaconda/navigator/getting-started/
# 2. PipEnv: https://docs.python-guide.org/dev/virtualenvs/
# Chronoclust packages are available from https://github.com/ghar1821/Chronoclust
use_condaenv("chronoclust-public")

# Set working directory
dirname(rstudioapi::getActiveDocumentContext()$path)
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
getwd()

# Setup Chronoclust
chronoclust <- import("chronoclust")  # Import Chronoclust API

# Assuming the data are stored within test_data directory within the 
# script's directory (working directory setup on line 16).
data_files <- c('test_data/synthetic_d0.csv.gz',
                'test_data/synthetic_d1.csv.gz',
                'test_data/synthetic_d2.csv.gz',
                'test_data/synthetic_d3.csv.gz',
                'test_data/synthetic_d4.csv.gz')

config <- list(beta= 0.2,
               delta= 0.05,
               epsilon= 0.03,
               lambda= 2,
               k= 4,
               mu= 0.01,
               pi= 3,
               omicron= 0.00000435,
               upsilon= 6.5)

# This is the directory (within your current working directory) where Chronoclust will output clustering result.
time_now <- Sys.time()
time_now <- gsub(":", "-", time_now)
time_now <- gsub(" ", "_", time_now)
outdir <- paste0("chronoclust", "_", time_now)

# run chronoclust with default config built in to chronoclust
chronoclust$app$run(data=data_files, 
                    output_directory=outdir)

# run chronoclust with pre-specified config above
chronoclust$app$run(data=data_files, 
                    output_directory=outdir, 
                    param_beta=config[['beta']], 
                    param_delta=config[['delta']],
                    param_epsilon=config[['epsilon']], 
                    param_lambda=config[['lambda']], 
                    param_k=config[['k']], 
                    param_mu=config[['mu']],
                    param_pi=config[['pi']], 
                    param_omicron=config[['omicron']], 
                    param_upsilon=config[['upsilon']])