source("renv/activate.R")
library(tidyverse)
source("shiny-app/R/data_loader.R")
train <- load_train_fd003(base_path = "../data/raw/CMaps")