# Lomax-delegate-racing-for-survival-analysis-with-competing-risks
Demo code and data for the corresponding NIPS 2008 paper.

The R code for Lomax delegate racing survival analysis with competing risks has been tested on R version 3.4.3 on Ubuntu 16.04. Follow the steps below for analysis of the synthetic data sets.

1. Download everything into a folder. Open an R console, set the working directory to the download folder by setwd("/path/to/folder")

2. install the following packages:
library(survival), 
library(cmprsk), 
library(threg), 
library(pec), 
library(riskRegression), 
library(coda), 
library(Rcpp), 
library(RcppArmadillo), 
library(doParallel), 
library(foreach), 
library(foreign), 
library(nnet), 
library(ggplot2), 
library(reshape2), 
library(MASS), 
library(fields), 
library(pracma).

3. install and load the package RNG by
install.packages("RNG_1.1.tar.gz",type="source"); 
library(RNG)

4. Copy and paste everything in "democode.r" to an R console.
