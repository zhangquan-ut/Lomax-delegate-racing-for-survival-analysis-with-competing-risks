# Lomax-delegate-racing-for-survival-analysis-with-competing-risks
Demo code and data for the corresponding NIPS 2018 paper.

The MAP estimations by pytorch for the synthetic datasets are provided. See MAP_pytorch.py for detail. But we highly recommend the implementation by Gibbs sampler, because it is not only more numerically stable, but also giving an explicit criterion of pruning redundant model capacity (see step 1 and 8 of Appendix B), which is our most significant contribution of interpretable nonlinearity by Bayesian nonparametrics.

The R code of Gibbs sampler for Lomax delegate racing survival analysis with competing risks has been tested on R version 3.4.3 on Ubuntu 16.04. Follow the steps below for analysis of the synthetic data sets.

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


