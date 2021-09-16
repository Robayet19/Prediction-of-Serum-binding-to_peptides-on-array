# Serum-binding-prediction-to peptides
This NN model predicts serum Antibody binding to 123,000 random peptide sequences of various lengths. More specifically, this script is written to evaluate the performance of the NN model as a function of number of peptide sequences used to train that model. Here, one can generate an individual NN model for each infected or uninfected serum sample.

The script was initially written by Alexander Taguchi to predict protein binding to these random peptide sequences(Taguchi, A. T., Boyd, J., Diehnelt, C. W., Legutki, J. B., Zhao, Z-G., and Woodbury, N. W. (2020) Combinatorial Science, 22 (10), 500-508 DOI: 10.1021/acscombsci.0c00003). However, the concept/NN architecture was implemented/modified by Robayet Chowdhury to explore the sequence space (123,000 or 10^5 peptides) used in this study.

