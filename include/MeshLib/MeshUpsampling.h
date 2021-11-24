#pragma once
#include <Eigen/Dense>
#include <Eigen/Sparse>

void meshUpSampling(Eigen::MatrixXd V, Eigen::MatrixXi F, Eigen::MatrixXd &NV, Eigen::MatrixXi &NF, int numSubdivs, Eigen::SparseMatrix<double> *mat = NULL, std::vector<int>* facemap = NULL);