#pragma once
#include <Eigen/Dense>
#include <Eigen/Sparse>

void meshUpSampling(Eigen::MatrixXd V, Eigen::MatrixXi F, Eigen::MatrixXd &NV, Eigen::MatrixXi &NF, int numSubdivs, Eigen::SparseMatrix<double> *mat = NULL, std::vector<int>* facemap = NULL, std::vector<std::pair<int, Eigen::Vector3d>> *bary = NULL);

void loopUpsampling(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, Eigen::MatrixXd& NV, Eigen::MatrixXi& NF, int upsampledTimes, Eigen::SparseMatrix<double> *loopMat = NULL);


void upsampleMeshZvals(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, const Eigen::MatrixXd& w, const std::vector<std::complex<double>>& zvals, Eigen::MatrixXd& NV, Eigen::MatrixXi& NF, Eigen::MatrixXd& upsampledW, std::vector<std::complex<double>>& upsampledZvals, int upsampledTimes = 2);