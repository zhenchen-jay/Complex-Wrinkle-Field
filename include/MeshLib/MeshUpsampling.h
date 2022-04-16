#pragma once
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "MeshConnectivity.h"
#include "../../dep/SecStencils/Mesh.h"

void meshUpSampling(Eigen::MatrixXd V, Eigen::MatrixXi F, Eigen::MatrixXd &NV, Eigen::MatrixXi &NF, int numSubdivs, Eigen::SparseMatrix<double> *mat = NULL, std::vector<int>* facemap = NULL, std::vector<std::pair<int, Eigen::Vector3d>> *bary = NULL);

void loopUpsampling(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, Eigen::MatrixXd& NV, Eigen::MatrixXi& NF, int upsampledTimes, Eigen::SparseMatrix<double> *loopMat = NULL);
