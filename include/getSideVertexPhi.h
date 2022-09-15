#pragma once
#include "CommonTools.h"

void getSideVertexPhi(const Eigen::MatrixXd& V, const MeshConnectivity& mesh, const Eigen::VectorXd& edgeOmega, const Eigen::VectorXd& vertPhi, Eigen::MatrixXd& upV, const MeshConnectivity& upMesh, Eigen::VectorXd& upPhi, int upLevel, int interpType = 0); // 0 for standard interp, 1 for cubric, 2 for Wojtan

void getSideVertexPhi(const Eigen::MatrixXd& V, const MeshConnectivity& mesh, const Eigen::MatrixXd& vertOmega, const Eigen::VectorXd& vertPhi, Eigen::MatrixXd& upV, const MeshConnectivity& upMesh, Eigen::VectorXd& upPhi, int upLevel, int interpType = 0); // 0 for standard interp, 1 for cubric, 2 for Wojtan