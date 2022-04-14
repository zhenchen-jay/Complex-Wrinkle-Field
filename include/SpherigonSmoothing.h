#pragma once
#include "CommonTools.h"

void spherigonSmoothing(const Eigen::MatrixXd& pos, const MeshConnectivity& mesh, const Eigen::MatrixXd& vertN, const std::vector<std::pair<int, Eigen::Vector3d>> bary, Eigen::MatrixXd& upPos, Eigen::MatrixXd& upN);