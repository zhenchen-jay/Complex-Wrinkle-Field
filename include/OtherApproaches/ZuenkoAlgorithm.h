#pragma once
#include <Eigen/Dense>
#include "../MeshLib/MeshConnectivity.h"

/*
 * In this file, I implemented the algorithm mentioned by [Zuenko et al] "Wrinkles, Folds, Creases, Buckles: Small-Scale Surface Deformations as Periodic Functions on 3D Meshes"
 */

namespace ZuenkoAlg
{
    void spherigonSmoothing(const Eigen::MatrixXd& pos, const MeshConnectivity& mesh, const Eigen::MatrixXd& vertN, const std::vector<std::pair<int, Eigen::Vector3d>> bary, Eigen::MatrixXd& upPos, Eigen::MatrixXd& upN);
}