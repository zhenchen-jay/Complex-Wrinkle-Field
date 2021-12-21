#pragma once
#include "../MeshLib/MeshConnectivity.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace IntrinsicFormula
{
    void computeMatriA(const MeshConnectivity &mesh, const Eigen::MatrixXd &edgew, const Eigen::VectorXd& faceArea, const Eigen::MatrixXd& cotEntries, const int nverts, Eigen::SparseMatrix<double>& A);
    void roundVertexZvalsFromHalfEdgeOmega(const MeshConnectivity &mesh, const Eigen::MatrixXd &edgew, const Eigen::VectorXd& faceArea, const Eigen::MatrixXd& cotEntries, const int nverts, std::vector<std::complex<double>>& zvals);
    void testRoundingEnergy(const MeshConnectivity &mesh, const Eigen::MatrixXd &edgew, const Eigen::VectorXd& faceArea, const Eigen::MatrixXd& cotEntries, const int nverts, std::vector<std::complex<double>> zvals);
}