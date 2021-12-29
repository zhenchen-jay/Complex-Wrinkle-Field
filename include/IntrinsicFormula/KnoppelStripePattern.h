#pragma once
#include "../MeshLib/MeshConnectivity.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace IntrinsicFormula
{
    void computeMatrixA(const MeshConnectivity &mesh, const Eigen::MatrixXd &halfEdgeW, const Eigen::VectorXd& faceArea, const Eigen::MatrixXd& cotEntries, const int nverts, Eigen::SparseMatrix<double>& A);
    void roundVertexZvalsFromHalfEdgeOmega(const MeshConnectivity &mesh, const Eigen::MatrixXd &halfEdgeW, const Eigen::VectorXd& faceArea, const Eigen::MatrixXd& cotEntries, const int nverts, std::vector<std::complex<double>>& zvals);
    void getUpsamplingTheta(const MeshConnectivity &mesh, const Eigen::MatrixXd& halfEdgeW, const std::vector<std::complex<double>>& zvals, const std::vector<std::pair<int, Eigen::Vector3d>> &bary, Eigen::VectorXd upTheta);
    double lArg(const long &n, const Eigen::Vector3d& bary);
    void testRoundingEnergy(const MeshConnectivity &mesh, const Eigen::MatrixXd &halfEdgeW, const Eigen::VectorXd& faceArea, const Eigen::MatrixXd& cotEntries, const int nverts, std::vector<std::complex<double>> zvals);
}