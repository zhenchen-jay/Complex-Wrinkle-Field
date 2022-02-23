#pragma once
#include "../MeshLib/MeshConnectivity.h"
#include "../CommonTools.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace IntrinsicFormula
{
    void computeMatrixA(const MeshConnectivity &mesh, const Eigen::MatrixXd &halfEdgeW, const Eigen::VectorXd& faceArea, const Eigen::MatrixXd& cotEntries, const int nverts, Eigen::SparseMatrix<double>& A);
    void computeMatrixAGivenMag(const MeshConnectivity &mesh, const Eigen::MatrixXd &halfEdgeW, const Eigen::VectorXd& vertAmp, const Eigen::VectorXd& faceArea, const Eigen::MatrixXd& cotEntries, const int nverts, Eigen::SparseMatrix<double>& A);


    double KnoppelEnergy(const MeshConnectivity& mesh, const Eigen::MatrixXd& halfEdgeW, const Eigen::VectorXd& faceArea, const Eigen::MatrixXd& cotEntries, const std::vector<std::complex<double>>& zvals, Eigen::VectorXd* deriv, std::vector<Eigen::Triplet<double>>* hess);
    double KnoppelEnergyGivenMag(const MeshConnectivity& mesh, const Eigen::MatrixXd& halfEdgeW, const Eigen::VectorXd& vertAmp, const Eigen::VectorXd& faceArea, const Eigen::MatrixXd& cotEntries, const std::vector<std::complex<double>>& zvals, Eigen::VectorXd* deriv, std::vector<Eigen::Triplet<double>>* hess);

    double KnoppelEnergyFor2DVertexOmegaPerEdge(const Eigen::MatrixXd& pos, const MeshConnectivity& mesh, const Eigen::VectorXd& faceArea, const Eigen::MatrixXd& cotEntries, const std::vector<std::complex<double>>& zvals, const Eigen::MatrixXd& vertexOmega, const double edgeWeight, int eid, Eigen::Matrix<double, 8, 1> *deriv, Eigen::Matrix<double, 8, 8>* hess, bool isProj = false);  // size(vertexOmega) = nverts, 2

    double KnoppelEnergyFor2DVertexOmega(const Eigen::MatrixXd& pos, const MeshConnectivity& mesh, const Eigen::VectorXd& faceArea, const Eigen::MatrixXd& cotEntries, const std::vector<std::complex<double>>& zvals, const Eigen::MatrixXd& vertexOmega, Eigen::VectorXd* deriv, std::vector<Eigen::Triplet<double>>* hess, bool isProj = false);  // size(vertexOmega) = nverts, 2

    void roundVertexZvalsFromHalfEdgeOmega(const MeshConnectivity &mesh, const Eigen::MatrixXd &halfEdgeW, const Eigen::VectorXd& faceArea, const Eigen::MatrixXd& cotEntries, const int nverts, std::vector<std::complex<double>>& zvals);
    void roundVertexZvalsFromHalfEdgeOmegaVertexMag(const MeshConnectivity &mesh, const Eigen::MatrixXd& halfEdgeW, const Eigen::VectorXd& vertAmp, const Eigen::VectorXd& faceArea, const Eigen::MatrixXd& cotEntries, const int nverts, std::vector<std::complex<double>>& zvals);

    void getUpsamplingTheta(const MeshConnectivity &mesh, const Eigen::MatrixXd& halfEdgeW, const std::vector<std::complex<double>>& zvals, const std::vector<std::pair<int, Eigen::Vector3d>> &bary, Eigen::VectorXd& upTheta);
    double lArg(const long &n, const Eigen::Vector3d& bary);
   


    // test functions
    void testRoundingEnergy(const MeshConnectivity& mesh, const Eigen::MatrixXd& halfEdgeW, const Eigen::VectorXd& faceArea, const Eigen::MatrixXd& cotEntries, const int nverts, std::vector<std::complex<double>> zvals);
    void testKnoppelEnergyFor2DVertexOmegaPerEdge(const Eigen::MatrixXd& pos, const MeshConnectivity& mesh, const Eigen::VectorXd& faceArea, const Eigen::MatrixXd& cotEntries, const std::vector<std::complex<double>>& zvals, const Eigen::MatrixXd& vertexOmega, const double edgeWeight, int eid);  // size(vertexOmega) = nverts, 2

    void testKnoppelEnergyFor2DVertexOmega(const Eigen::MatrixXd& pos, const MeshConnectivity& mesh, const Eigen::VectorXd& faceArea, const Eigen::MatrixXd& cotEntries, const std::vector<std::complex<double>>& zvals, const Eigen::MatrixXd& vertexOmega);  // size(vertexOmega) = nverts, 2


}