#pragma once
#include "../MeshLib/MeshConnectivity.h"
#include "../CommonTools.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace IntrinsicFormula
{
    void computeHalfEdgeMatrix(const MeshConnectivity &mesh, const Eigen::MatrixXd &halfEdgeW, const Eigen::VectorXd& faceArea, const Eigen::MatrixXd& cotEntries, const int nverts, Eigen::SparseMatrix<double>& A);
    void computeHalfEdgeMatrixGivenMag(const MeshConnectivity &mesh, const Eigen::MatrixXd &halfEdgeW, const Eigen::VectorXd& vertAmp, const Eigen::VectorXd& faceArea, const Eigen::MatrixXd& cotEntries, const int nverts, Eigen::SparseMatrix<double>& A);


    double KnoppelHalfEdgeEnergy(const MeshConnectivity& mesh, const Eigen::MatrixXd& halfEdgeW, const Eigen::VectorXd& faceArea, const Eigen::MatrixXd& cotEntries, const std::vector<std::complex<double>>& zvals, Eigen::VectorXd* deriv, std::vector<Eigen::Triplet<double>>* hess);
    double KnoppelHalfEdgeEnergyGivenMag(const MeshConnectivity& mesh, const Eigen::MatrixXd& halfEdgeW, const Eigen::VectorXd& vertAmp, const Eigen::VectorXd& faceArea, const Eigen::MatrixXd& cotEntries, const std::vector<std::complex<double>>& zvals, Eigen::VectorXd* deriv, std::vector<Eigen::Triplet<double>>* hess);

    void roundZvalsFromHalfEdgeOmega(const MeshConnectivity &mesh, const Eigen::MatrixXd &halfEdgeW, const Eigen::VectorXd& faceArea, const Eigen::MatrixXd& cotEntries, const int nverts, std::vector<std::complex<double>>& zvals);
    void roundZvalsFromHalfEdgeOmegaVertexMag(const MeshConnectivity &mesh, const Eigen::MatrixXd& halfEdgeW, const Eigen::VectorXd& vertAmp, const Eigen::VectorXd& faceArea, const Eigen::MatrixXd& cotEntries, const int nverts, std::vector<std::complex<double>>& zvals);

    // we compute the zvals for seperately
    void roundZvalsForSpecificDomainFromHalfEdgeOmegaGivenMag(const MeshConnectivity& mesh, const Eigen::MatrixXd& halfEdgeW, const Eigen::VectorXd& vertAmp, const Eigen::VectorXi& vertFlags, const Eigen::VectorXd& faceArea, const Eigen::MatrixXd& cotEntries, const int nverts, std::vector<std::complex<double>>& zvals);
    void roundZvalsForSpecificDomainFromHalfEdgeOmegaBndValues(const Eigen::MatrixXd& pos, const MeshConnectivity& mesh, const Eigen::MatrixXd& halfEdgeW, const Eigen::VectorXi& vertFlags, const Eigen::VectorXd& faceArea, const Eigen::MatrixXd& cotEntries, const int nverts, std::vector<std::complex<double>>& zvals, double smoothnessCoeff = 1e-3);


    void getUpsamplingThetaFromHalfEdgeOmega(const MeshConnectivity &mesh, const Eigen::MatrixXd& halfEdgeW, const std::vector<std::complex<double>>& zvals, const std::vector<std::pair<int, Eigen::Vector3d>> &bary, Eigen::VectorXd& upTheta);
    double lArg(const long &n, const Eigen::Vector3d& bary);
  

}