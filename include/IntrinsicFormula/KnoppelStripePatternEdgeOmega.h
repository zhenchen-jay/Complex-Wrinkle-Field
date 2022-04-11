#pragma once
#include "../MeshLib/MeshConnectivity.h"
#include "../CommonTools.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace IntrinsicFormula
{
    void computeEdgeMatrix(const MeshConnectivity &mesh, const Eigen::VectorXd& edgeW, const Eigen::VectorXd& faceArea, const Eigen::MatrixXd& cotEntries, const int nverts, Eigen::SparseMatrix<double>& A);
    void computeEdgeMatrixGivenMag(const MeshConnectivity &mesh, const Eigen::VectorXd& edgeW, const Eigen::VectorXd& vertAmp, const Eigen::VectorXd& faceArea, const Eigen::MatrixXd& cotEntries, const int nverts, Eigen::SparseMatrix<double>& A);


    double KnoppelEdgeEnergy(const MeshConnectivity& mesh, const Eigen::VectorXd& edgeW, const Eigen::VectorXd& faceArea, const Eigen::MatrixXd& cotEntries, const std::vector<std::complex<double>>& zvals, Eigen::VectorXd* deriv, std::vector<Eigen::Triplet<double>>* hess);
    double KnoppelEdgeEnergyGivenMag(const MeshConnectivity& mesh, const Eigen::VectorXd& edgeW, const Eigen::VectorXd& vertAmp, const Eigen::VectorXd& edgeWeight, const std::vector<std::complex<double>>& zvals, Eigen::VectorXd* deriv, std::vector<Eigen::Triplet<double>>* hess);

    void roundZvalsFromEdgeOmega(const MeshConnectivity &mesh, const Eigen::VectorXd& edgeW, const Eigen::VectorXd& faceArea, const Eigen::MatrixXd& cotEntries, const int nverts, std::vector<std::complex<double>>& zvals);
    void roundZvalsFromEdgeOmegaVertexMag(const MeshConnectivity &mesh, const Eigen::VectorXd& edgeW, const Eigen::VectorXd& vertAmp, const Eigen::VectorXd& faceArea, const Eigen::MatrixXd& cotEntries, const int nverts, std::vector<std::complex<double>>& zvals);

    // we compute the zvals for seperately
    void roundZvalsForSpecificDomainFromEdgeOmegaGivenMag(const MeshConnectivity& mesh, const Eigen::VectorXd& edgeW, const Eigen::VectorXd& vertAmp, const Eigen::VectorXi& vertFlags, const Eigen::VectorXd& faceArea, const Eigen::MatrixXd& cotEntries, const int nverts, std::vector<std::complex<double>>& zvals);
    void roundZvalsForSpecificDomainFromEdgeOmegaBndValues(const Eigen::MatrixXd& pos, const MeshConnectivity& mesh, const Eigen::VectorXd& edgeW, const Eigen::VectorXi& vertFlags, const Eigen::VectorXd& faceArea, const Eigen::MatrixXd& cotEntries, const int nverts, std::vector<std::complex<double>>& zvals);


    void getUpsamplingThetaFromEdgeOmega(const MeshConnectivity &mesh, const Eigen::VectorXd& edgeW, const std::vector<std::complex<double>>& zvals, const std::vector<std::pair<int, Eigen::Vector3d>> &bary, Eigen::VectorXd& upTheta);

}