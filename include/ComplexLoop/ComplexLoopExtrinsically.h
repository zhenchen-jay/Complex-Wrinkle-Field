#pragma once
#include "ComplexLoop.h"

class ComplexLoopZuenkoExtrinsically : public ComplexLoop	// We modify the Loop.h
{
public:
    void virtual Subdivide(const Eigen::VectorXd& omega, const std::vector<std::complex<double>>& zvals, Eigen::VectorXd& omegaNew, std::vector<std::complex<double>>& upZvals, int level) override;

private:
    std::complex<double> interpZ(const std::vector<std::complex<double>>& zList, const std::vector<Eigen::Vector3d>& gradThetaList, std::vector<double>& coords, const std::vector<Eigen::Vector3d>& pList);
    void updateLoopedZvals(const Eigen::VectorXd& omega, const std::vector<std::complex<double>>& zvals, std::vector<std::complex<double>>& upZvals);
    void computeEdgeOmega2ExtrinsicalGradient(const Eigen::VectorXd& omega, Eigen::MatrixXd& gradTheta);
};
