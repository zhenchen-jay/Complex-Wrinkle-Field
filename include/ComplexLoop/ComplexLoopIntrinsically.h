#pragma once
#include "ComplexLoop.h"

class ComplexLoopIntrinsically : public ComplexLoop	// We modify the Loop.h
{
public:
    void virtual Subdivide(const Eigen::VectorXd& omega, const std::vector<std::complex<double>>& zvals, Eigen::VectorXd& omegaNew, std::vector<std::complex<double>>& upZvals, int level) override;

private:
    std::complex<double> computeBarycentricZval(const Eigen::VectorXd& omega, const std::vector<std::complex<double>>& zvals, int fid, const Eigen::Vector3d& bary);
    std::complex<double> computeZvalAtPoint(const Eigen::VectorXd& omega, const std::vector<std::complex<double>>& zvals, int fid, const Eigen::Vector3d& p, int startviInFace = 0);
    void updateLoopedZvals(const Eigen::VectorXd& omega, const std::vector<std::complex<double>>& zvals, std::vector<std::complex<double>>& upZvals);
};
