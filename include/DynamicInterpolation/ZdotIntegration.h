#pragma once
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <complex>

class ZdotIntegration{
public:
    ZdotIntegration(){}


    std::complex<double> computeBiBj(const Eigen::Vector2d& wi, const Eigen::Vector2d& wj, const Eigen::Vector3d& P0, const Eigen::Vector3d& P1, const Eigen::Vector3d& P2, int i, int j, Eigen::Vector4cd* deriv, Eigen::Matrix4cd* hess);

    void testComputeBiBj(const Eigen::Vector2d& wi, const Eigen::Vector2d& wj, const Eigen::Vector3d& P0, const Eigen::Vector3d& P1, const Eigen::Vector3d& P2, int i, int j);

private:
    double triangleArea(const Eigen::Vector3d& P0, const Eigen::Vector3d& P1, const Eigen::Vector3d& P2);
};