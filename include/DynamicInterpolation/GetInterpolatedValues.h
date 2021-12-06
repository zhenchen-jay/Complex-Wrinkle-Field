#pragma once

#include "../MeshLib/MeshConnectivity.h"

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>
#include <complex>

// TODO: make it work for 3D using intrinsic information

class GetInterpolatedValues{
public:
    GetInterpolatedValues(const Eigen::MatrixXd &basePos, const Eigen::MatrixXi &baseF, const Eigen::MatrixXd &upsampledPos, const Eigen::MatrixXi &upsampledF, const std::vector<std::pair<int, Eigen::Vector3d>>& baryCoords) : _basePos(basePos), _upsampledPos(upsampledPos), _baryCoords(baryCoords)
    {
        _baseMesh = MeshConnectivity(baseF);
        _upsampledMesh = MeshConnectivity(upsampledF);
    }
    GetInterpolatedValues() {}

    std::complex<double> planeWaveBasis(Eigen::VectorXd p, Eigen::VectorXd v, Eigen::Vector2d omega, Eigen::VectorXcd *deriv, Eigen::MatrixXcd* hess);

    std::complex<double> planeWaveValue(const Eigen::MatrixXd& w, const std::vector<std::complex<double>>& vertVals, int vid, Eigen::VectorXcd* deriv, Eigen::MatrixXcd* hess, bool isProj = false);
    std::vector<std::complex<double>> getZValues(const Eigen::MatrixXd& w, const std::vector<std::complex<double>>& vertVals, std::vector<Eigen::VectorXcd> *deriv, std::vector<Eigen::MatrixXcd> *H, bool isProj = false);

    // we use (x_{t+1} - x_t) / dt to approximate x'(t)
    std::complex<double> planeWaveValueDot(const Eigen::MatrixXd& w1, const Eigen::MatrixXd& w2, const std::vector<std::complex<double>>& vertVals1, const std::vector<std::complex<double>>& vertVals2, const double dt, int vid, Eigen::VectorXcd* deriv, Eigen::MatrixXcd* hess, bool isProj = false);
    std::vector<std::complex<double>> getZDotValues(const Eigen::MatrixXd& w1, const Eigen::MatrixXd& w2, const std::vector<std::complex<double>>& vertVals1, const std::vector<std::complex<double>>& vertVals2, const double dt, std::vector<Eigen::VectorXcd>* deriv, std::vector<Eigen::MatrixXcd> *H, bool isProj = false);

private:
    Eigen::Vector3d computeWeight(Eigen::Vector3d bary)
    {
        Eigen::Vector3d weights;
        for(int i = 0; i < 3; i++)
        {
            weights(i) = 3 * bary(i) * bary(i) - 2 * bary(i) * bary(i) + 2 * bary(i) * bary((i+1)%3) * bary((i+2)%3);
        }
        return weights;
    }

private:
    Eigen::MatrixXd _basePos;
    MeshConnectivity _baseMesh;
    Eigen::MatrixXd _upsampledPos;
    MeshConnectivity _upsampledMesh;

    std::vector<std::pair<int, Eigen::Vector3d>> _baryCoords;
};