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
        for (auto& it : _baryCoords)
        {
            Eigen::Vector3d weights = computeWeight(it.second);
            _baryWeights.push_back({ it.first, weights });
        }
    }
    GetInterpolatedValues() {}

    std::complex<double> planeWaveBasis(Eigen::VectorXd p, Eigen::VectorXd pi, Eigen::Vector2d omega, Eigen::VectorXcd *deriv, Eigen::MatrixXcd* hess, std::vector<Eigen::MatrixXcd>* derivHess);

    std::complex<double> planeWaveValue(const Eigen::MatrixXd& w, const std::vector<std::complex<double>>& vertVals, int vid, Eigen::VectorXcd* deriv, Eigen::MatrixXcd* hess, std::vector<Eigen::MatrixXcd> *derivHess);
    std::vector<std::complex<double>> getZValues(const Eigen::MatrixXd& w, const std::vector<std::complex<double>>& vertVals, std::vector<Eigen::VectorXcd> *deriv, std::vector<Eigen::MatrixXcd> *H);

    // we use (x_{t+1} - x_t) / dt to approximate x'(t)
    std::complex<double> planeWaveValueDot(const Eigen::MatrixXd& w1, const Eigen::MatrixXd& w2, const std::vector<std::complex<double>>& vertVals1, const std::vector<std::complex<double>>& vertVals2, const double dt, int vid, Eigen::VectorXcd* deriv, Eigen::MatrixXcd* hess);
    std::vector<std::complex<double>> getZDotValues(const Eigen::MatrixXd& w1, const Eigen::MatrixXd& w2, const std::vector<std::complex<double>>& vertVals1, const std::vector<std::complex<double>>& vertVals2, const double dt, std::vector<Eigen::VectorXcd>* deriv, std::vector<Eigen::MatrixXcd> *H);

    double zDotSquarePerVertex(const Eigen::MatrixXd& w1, const Eigen::MatrixXd& w2, const std::vector<std::complex<double>>& vertVals1, const std::vector<std::complex<double>>& vertVals2, const double dt, int vid, Eigen::VectorXd* deriv, Eigen::MatrixXd* hess, bool isProj = false);
    double zDotSquareIntegration(const Eigen::MatrixXd& w1, const Eigen::MatrixXd& w2, const std::vector<std::complex<double>>& vertVals1, const std::vector<std::complex<double>>& vertVals2, const double dt, Eigen::VectorXd* deriv, std::vector<Eigen::Triplet<double>> *hessT, bool isProj = false);

    // test function
    void testPlaneWaveValue(const Eigen::MatrixXd& w, const std::vector<std::complex<double>>& vertVals, int vid);
    void testPlaneWaveValueDot(const Eigen::MatrixXd& w1, const Eigen::MatrixXd& w2, const std::vector<std::complex<double>>& vertVals1, const std::vector<std::complex<double>>& vertVals2, const double dt, int vid);
    void testPlaneWaveBasis(Eigen::VectorXd p, Eigen::VectorXd pi, Eigen::Vector2d omega);
    void testZDotSquarePerVertex(const Eigen::MatrixXd& w1, const Eigen::MatrixXd& w2, const std::vector<std::complex<double>>& vertVals1, const std::vector<std::complex<double>>& vertVals2, const double dt, int vid);
    void testZDotSquareIntegration(const Eigen::MatrixXd& w1, const Eigen::MatrixXd& w2, const std::vector<std::complex<double>>& vertVals1, const std::vector<std::complex<double>>& vertVals2, const double dt);

private:
    Eigen::Vector3d computeWeight(Eigen::Vector3d bary)
    {
        Eigen::Vector3d weights;
        for(int i = 0; i < 3; i++)
        {
            weights(i) = 3 * bary(i) * bary(i) - 2 * bary(i) * bary(i) * bary(i) + 2 * bary(i) * bary((i+1)%3) * bary((i+2)%3);
        }
        return weights;
    }

    Eigen::MatrixXd SPDProjection(Eigen::MatrixXd A);

private:
    Eigen::MatrixXd _basePos;
    MeshConnectivity _baseMesh;
    Eigen::MatrixXd _upsampledPos;
    MeshConnectivity _upsampledMesh;

    std::vector<std::pair<int, Eigen::Vector3d>> _baryCoords;
    std::vector<std::pair<int, Eigen::Vector3d>> _baryWeights;
};