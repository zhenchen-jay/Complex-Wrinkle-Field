#pragma once

#include "../MeshLib/MeshConnectivity.h"
#include "../CommonTools.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>
#include <complex>
#include <igl/doublearea.h>

// TODO: make it work for 3D using intrinsic information

class ComputeZandZdot {
public:
    ComputeZandZdot(const Eigen::MatrixXd& basePos, const Eigen::MatrixXi& baseF, int quadOrder) : _basePos(basePos)
    {
        _baseMesh = MeshConnectivity(baseF);
        _quadPoints = buildQuadraturePoints(quadOrder);
        std::cout << "num of quad points: " << _quadPoints.size() << std::endl;
        _hatWeights.clear();

        for(auto &it : _quadPoints)
        {
            Eigen::Vector3d hw = computeHatWeight(it.u, it.v);
            _hatWeights.push_back(hw);
        }

        igl::doublearea(basePos, baseF, _doubleFarea);
    }
    ComputeZandZdot() {}


    Eigen::Vector3d getQuadPosition(int faceId, int quadId)
    {
        Eigen::Vector3d p = (1 - _quadPoints[quadId].u - _quadPoints[quadId].v) * _basePos.row(_baseMesh.faceVertex(faceId, 0)) + _quadPoints[quadId].u * _basePos.row(_baseMesh.faceVertex(faceId, 1)) + _quadPoints[quadId].v * _basePos.row(_baseMesh.faceVertex(faceId, 2));
        return p;
    }

    std::vector<QuadraturePoints> getQuadPts()
    {
        return _quadPoints;
    }

    std::complex<double> planeWaveBasis(Eigen::Vector3d p, Eigen::Vector3d pi, Eigen::Vector2d omega, Eigen::Vector2cd* deriv, Eigen::Matrix2cd* hess, std::vector<Eigen::Matrix2cd>* derivHess);

    std::complex<double> planeWaveValueFromQuad(const Eigen::MatrixXd& w,
        const std::vector<std::complex<double>>& vertVals, int faceId, int quadId,
        Eigen::Matrix<std::complex<double>, 12, 1>* deriv, 
        Eigen::Matrix<std::complex<double>, 12, 12>* hess, 
        std::vector<Eigen::Matrix<std::complex<double>, 12, 12>>* derivHess);

    // we use (x_{t+1} - x_t) / dt to approximate x'(t)
    std::complex<double> planeWaveValueDotFromQuad(const Eigen::MatrixXd& w1, 
        const Eigen::MatrixXd& w2, 
        const std::vector<std::complex<double>>& vertVals1, 
        const std::vector<std::complex<double>>& vertVals2, 
        const double dt, int faceId, int quadId, 
        Eigen::Matrix<std::complex<double>, 24, 1>* deriv, 
        Eigen::Matrix<std::complex<double>, 24, 24>* hess);

    double zDotSquarePerface(const Eigen::MatrixXd& w1, 
        const Eigen::MatrixXd& w2, 
        const std::vector<std::complex<double>>& vertVals1, 
        const std::vector<std::complex<double>>& vertVals2, 
        const double dt, int faceId, Eigen::Matrix<double, 24, 1>* deriv, Eigen::Matrix<double, 24, 24>* hess, bool isProj = false);

    double zDotSquareIntegration(const Eigen::MatrixXd& w1, 
        const Eigen::MatrixXd& w2, 
        const std::vector<std::complex<double>>& vertVals1, 
        const std::vector<std::complex<double>>& vertVals2, 
        const double dt, Eigen::VectorXd* deriv, std::vector<Eigen::Triplet<double>>* hessT, bool isProj = false);

    // test function
    void testPlaneWaveValueFromQuad(const Eigen::MatrixXd& w, const std::vector<std::complex<double>>& vertVals, int faceId, int quadId);
    void testPlaneWaveValueDotFromQuad(const Eigen::MatrixXd& w1, const Eigen::MatrixXd& w2, const std::vector<std::complex<double>>& vertVals1, const std::vector<std::complex<double>>& vertVals2, const double dt, int faceId, int quadId);
    void testPlaneWaveBasis(Eigen::VectorXd p, Eigen::VectorXd pi, Eigen::Vector2d omega);
    void testZDotSquarePerface(const Eigen::MatrixXd& w1, const Eigen::MatrixXd& w2, const std::vector<std::complex<double>>& vertVals1, const std::vector<std::complex<double>>& vertVals2, const double dt, int faceId);
    void testZDotSquareIntegration(const Eigen::MatrixXd& w1, const Eigen::MatrixXd& w2, const std::vector<std::complex<double>>& vertVals1, const std::vector<std::complex<double>>& vertVals2, const double dt);

private:
    Eigen::MatrixXd _basePos;
    MeshConnectivity _baseMesh;
    std::vector<QuadraturePoints> _quadPoints;
    std::vector<Eigen::Vector3d> _hatWeights;
    Eigen::VectorXd _doubleFarea;

};