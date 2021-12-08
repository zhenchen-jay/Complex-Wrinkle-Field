#pragma once

#include "../MeshLib/MeshConnectivity.h"

#include <Eigen/Dense>
#include <Eigen/Sparse>

// we use IE to do this
class TimeIntegratedFrames{
public:
    TimeIntegratedFrames(){}
    TimeIntegratedFrames(const Eigen::MatrixXd& basePos, const Eigen::MatrixXi& baseF, const Eigen::MatrixXd& upsampledPos, const Eigen::MatrixXi& upsampledF, const std::vector<std::pair<int, Eigen::Vector3d>>& baryCoords, const Eigen::MatrixXd &w0, const Eigen::MatrixXd &w1, const std::vector<std::complex<double>> &vertVals0, const std::vector<std::complex<double>> &vertVals1, int numFrames):_basePos(basePos), _upsampledPos(upsampledPos), _baryCoords(baryCoords)
    {
        _baseMesh = MeshConnectivity(baseF);
        _upsampledMesh = MeshConnectivity(upsampledF);

        _wList.push_back(w0);
        _vertValsList.push_back(vertVals0);

        _dt = 1.0 / (numFrames + 1);
        _numFrames = numFrames + 1;

        int nverts = _basePos.rows();
        _initX.setZero(4 * nverts);
        _tarX.setZero(4 * nverts);
        for(int i =0; i < nverts; i++)
        {
            _initX(2 * i) = vertVals0[i].real();
            _initX(2 * i + 1) = vertVals0[i].imag();
            _initX.segment<2>(2 * i + 2 * nverts) = w0.row(i).transpose();

            _tarX(2 * i) = vertVals1[i].real();
            _tarX(2 * i + 1) = vertVals1[i].imag();
            _tarX.segment<2>(2 * i + 2 * nverts) = w1.row(i).transpose();
        }
        _curX = _initX;
        _curV = _curX;
        _curV.setZero();

    }

    void updateWZList();

    std::vector<Eigen::MatrixXd> getWList() { return _wList; }
    std::vector<std::vector<std::complex<double>>> getVertValsList() { return _vertValsList; }

    double computeEnergy(const Eigen::VectorXd& x, Eigen::VectorXd* deriv = NULL, Eigen::SparseMatrix<double>* hess = NULL, bool isProj = false);

    void solveInterpFrames();

    void testEnergy(Eigen::VectorXd x);

private:
    Eigen::MatrixXd _basePos;
    MeshConnectivity _baseMesh;
    Eigen::MatrixXd _upsampledPos;
    MeshConnectivity _upsampledMesh;
    std::vector<std::pair<int, Eigen::Vector3d>> _baryCoords;
    std::vector<Eigen::MatrixXd> _wList;
    std::vector<std::vector<std::complex<double>>> _vertValsList;
    double _dt;
    double _numFrames;

    Eigen::VectorXd _curX;
    Eigen::VectorXd _curV;

    Eigen::VectorXd _initX;
    Eigen::VectorXd _tarX;
};