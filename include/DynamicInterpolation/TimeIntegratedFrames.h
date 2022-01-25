#pragma once

#include "../MeshLib/MeshConnectivity.h"
#include "../CommonTools.h"

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <igl/doublearea.h>
#include <igl/cotmatrix.h>

// we use IE to do this
class TimeIntegratedFrames{
public:
    TimeIntegratedFrames(){}
    TimeIntegratedFrames(const Eigen::MatrixXd& basePos, const Eigen::MatrixXi& baseF, const Eigen::MatrixXd &w0, const Eigen::MatrixXd &w1, const std::vector<std::complex<double>> &vertVals0, const std::vector<std::complex<double>> &vertVals1, int numFrames, double K, bool useInertial = false):_basePos(basePos), _K(K), _useInertial(useInertial)
    {
        _baseMesh = MeshConnectivity(baseF);

        _wList.push_back(w0);
        _vertValsList.push_back(vertVals0);

        _dt = 1.0 / (numFrames + 1);
        _numFrames = numFrames + 1;
        _w0 = w0;
        _w1 = w1;

        _wTar.setZero(_basePos.rows(), 3);
        _wTar.block(0, 0, _wTar.rows(), 2) = (1 - _dt) * _w0 + _dt * _w1;
        _halfEdgewTar = vertexVec2IntrinsicHalfEdgeVec(_wTar, _basePos, _baseMesh);


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

        igl::doublearea(_basePos, baseF, _faceArea);
        _faceArea /= 2.0;
        igl::cotmatrix_entries(_basePos, baseF, _cotEntries);
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
    std::vector<Eigen::MatrixXd> _wList;
    Eigen::MatrixXd _w0;
    Eigen::MatrixXd _w1;
    Eigen::MatrixXd _wTar;
    Eigen::MatrixXd _halfEdgewTar;
    std::vector<std::vector<std::complex<double>>> _vertValsList;
    double _dt;
    double _numFrames;
    double _K;

    Eigen::VectorXd _curX;
    Eigen::VectorXd _curV;

    Eigen::VectorXd _initX;
    Eigen::VectorXd _tarX;

    Eigen::VectorXd _faceArea;
    Eigen::MatrixXd _cotEntries;

    bool _useInertial;
};