#pragma once

#include "../MeshLib/MeshConnectivity.h"
#include "../CommonTools.h"

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <igl/doublearea.h>
#include <igl/cotmatrix.h>

// we use IE to do this
class TimeIntegratedStrainDrivenModel{
public:
    TimeIntegratedStrainDrivenModel(){}
    TimeIntegratedStrainDrivenModel(const std::vector<Eigen::MatrixXd> &wTar, const std::vector<Eigen::VectorXd> &aTar, const Eigen::MatrixXd& basePos, const Eigen::MatrixXi& baseF, const Eigen::MatrixXd &w0, const std::vector<std::complex<double>> &vertVals0, int numFrames, int quadOrder, double spatialRate, double fakeThickness, int modelFlag):
     _basePos(basePos), _w0(w0), _spatialRate(spatialRate), _fakeThickness(fakeThickness), _modelFlag(modelFlag), _wTarList(wTar), _aTarList(aTar)
    {
        _baseMesh = MeshConnectivity(baseF);

        _wList.push_back(w0);
        _vertValsList.push_back(vertVals0);

        _dt = 1.0 / (numFrames + 1);
        _numFrames = numFrames + 1;
        _w0 = w0;

        int nverts = _basePos.rows();
        _initX.setZero(4 * nverts);
        for(int i =0; i < nverts; i++)
        {
            _initX(2 * i) = vertVals0[i].real();
            _initX(2 * i + 1) = vertVals0[i].imag();
            _initX.segment<2>(2 * i + 2 * nverts) = w0.row(i).transpose();
        }
        _curX = _initX;
        _curV = _curX;
        _curV.setZero();


        igl::doublearea(_basePos, baseF, _faceArea);
        _faceArea /= 2.0;
        igl::cotmatrix_entries(_basePos, baseF, _cotEntries);


        _compressDir = wTar;
        for(int i = 0; i < aTar.size(); i++)
        {
            for(int j = 0; j < aTar[i].rows(); j++)
            {
                _compressDir[i].row(j) *= aTar[i][j];
            }

        }
        _curCompDir = _compressDir[0];
        _curWTar = _wTarList[0];
        _curAmpTar = _aTarList[0];

    }

    void updateWZList();

    std::vector<Eigen::MatrixXd> getWList() { return _wList; }
    std::vector<std::vector<std::complex<double>>> getVertValsList() { return _vertValsList; }

    double computeEnergy(const Eigen::VectorXd& x, Eigen::VectorXd* deriv = NULL, Eigen::SparseMatrix<double>* hess = NULL, bool isProj = false);
    double computeSpatialEnergyPerVertex(int vertId, const std::vector<std::complex<double>> &zvals, const Eigen::MatrixXd &w, Eigen::Vector4d *deriv, Eigen::Matrix4d *hess, bool isProj);
    double computeSpatialEnergy(const std::vector<std::complex<double>> &zvals, const Eigen::MatrixXd &w, Eigen::VectorXd *deriv,
                                                                 std::vector<Eigen::Triplet<double>> *hessT, bool isProj);

    void solveInterpFrames();

    void testEnergy(Eigen::VectorXd x);

private:
    Eigen::MatrixXd _basePos;
    MeshConnectivity _baseMesh;
    std::vector<Eigen::MatrixXd> _wList;
    std::vector<Eigen::MatrixXd> _wTarList;
    std::vector<Eigen::VectorXd> _aTarList;
    Eigen::MatrixXd _w0;
    std::vector<std::vector<std::complex<double>>> _vertValsList;
    double _dt;
    double _numFrames;

    std::vector<Eigen::MatrixXd> _compressDir;
    Eigen::MatrixXd _curCompDir;
    Eigen::MatrixXd _curWTar;
    Eigen::VectorXd _curAmpTar;
    double _spatialRate;

    Eigen::VectorXd _curX;
    Eigen::VectorXd _curV;

    Eigen::VectorXd _initX;

    Eigen::VectorXd _faceArea;
    Eigen::MatrixXd _cotEntries;

    double _fakeThickness;
    int _modelFlag;

    bool _useInertial;
};