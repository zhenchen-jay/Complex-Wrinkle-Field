#pragma once

#include <Eigen/Dense>
#include <igl/cotmatrix.h>

#include "GetInterpolatedValues.h"
#include "ComputeZandZdot.h"

class StrainDrivenModel
{
public:
    StrainDrivenModel() {}
    ~StrainDrivenModel() {}

    StrainDrivenModel(const std::vector<Eigen::MatrixXd> &wTar, const std::vector<Eigen::VectorXd> &aTar, const Eigen::MatrixXd& basePos, const Eigen::MatrixXi& baseF, const Eigen::MatrixXd &w0, const Eigen::MatrixXd &w1, const std::vector<std::complex<double>> &vertVals0, const std::vector<std::complex<double>> &vertVals1, int numFrames, int quadOrder, double spatialRate, double fakeThickness, int modelFlag):_basePos(basePos), _w0(w0), _w1(w1), _vertVals0(vertVals0), _vertVals1(vertVals1), _fakeThickness(fakeThickness), _modelFlag(modelFlag)
    {
        _baseMesh = MeshConnectivity(baseF);
        _compressDir = wTar;
        for(int i = 0; i < aTar.size(); i++)
        {
            for(int j = 0; j < aTar[i].rows(); j++)
            {
                _compressDir[i].row(j) *= aTar[i][j];
            }

        }

        _wList.resize(numFrames + 2);
        _vertValsList.resize(numFrames + 2);

        _wList[0] = w0;
        _wList[numFrames + 1] = w1;

        _vertValsList[0] = vertVals0;
        _vertValsList[numFrames + 1] = vertVals1;

        _dt = 1.0 / (numFrames + 1);

        // linear interpolate in between
        for (int i = 1; i <= numFrames; i++)
        {
            double t = _dt * i;
            _wList[i] = (1 - t) * w0 + t * w1;
            _vertValsList[i] = vertVals0;

            for (int j = 0; j < _vertValsList[i].size(); j++)
            {
                _vertValsList[i][j] = (1 - t) * vertVals0[j] + t * vertVals1[j];
                //_vertValsList[i][j] = _vertValsList[i][j] / (_wList[i].row(j).norm() * std::abs(_vertValsList[i][j]));
            }

        }
        _model = ComputeZandZdot(basePos, baseF, quadOrder);

        igl::cotmatrix(basePos, baseF, _cotMat);

        igl::doublearea(_basePos, baseF, _faceArea);
        _faceArea /= 2.0;
        igl::cotmatrix_entries(_basePos, baseF, _cotEntries);

        _spatialRate = spatialRate;
        _wTar = wTar;
        _aTar = aTar;

    }

    void convertVariable2List(const Eigen::VectorXd& x);
    void convertList2Variable(Eigen::VectorXd& x);


    std::vector<Eigen::MatrixXd> getWList() { return _wList; }
    std::vector<std::vector<std::complex<double>>> getVertValsList() { return _vertValsList; }

    void setWList(const std::vector<Eigen::MatrixXd>& wList) { _wList = wList; }
    void setVertValsList(const std::vector<std::vector<std::complex<double>>>& zvals) { _vertValsList = zvals; }

    void getComponentNorm(const Eigen::VectorXd& x, double &znorm, double &wnorm)
    {
        int nbaseVerts = _basePos.rows();
        int numFrames = _vertValsList.size() - 2;

        znorm = 0;
        wnorm = 0;

        for (int i = 0; i < numFrames; i++)
        {
            for (int j = 0; j < nbaseVerts; j++)
            {
                znorm = std::max(znorm, std::abs(x(i * 4 * nbaseVerts + 2 * j)));
                znorm = std::max(znorm, std::abs(x(i * 4 * nbaseVerts + 2 * j + 1)));

                wnorm = std::max(wnorm, std::abs(x(i * 4 * nbaseVerts + 2 * nbaseVerts + 2 * j)));
                wnorm = std::max(wnorm, std::abs(x(i * 4 * nbaseVerts + 2 * nbaseVerts + 2 * j + 1)));
            }
        }
    }

    double computeEnergy(const Eigen::VectorXd& x, Eigen::VectorXd* deriv = NULL, Eigen::SparseMatrix<double>* hess = NULL, bool isProj = false);
    double computeSpatialEnergyPerFrame(int frameId, Eigen::VectorXd* deriv = NULL, std::vector<Eigen::Triplet<double>>* hessT = NULL, bool isProj = false);
    double computeSpatialEnergyPerFramePerVertex(int frameId, int vertId, Eigen::Vector4d *deriv = NULL, Eigen::Matrix4d *hess = NULL, bool isProj = false);

    void testEnergy(Eigen::VectorXd x);
    void testSpatialEnergyPerFrame(int frameId);
    void testSpatialEnergyPerFramePerVertex(int frameId, int vertId);

    Eigen::VectorXd getEntries(const std::vector<std::complex<double>> &zvals, int entryId);

public:
    Eigen::MatrixXd _basePos;
    MeshConnectivity _baseMesh;
    Eigen::MatrixXd _w0;
    Eigen::MatrixXd _w1;
    std::vector<Eigen::MatrixXd> _wList;

    std::vector<std::complex<double>> _vertVals0;
    std::vector<std::complex<double>> _vertVals1;
    std::vector<std::vector<std::complex<double>>> _vertValsList;

    ComputeZandZdot _model;
    double _dt;

    Eigen::SparseMatrix<double> _cotMat;
    Eigen::MatrixXd _cotEntries;
    Eigen::VectorXd _faceArea;

    std::vector<Eigen::MatrixXd> _compressDir;
    std::vector<Eigen::MatrixXd> _wTar;
    std::vector<Eigen::VectorXd> _aTar;
    double _spatialRate;
    double _fakeThickness;
    int _modelFlag; // 0 use a * w as guidance, 1 use atar and wTar seperately.

};
