#pragma once

#include <Eigen/Dense>

#include "GetInterpolatedValues.h"

class InterpolateKeyFrames
{
public:
	InterpolateKeyFrames() {}
	~InterpolateKeyFrames() {}

	InterpolateKeyFrames(const Eigen::MatrixXd& basePos, const Eigen::MatrixXi& baseF, const Eigen::MatrixXd& upsampledPos, const Eigen::MatrixXi& upsampledF, const std::vector<std::pair<int, Eigen::Vector3d>>& baryCoords, const Eigen::MatrixXd &w0, const Eigen::MatrixXd &w1, const std::vector<std::complex<double>> &vertVals0, const std::vector<std::complex<double>> &vertVals1, int numFrames):_basePos(basePos), _upsampledPos(upsampledPos), _baryCoords(baryCoords), _w0(w0), _w1(w1), _vertVals0(vertVals0), _vertVals1(vertVals1)
	{
		_baseMesh = MeshConnectivity(baseF);
		_upsampledMesh = MeshConnectivity(upsampledF);

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
				_vertValsList[i][j] = (1 - t) * vertVals0[j] + t * vertVals1[j];
		}

		_model = GetInterpolatedValues(basePos, baseF, upsampledPos, upsampledF, baryCoords);
		
	}

	void convertVariable2List(const Eigen::VectorXd& x);
	void convertList2Variable(Eigen::VectorXd& x);

	std::vector<Eigen::MatrixXd> getWList() { return _wList; }
	std::vector<std::vector<std::complex<double>>> getVertValsList() { return _vertValsList; }

	double computeEnergy(const Eigen::VectorXd& x, Eigen::VectorXd* deriv = NULL, Eigen::SparseMatrix<double>* hess = NULL, bool isProj = false);
	void testEnergy(Eigen::VectorXd x);

public:
	Eigen::MatrixXd _basePos;
	MeshConnectivity _baseMesh;
	Eigen::MatrixXd _upsampledPos;
	MeshConnectivity _upsampledMesh;
	std::vector<std::pair<int, Eigen::Vector3d>> _baryCoords;
	std::vector<Eigen::MatrixXd> _wList;
	Eigen::MatrixXd _w0;
	Eigen::MatrixXd _w1;
	std::vector<std::complex<double>> _vertVals0;
	std::vector<std::complex<double>> _vertVals1;
	std::vector<std::vector<std::complex<double>>> _vertValsList;
	GetInterpolatedValues _model;
	double _dt;

};
