#pragma once

#include <Eigen/Dense>

#include "GetInterpolatedValues.h"
#include "ComputeZandZdot.h"

class InterpolateKeyFrames
{
public:
	InterpolateKeyFrames() {}
	~InterpolateKeyFrames() {}

	InterpolateKeyFrames(const Eigen::MatrixXd& basePos, const Eigen::MatrixXi& baseF, const Eigen::MatrixXd& upsampledPos, const Eigen::MatrixXi& upsampledF, const std::vector<std::pair<int, Eigen::Vector3d>>& baryCoords, const Eigen::MatrixXd &w0, const Eigen::MatrixXd &w1, const std::vector<std::complex<double>> &vertVals0, const std::vector<std::complex<double>> &vertVals1, int numFrames, int quadOrder, bool isUseUpmesh, double penaltyCoef = 0):_basePos(basePos), _upsampledPos(upsampledPos), _baryCoords(baryCoords), _w0(w0), _w1(w1), _vertVals0(vertVals0), _vertVals1(vertVals1), _isUseUpMesh(isUseUpmesh), _penaltyCoef(penaltyCoef)
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
		_newmodel = ComputeZandZdot(basePos, baseF, quadOrder);
		
	}

	void convertVariable2List(const Eigen::VectorXd& x);
	void convertList2Variable(Eigen::VectorXd& x);

	void initializeLamdaMu(Eigen::VectorXd& lambda, Eigen::VectorXd& mu, double initMu = 1.0);

	std::vector<Eigen::MatrixXd> getWList() { return _wList; }
	std::vector<std::vector<std::complex<double>>> getVertValsList() { return _vertValsList; }

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

	Eigen::VectorXd getConstraints(const Eigen::VectorXd& x, std::vector<Eigen::SparseVector<double> > *deriv = NULL, std::vector<Eigen::SparseMatrix<double>>* hess = NULL, bool isProj = false);
	Eigen::VectorXd getConstraintsPenalty(const Eigen::VectorXd& x, std::vector< Eigen::SparseVector<double> > *deriv = NULL, std::vector<Eigen::SparseMatrix<double>>* hess = NULL, bool isProj = false);

	double computePerVertexPenalty(const std::vector<std::complex<double>>& zvals, const Eigen::MatrixXd& w, int vid, Eigen::Vector4d* deriv = NULL, Eigen::Matrix4d* hess = NULL, bool isProj = false);
	double computePerVertexConstraint(const std::vector<std::complex<double>>& zvals, const Eigen::MatrixXd& w, int vid, Eigen::Vector4d* deriv = NULL, Eigen::Matrix4d* hess = NULL, bool isProj = false);

	double computePerFramePenalty(const std::vector<std::complex<double>>& zvals, const Eigen::MatrixXd& w, Eigen::VectorXd* deriv = NULL, std::vector<Eigen::Triplet<double>>* hessT = NULL, bool isProj = false);
	double computePenalty(const Eigen::VectorXd& x, Eigen::VectorXd* deriv = NULL, Eigen::SparseMatrix<double>* hess = NULL, bool isProj = false);

	

	double computeEnergy(const Eigen::VectorXd& x, Eigen::VectorXd* deriv = NULL, Eigen::SparseMatrix<double>* hess = NULL, bool isProj = false);

	double computePerFrameConstraints(const std::vector<std::complex<double>>& zvals, const Eigen::MatrixXd& w, const Eigen::VectorXd& lambda, Eigen::VectorXd* deriv = NULL, std::vector<Eigen::Triplet<double>>* hessT = NULL, bool isProj = false);
	double computeConstraints(const Eigen::VectorXd& x, const Eigen::VectorXd& lambda, Eigen::VectorXd* deriv = NULL, Eigen::SparseMatrix<double>* hess = NULL, bool isProj = false);

	double computePerFrameConstraintsPenalty(const std::vector<std::complex<double>>& zvals, const Eigen::MatrixXd& w, const Eigen::VectorXd& mu, Eigen::VectorXd* deriv = NULL, std::vector<Eigen::Triplet<double>>* hessT = NULL, bool isProj = false);
	double computeConstraintsPenalty(const Eigen::VectorXd& x, const Eigen::VectorXd& mu, Eigen::VectorXd* deriv = NULL, Eigen::SparseMatrix<double>* hess = NULL, bool isProj = false);

	void testPerVertexPenalty(const std::vector<std::complex<double>>& zvals, const Eigen::MatrixXd& w, int vid);
	void testPerFramePenalty(const std::vector<std::complex<double>>& zvals, const Eigen::MatrixXd& w);
	void testPenalty(Eigen::VectorXd x);
	void testConstraintsPenalty(Eigen::VectorXd x, Eigen::VectorXd mu);
	void testConstraints(Eigen::VectorXd x, Eigen::VectorXd lambda);
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
	ComputeZandZdot _newmodel;
	double _dt;
	bool _isUseUpMesh;
	double _penaltyCoef;

};
