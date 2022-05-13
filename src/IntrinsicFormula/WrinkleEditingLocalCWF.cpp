#include "../../include/IntrinsicFormula/WrinkleEditingLocalCWF.h"
#include "../../include/Optimization/NewtonDescent.h"
#include "../../include/IntrinsicFormula/KnoppelStripePatternEdgeOmega.h"
#include "../../include/WrinkleFieldsEditor.h"
#include <igl/cotmatrix_entries.h>
#include <igl/cotmatrix.h>
#include <igl/doublearea.h>
#include <igl/boundary_loop.h>
#include <Eigen/SPQRSupport>

using namespace IntrinsicFormula;


void WrinkleEditingLocalCWF::convertList2Variable(Eigen::VectorXd& x)
{
	int nFreeVerts = _freeVids.size();
	int nFreeEdges = _freeEids.size();

	int numFrames = _zvalsList.size() - 2;

	int DOFsPerframe = (2 * nFreeVerts + nFreeEdges);

	int DOFs = numFrames * DOFsPerframe;

	x.setZero(DOFs);

	for (int i = 0; i < numFrames; i++)
	{
		for (int j = 0; j < nFreeVerts; j++)
		{
			x(i * DOFsPerframe + 2 * j) = _zvalsList[i + 1][_freeVids[j]].real();
			x(i * DOFsPerframe + 2 * j + 1) = _zvalsList[i + 1][_freeVids[j]].imag();
		}

		for (int j = 0; j < nFreeEdges; j++)
		{
			x(i * DOFsPerframe + 2 * nFreeVerts + j) = _edgeOmegaList[i + 1](_freeEids[j]);
		}
	}
}

void WrinkleEditingLocalCWF::convertVariable2List(const Eigen::VectorXd& x)
{
	int nFreeVerts = _freeVids.size();
	int nFreeEdges = _freeEids.size();

	int numFrames = _zvalsList.size() - 2;

	int DOFsPerframe = (2 * nFreeVerts + nFreeEdges);

	for (int i = 0; i < numFrames; i++)
	{
		for (int j = 0; j < nFreeVerts; j++)
		{
			_zvalsList[i + 1][_freeVids[j]] = std::complex<double>(x(i * DOFsPerframe + 2 * j), x(i * DOFsPerframe + 2 * j + 1));
		}

		for (int j = 0; j < nFreeEdges; j++)
		{
			_edgeOmegaList[i + 1](_freeEids[j]) = x(i * DOFsPerframe + 2 * nFreeVerts + j);
		}
	}
}


double WrinkleEditingLocalCWF::temporalAmpDifference(int frameId, Eigen::VectorXd* deriv, std::vector<Eigen::Triplet<double>>* hessT, bool isProj)
{
	int nFreeVerts = _freeVids.size();
	double energy = 0;

	if (deriv)
		deriv->setZero(2 * nFreeVerts);
	if (hessT)
		hessT->clear();

	for (int i = 0; i < nFreeVerts; i++)
	{
		int vid = _freeVids[i];
		double ampSq = _zvalsList[frameId][vid].real() * _zvalsList[frameId][vid].real() +
			_zvalsList[frameId][vid].imag() * _zvalsList[frameId][vid].imag();
		double refAmpSq = _combinedRefAmpList[frameId][vid] * _combinedRefAmpList[frameId][vid];
		double ca = _spatialAmpRatio * _vertArea(vid) / (_refAmpAveList[frameId] * _refAmpAveList[frameId]);

		energy += ca * (ampSq - refAmpSq) * (ampSq - refAmpSq);

		if (deriv)
		{
			(*deriv)(2 * i) += 2.0 * ca * (ampSq - refAmpSq) * (2.0 * _zvalsList[frameId][vid].real());
			(*deriv)(2 * i + 1) += 2.0 * ca * (ampSq - refAmpSq) * (2.0 * _zvalsList[frameId][vid].imag());
		}

		if (hessT)
		{
			Eigen::Matrix2d tmpHess;
			tmpHess << 
				2.0 * _zvalsList[frameId][vid].real() * 2.0 * _zvalsList[frameId][vid].real(),
				2.0 * _zvalsList[frameId][vid].real() * 2.0 * _zvalsList[frameId][vid].imag(),
				2.0 * _zvalsList[frameId][vid].real() * 2.0 * _zvalsList[frameId][vid].imag(),
				2.0 * _zvalsList[frameId][vid].imag() * 2.0 * _zvalsList[frameId][vid].imag();

			tmpHess *= 2.0 * ca;
			tmpHess += 2.0 * ca * (ampSq - refAmpSq) * (2.0 * Eigen::Matrix2d::Identity());


			if (isProj)
				tmpHess = SPDProjection(tmpHess);

			for (int k = 0; k < 2; k++)
				for (int l = 0; l < 2; l++)
					hessT->push_back({ 2 * i + k, 2 * i + l, tmpHess(k, l) });
		}
	}
	return energy;
}

double WrinkleEditingLocalCWF::temporalOmegaDifference(int frameId, Eigen::VectorXd* deriv, std::vector<Eigen::Triplet<double>>* hessT, bool isProj)
{
	int nFreeEdges = _freeEids.size();
	double energy = 0;

	if (deriv)
		deriv->setZero(nFreeEdges);

	for (int i = 0; i < nFreeEdges; i++)
	{
		int eid = _freeEids[i];
		double ce = _spatialEdgeRatio * _edgeArea(eid) * (_refAmpAveList[frameId] * _refAmpAveList[frameId]);

		energy += ce * (_edgeOmegaList[frameId](eid) - _combinedRefOmegaList[frameId](eid)) * (_edgeOmegaList[frameId](eid) - _combinedRefOmegaList[frameId](eid));

		if (deriv) 
		{
			(*deriv)(i) += 2 * ce * (_edgeOmegaList[frameId](eid) - _combinedRefOmegaList[frameId](eid));
		}

		if (hessT) 
		{
			hessT->push_back(Eigen::Triplet<double>(i, i, 2 * ce));
		}
	}

	return energy;
}

double WrinkleEditingLocalCWF::spatialKnoppelEnergy(int frameId, Eigen::VectorXd* deriv, std::vector<Eigen::Triplet<double>>* hessT, bool isProj)
{
	double energy = 0;
	int nFreeEdges = _freeEids.size();
	int nFreeVerts = _freeVids.size();
	double aveAmp = _refAmpAveList[frameId];
	std::vector<Eigen::Triplet<double>> AT;
	AT.clear();

	int maxFreeVid = 0;

	for (int fe = 0; fe < nFreeEdges; fe++)
	{
		int eid = _freeEids[fe];
		int vid0 = _mesh.edgeVertex(eid, 0);
		int vid1 = _mesh.edgeVertex(eid, 1);

		double r0 = _combinedRefAmpList[frameId](vid0) / aveAmp;
		double r1 = _combinedRefAmpList[frameId](vid1) / aveAmp;

		std::complex<double> expw0 = std::complex<double>(std::cos(_combinedRefOmegaList[frameId](eid)), std::sin(_combinedRefOmegaList[frameId](eid)));

		std::complex<double> z0 = _zvalsList[frameId][vid0];
		std::complex<double> z1 = _zvalsList[frameId][vid1];

		double ce = _spatialKnoppelRatio * _edgeArea(eid);

		energy += 0.5 * norm((r1 * z0 * expw0 - r0 * z1)) * ce;

		if (deriv || hessT)
		{
			int freeV0 = _actualVid2Free[vid0];
			int freeV1 = _actualVid2Free[vid1];


			AT.push_back({ 2 * freeV0, 2 * freeV0, r1 * r1 * ce });
			AT.push_back({ 2 * freeV0 + 1, 2 * freeV0 + 1, r1 * r1 * ce });

			AT.push_back({ 2 * freeV1, 2 * freeV1, r0 * r0 * ce });
			AT.push_back({ 2 * freeV1 + 1, 2 * freeV1 + 1, r0 * r0 * ce });


			AT.push_back({ 2 * freeV0, 2 * freeV1, -ce * (expw0.real()) * r0 * r1 });
			AT.push_back({ 2 * freeV0 + 1, 2 * freeV1, -ce * (-expw0.imag()) * r0 * r1 });
			AT.push_back({ 2 * freeV0, 2 * freeV1 + 1, -ce * (expw0.imag()) * r0 * r1 });
			AT.push_back({ 2 * freeV0 + 1, 2 * freeV1 + 1, -ce * (expw0.real()) * r0 * r1 });

			AT.push_back({ 2 * freeV1, 2 * freeV0, -ce * (expw0.real()) * r0 * r1 });
			AT.push_back({ 2 * freeV1, 2 * freeV0 + 1, -ce * (-expw0.imag()) * r0 * r1 });
			AT.push_back({ 2 * freeV1 + 1, 2 * freeV0, -ce * (expw0.imag()) * r0 * r1 });
			AT.push_back({ 2 * freeV1 + 1, 2 * freeV0 + 1, -ce * (expw0.real()) * r0 * r1 });
		}
	}

	if (deriv || hessT)
	{
		Eigen::SparseMatrix<double> A;
		
		A.resize(2 * nFreeVerts, 2 * nFreeVerts);
		A.setFromTriplets(AT.begin(), AT.end());

		// check whether A is PD


		if (deriv)
		{
			Eigen::VectorXd fvals(2 * nFreeVerts);
			for (int i = 0; i < nFreeVerts; i++)
			{
				fvals(2 * i) = _zvalsList[frameId][_freeVids[i]].real();
				fvals(2 * i + 1) = _zvalsList[frameId][_freeVids[i]].imag();
			}
			(*deriv) = A * fvals;
		}

		if (hessT)
			(*hessT) = AT;
	}

	return energy;
}


double WrinkleEditingLocalCWF::kineticEnergy(int frameId, Eigen::VectorXd* deriv, std::vector<Eigen::Triplet<double>>* hessT, bool isProj)
{
	double energy = 0;
	if (frameId >= _zvalsList.size() - 1)
	{
		std::cerr << "frame id overflows" << std::endl;
		exit(EXIT_FAILURE);
	}
	int nFreeEdges = _freeEids.size();
	int nFreeVerts = _freeVids.size();
	int nFreeFaces = _freeFids.size();
	int nDOFs = 2 * nFreeVerts + nFreeEdges;

	if (deriv)
		deriv->setZero(2 * nDOFs);
	if (hessT)
		hessT->clear();

	for (int i = 0; i < nFreeFaces; i++)
	{
		int fid = _freeFids[i];

		Eigen::Matrix<double, 18, 1> faceDeriv;
		Eigen::Matrix<double, 18, 18> faceHess;

		energy += _zdotModel.computeZdotIntegrationPerface(_zvalsList[frameId], _edgeOmegaList[frameId], _zvalsList[frameId + 1], _edgeOmegaList[frameId + 1], fid, deriv ? &faceDeriv : NULL, hessT ? &faceHess : NULL, isProj);

		if (deriv)
		{
			for (int j = 0; j < 3; j++)
			{
				int freeVid = _actualVid2Free[_mesh.faceVertex(fid, j)];
				int freeEid = _actualEid2Free[_mesh.faceEdge(fid, j)];

				(*deriv)(2 * freeVid) += faceDeriv(2 * j);
				(*deriv)(2 * freeVid + 1) += faceDeriv(2 * j + 1);
				(*deriv)(freeEid + 2 * nFreeVerts) += faceDeriv(6 + j);

				(*deriv)(2 * freeVid + nDOFs) += faceDeriv(9 + 2 * j);
				(*deriv)(2 * freeVid + 1 + nDOFs) += faceDeriv(9 + 2 * j + 1);
				(*deriv)(freeEid + 2 * nFreeVerts + nDOFs) += faceDeriv(15 + j);
			}
		}

		if (hessT)
		{
			for (int j = 0; j < 3; j++)
			{
				int vid = _actualVid2Free[_mesh.faceVertex(fid, j)];
				int eid = _actualEid2Free[_mesh.faceEdge(fid, j)];

				for (int k = 0; k < 3; k++)
				{
					int vid1 = _actualVid2Free[_mesh.faceVertex(fid, k)];
					int eid1 = _actualEid2Free[_mesh.faceEdge(fid, k)];

					for (int m1 = 0; m1 < 2; m1++)
					{
						for (int m2 = 0; m2 < 2; m2++)
						{
							hessT->push_back({ 2 * vid + m1, 2 * vid1 + m2,  faceHess(2 * j + m1, 2 * k + m2) });
							hessT->push_back({ 2 * vid + m1, 2 * vid1 + m2 + nDOFs,  faceHess(2 * j + m1, 9 + 2 * k + m2) });

							hessT->push_back({ 2 * vid + m1 + nDOFs, 2 * vid1 + m2,  faceHess(9 + 2 * j + m1, 2 * k + m2) });
							hessT->push_back({ 2 * vid + m1 + nDOFs, 2 * vid1 + m2 + nDOFs,  faceHess(9 + 2 * j + m1, 9 + 2 * k + m2) });
						}

						hessT->push_back({ 2 * vid + m1, eid1 + 2 * nFreeVerts,  faceHess(2 * j + m1, 6 + k) });
						hessT->push_back({ eid + 2 * nFreeVerts, 2 * vid1 + m1, faceHess(6 + j, 2 * k + m1) });

						hessT->push_back({ 2 * vid + m1, eid1 + 2 * nFreeVerts + nDOFs, faceHess(2 * j + m1, 15 + k) });
						hessT->push_back({ eid + 2 * nFreeVerts + nDOFs, 2 * vid1 + m1, faceHess(15 + j, 2 * k + m1) });

						hessT->push_back({ 2 * vid + m1 + nDOFs, eid1 + 2 * nFreeVerts,  faceHess(9 + 2 * j + m1, 6 + k) });
						hessT->push_back({ eid + 2 * nFreeVerts, 2 * vid1 + m1 + nDOFs, faceHess(6 + j, 9 + 2 * k + m1) });

						hessT->push_back({ 2 * vid + m1 + nDOFs, eid1 + 2 * nFreeVerts + nDOFs,  faceHess(9 + 2 * j + m1, 15 + k) });
						hessT->push_back({ eid + 2 * nFreeVerts + nDOFs, 2 * vid1 + m1 + nDOFs,  faceHess(15 + j, 9 + 2 * k + m1) });

					}
					hessT->push_back({ eid + 2 * nFreeVerts, eid1 + 2 * nFreeVerts, faceHess(6 + j, 6 + k) });
					hessT->push_back({ eid + 2 * nFreeVerts, eid1 + 2 * nFreeVerts + nDOFs, faceHess(6 + j, 15 + k) });
					hessT->push_back({ eid + 2 * nFreeVerts + nDOFs, eid1 + 2 * nFreeVerts, faceHess(15 + j, 6 + k) });
					hessT->push_back({ eid + 2 * nFreeVerts + nDOFs, eid1 + 2 * nFreeVerts + nDOFs, faceHess(15 + j, 15 + k) });
				}


			}
		}
	}

	return energy;
}

double WrinkleEditingLocalCWF::computeEnergy(const Eigen::VectorXd& x, Eigen::VectorXd* deriv, Eigen::SparseMatrix<double>* hess, bool isProj)
{
	int nFreeVerts = _freeVids.size();
	int nFreeEdges = _freeEids.size();

	int numFrames = _zvalsList.size() - 2;

	int DOFsPerframe = (2 * nFreeVerts + nFreeEdges);

	int DOFs = numFrames * DOFsPerframe;

	convertVariable2List(x);

	Eigen::VectorXd curDeriv;
	std::vector<Eigen::Triplet<double>> T, curT;

	double energy = 0;
	if (deriv)
	{
		deriv->setZero(DOFs);
	}

	std::vector<Eigen::VectorXd> curKDerivList(numFrames + 1);
	std::vector<std::vector<Eigen::Triplet<double>>> curKTList(numFrames + 1);
	std::vector<double> keList(numFrames + 1);

	auto kineticEnergyPerframe = [&](const tbb::blocked_range<uint32_t>& range) 
	{
		for (uint32_t i = range.begin(); i < range.end(); ++i)
		{
			keList[i] = kineticEnergy(i, deriv ? &curKDerivList[i] : NULL, hess ? &curKTList[i] : NULL, isProj);
		}
	};

	tbb::blocked_range<uint32_t> rangex(0u, (uint32_t)numFrames + 1, GRAIN_SIZE);
	tbb::parallel_for(rangex, kineticEnergyPerframe);

	for (int i = 0; i < _zvalsList.size() - 1; i++)
	{
		energy += keList[i];

		if (deriv)
		{
			if (i == 0)
				deriv->segment(0, DOFsPerframe) += curKDerivList[i].segment(DOFsPerframe, DOFsPerframe);
			else if (i == _zvalsList.size() - 2)
				deriv->segment((i - 1) * DOFsPerframe, DOFsPerframe) += curKDerivList[i].segment(0, DOFsPerframe);
			else
			{
				deriv->segment((i - 1) * DOFsPerframe, 2 * DOFsPerframe) += curKDerivList[i];
			}
		}

		if (hess)
		{
			for (auto& it : curKTList[i])
			{

				if (i == 0)
				{
					if (it.row() >= DOFsPerframe && it.col() >= DOFsPerframe)
						T.push_back({ it.row() - DOFsPerframe, it.col() - DOFsPerframe, it.value() });
				}
				else if (i == _zvalsList.size() - 2)
				{
					if (it.row() < DOFsPerframe && it.col() < DOFsPerframe)
						T.push_back({ it.row() + (i - 1) * DOFsPerframe, it.col() + (i - 1) * DOFsPerframe, it.value() });
				}
				else
				{
					T.push_back({ it.row() + (i - 1) * DOFsPerframe, it.col() + (i - 1) * DOFsPerframe, it.value() });
				}


			}
		}
	}

	std::vector<Eigen::VectorXd> ampDerivList(numFrames), omegaDerivList(numFrames), knoppelDerivList(numFrames);
	std::vector<std::vector<Eigen::Triplet<double>>> ampTList(numFrames), omegaTList(numFrames), knoppelTList(numFrames);
	std::vector<double> ampEnergyList(numFrames), omegaEnergyList(numFrames), knoppelEnergyList(numFrames);

	auto otherEnergiesPerframe = [&](const tbb::blocked_range<uint32_t>& range)
	{
		for (uint32_t i = range.begin(); i < range.end(); ++i)
		{
			ampEnergyList[i] = temporalAmpDifference(i + 1, deriv ? &ampDerivList[i] : NULL, hess ? &ampTList[i] : NULL, isProj);
			omegaEnergyList[i] = temporalOmegaDifference(i + 1, deriv ? &omegaDerivList[i] : NULL, hess ? &omegaTList[i] : NULL, isProj);
			knoppelEnergyList[i] = spatialKnoppelEnergy(i + 1, deriv ? &knoppelDerivList[i] : NULL, hess ? &knoppelTList[i] : NULL, isProj);
		}
	};

	tbb::blocked_range<uint32_t> rangex1(0u, (uint32_t)numFrames, GRAIN_SIZE);
	tbb::parallel_for(rangex1, otherEnergiesPerframe);
	

	for (int i = 0; i < numFrames; i++)
	{
		energy += ampEnergyList[i];
		energy += omegaEnergyList[i];
		energy += knoppelEnergyList[i];

		if (deriv) 
		{
			deriv->segment(i * DOFsPerframe, 2 * nFreeVerts) += ampDerivList[i] + knoppelDerivList[i];
			deriv->segment(i * DOFsPerframe + 2 * nFreeVerts, nFreeEdges) += omegaDerivList[i];
		}

		if (hess) 
		{
			for (auto& it : ampTList[i])
			{
				T.push_back({ i * DOFsPerframe + it.row(), i * DOFsPerframe + it.col(), it.value() });
			}
			for (auto& it : knoppelTList[i])
			{
				T.push_back({ i * DOFsPerframe + it.row(), i * DOFsPerframe + it.col(), it.value() });
			}
			for (auto& it : omegaTList[i])
			{
				T.push_back({ i * DOFsPerframe + it.row() + 2 * nFreeVerts, i * DOFsPerframe + it.col() + 2 * nFreeVerts, it.value() });
			}
		}
	}

	if (hess)
	{
		//std::cout << "num of triplets: " << T.size() << std::endl;
		hess->resize(DOFs, DOFs);
		hess->setFromTriplets(T.begin(), T.end());
	}
	return energy;
}

void WrinkleEditingLocalCWF::solveIntermeditateFrames(Eigen::VectorXd& x, int numIter, double gradTol, double xTol, double fTol, bool isdisplayInfo, std::string workingFolder)
{
	std::cout << "Local CWF model: " << std::endl;
	auto funVal = [&](const Eigen::VectorXd& x, Eigen::VectorXd* grad, Eigen::SparseMatrix<double>* hess, bool isProj) {
		Eigen::VectorXd deriv;
		Eigen::SparseMatrix<double> H;
		double E = computeEnergy(x, grad ? &deriv : NULL, hess ? &H : NULL, isProj);

		if (grad)
		{
			(*grad) = deriv;
		}

		if (hess)
		{
			(*hess) = H;
		}

		return E;
	};
	auto maxStep = [&](const Eigen::VectorXd& x, const Eigen::VectorXd& dir) {
		return 1.0;
	};

	auto getVecNorm = [&](const Eigen::VectorXd& x, double& znorm, double& wnorm) {
		getComponentNorm(x, znorm, wnorm);
	};
	auto saveTmpRes = [&](const Eigen::VectorXd& x, std::string* folder)
	{
		save(x, folder);
	};



	OptSolver::testFuncGradHessian(funVal, x);

	auto x0 = x;
	Eigen::VectorXd grad;
	Eigen::SparseMatrix<double> hess;
	double f0 = funVal(x0, &grad, &hess, false);
	std::cout << "initial f: " << f0 << ", grad norm: " << grad.norm() << ", hess norm: " << hess.norm() << std::endl;
	OptSolver::newtonSolver(funVal, maxStep, x, numIter, gradTol, std::max(1e-16, xTol), std::max(1e-16, fTol), true, getVecNorm, &workingFolder, saveTmpRes);
	std::cout << "before optimization: " << x0.norm() << ", after optimization: " << x.norm() << std::endl;
	std::cout << "solve finished." << std::endl;
}