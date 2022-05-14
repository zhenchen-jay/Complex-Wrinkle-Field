#include "../../include/IntrinsicFormula/WrinkleEditingNaiveCWF.h"
#include "../../include/Optimization/NewtonDescent.h"
#include "../../include/IntrinsicFormula/KnoppelStripePatternEdgeOmega.h"
#include "../../include/WrinkleFieldsEditor.h"
#include <igl/cotmatrix_entries.h>
#include <igl/cotmatrix.h>
#include <igl/doublearea.h>
#include <igl/boundary_loop.h>
#include <Eigen/SPQRSupport>

using namespace IntrinsicFormula;

void WrinkleEditingNaiveCWF::convertList2Variable(Eigen::VectorXd& x)
{
	int nverts = _pos.rows();
	int nedges = _mesh.nEdges();

	int numFrames = _zvalsList.size() - 2;

	int DOFsPerframe = 2 * nverts;

	int DOFs = numFrames * DOFsPerframe;

	x.setZero(DOFs);

	for (int i = 0; i < numFrames; i++)
	{
		for (int j = 0; j < nverts; j++)
		{
			x(i * DOFsPerframe + 2 * j) = _zvalsList[i + 1][j].real();
			x(i * DOFsPerframe + 2 * j + 1) = _zvalsList[i + 1][j].imag();
		}
	}
}

void WrinkleEditingNaiveCWF::convertVariable2List(const Eigen::VectorXd& x)
{
	int nverts = _pos.rows();
	int nedges = _mesh.nEdges();

	int numFrames = _zvalsList.size() - 2;

	int DOFsPerframe = 2 * nverts;

	int DOFs = numFrames * DOFsPerframe;

	for (int i = 0; i < numFrames; i++)
	{
		for (int j = 0; j < nverts; j++)
		{
			_zvalsList[i + 1][j] = std::complex<double>(x(i * DOFsPerframe + 2 * j), x(i * DOFsPerframe + 2 * j + 1));
		}
	}
}

double WrinkleEditingNaiveCWF::temporalAmpDifference(int frameId, Eigen::VectorXd* deriv, std::vector<Eigen::Triplet<double>>* hessT, bool isProj)
{
	int nverts = _pos.rows();
	double energy = 0;

	if (deriv)
		deriv->setZero(2 * nverts);
	if (hessT)
		hessT->clear();

	for (int vid = 0; vid < nverts; vid++)
	{
		double ampSq = _zvalsList[frameId][vid].real() * _zvalsList[frameId][vid].real() +
			_zvalsList[frameId][vid].imag() * _zvalsList[frameId][vid].imag();
		double refAmpSq = _combinedRefAmpList[frameId][vid] * _combinedRefAmpList[frameId][vid];
		double ca = _spatialAmpRatio * _vertArea(vid) / (_refAmpAveList[frameId] * _refAmpAveList[frameId]);

		energy += ca * (ampSq - refAmpSq) * (ampSq - refAmpSq);

		if (deriv)
		{
			(*deriv)(2 * vid) += 2.0 * ca * (ampSq - refAmpSq) * (2.0 * _zvalsList[frameId][vid].real());
			(*deriv)(2 * vid + 1) += 2.0 * ca * (ampSq - refAmpSq) * (2.0 * _zvalsList[frameId][vid].imag());
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
					hessT->push_back({ 2 * vid + k, 2 * vid + l, tmpHess(k, l) });
		}
	}
	return energy;
}


double WrinkleEditingNaiveCWF::spatialKnoppelEnergy(int frameId, Eigen::VectorXd* deriv, std::vector<Eigen::Triplet<double>>* hessT, bool isProj)
{
	double energy = 0;
	int nedges = _mesh.nEdges();
	int nverts = _pos.rows();
	double aveAmp = _refAmpAveList[frameId];
	std::vector<Eigen::Triplet<double>> AT;
	AT.clear();

	int maxFreeVid = 0;

	for (int eid = 0; eid < nedges; eid++)
	{
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
			AT.push_back({ 2 * vid0, 2 * vid0, r1 * r1 * ce });
			AT.push_back({ 2 * vid0 + 1, 2 * vid0 + 1, r1 * r1 * ce });

			AT.push_back({ 2 * vid1, 2 * vid1, r0 * r0 * ce });
			AT.push_back({ 2 * vid1 + 1, 2 * vid1 + 1, r0 * r0 * ce });


			AT.push_back({ 2 * vid0, 2 * vid1, -ce * (expw0.real()) * r0 * r1 });
			AT.push_back({ 2 * vid0 + 1, 2 * vid1, -ce * (-expw0.imag()) * r0 * r1 });
			AT.push_back({ 2 * vid0, 2 * vid1 + 1, -ce * (expw0.imag()) * r0 * r1 });
			AT.push_back({ 2 * vid0 + 1, 2 * vid1 + 1, -ce * (expw0.real()) * r0 * r1 });

			AT.push_back({ 2 * vid1, 2 * vid0, -ce * (expw0.real()) * r0 * r1 });
			AT.push_back({ 2 * vid1, 2 * vid0 + 1, -ce * (-expw0.imag()) * r0 * r1 });
			AT.push_back({ 2 * vid1 + 1, 2 * vid0, -ce * (expw0.imag()) * r0 * r1 });
			AT.push_back({ 2 * vid1 + 1, 2 * vid0 + 1, -ce * (expw0.real()) * r0 * r1 });
		}
	}

	if (deriv || hessT)
	{
		Eigen::SparseMatrix<double> A;
		
		A.resize(2 * nverts, 2 * nverts);
		A.setFromTriplets(AT.begin(), AT.end());

		// check whether A is PD


		if (deriv)
		{
			Eigen::VectorXd fvals(2 * nverts);
			for (int i = 0; i < nverts; i++)
			{
				fvals(2 * i) = _zvalsList[frameId][i].real();
				fvals(2 * i + 1) = _zvalsList[frameId][i].imag();
			}
			(*deriv) = A * fvals;
		}

		if (hessT)
			(*hessT) = AT;
	}

	return energy;
}


double WrinkleEditingNaiveCWF::kineticEnergy(int frameId, Eigen::VectorXd* deriv, std::vector<Eigen::Triplet<double>>* hessT, bool isProj)
{
	int nverts = _pos.rows();
	double dt = 1. / (_zvalsList.size() - 1);
	double energy = 0;

	int DOFsPerframe = 2 * nverts;

	if (deriv)
		deriv->setZero(4 * nverts);

	for (int vid = 0; vid < nverts; vid++)
	{
		Eigen::Vector2d diff;
		double coeff = _vertWeight(vid) / (dt * dt) * _vertArea[vid];
		diff << (_zvalsList[frameId + 1][vid] - _zvalsList[frameId][vid]).real(), (_zvalsList[frameId + 1][vid] - _zvalsList[frameId][vid]).imag();
		energy += 0.5 * coeff * diff.squaredNorm();

		if (deriv)
		{
			deriv->segment<2>(2 * vid) += -coeff * diff;
			deriv->segment<2>(2 * vid + DOFsPerframe) += coeff * diff;
		}

		if (hessT)
		{
			hessT->push_back({ 2 * vid, 2 * vid, coeff });
			hessT->push_back({ 2 * vid, DOFsPerframe + 2 * vid, -coeff });

			hessT->push_back({ 2 * vid + 1, 2 * vid + 1, coeff });
			hessT->push_back({ 2 * vid + 1, DOFsPerframe + 2 * vid + 1, -coeff });

			hessT->push_back({ DOFsPerframe + 2 * vid, DOFsPerframe + 2 * vid, coeff });
			hessT->push_back({ DOFsPerframe + 2 * vid, 2 * vid, -coeff });
			
			hessT->push_back({ DOFsPerframe + 2 * vid + 1, DOFsPerframe + 2 * vid + 1, coeff });
			hessT->push_back({ DOFsPerframe + 2 * vid + 1, 2 * vid + 1, -coeff });

		}
	}
	return energy;
}

double WrinkleEditingNaiveCWF::fullKneticEnergy(Eigen::VectorXd* deriv, Eigen::SparseMatrix<double>* hess)
{
	int nverts = _pos.rows();
	int numFrames = _zvalsList.size();

	int DOFsPerframe = 2 * nverts;

	int DOFs = numFrames * DOFsPerframe;

	std::vector<Eigen::Triplet<double>> T;

	double energy = 0;
	if (deriv)
	{
		deriv->setZero(DOFs);
	}

	for (int i = 0; i < _zvalsList.size() - 1; i++)
	{
		Eigen::VectorXd curDeriv;
		std::vector<Eigen::Triplet<double>> curT;
		energy += kineticEnergy(i, deriv ? &curDeriv : NULL, hess ? &curT : NULL, false);

		if (deriv)
		{
			deriv->segment(i * DOFsPerframe, 2 * DOFsPerframe) += curDeriv;	
		}
		if (hess)
		{
			for (auto& it : curT)
			{
				T.push_back({ it.row() + i * DOFsPerframe, it.col() + i * DOFsPerframe , it.value() });
			}
		}
	}

	if (hess)
	{
		hess->resize(DOFs, DOFs);
		hess->setFromTriplets(T.begin(), T.end());
	}
	return energy;
}

double WrinkleEditingNaiveCWF::computeEnergy(const Eigen::VectorXd& x, Eigen::VectorXd* deriv, Eigen::SparseMatrix<double>* hess, bool isProj)
{
	int nverts = _pos.rows();
	int nedges = _mesh.nEdges();

	int numFrames = _zvalsList.size() - 2;

	int DOFsPerframe = 2 * nverts;

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
	

	/*
	std::vector<Eigen::VectorXd> ampDerivList(numFrames), knoppelDerivList(numFrames);
	std::vector<std::vector<Eigen::Triplet<double>>> ampTList(numFrames), knoppelTList(numFrames);
	std::vector<double> ampEnergyList(numFrames), knoppelEnergyList(numFrames);

	auto otherEnergiesPerframe = [&](const tbb::blocked_range<uint32_t>& range)
	{
		for (uint32_t i = range.begin(); i < range.end(); ++i)
		{
			ampEnergyList[i] = temporalAmpDifference(i + 1, deriv ? &ampDerivList[i] : NULL, hess ? &ampTList[i] : NULL, isProj);
			knoppelEnergyList[i] = spatialKnoppelEnergy(i + 1, deriv ? &knoppelDerivList[i] : NULL, hess ? &knoppelTList[i] : NULL, isProj);
		}
	};

	tbb::blocked_range<uint32_t> rangex1(0u, (uint32_t)numFrames, GRAIN_SIZE);
	tbb::parallel_for(rangex1, otherEnergiesPerframe);
	

	for (int i = 0; i < numFrames; i++)
	{
		energy += ampEnergyList[i];
		energy += knoppelEnergyList[i];

		if (deriv) 
		{
			deriv->segment(i * DOFsPerframe, DOFsPerframe) += knoppelDerivList[i];
			deriv->segment(i * DOFsPerframe, 2 * nverts) += ampDerivList[i];
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
		}
	}
	*/
	if (hess)
	{
		//std::cout << "num of triplets: " << T.size() << std::endl;
		hess->resize(DOFs, DOFs);
		hess->setFromTriplets(T.begin(), T.end());
	}
	return energy;
}

void WrinkleEditingNaiveCWF::solveIntermeditateFrames(Eigen::VectorXd& x, int numIter, double gradTol, double xTol, double fTol, bool isdisplayInfo, std::string workingFolder)
{
	std::cout << "Naive CWF model: " << std::endl;
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

void WrinkleEditingNaiveCWF::testFullKneticEnergy()
{
	Eigen::VectorXd deriv;
	Eigen::SparseMatrix<double> hess;

	double ke = fullKneticEnergy(&deriv, &hess);

	std::cout << "ke: " << ke << ", deriv: " << deriv.norm() << ", hess norm: " << hess.norm() << std::endl;

	std::vector<std::vector<std::complex<double>>> zLists = _zvalsList;

	int nverts = _zvalsList[0].size();
	
	auto convertList2Vec = [&](const std::vector<std::vector<std::complex<double>>>& zlist, Eigen::VectorXd& x)
	{
		int nframes = zlist.size();
		int npoints = zlist[0].size();
		x.resize(2 * npoints * nframes);

		for (int i = 0; i < nframes; i++)
		{
			for (int j = 0; j < npoints; j++)
			{
				x(i * 2 * npoints + 2 * j) = zlist[i][j].real();
				x(i * 2 * npoints + 2 * j + 1) = zlist[i][j].imag();
			}
		}
	};

	auto convertVec2List = [&](const Eigen::VectorXd& x, std::vector<std::vector<std::complex<double>>>& zlist)
	{
		int nframes = zlist.size();
		int npoints = zlist[0].size();
		
		for (int i = 0; i < nframes; i++)
		{
			for (int j = 0; j < npoints; j++)
			{
				zlist[i][j] = { x(i * 2 * npoints + 2 * j) , x(i * 2 * npoints + 2 * j + 1) };
			}
		}
	};

	Eigen::VectorXd x0, x;
	convertList2Vec(_zvalsList, x);

	x0 = x;
	x0.setZero();
	convertVec2List(x0, _zvalsList);

	Eigen::VectorXd deriv0;
	Eigen::SparseMatrix<double> hess0;
	double ke0 = fullKneticEnergy(&deriv0, &hess0);

	std::cout << "hess check: " << (hess - hess0).norm() << ", grad check: " << deriv0.norm() << ", grad-hess: " << (deriv - hess0 * x).norm() << ", energy-hess: " << (ke - 0.5 * x.dot(hess0 * x)) << std::endl;
	_zvalsList = zLists;

	std::vector<Eigen::Triplet<double>> projT;
	int row = 0;
	for (int i = 0; i < _zvalsList.size(); i++)
	{
		if (i == 0 || i == _zvalsList.size() - 1)
			continue;
		for (int j = 0; j < nverts; j++)
		{
			projT.push_back({ row, i * (2 * nverts) + 2 * j, 1.0 });
			projT.push_back({ row + 1, i * (2 * nverts) + 2 * j + 1, 1.0 });
			row += 2;
		}
	}

	Eigen::SparseMatrix<double> projM;
	projM.resize(row, _zvalsList.size() * 2 * nverts);

	projM.setFromTriplets(projT.begin(), projT.end());

	Eigen::VectorXd testx, testDeriv, testDeriv0;
	Eigen::SparseMatrix<double> testHess, testHess0;
	convertList2Variable(testx);
	double ke1 = computeEnergy(testx, &testDeriv, &testHess, false);

	Eigen::SparseMatrix<double> projHess = projM * hess * projM.transpose();

	Eigen::VectorXd fixedVar;
	convertList2Vec(_zvalsList, fixedVar);
	fixedVar = fixedVar - projM.transpose() * testx;

	Eigen::VectorXd zeroX = testx;
	zeroX.setZero();
	double testke0 = computeEnergy(zeroX, &testDeriv0, &testHess0, false);
	_zvalsList = zLists;

	std::cout << "testKe0 - quad: " << testke0 - 0.5 * fixedVar.dot(hess0 * fixedVar) << std::endl;
	Eigen::VectorXd by = projM * hess0 * fixedVar;
	double cy = 0.5 * fixedVar.dot(hess0 * fixedVar);
	
	std::cout << "ke1 - ke: " << (ke - ke1) << ", grad check: " << (testDeriv - projM * deriv).norm() << " " << (testDeriv - by - projHess * testx).norm() << ", hess check: " << (testHess - projHess).norm() << std::endl;

}

void WrinkleEditingNaiveCWF::testKneticEnergy(int frameId)
{
	Eigen::VectorXd deriv;
	std::vector<Eigen::Triplet<double>> hessT;
	Eigen::SparseMatrix<double> hess;

	std::vector<std::complex<double>> prevZvals = _zvalsList[frameId];
	std::vector<std::complex<double>> curZvals = _zvalsList[frameId + 1];

	double ke = kineticEnergy(frameId, &deriv, &hessT, false);
	hess.resize(deriv.rows(), deriv.rows());
	hess.setFromTriplets(hessT.begin(), hessT.end());

	std::cout << "ke: " << ke << ", deriv: " << deriv.norm() << ", hess norm: " << hess.norm() << std::endl;

	int nverts = _zvalsList[frameId].size();

	std::vector<std::complex<double>> zeroZvals;
	zeroZvals.resize(nverts, 0);
	_zvalsList[frameId] = zeroZvals;
	_zvalsList[frameId + 1] = zeroZvals;

	

	Eigen::VectorXd deriv0;
	std::vector<Eigen::Triplet<double>> hessT0;
	Eigen::SparseMatrix<double> hess0;

	double ke0 = kineticEnergy(frameId, &deriv0, &hessT0, false);
	hess0.resize(deriv.rows(), deriv.rows());
	hess0.setFromTriplets(hessT0.begin(), hessT0.end());

	Eigen::VectorXd x(4 * nverts);
	for (int i = 0; i < nverts; i++)
	{
		x(2 * i) = prevZvals[i].real();
		x(2 * i + 1) = prevZvals[i].imag();

		x(2 * nverts + 2 * i) = curZvals[i].real();
		x(2 * nverts + 2 * i + 1) = curZvals[i].imag();
	}


	std::cout << "hess const check: " << (hess - hess0).norm() << std::endl;
	std::cout << "hess gradient check: " << (deriv - hess0 * x).norm() << std::endl;
	std::cout << "energy-hess check: " << (0.5 * x.dot(hess0 * x) + x.dot(deriv0) + ke0 - ke) << ", deriv0: " << deriv0.norm() << std::endl;

	Eigen::VectorXd dir = deriv;
	dir.setRandom();
	_zvalsList[frameId] = prevZvals;
	_zvalsList[frameId + 1] = curZvals;

	deriv = hess0 * x;
	
	for (int i = 3; i < 10; i++)
	{
		double eps = std::pow(0.1, i);
		for (int j = 0; j < nverts; j++)
		{
			_zvalsList[frameId][j] = { prevZvals[j].real() + eps * dir(2 * j), prevZvals[j].imag() + eps * dir(2 * j + 1) };
			_zvalsList[frameId + 1][j] = { curZvals[j].real() + eps * dir(2 * j + 2 * nverts), curZvals[j].imag() + eps * dir(2 * j + 1 + 2 * nverts) };
		}

		Eigen::VectorXd x(4 * nverts);
		for (int i = 0; i < nverts; i++)
		{
			x(2 * i) = _zvalsList[frameId][i].real();
			x(2 * i + 1) = _zvalsList[frameId][i].imag();

			x(2 * nverts + 2 * i) = _zvalsList[frameId + 1][i].real();
			x(2 * nverts + 2 * i + 1) = _zvalsList[frameId + 1][i].imag();
		}

		Eigen::VectorXd deriv1;
		double ke1 = kineticEnergy(frameId, &deriv1, NULL, false);
		deriv1 = hess0 * x;
		std::cout << "eps: " << eps << std::endl;
		std::cout << "energy-gradient: " << (ke1 - ke) / eps - dir.dot(deriv) << std::endl;
		std::cout << "gradient-hessian: " << ((deriv1 - deriv) / eps - hess * dir).norm() << std::endl;
	}
	_zvalsList[frameId] = prevZvals;
	_zvalsList[frameId + 1] = curZvals;

}