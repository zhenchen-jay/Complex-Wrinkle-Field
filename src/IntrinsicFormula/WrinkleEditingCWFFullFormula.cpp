#include "../../include/IntrinsicFormula/WrinkleEditingCWFFullFormula.h"
#include "../../include/Optimization/NewtonDescent.h"
#include "../../include/IntrinsicFormula/KnoppelStripePatternEdgeOmega.h"
#include "../../include/WrinkleFieldsEditor.h"
#include <igl/cotmatrix_entries.h>
#include <igl/cotmatrix.h>
#include <igl/doublearea.h>
#include <igl/boundary_loop.h>
#include <Eigen/SPQRSupport>

using namespace IntrinsicFormula;

void WrinkleEditingCWFFullFormula::convertList2Variable(Eigen::VectorXd& x)
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

void WrinkleEditingCWFFullFormula::convertVariable2List(const Eigen::VectorXd& x)
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


double WrinkleEditingCWFFullFormula::spatialKnoppelEnergy(int frameId, Eigen::VectorXd* deriv, std::vector<Eigen::Triplet<double>>* hessT, bool isProj)
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

		std::complex<double> expw0 = std::complex<double>(std::cos(_edgeOmegaList[frameId](eid)), std::sin(_edgeOmegaList[frameId](eid)));

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


double WrinkleEditingCWFFullFormula::kineticEnergy(int frameId, Eigen::VectorXd* deriv, std::vector<Eigen::Triplet<double>>* hessT, bool isProj)
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

void WrinkleEditingCWFFullFormula::computeFaceGradTheta()
{
	int nfaces = _mesh.nFaces();
	int nframes = _zvalsList.size();

    _faceIuInvOmegaOmega.resize(nframes);

	auto computeGradThetaPerframe = [&](const tbb::blocked_range<uint32_t>& range)
	{
		for (uint32_t i = range.begin(); i < range.end(); ++i)
		{
            _faceIuInvOmegaOmega[i].resize(nfaces);
			for (int fid = 0; fid < nfaces; fid++)
			{
                _faceIuInvOmegaOmega[i][fid].resize(3);
				for (int vInF = 0; vInF < 3; vInF++)
				{
					int eid0 = _mesh.faceEdge(fid, (vInF + 1) % 3);
					int eid1 = _mesh.faceEdge(fid, (vInF + 2) % 3);
					Eigen::RowVector3d r0 = _pos.row(_mesh.edgeVertex(eid0, 1)) - _pos.row(_mesh.edgeVertex(eid0, 0));
					Eigen::RowVector3d r1 = _pos.row(_mesh.edgeVertex(eid1, 1)) - _pos.row(_mesh.edgeVertex(eid1, 0));

					Eigen::Matrix2d Iinv, I;
					I << r0.dot(r0), r0.dot(r1), r1.dot(r0), r1.dot(r1);
					Iinv = I.inverse();

					Eigen::Vector2d rhs;
					double w1 = _combinedRefOmegaList[i](eid0);
					double w2 = _combinedRefOmegaList[i](eid1);
					rhs << w1, w2;

                    _faceIuInvOmegaOmega[i][fid][vInF] = Iinv * rhs * rhs.transpose();

				}
			}
		}
	};

	tbb::blocked_range<uint32_t> rangex(0u, (uint32_t)nframes);
	tbb::parallel_for(rangex, computeGradThetaPerframe);
}

double WrinkleEditingCWFFullFormula::workLoadEnergy(int frameId, Eigen::VectorXd* deriv, std::vector<Eigen::Triplet<double>>* hessT, bool isProj)
{
	int nverts = _pos.rows();
	int nfaces = _mesh.nFaces();
	double dt = 1. / (_zvalsList.size() - 1);
	double energy = 0;

	int DOFsPerframe = 2 * nverts;

	if (deriv)
		deriv->setZero(4 * nverts);

	double aveAmp = _refAmpAveList[frameId];

	for (int fid = 0; fid < nfaces; fid++)
	{
		for (int j = 0; j < 3; j++)
		{
			int vid = _mesh.faceVertex(fid, j);
            double coeff = _faceArea[fid];
            coeff *= 100;
			Eigen::Vector2d zvecNew, zvec;
			zvecNew << _zvalsList[frameId + 1][vid].real(), _zvalsList[frameId + 1][vid].imag();
			zvec << _zvalsList[frameId][vid].real(), _zvalsList[frameId][vid].imag();

            Eigen::Matrix2d curMat = _faceIuInvOmegaOmega[frameId][fid][j];
            Eigen::Matrix2d diffMat = _faceIuInvOmegaOmega[frameId + 1][fid][j] - _faceIuInvOmegaOmega[frameId][fid][j];

            double fbCurMat = (curMat.transpose() * curMat).trace();
            double fbDiffMat = (diffMat.transpose() * diffMat).trace();

            double f1 = zvec.dot(zvecNew - zvec);
            double f2 = zvec.squaredNorm();

			double work = f1 * f1 / f2 * fbCurMat + f2 * fbDiffMat;
			fbCurMat = 1000;
            work = fbCurMat * f1 * f1 / f2;
//            work = f2 * fbDiffMat;
			energy += 0.5 * coeff * work;

			if (deriv || hessT)
			{
				Eigen::Vector4d df1, df2;
				df1 << zvecNew[0] - 2 * zvec[0], zvecNew[1] - 2 * zvec[1], zvec[0], zvec[1];
                df2 << 2 * zvec[0], 2 * zvec[1], 0, 0;

				if (deriv)
				{
                    Eigen::Vector4d dwork;
                    dwork = fbCurMat * (2 * f1 / f2 * df1 - f1 * f1 / f2 / f2 * df2) + fbDiffMat * df2;
                    dwork = fbCurMat * (2 * f1 / f2 * df1 - f1 * f1 / f2 / f2 * df2);
					deriv->segment<2>(2 * vid) += 0.5 * coeff * dwork.segment<2>(0);
					deriv->segment<2>(2 * vid + DOFsPerframe) += 0.5 * coeff * dwork.segment<2>(2);
				}
				
				if (hessT)
				{
					Eigen::Matrix4d hf1, hf2, hwork;
					hf1 <<
						-2, 0, 1, 0,
						0, -2, 0, 1,
						1, 0, 0, 0,
						0, 1, 0, 0;

                    hf2 <<
                        2, 0, 0, 0,
                        0, 2, 0, 0,
                        0, 0, 0, 0,
                        0, 0, 0, 0;

                    hwork = fbCurMat * (  2.0 / f2 * (df1 * df1.transpose() + f1 * hf1)
                                        - 2.0 * f1 / f2 / f2 * (df1 * df2.transpose() + df2 * df1.transpose())
                                        - f1 * f1 / f2 / f2 * hf2
                                        + 2.0 * f1 * f1 / f2 / f2 / f2 * (df2 * df2.transpose()))
                          + fbDiffMat * hf2;

                    hwork = fbCurMat * (  2.0 / f2 * (df1 * df1.transpose() + f1 * hf1)
                                          - 2.0 * f1 / f2 / f2 * (df1 * df2.transpose() + df2 * df1.transpose())
                                          - f1 * f1 / f2 / f2 * hf2
                                          + 2.0 * f1 * f1 / f2 / f2 / f2 * (df2 * df2.transpose()));
//                    hwork = fbDiffMat * hf2;
//                    hwork = fbCurMat * (  2.0 / f2 * (df1 * df1.transpose() + f1 * hf1) - 2 * f1 / f2 / f2 * (df1 * df2.transpose()) );

					if (isProj)
                        hwork = SPDProjection(hwork);

					for(int p = 0; p < 2; p++)
						for (int q = 0; q < 2; q++)
						{
							hessT->push_back({ 2 * vid + p, 2 * vid + q, 0.5 * coeff * hwork(p, q) });
							hessT->push_back({ 2 * vid + p + DOFsPerframe, 2 * vid + q + DOFsPerframe, 0.5 * coeff * hwork(2 + p, 2 + q) });
							hessT->push_back({ 2 * vid + p + DOFsPerframe, 2 * vid + q, 0.5 * coeff * hwork(2 + p, q) });
							hessT->push_back({ 2 * vid + p, 2 * vid + q + DOFsPerframe, 0.5 * coeff * hwork(p, 2 + q) });
						}

				}
			}
		}
	}

	return energy;
}

double WrinkleEditingCWFFullFormula::computeEnergy(const Eigen::VectorXd& x, Eigen::VectorXd* deriv, Eigen::SparseMatrix<double>* hess, bool isProj)
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
	
	std::vector<Eigen::VectorXd> curKDerivList(numFrames + 1), curWDerivList(numFrames + 1);
	std::vector<std::vector<Eigen::Triplet<double>>> curKTList(numFrames + 1), curWTList(numFrames + 1);
	std::vector<double> keList(numFrames + 1), woList(numFrames + 1);

	auto kineticEnergyPerframe = [&](const tbb::blocked_range<uint32_t>& range) 
	{
		for (uint32_t i = range.begin(); i < range.end(); ++i)
		{
			keList[i] = kineticEnergy(i, deriv ? &curKDerivList[i] : NULL, hess ? &curKTList[i] : NULL, isProj);
		}
	};

	tbb::blocked_range<uint32_t> rangex(0u, (uint32_t)numFrames + 1, GRAIN_SIZE);
	tbb::parallel_for(rangex, kineticEnergyPerframe);

//	for (uint32_t i =0; i < numFrames + 1; ++i)
//	{
//		woList[i] = workLoadEnergy(i, deriv ? &curWDerivList[i] : NULL, hess ? &curWTList[i] : NULL, isProj);
//	}

	auto workloadEnergyPerframe = [&](const tbb::blocked_range<uint32_t>& range)
	{
		for (uint32_t i = range.begin(); i < range.end(); ++i)
		{
			woList[i] = workLoadEnergy(i, deriv ? &curWDerivList[i] : NULL, hess ? &curWTList[i] : NULL, isProj);
		}
	};

	tbb::parallel_for(rangex, workloadEnergyPerframe);


	for (int i = 0; i < _zvalsList.size() - 1; i++)
	{
		energy += keList[i] + woList[i];

		if (deriv)
		{
			if (i == 0)
				deriv->segment(0, DOFsPerframe) += curKDerivList[i].segment(DOFsPerframe, DOFsPerframe) + curWDerivList[i].segment(DOFsPerframe, DOFsPerframe);
			else if (i == _zvalsList.size() - 2)
				deriv->segment((i - 1) * DOFsPerframe, DOFsPerframe) += curKDerivList[i].segment(0, DOFsPerframe) + curWDerivList[i].segment(0, DOFsPerframe);
			else
			{
				deriv->segment((i - 1) * DOFsPerframe, 2 * DOFsPerframe) += curKDerivList[i] + curWDerivList[i];
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

			for (auto& it : curWTList[i])
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

	std::vector<Eigen::VectorXd> ampDerivList(numFrames), knoppelDerivList(numFrames);
	std::vector<std::vector<Eigen::Triplet<double>>> ampTList(numFrames), knoppelTList(numFrames);
	std::vector<double> ampEnergyList(numFrames), knoppelEnergyList(numFrames);

	auto otherEnergiesPerframe = [&](const tbb::blocked_range<uint32_t>& range)
	{
		for (uint32_t i = range.begin(); i < range.end(); ++i)
		{
			//ampEnergyList[i] = temporalAmpDifference(i + 1, deriv ? &ampDerivList[i] : NULL, hess ? &ampTList[i] : NULL, isProj);
			knoppelEnergyList[i] = spatialKnoppelEnergy(i + 1, deriv ? &knoppelDerivList[i] : NULL, hess ? &knoppelTList[i] : NULL, isProj);
		}
	};

	tbb::blocked_range<uint32_t> rangex1(0u, (uint32_t)numFrames, GRAIN_SIZE);
	tbb::parallel_for(rangex1, otherEnergiesPerframe);
	

	for (int i = 0; i < numFrames; i++)
	{
		//energy += ampEnergyList[i];
		energy += knoppelEnergyList[i];

		if (deriv) 
		{
			deriv->segment(i * DOFsPerframe, DOFsPerframe) += knoppelDerivList[i];
			//deriv->segment(i * DOFsPerframe, 2 * nverts) += ampDerivList[i];
		}

		if (hess) 
		{
			/*for (auto& it : ampTList[i])
			{
				T.push_back({ i * DOFsPerframe + it.row(), i * DOFsPerframe + it.col(), it.value() });
			}*/
			for (auto& it : knoppelTList[i])
			{
				T.push_back({ i * DOFsPerframe + it.row(), i * DOFsPerframe + it.col(), it.value() });
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

void WrinkleEditingCWFFullFormula::solveIntermeditateFrames(Eigen::VectorXd& x, int numIter, double gradTol, double xTol, double fTol, bool isdisplayInfo, std::string workingFolder)
{
	std::cout << "Full formula CWF model: " << std::endl;
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

	computeFaceGradTheta();

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