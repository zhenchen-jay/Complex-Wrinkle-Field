#include <iostream>
#include "../../include/DynamicInterpolation/InterpolateKeyFrames.h"

void InterpolateKeyFrames::convertList2Variable(Eigen::VectorXd& x)
{
	int nbaseVerts = _basePos.rows();
	int numFrames = _vertValsList.size() - 2;

	int DOFs = numFrames * nbaseVerts * 4;

	x.setZero(DOFs);

	for (int i = 0; i < numFrames; i++)
	{
		for (int j = 0; j < nbaseVerts; j++)
		{
			x(i * 4 * nbaseVerts + 2 * j) = _vertValsList[i + 1][j].real();
			x(i * 4 * nbaseVerts + 2 * j + 1) = _vertValsList[i + 1][j].imag();

			x(i * 4 * nbaseVerts + 2 * nbaseVerts + 2 * j) = _wList[i + 1](j, 0);
			x(i * 4 * nbaseVerts + 2 * nbaseVerts + 2 * j + 1) = _wList[i + 1](j, 1);
		}
	}
}

void InterpolateKeyFrames::convertVariable2List(const Eigen::VectorXd& x)
{
	int nbaseVerts = _basePos.rows();
	int numFrames = _vertValsList.size() - 2;

	for (int i = 0; i < numFrames; i++)
	{
		for (int j = 0; j < nbaseVerts; j++)
		{
			_vertValsList[i + 1][j] = std::complex<double>(x(i * 4 * nbaseVerts + 2 * j), x(i * 4 * nbaseVerts + 2 * j + 1));
			_wList[i + 1](j, 0) = x(i * 4 * nbaseVerts + 2 * nbaseVerts + 2 * j);
			_wList[i + 1](j, 1) = x(i * 4 * nbaseVerts + 2 * nbaseVerts + 2 * j + 1);
		}
	}
}

void InterpolateKeyFrames::initializeLamdaMu(Eigen::VectorXd& lambda, double& mu, double initMu)
{
	int nbaseVerts = _basePos.rows();
	int numFrames = _vertValsList.size() - 2;

	lambda.setZero(nbaseVerts * numFrames);
	mu = initMu;
}

double InterpolateKeyFrames::computePerVertexPenalty(const std::vector<std::complex<double>>& zvals, const Eigen::MatrixXd& w, int vid, Eigen::Vector4d* deriv, Eigen::Matrix4d* hess, bool isProj)
{
	double rsq = zvals[vid].real() * zvals[vid].real() + zvals[vid].imag() * zvals[vid].imag();
	double wsq = w(vid, 0) * w(vid, 0) + w(vid, 1) * w(vid, 1);
	double energy = (rsq * wsq - 1) * (rsq * wsq - 1);

	if (deriv || hess)
	{
		Eigen::Vector4d drsq, dwsq, drwsq;
		Eigen::Matrix4d hrsq, hwsq, hrwsq;

		drsq << 2.0 * zvals[vid].real(), 2.0 * zvals[vid].imag(), 0 , 0;
		dwsq << 0, 0, 2.0 * w(vid, 0), 2.0 * w(vid, 1);

		drwsq = drsq * wsq + dwsq * rsq;
		
		if (deriv)
		{
			deriv->setZero();
			(*deriv) = 2 * (rsq * wsq - 1) * drwsq;
		}
		
		if (hess)
		{
			hess->setZero();

			hrsq.setIdentity();
			hwsq.setIdentity();

			hrsq *= 2.0;
			hwsq *= 2.0;

			hrsq(2, 2) = 0;
			hrsq(3, 3) = 0;
			hwsq(0, 0) = 0;
			hwsq(1, 1) = 0;

			hrwsq = hrsq * wsq + hwsq * rsq + drsq * dwsq.transpose() + dwsq * drsq.transpose();

			(*hess) = 2 * (rsq * wsq - 1) * hrwsq + 2 * drwsq * drwsq.transpose();
			
			if (isProj)
			{
				//(*hess) = SPDProjection(*hess);
//				if((rsq * wsq - 1) > 0)
//					(*hess) = 2 * (rsq * wsq - 1) * (hrsq * wsq + hwsq * rsq) + 2 * drwsq * drwsq.transpose();
//				else
					(*hess) = SPDProjection(*hess);
			}
				
		}
		
	}

	return energy;
}

double InterpolateKeyFrames::computePerVertexConstraint(const std::vector<std::complex<double>>& zvals, const Eigen::MatrixXd& w, int vid, Eigen::Vector4d* deriv, Eigen::Matrix4d* hess, bool isProj)
{
	double rsq = zvals[vid].real() * zvals[vid].real() + zvals[vid].imag() * zvals[vid].imag();
	double wsq = w(vid, 0) * w(vid, 0) + w(vid, 1) * w(vid, 1);
	double energy = (rsq * wsq - 1);

	if (deriv || hess)
	{
		Eigen::Vector4d drsq, dwsq, drwsq;
		Eigen::Matrix4d hrsq, hwsq, hrwsq;

		drsq << 2.0 * zvals[vid].real(), 2.0 * zvals[vid].imag(), 0, 0;
		dwsq << 0, 0, 2.0 * w(vid, 0), 2.0 * w(vid, 1);

		drwsq = drsq * wsq + dwsq * rsq;

		if (deriv)
		{
			deriv->setZero();
			(*deriv) = drwsq;
		}

		if (hess)
		{
			hess->setZero();

			hrsq.setIdentity();
			hwsq.setIdentity();

			hrsq *= 2.0;
			hwsq *= 2.0;

			hrsq(2, 2) = 0;
			hrsq(3, 3) = 0;
			hwsq(0, 0) = 0;
			hwsq(1, 1) = 0;

			hrwsq = hrsq * wsq + hwsq * rsq + drsq * dwsq.transpose() + dwsq * drsq.transpose();

			(*hess) = hrwsq;

			if (isProj)
			{
				(*hess) = SPDProjection(*hess);
			}

		}

	}

	return energy;
}

double InterpolateKeyFrames::computePerFramePenalty(const std::vector<std::complex<double>>& zvals, const Eigen::MatrixXd& w, Eigen::VectorXd* deriv, std::vector<Eigen::Triplet<double>>* hessT, bool isProj)
{
	int nbaseVerts = _basePos.rows();
	double energy = 0;

	std::vector<double> energyList(nbaseVerts);
	std::vector<Eigen::Vector4d> derivList(nbaseVerts);
	std::vector<Eigen::Matrix4d> hessList(nbaseVerts);

	auto computeEnergy = [&](const tbb::blocked_range<uint32_t>& range) {
		for (uint32_t i = range.begin(); i < range.end(); ++i)
		{
			energyList[i] = computePerVertexPenalty(zvals, w, i, deriv ? &derivList[i] : NULL, hessT ? &hessList[i] : NULL, isProj);
		}
	};

	tbb::blocked_range<uint32_t> rangex(0u, (uint32_t)nbaseVerts, GRAIN_SIZE);
	tbb::parallel_for(rangex, computeEnergy);

	if (deriv)
	{
		deriv->setZero(4 * nbaseVerts);
	}
	if (hessT)
	{
		hessT->clear();
	}

	for (int i = 0; i < nbaseVerts; i++)
	{
		energy += energyList[i];
		if (deriv)
		{
			for (int j = 0; j < 2; j++)
			{
				(*deriv)(2 * i + j) = derivList[i](j);
				(*deriv)(2 * nbaseVerts + 2 * i + j) = derivList[i](2 + j);
			}
		}

		if (hessT)
		{
			for (int j = 0; j < 2; j++)
			{
				for (int k = 0; k < 2; k++)
				{
					hessT->push_back({ 2 * i + j, 2 * i + k, hessList[i](j, k) });
					hessT->push_back({ 2 * nbaseVerts + 2 * i + j, 2 * i + k, hessList[i](2 + j, k) });
					hessT->push_back({ 2 * i + j, 2 * nbaseVerts + 2 * i + k, hessList[i](j, 2 + k) });
					hessT->push_back({ 2 * nbaseVerts + 2 * i + j, 2 * nbaseVerts + 2 * i + k, hessList[i](2 + j, 2 + k) });
				}
			}
				
		}
	}
	return energy;
}

double InterpolateKeyFrames::computePenalty(const Eigen::VectorXd& x, Eigen::VectorXd* deriv, Eigen::SparseMatrix<double>* hess, bool isProj)
{
	int nbaseVerts = _basePos.rows();
	int numFrames = _vertValsList.size() - 2;
	int DOFs = numFrames * nbaseVerts * 4;

	convertVariable2List(x);

	Eigen::VectorXd curDeriv;
	std::vector<Eigen::Triplet<double>> T, curT;

	double energy = 0;
	if (deriv)
	{
		deriv->setZero(DOFs);
	}
	
	for (int i = 1; i < _vertValsList.size() - 1; i++)
	{
		energy += computePerFramePenalty(_vertValsList[i], _wList[i], deriv ? &curDeriv : NULL, hess ? &curT : NULL, isProj);
		if (deriv)
		{
			deriv->segment(4 * (i - 1) * nbaseVerts, 4 * nbaseVerts) = curDeriv;
		}
		if (hess)
		{
			for (auto& it : curT)
				T.push_back({ it.row() + 4 * (i - 1) * nbaseVerts, it.col() + 4 * (i - 1) * nbaseVerts, it.value() });
			curT.clear();
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

Eigen::VectorXd InterpolateKeyFrames::getConstraints(const Eigen::VectorXd& x, std::vector<Eigen::SparseVector<double>>* deriv, std::vector<Eigen::SparseMatrix<double>>* hess, bool isProj)
{
	convertVariable2List(x);
	int nbaseVerts = _basePos.rows();
	int numFrames = _vertValsList.size() - 2;

	int nEqs = nbaseVerts * numFrames;
	int nDOFs = x.size();

	Eigen::VectorXd constraints(nEqs);
	if (deriv)
		deriv->resize(nEqs);
	if (hess)
		hess->resize(nEqs);

	auto computeConstraintPerFrame = [&](const tbb::blocked_range<uint32_t>& range)
	{
		for (uint32_t i = range.begin(); i < range.end(); ++i)
		{
			for (int j = 0; j < nbaseVerts; ++j)
			{
				double constraint;
				Eigen::Vector4d dconstraints;
				Eigen::Matrix4d hconstarints;
				constraint = computePerVertexConstraint(_vertValsList[i], _wList[i], j, deriv ? &dconstraints : NULL, hess ? &hconstarints : NULL, isProj);

				int baseDOFs = 4 * (i - 1) * nbaseVerts;
				int basenEqs = (i - 1) * nbaseVerts;
				constraints(basenEqs + j) = constraint;
				std::vector<Eigen::Triplet<double>> hT;

				if (deriv)
				{
					deriv->at(basenEqs + j).resize(nDOFs);
					deriv->at(basenEqs + j).setZero();
					for (int k = 0; k < 2; k++)
					{
						deriv->at(basenEqs + j).coeffRef(baseDOFs + 2 * i + k) = dconstraints(k);
						deriv->at(basenEqs + j).coeffRef(baseDOFs + 2 * nbaseVerts + 2 * i + k) = dconstraints(2 + k);
					}
				}

				if (hess)
				{
					for (int j = 0; j < 2; j++)
					{
						for (int k = 0; k < 2; k++)
						{
							hT.push_back(Eigen::Triplet<double>(baseDOFs + 2 * i + j, baseDOFs + 2 * i + k, hconstarints(j, k)));
							hT.push_back(Eigen::Triplet<double>(baseDOFs + 2 * nbaseVerts + 2 * i + j, baseDOFs + 2 * i + k, hconstarints(2 + j, k) ));
							hT.push_back(Eigen::Triplet<double>(baseDOFs + 2 * i + j, baseDOFs + 2 * nbaseVerts + 2 * i + k, hconstarints(j, 2 + k) ));
							hT.push_back(Eigen::Triplet<double>(baseDOFs + 2 * nbaseVerts + 2 * i + j, baseDOFs + 2 * nbaseVerts + 2 * i + k, hconstarints(2 + j, 2 + k) ));
						}
					}
					hess->at(basenEqs + j).resize(nDOFs, nDOFs);
					hess->at(basenEqs + j).setFromTriplets(hT.begin(), hT.end());
				}
			}
		}
	};

	tbb::blocked_range<uint32_t> rangex(1u, (uint32_t)(_vertValsList.size() - 1), GRAIN_SIZE);
	tbb::parallel_for(rangex, computeConstraintPerFrame);
	return constraints;
}

Eigen::VectorXd InterpolateKeyFrames::getConstraintsPenalty(const Eigen::VectorXd& x, std::vector<Eigen::SparseVector<double>>* deriv, std::vector<Eigen::SparseMatrix<double>>* hess, bool isProj)
{
	convertVariable2List(x);
	int nbaseVerts = _basePos.rows();
	int numFrames = _vertValsList.size() - 2;

	int nEqs = nbaseVerts * numFrames;
	int nDOFs = x.size();

	Eigen::VectorXd constraints(nEqs);
	if (deriv)
		deriv->resize(nEqs);
	if (hess)
		hess->resize(nEqs);

	auto computeConstraintPerFrame = [&](const tbb::blocked_range<uint32_t>& range)
	{
		for (uint32_t i = range.begin(); i < range.end(); ++i)
		{
			for (int j = 0; j < nbaseVerts; ++j)
			{
				double constraint;
				Eigen::Vector4d dconstraints;
				Eigen::Matrix4d hconstarints;
				constraint = computePerVertexPenalty(_vertValsList[i], _wList[i], j, deriv ? &dconstraints : NULL, hess ? &hconstarints : NULL, isProj);

				int baseDOFs = 4 * (i - 1) * nbaseVerts;
				int basenEqs = (i - 1) * nbaseVerts;
				constraints(basenEqs + j) = constraint;
				std::vector<Eigen::Triplet<double>> hT;

				if (deriv)
				{
					deriv->at(basenEqs + j).resize(nDOFs);
					deriv->at(basenEqs + j).setZero();
					for (int k = 0; k < 2; k++)
					{
						deriv->at(basenEqs + j).coeffRef(baseDOFs + 2 * i + k) = dconstraints(k);
						deriv->at(basenEqs + j).coeffRef(baseDOFs + 2 * nbaseVerts + 2 * i + k) = dconstraints(2 + k);
					}
				}

				if (hess)
				{
					for (int j = 0; j < 2; j++)
					{
						for (int k = 0; k < 2; k++)
						{
							hT.push_back(Eigen::Triplet<double>(baseDOFs + 2 * i + j, baseDOFs + 2 * i + k, hconstarints(j, k)));
							hT.push_back(Eigen::Triplet<double>(baseDOFs + 2 * nbaseVerts + 2 * i + j, baseDOFs + 2 * i + k, hconstarints(2 + j, k)));
							hT.push_back(Eigen::Triplet<double>(baseDOFs + 2 * i + j, baseDOFs + 2 * nbaseVerts + 2 * i + k, hconstarints(j, 2 + k)));
							hT.push_back(Eigen::Triplet<double>(baseDOFs + 2 * nbaseVerts + 2 * i + j, baseDOFs + 2 * nbaseVerts + 2 * i + k, hconstarints(2 + j, 2 + k)));
						}
					}
					hess->at(basenEqs + j).resize(nDOFs, nDOFs);
					hess->at(basenEqs + j).setFromTriplets(hT.begin(), hT.end());
				}
			}
		}
	};

	tbb::blocked_range<uint32_t> rangex(1u, (uint32_t)(_vertValsList.size() - 1), GRAIN_SIZE);
	tbb::parallel_for(rangex, computeConstraintPerFrame);
	return constraints;
}

Eigen::VectorXd InterpolateKeyFrames::getEntries(const std::vector<std::complex<double>>& zvals, int entryId)
{
	if (entryId != 0 && entryId != 1)
	{
		std::cerr << "Error in get entry!" << std::endl;
		exit(1);
	}
	int size = zvals.size();
	Eigen::VectorXd vals(size);
	for (int i = 0; i < size; i++)
	{
		if (entryId == 0)
			vals(i) = zvals[i].real();
		else
			vals(i) = zvals[i].imag();
	}
	return vals;
}

double InterpolateKeyFrames::computeSmoothnessEnergy(const Eigen::VectorXd& x, Eigen::VectorXd* deriv, Eigen::SparseMatrix<double>* hess)
{
	int nbaseVerts = _basePos.rows();
	int numFrames = _vertValsList.size() - 2;
	int DOFs = numFrames * nbaseVerts * 4;

	convertVariable2List(x);
	std::vector<double> energyList(numFrames);
	std::vector<Eigen::VectorXd> derivList(numFrames);
	std::vector<std::vector<Eigen::Triplet<double>>> hessList(numFrames);

	auto computeSmoothnessEnergyPerFrame = [&](const tbb::blocked_range<uint32_t>& range)
	{
		for (uint32_t i = range.begin(); i < range.end(); ++i)
		{
			energyList[i - 1] = _newmodel.gradZSquareIntegration(_wList[i], _vertValsList[i], deriv ? &(derivList[i - 1]) : NULL, hess ? &(hessList[i - 1]) : NULL);
		}
	};

	tbb::blocked_range<uint32_t> rangex(1u, (uint32_t)(_vertValsList.size() - 1), GRAIN_SIZE);
	tbb::parallel_for(rangex, computeSmoothnessEnergyPerFrame);

	/*for (uint32_t i = 1; i < (uint32_t)(_vertValsList.size() - 1); ++i)
	{
		energyList[i - 1] = _newmodel.gradZSquareIntegration(_wList[i], _vertValsList[i], deriv ? &(derivList[i - 1]) : NULL, hess ? &(hessList[i - 1]) : NULL);
	}*/

	double energy = 0;
	if (deriv)
	{
		deriv->setZero(DOFs);
	}
	std::vector<Eigen::Triplet<double>> T;

	for (int i = 0; i < numFrames; i++)
	{
		energy += energyList[i];

		int curDOFs = i * 4 * nbaseVerts;
		if (deriv)
		{
			deriv->segment(curDOFs, 4 * nbaseVerts) += derivList[i];
		}

		if (hess)
		{
			for (auto& it : hessList[i])
			{
				
				T.push_back({ it.row() + curDOFs, it.col() + curDOFs, it.value() });
	
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

double InterpolateKeyFrames::computeEnergy(const Eigen::VectorXd& x, Eigen::VectorXd* deriv, Eigen::SparseMatrix<double>* hess, bool isProj)
{
	int nbaseVerts = _basePos.rows();
	int numFrames = _vertValsList.size() - 2;
	int DOFs = numFrames * nbaseVerts * 4;

	convertVariable2List(x);

	Eigen::VectorXd curDeriv;
	std::vector<Eigen::Triplet<double>> T, curT;

	double energy = 0;
	if (deriv)
	{
		deriv->setZero(DOFs);
	}

	for (int i = 0; i < _vertValsList.size() - 1; i++)
	{
		if(_isUseUpMesh)
			energy += _model.zDotSquareIntegration(_wList[i], _wList[i + 1], _vertValsList[i], _vertValsList[i + 1], _dt, deriv ? &curDeriv : NULL, hess ? &curT : NULL, isProj);
		else
			energy += _newmodel.zDotSquareIntegration(_wList[i], _wList[i + 1], _vertValsList[i], _vertValsList[i + 1], _dt, deriv ? &curDeriv : NULL, hess ? &curT : NULL, isProj);
		
		Eigen::VectorXd xvec, yvec;
		xvec = getEntries(_vertValsList[i], 0);
		yvec = getEntries(_vertValsList[i], 1);
		
		/*if (i > 0)
		{
			energy += -0.5 * _smoothCoeff * (xvec.dot(_cotMat * xvec) + yvec.dot(_cotMat * yvec));
		}*/

		if (deriv)
		{
			if(i == 0)
				deriv->segment(0, 4 * nbaseVerts) += curDeriv.segment(4 * nbaseVerts, 4 * nbaseVerts);
			else if(i == _vertValsList.size() - 2)
				deriv->segment(4 * (i - 1) * nbaseVerts, 4 * nbaseVerts) += curDeriv.segment(0, 4 * nbaseVerts);
			else
			{
				deriv->segment(4 * (i - 1) * nbaseVerts, 8 * nbaseVerts) += curDeriv;
			}

			/*if (i > 0)
			{
				Eigen::VectorXd xDeriv = -_smoothCoeff * _cotMat * xvec;
				Eigen::VectorXd yDeriv = -_smoothCoeff * _cotMat * yvec;
				for (int j = 0; j < nbaseVerts; j++)
				{
					(*deriv)(4 * (i - 1) * nbaseVerts + 2 * j) += xDeriv(j);
					(*deriv)(4 * (i - 1) * nbaseVerts + 2 * j + 1) += yDeriv(j);
				}
				
			}*/

		}

		if (hess)
		{
			for (auto& it : curT)
			{
				if (i == 0)
				{
					if (it.row() >= 4 * nbaseVerts && it.col() >= 4 * nbaseVerts)
						T.push_back({ it.row() - 4 * nbaseVerts, it.col() - 4 * nbaseVerts, it.value() });
				}
				else if (i == _vertValsList.size() - 2)
				{
					if (it.row() < 4 * nbaseVerts && it.col() < 4 * nbaseVerts)
						T.push_back({ it.row() + 4 * (i - 1) * nbaseVerts, it.col() + 4 * (i - 1) * nbaseVerts, it.value() });
				}
				else
				{
					T.push_back({ it.row() + 4 * (i - 1) * nbaseVerts, it.col() + 4 * (i - 1) * nbaseVerts, it.value() });
				}
			}

			/*if (i > 0)
			{
				for (int k = 0; k < _cotMat.outerSize(); ++k)
					for (Eigen::SparseMatrix<double>::InnerIterator it(_cotMat, k); it; ++it)
					{
						T.push_back(Eigen::Triplet<double>(2 * it.row() + 4 * (i - 1) * nbaseVerts, 2 * it.col() + 4 * (i - 1) * nbaseVerts, -_smoothCoeff * it.value()));
						T.push_back(Eigen::Triplet<double>(2 * it.row() + 1 + 4 * (i - 1) * nbaseVerts, 2 * it.col() + 1 + 4 * (i - 1) * nbaseVerts, -_smoothCoeff * it.value()));
					}
			}*/
			curT.clear();
		}
	}
	if (hess)
	{
	    //std::cout << "num of triplets: " << T.size() << std::endl;
		hess->resize(DOFs, DOFs);
		hess->setFromTriplets(T.begin(), T.end());
	}


	if (_smoothCoeff > 0)
	{
		Eigen::VectorXd sDeriv;
		Eigen::SparseMatrix<double> sHess;
		double smoothTerm = computeSmoothnessEnergy(x, deriv ? &sDeriv : NULL, hess ? &sHess : NULL);

		energy += _smoothCoeff * smoothTerm;
		if (deriv)
			(*deriv) += _smoothCoeff * sDeriv;
		if (hess)
			(*hess) += _smoothCoeff * sHess;
	}

	return energy;
}

double InterpolateKeyFrames::computePerFrameConstraints(const std::vector<std::complex<double>>& zvals, const Eigen::MatrixXd& w, const Eigen::VectorXd& lambda, Eigen::VectorXd* deriv, std::vector<Eigen::Triplet<double>>* hessT, bool isProj)
{
	int nbaseVerts = _basePos.rows();
	double energy = 0;

	std::vector<double> energyList(nbaseVerts);
	std::vector<Eigen::Vector4d> derivList(nbaseVerts);
	std::vector<Eigen::Matrix4d> hessList(nbaseVerts);

	auto computeEnergy = [&](const tbb::blocked_range<uint32_t>& range) {
		for (uint32_t i = range.begin(); i < range.end(); ++i)
		{
			energyList[i] = computePerVertexConstraint(zvals, w, i, deriv ? &derivList[i] : NULL, hessT ? &hessList[i] : NULL, false);
		}
	};

	tbb::blocked_range<uint32_t> rangex(0u, (uint32_t)nbaseVerts, GRAIN_SIZE);
	tbb::parallel_for(rangex, computeEnergy);

	if (deriv)
	{
		deriv->setZero(4 * nbaseVerts);
	}
	if (hessT)
	{
		hessT->clear();
	}

	for (int i = 0; i < nbaseVerts; i++)
	{
		energy = energy + lambda(i) * energyList[i];
		if (deriv)
		{
			for (int j = 0; j < 2; j++)
			{
				(*deriv)(2 * i + j) += lambda(i) * derivList[i](j);
				(*deriv)(2 * nbaseVerts + 2 * i + j) += lambda(i) * derivList[i](2 + j);
			}
		}

		if (hessT)
		{
			if (isProj)
				hessList[i] = SPDProjection(lambda(i) * hessList[i]);
			else
				hessList[i] *= lambda(i);
			for (int j = 0; j < 2; j++)
			{
				for (int k = 0; k < 2; k++)
				{
					hessT->push_back({ 2 * i + j, 2 * i + k, hessList[i](j, k) });
					hessT->push_back({ 2 * nbaseVerts + 2 * i + j, 2 * i + k, hessList[i](2 + j, k) });
					hessT->push_back({ 2 * i + j, 2 * nbaseVerts + 2 * i + k, hessList[i](j, 2 + k) });
					hessT->push_back({ 2 * nbaseVerts + 2 * i + j, 2 * nbaseVerts + 2 * i + k, hessList[i](2 + j, 2 + k) });
				}
			}

		}
	}
	return energy;
}

double InterpolateKeyFrames::computeConstraints(const Eigen::VectorXd& x, const Eigen::VectorXd& lambda, Eigen::VectorXd* deriv, Eigen::SparseMatrix<double>* hess, bool isProj, Eigen::VectorXd* constraints)
{
	int nbaseVerts = _basePos.rows();
	int numFrames = _vertValsList.size() - 2;
	int DOFs = numFrames * nbaseVerts * 4;

	convertVariable2List(x);

	Eigen::VectorXd curDeriv;
	std::vector<Eigen::Triplet<double>> T, curT;

	double energy = 0;
	if (deriv)
	{
		deriv->setZero(DOFs);
	}

    if(constraints)
    {
        constraints->setZero(numFrames * nbaseVerts);
    }

	for (int i = 1; i < _vertValsList.size() - 1; i++)
	{
		Eigen::VectorXd frameLamba = lambda.segment((i - 1) * nbaseVerts, nbaseVerts);
		energy += computePerFrameConstraints(_vertValsList[i], _wList[i], frameLamba, deriv ? &curDeriv : NULL, hess ? &curT : NULL, isProj);

        if(constraints)
        {
            (*constraints) = getConstraints(x);
        }

		if (deriv)
		{
			deriv->segment(4 * (i - 1) * nbaseVerts, 4 * nbaseVerts) = curDeriv;
		}
		if (hess)
		{
			for (auto& it : curT)
				T.push_back({ it.row() + 4 * (i - 1) * nbaseVerts, it.col() + 4 * (i - 1) * nbaseVerts, it.value() });
			curT.clear();
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


double InterpolateKeyFrames::computePerFrameConstraintsPenalty(const std::vector<std::complex<double>>& zvals, const Eigen::MatrixXd& w, const double& mu, Eigen::VectorXd* deriv, std::vector<Eigen::Triplet<double>>* hessT, bool isProj)
{
	int nbaseVerts = _basePos.rows();
	double energy = 0;

	std::vector<double> energyList(nbaseVerts);
	std::vector<Eigen::Vector4d> derivList(nbaseVerts);
	std::vector<Eigen::Matrix4d> hessList(nbaseVerts);

	auto computeEnergy = [&](const tbb::blocked_range<uint32_t>& range) {
		for (uint32_t i = range.begin(); i < range.end(); ++i)
		{
			energyList[i] = computePerVertexPenalty(zvals, w, i, deriv ? &derivList[i] : NULL, hessT ? &hessList[i] : NULL, isProj);
		}
	};

	tbb::blocked_range<uint32_t> rangex(0u, (uint32_t)nbaseVerts, GRAIN_SIZE);
	tbb::parallel_for(rangex, computeEnergy);

	if (deriv)
	{
		deriv->setZero(4 * nbaseVerts);
	}
	if (hessT)
	{
		hessT->clear();
	}

	for (int i = 0; i < nbaseVerts; i++)
	{
		energy = energy + mu / 2.0 * energyList[i];
		if (deriv)
		{
			for (int j = 0; j < 2; j++)
			{
				(*deriv)(2 * i + j) += mu / 2.0 * derivList[i](j);
				(*deriv)(2 * nbaseVerts + 2 * i + j) += mu / 2.0 * derivList[i](2 + j);
			}
		}

		if (hessT)
		{
			if (isProj)
				hessList[i] = SPDProjection(mu / 2.0 * hessList[i]);
			else
				hessList[i] = mu / 2.0 * hessList[i];
			for (int j = 0; j < 2; j++)
			{
				for (int k = 0; k < 2; k++)
				{
					hessT->push_back({ 2 * i + j, 2 * i + k, hessList[i](j, k) });
					hessT->push_back({ 2 * nbaseVerts + 2 * i + j, 2 * i + k, hessList[i](2 + j, k) });
					hessT->push_back({ 2 * i + j, 2 * nbaseVerts + 2 * i + k, hessList[i](j, 2 + k) });
					hessT->push_back({ 2 * nbaseVerts + 2 * i + j, 2 * nbaseVerts + 2 * i + k, hessList[i](2 + j, 2 + k) });
				}
			}

		}
	}
	return energy;
}

double InterpolateKeyFrames::computeConstraintsPenalty(const Eigen::VectorXd& x, const double& mu, Eigen::VectorXd* deriv, Eigen::SparseMatrix<double>* hess, bool isProj)
{
	int nbaseVerts = _basePos.rows();
	int numFrames = _vertValsList.size() - 2;
	int DOFs = numFrames * nbaseVerts * 4;

	convertVariable2List(x);

	Eigen::VectorXd curDeriv;
	std::vector<Eigen::Triplet<double>> T, curT;

	double energy = 0;
	if (deriv)
	{
		deriv->setZero(DOFs);
	}

	for (int i = 1; i < _vertValsList.size() - 1; i++)
	{
		energy += computePerFrameConstraintsPenalty(_vertValsList[i], _wList[i], mu, deriv ? &curDeriv : NULL, hess ? &curT : NULL, isProj);
		if (deriv)
		{
			deriv->segment(4 * (i - 1) * nbaseVerts, 4 * nbaseVerts) = curDeriv;
		}
		if (hess)
		{
			for (auto& it : curT)
				T.push_back({ it.row() + 4 * (i - 1) * nbaseVerts, it.col() + 4 * (i - 1) * nbaseVerts, it.value() });
			curT.clear();
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

void InterpolateKeyFrames::testEnergy(Eigen::VectorXd x)
{
	Eigen::VectorXd deriv;
	Eigen::SparseMatrix<double> hess;

	double e = computeEnergy(x, &deriv, &hess, false);
	std::cout << "energy: " << e << std::endl;
	
	Eigen::VectorXd dir = deriv;
	dir.setRandom();

	for (int i = 3; i < 9; i++)
	{
		double eps = std::pow(0.1, i);

		Eigen::VectorXd deriv1;
		double e1 = computeEnergy(x + eps * dir, &deriv1, NULL, false);

		std::cout << "eps: " << eps << std::endl;
		std::cout << "value-gradient check: " << (e1 - e) / eps - dir.dot(deriv) << std::endl;
		std::cout << "gradient-hessian check: " << ((deriv1 - deriv) / eps - hess * dir).norm() << std::endl;
	}
}

void InterpolateKeyFrames::testSmoothnessEnergy(Eigen::VectorXd x)
{
	Eigen::VectorXd deriv;
	Eigen::SparseMatrix<double> hess;

	double e = computeSmoothnessEnergy(x, &deriv, &hess);
	std::cout << "energy: " << e << std::endl;

	Eigen::VectorXd dir = deriv;
	dir.setRandom();

	for (int i = 3; i < 9; i++)
	{
		double eps = std::pow(0.1, i);

		Eigen::VectorXd deriv1;
		double e1 = computeSmoothnessEnergy(x + eps * dir, &deriv1, NULL);

		std::cout << "eps: " << eps << std::endl;
		std::cout << "value-gradient check: " << (e1 - e) / eps - dir.dot(deriv) << std::endl;
		std::cout << "gradient-hessian check: " << ((deriv1 - deriv) / eps - hess * dir).norm() << std::endl;
	}
}

void InterpolateKeyFrames::testPenalty(Eigen::VectorXd x)
{
	Eigen::VectorXd deriv;
	Eigen::SparseMatrix<double> hess;

	double e = computePenalty(x, &deriv, &hess, false);
	std::cout << "penalty energy: " << e << std::endl;

	Eigen::VectorXd dir = deriv;
	dir.setRandom();

	for (int i = 3; i < 9; i++)
	{
		double eps = std::pow(0.1, i);

		Eigen::VectorXd deriv1;
		double e1 = computePenalty(x + eps * dir, &deriv1, NULL, false);

		std::cout << "eps: " << eps << std::endl;
		std::cout << "value-gradient check: " << (e1 - e) / eps - dir.dot(deriv) << std::endl;
		std::cout << "gradient-hessian check: " << ((deriv1 - deriv) / eps - hess * dir).norm() << std::endl;
	}
}

void InterpolateKeyFrames::testConstraints(Eigen::VectorXd x, Eigen::VectorXd lambda)
{
	Eigen::VectorXd deriv;
	Eigen::SparseMatrix<double> hess;

	double e = computeConstraints(x, lambda, &deriv, &hess, false);
	std::cout << "penalty energy: " << e << std::endl;

	Eigen::VectorXd dir = deriv;
	dir.setRandom();

	for (int i = 3; i < 9; i++)
	{
		double eps = std::pow(0.1, i);

		Eigen::VectorXd deriv1;
		double e1 = computeConstraints(x + eps * dir, lambda, &deriv1, NULL, false);

		std::cout << "eps: " << eps << std::endl;
		std::cout << "value-gradient check: " << (e1 - e) / eps - dir.dot(deriv) << std::endl;
		std::cout << "gradient-hessian check: " << ((deriv1 - deriv) / eps - hess * dir).norm() << std::endl;
	}
}

void InterpolateKeyFrames::testConstraintsPenalty(Eigen::VectorXd x, double mu)
{
	Eigen::VectorXd deriv;
	Eigen::SparseMatrix<double> hess;

	double e = computeConstraintsPenalty(x, mu, &deriv, &hess, false);
	std::cout << "penalty energy: " << e << std::endl;

	Eigen::VectorXd dir = deriv;
	dir.setRandom();

	for (int i = 3; i < 9; i++)
	{
		double eps = std::pow(0.1, i);

		Eigen::VectorXd deriv1;
		double e1 = computeConstraintsPenalty(x + eps * dir, mu, &deriv1, NULL, false);

		std::cout << "eps: " << eps << std::endl;
		std::cout << "value-gradient check: " << (e1 - e) / eps - dir.dot(deriv) << std::endl;
		std::cout << "gradient-hessian check: " << ((deriv1 - deriv) / eps - hess * dir).norm() << std::endl;
	}
}

void InterpolateKeyFrames::testPerVertexPenalty(const std::vector<std::complex<double>>& zvals, const Eigen::MatrixXd& w, int vid)
{
	Eigen::Vector4d deriv;
	Eigen::Matrix4d hess;

	double e = computePerVertexPenalty(zvals, w, vid, &deriv, &hess, false);
	std::cout << "penalty energy: " << e << std::endl;

	Eigen::Vector4d dir = deriv;
	dir.setRandom();
	auto backupz = zvals;
	auto backupw = w;

	for (int i = 3; i < 9; i++)
	{
		double eps = std::pow(0.1, i);

		Eigen::Vector4d deriv1;
		backupz[vid] = std::complex<double>(zvals[vid].real() + eps * dir(0), zvals[vid].imag() + eps * dir(1));
		backupw.row(vid) << w(vid, 0) + eps * dir(2), w(vid, 1) + eps * dir(3);
		double e1 = computePerVertexPenalty(backupz, backupw, vid, &deriv1, NULL, false);

		std::cout << "eps: " << eps << std::endl;
		std::cout << "value-gradient check: " << (e1 - e) / eps - dir.dot(deriv) << std::endl;
		std::cout << "gradient-hessian check: " << ((deriv1 - deriv) / eps - hess * dir).norm() << std::endl;
	}
}


void InterpolateKeyFrames::testPerFramePenalty(const std::vector<std::complex<double>>& zvals, const Eigen::MatrixXd& w)
{
	Eigen::VectorXd deriv;
	Eigen::SparseMatrix<double> hess;
	std::vector<Eigen::Triplet<double>> T;

	double e = computePerFramePenalty(zvals, w, &deriv, &T, false);
	std::cout << "penalty energy: " << e << std::endl;

	Eigen::VectorXd dir = deriv;
	dir.setRandom();
	auto backupz = zvals;
	auto backupw = w;
	int nverts = zvals.size();

	hess.resize(4 * nverts, 4 * nverts);
	hess.setFromTriplets(T.begin(), T.end());

	for (int i = 3; i < 9; i++)
	{
		double eps = std::pow(0.1, i);

		Eigen::VectorXd deriv1;
		for (int vid = 0; vid < zvals.size(); vid++)
		{
			backupz[vid] = std::complex<double>(zvals[vid].real() + eps * dir(2 * vid), zvals[vid].imag() + eps * dir(2 * vid + 1));
			backupw.row(vid) << w(vid, 0) + eps * dir(2 * vid + 2 * nverts), w(vid, 1) + eps * dir(2 * vid + 2 * nverts + 1);
		}
		
		double e1 = computePerFramePenalty(backupz, backupw, &deriv1, NULL, false);

		std::cout << "eps: " << eps << std::endl;
		std::cout << "value-gradient check: " << (e1 - e) / eps - dir.dot(deriv) << std::endl;
		std::cout << "gradient-hessian check: " << ((deriv1 - deriv) / eps - hess * dir).norm() << std::endl;
	}
}