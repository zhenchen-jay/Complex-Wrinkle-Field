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
		energy += _model.zDotSquareIntegration(_wList[i], _wList[i + 1], _vertValsList[i], _vertValsList[i + 1], _dt, deriv ? &curDeriv : NULL, hess ? &curT : NULL, isProj);

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
			curT.clear();
		}
	}
	if (hess)
	{
	    std::cout << "num of triplets: " << T.size() << std::endl;
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