#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <iostream>
#include <iomanip>

#ifndef GRAIN_SIZE
#define GRAIN_SIZE 10
#endif

#include "../../include/DynamicInterpolation/ComputeZandZdot.h"

std::complex<double> ComputeZandZdot::planeWaveBasis(Eigen::Vector3d p, Eigen::Vector3d pi, Eigen::Vector2d omega, Eigen::Vector2cd* deriv, Eigen::Matrix2cd* hess, std::vector<Eigen::Matrix2cd>* derivHess)
{
	if (deriv)
		deriv->setZero();
	if (hess)
		hess->setZero();
	if (derivHess)
	{
		derivHess->resize(omega.size());
		for (auto& m : *derivHess)
			m.setZero(omega.size(), omega.size());
	}
	if (omega.norm() == 0)
		return 1;
	Eigen::Vector2d pv = (p - pi).segment<2>(0);
	double deltatheta = omega.dot(pv);
	Eigen::Vector2d gradDeltaTheta = pv;

	std::complex<double> z = std::complex<double>(std::cos(deltatheta), std::sin(deltatheta));

	if (deriv)
	{
		(*deriv) = std::complex<double>(0, 1) * z * pv;
	}
	if (hess)
	{
		(*hess) = -z * gradDeltaTheta * gradDeltaTheta.transpose();
	}
	if (derivHess)
	{
		for (int i = 0; i < pv.size(); i++)
		{
			(*derivHess)[i] = -std::complex<double>(-std::sin(deltatheta), std::cos(deltatheta)) * pv(i) * gradDeltaTheta * gradDeltaTheta.transpose();
		}
	}

	return z;
}

std::complex<double> ComputeZandZdot::planeWaveValueFromQuad(const Eigen::MatrixXd& w,
	const std::vector<std::complex<double>>& vertVals, int faceId, int quadId,
	Eigen::Matrix<std::complex<double>, 12, 1>* deriv,
	Eigen::Matrix<std::complex<double>, 12, 12>* hess,
	std::vector<Eigen::Matrix<std::complex<double>, 12, 12>>* derivHess)
{
	std::complex<double> z = 0;

	if (deriv)
	{
		deriv->setZero();
	}
	if (hess)
	{
		hess->setZero();
	}
	if (derivHess)
	{
		derivHess->resize(12);
		for (auto& m : *derivHess)
			m.setZero();
	}
	Eigen::Vector3d p = getQuadPosition(faceId, quadId);
	

	for (int j = 0; j < 3; j++)
	{
		int baseVid = _baseMesh.faceVertex(faceId, j);
		Eigen::Vector2d wi = w.row(baseVid);
		std::complex<double> fi = vertVals[baseVid];
		Eigen::Vector2cd expDeriv;
		Eigen::Matrix2cd expHess;
		std::vector<Eigen::Matrix2cd> expDerivHess;


		std::complex<double> expPart = planeWaveBasis(p, _basePos.row(baseVid), wi, (deriv || hess || derivHess) ? &expDeriv : NULL, (hess || derivHess) ? &expHess : NULL, derivHess ? &expDerivHess : NULL);

		z += _hatWeights[quadId](j) * fi * expPart;

		if (deriv || hess || derivHess)
		{
			Eigen::Vector4cd gradFi;
			gradFi << 1, std::complex<double>(0, 1), 0, 0;

			Eigen::Vector4cd fullExpDeriv;
			fullExpDeriv << 0, 0, expDeriv(0), expDeriv(1);

			if (deriv)
			{
			    deriv->segment<4>(4 * j) += _hatWeights[quadId](j) * (expPart * gradFi + fullExpDeriv * fi);
			}
			if (hess || derivHess)
			{
				Eigen::Matrix4cd fullExpHess;
				fullExpHess.setZero();
				fullExpHess.block<2, 2>(2, 2) = expHess;

				if (hess)
				    hess->block<4, 4>(4 * j, 4 * j) += _hatWeights[quadId](j) * (gradFi * fullExpDeriv.transpose() + fullExpDeriv * gradFi.transpose() + fi * fullExpHess);

				if (derivHess)
				{
					std::vector<Eigen::Matrix4cd> fullExpDerivHess;
					fullExpDerivHess.resize(4);
					for (int p = 0; p < 2; p++)
					{
						fullExpDerivHess[p].setZero();
						fullExpDerivHess[p + 2].block<2, 2>(2, 2) = expDerivHess[p];
					}

					for (int p = 0; p < 4; p++)
						for (int m = 0; m < 4; m++)
							for (int n = 0; n < 4; n++)
							{
							    (*derivHess)[4 * j + p](4 * j + m, 4 * j + n) += _hatWeights[quadId](j) * (fullExpHess(p, m) * gradFi(n) + fullExpHess(m, n) * gradFi(p) + fullExpHess(n, p) * gradFi(m) + fi * fullExpDerivHess[p](m, n));
							}
				}

			}
		}
	}
	return z;
}


std::complex<double> ComputeZandZdot::planeWaveValueDotFromQuad(const Eigen::MatrixXd& w1,
	const Eigen::MatrixXd& w2,
	const std::vector<std::complex<double>>& vertVals1,
	const std::vector<std::complex<double>>& vertVals2,
	const double dt, int faceId, int quadId,
	Eigen::Matrix<std::complex<double>, 24, 1>* deriv,
	Eigen::Matrix<std::complex<double>, 24, 24>* hess)
{
	Eigen::Matrix<std::complex<double>, 12, 1> z1Deriv, z2Deriv;
	Eigen::Matrix<std::complex<double>, 12, 12> z1Hess, z2Hess;

	std::complex<double> z1 = planeWaveValueFromQuad(w1, vertVals1, faceId, quadId, (deriv || hess) ? &z1Deriv : NULL, hess ? &z1Hess : NULL, NULL);
	std::complex<double> z2 = planeWaveValueFromQuad(w2, vertVals2, faceId, quadId, (deriv || hess) ? &z2Deriv : NULL, hess ? &z2Hess : NULL, NULL);

	std::complex<double> zdot = (z2 - z1) / dt;
	if (deriv)
	{
		deriv->setZero();
		deriv->segment<12>(0) = -z1Deriv / dt;
		deriv->segment<12>(12) = z2Deriv / dt;

	}
	if (hess)
	{
		hess->setZero();
		hess->block<12, 12>(0, 0) = -z1Hess / dt;
		hess->block<12, 12>(12, 12) = z2Hess / dt;
	}
	return zdot;
}

double ComputeZandZdot::zDotSquarePerface(const Eigen::MatrixXd& w1,
	const Eigen::MatrixXd& w2,
	const std::vector<std::complex<double>>& vertVals1,
	const std::vector<std::complex<double>>& vertVals2,
	const double dt, int faceId, Eigen::Matrix<double, 24, 1>* deriv, Eigen::Matrix<double, 24, 24>* hess, bool isProj)
{
	double zdotSquare = 0;

	if (deriv)
	{
		deriv->setZero();
	}

	if (hess)
	{
		hess->setZero();
	}

	for (int qid = 0; qid < _quadPoints.size(); qid++)
	{
		Eigen::Matrix<std::complex<double>, 24, 1> zdotDeriv;
		Eigen::Matrix<std::complex<double>, 24, 24> zdotHess;
		std::complex<double> zdot = planeWaveValueDotFromQuad(w1, w2, vertVals1, vertVals2, dt, faceId, qid, (deriv || hess) ? &zdotDeriv : NULL, hess ? &zdotHess : NULL);

		double componentWeights = (_doubleFarea(faceId) / 2) * _quadPoints[qid].weight;
		zdotSquare += (0.5 * zdot.real() * zdot.real() + 0.5 * zdot.imag() * zdot.imag()) * componentWeights;

		if (deriv)
		{
			(*deriv) += (zdot.real() * zdotDeriv.real() + zdot.imag() * zdotDeriv.imag()) * componentWeights;
		}

		if (hess)
		{
			(*hess) += (zdotDeriv.real() * zdotDeriv.real().transpose() + zdotDeriv.imag() * zdotDeriv.imag().transpose() + zdot.real() * zdotHess.real() + zdot.imag() * zdotHess.imag()) * componentWeights;
		}
	}

	if (hess && isProj)
	{
		(*hess) = SPDProjection(*hess);
	}

	return zdotSquare;
}

double ComputeZandZdot::zDotSquareIntegration(const Eigen::MatrixXd& w1,
	const Eigen::MatrixXd& w2,
	const std::vector<std::complex<double>>& vertVals1,
	const std::vector<std::complex<double>>& vertVals2,
	const double dt, Eigen::VectorXd* deriv, std::vector<Eigen::Triplet<double>>* hessT, bool isProj)
{
	double energy = 0;
	int nverts = _basePos.rows();
	int nfaces = _baseMesh.nFaces();

	if (deriv)
		deriv->setZero(8 * nverts);
	if (hessT)
		hessT->clear();

	std::vector<double> energyList(nfaces);
	std::vector<Eigen::Matrix<double, 24, 1>> derivList(nfaces);
	std::vector<Eigen::Matrix<double, 24, 24>> hessList(nfaces);

	 auto computeEnergy = [&](const tbb::blocked_range<uint32_t>& range) {
	     for (uint32_t i = range.begin(); i < range.end(); ++i)
	     {
			 energyList[i] = zDotSquarePerface(w1, w2, vertVals1, vertVals2, dt, i, deriv ? &derivList[i] : NULL, hessT ? &hessList[i] : NULL, isProj);
	     }
	 };

	 tbb::blocked_range<uint32_t> rangex(0u, (uint32_t)nfaces, GRAIN_SIZE);
	 tbb::parallel_for(rangex, computeEnergy);

	/*for (uint32_t i = 0; i < nfaces; ++i)
	{
		energyList[i] = zDotSquarePerface(w1, w2, vertVals1, vertVals2, dt, i, deriv ? &derivList[i] : NULL, hessT ? &hessList[i] : NULL, isProj);
	}*/

	for (int i = 0; i < nfaces; i++)
	{
		energy += energyList[i];

		if (deriv || hessT)
		{
			for (int j = 0; j < 3; j++)
			{
				int baseVid = _baseMesh.faceVertex(i, j);

				if (deriv)
				{
					for (int k = 0; k < 2; k++)
					{
						(*deriv)(2 * baseVid + k) += derivList[i](4 * j + k);
						(*deriv)(2 * baseVid + 2 * nverts + k) += derivList[i](4 * j + 2 + k);

						(*deriv)(2 * baseVid + 4 * nverts + k) += derivList[i](4 * j + 12 + k);
						(*deriv)(2 * baseVid + 6 * nverts + k) += derivList[i](4 * j + 14 + k);
					}
				}
				if (hessT)
				{
					for (int k = 0; k < 3; k++)
					{
						int baseVid1 = _baseMesh.faceVertex(i, k);
						for (int m1 = 0; m1 < 2; m1++)
							for (int m2 = 0; m2 < 2; m2++)
							{
								hessT->push_back({ 2 * baseVid + m1, 2 * baseVid1 + m2, hessList[i](4 * j + m1, 4 * k + m2) });
								hessT->push_back({ 2 * baseVid + m1, 2 * baseVid1 + 2 * nverts + m2, hessList[i](4 * j + m1, 4 * k + 2 + m2) });
								hessT->push_back({ 2 * baseVid + m1, 2 * baseVid1 + 4 * nverts + m2, hessList[i](4 * j + m1, 4 * k + 12 + m2) });
								hessT->push_back({ 2 * baseVid + m1, 2 * baseVid1 + 6 * nverts + m2, hessList[i](4 * j + m1, 4 * k + 14 + m2) });


								hessT->push_back({ 2 * baseVid + 2 * nverts + m1, 2 * baseVid1 + m2, hessList[i](4 * j + 2 + m1, 4 * k + m2) });
								hessT->push_back({ 2 * baseVid + 2 * nverts + m1, 2 * baseVid1 + 2 * nverts + m2, hessList[i](4 * j + 2 + m1, 4 * k + 2 + m2) });
								hessT->push_back({ 2 * baseVid + 2 * nverts + m1, 2 * baseVid1 + 4 * nverts + m2, hessList[i](4 * j + 2 + m1, 4 * k + 12 + m2) });
								hessT->push_back({ 2 * baseVid + 2 * nverts + m1, 2 * baseVid1 + 6 * nverts + m2, hessList[i](4 * j + 2 + m1, 4 * k + 14 + m2) });

								hessT->push_back({ 2 * baseVid + 4 * nverts + m1, 2 * baseVid1 + m2, hessList[i](4 * j + 12 + m1, 4 * k + m2) });
								hessT->push_back({ 2 * baseVid + 4 * nverts + m1, 2 * baseVid1 + 2 * nverts + m2, hessList[i](4 * j + 12 + m1, 4 * k + 2 + m2) });
								hessT->push_back({ 2 * baseVid + 4 * nverts + m1, 2 * baseVid1 + 4 * nverts + m2, hessList[i](4 * j + 12 + m1, 4 * k + 12 + m2) });
								hessT->push_back({ 2 * baseVid + 4 * nverts + m1, 2 * baseVid1 + 6 * nverts + m2, hessList[i](4 * j + 12 + m1, 4 * k + 14 + m2) });

								hessT->push_back({ 2 * baseVid + 6 * nverts + m1, 2 * baseVid1 + m2, hessList[i](4 * j + 14 + m1, 4 * k + m2) });
								hessT->push_back({ 2 * baseVid + 6 * nverts + m1, 2 * baseVid1 + 2 * nverts + m2, hessList[i](4 * j + 14 + m1, 4 * k + 2 + m2) });
								hessT->push_back({ 2 * baseVid + 6 * nverts + m1, 2 * baseVid1 + 4 * nverts + m2, hessList[i](4 * j + 14 + m1, 4 * k + 12 + m2) });
								hessT->push_back({ 2 * baseVid + 6 * nverts + m1, 2 * baseVid1 + 6 * nverts + m2, hessList[i](4 * j + 14 + m1, 4 * k + 14 + m2) });

							}
					}
				}
			}
		}

	}
	return energy;
}


///////////////////////////////////// test functions //////////////////////////////////////
void ComputeZandZdot::testPlaneWaveBasis(Eigen::VectorXd p, Eigen::VectorXd pi, Eigen::Vector2d omega)
{
	Eigen::Vector2cd deriv;
	Eigen::Matrix2cd hess;
	std::vector<Eigen::Matrix2cd> derivHess;

	std::complex<double> z = planeWaveBasis(p, pi, omega, &deriv, &hess, &derivHess);
	Eigen::VectorXd dir = Eigen::VectorXd::Random(deriv.rows());
	std::cout << "dir: " << dir.transpose() << std::endl;
	std::cout << "p: " << p.transpose() << std::endl;
	std::cout << "pi: " << pi.transpose() << std::endl;
	std::cout << "w: " << omega.transpose() << std::endl;

	std::cout << "z: " << z << std::endl;
	std::cout << "deriv: \n" << deriv << std::endl;
	std::cout << "hess: \n" << hess << std::endl;
	for (int j = 0; j < deriv.rows(); j++)
	{
		std::cout << "i-th row: \n" << derivHess[j] << std::endl;
	}

	Eigen::Vector2d omegaBackup = omega;
	for (int i = 3; i <= 10; i++)
	{
		double eps = std::pow(0.1, i);
		omegaBackup = omega + eps * dir;

		Eigen::Vector2cd deriv1;
		Eigen::Matrix2cd hess1;
		std::complex<double> z1 = planeWaveBasis(p, pi, omegaBackup, &deriv1, &hess1, NULL);

		std::cout << "eps: " << eps << std::endl;

		std::cout << "value-gradient check: " << (z1 - z) / eps - dir.dot(deriv) << std::endl;
		std::cout << "gradient-hessian check: " << ((deriv1 - deriv) / eps - hess * dir).norm() << std::endl;
		std::cout << "hessian-3rd derivative check: " << std::endl;
		for (int j = 0; j < deriv.rows(); j++)
		{
			std::cout << j << "-th row: " << ((hess1 - hess).row(j) / eps - dir.transpose() * derivHess[j]).norm() << std::endl;
		}
	}

}

void ComputeZandZdot::testPlaneWaveValueFromQuad(const Eigen::MatrixXd& w, const std::vector<std::complex<double>>& vertVals, int faceId, int quadId)
{
	Eigen::Matrix<std::complex<double>, 12, 1> deriv;
	Eigen::Matrix<std::complex<double>, 12, 12> hess;
	std::vector<Eigen::Matrix<std::complex<double>, 12, 12>> derivHess;
	std::complex<double> z = planeWaveValueFromQuad(w, vertVals, faceId, quadId, &deriv, &hess, &derivHess);

	Eigen::VectorXd dir = Eigen::VectorXd::Random(deriv.rows());

	Eigen::MatrixXd backupW = w;
	std::vector<std::complex<double>> backupVertVals = vertVals;
	std::cout << "faceId: " << faceId << ", quadId: " << quadId << std::endl;

	for (int i = 3; i <= 10; i++)
	{
		double eps = std::pow(0.1, i);
		for (int j = 0; j < 3; j++)
		{
			int baseVid = _baseMesh.faceVertex(faceId, j);
			backupVertVals[baseVid] = std::complex<double>(vertVals[baseVid].real() + eps * dir(4 * j), vertVals[baseVid].imag() + eps * dir(4 * j + 1));
			backupW(baseVid, 0) = w(baseVid, 0) + eps * dir(4 * j + 2);
			backupW(baseVid, 1) = w(baseVid, 1) + eps * dir(4 * j + 3);
		}
		Eigen::Matrix<std::complex<double>, 12, 1> deriv1;
		Eigen::Matrix<std::complex<double>, 12, 12> hess1;
		std::complex<double> z1 = planeWaveValueFromQuad(backupW, backupVertVals, faceId, quadId, &deriv1, &hess1, NULL);

		std::cout << "eps: " << eps << std::endl;

		std::cout << "value-gradient check: " << (z1 - z) / eps - dir.dot(deriv) << std::endl;
		std::cout << "gradient-hessian check: " << ((deriv1 - deriv) / eps - hess * dir).norm() << std::endl;
		std::cout << "hessian-3rd derivative check: " << std::endl;
		for (int j = 0; j < deriv.rows(); j++)
		{
			//Eigen::VectorXcd finiteDiff = (hess1 - hess).row(j);
			std::cout << j << "-th row: " << ((hess1 - hess).row(j) / eps - dir.transpose() * derivHess[j]).norm() << std::endl;
		}
	}
}


void ComputeZandZdot::testPlaneWaveValueDotFromQuad(const Eigen::MatrixXd& w1, const Eigen::MatrixXd& w2, const std::vector<std::complex<double>>& vertVals1, const std::vector<std::complex<double>>& vertVals2, const double dt, int faceId, int quadId)
{
	Eigen::Matrix<std::complex<double>, 24, 1> deriv;
	Eigen::Matrix<std::complex<double>, 24, 24> hess;

	std::complex<double> z = planeWaveValueDotFromQuad(w1, w2, vertVals1, vertVals2, dt, faceId, quadId, &deriv, &hess);
	//	std::cout << z.real() << std::endl;
	//	std::cout << "deriv: \n" << deriv.real().transpose() << std::endl;
	//	std::cout << "hessian: \n" << hess.real() << std::endl;

	std::cout << "faceId: " << faceId << ", quadId: " << quadId << std::endl;


	Eigen::VectorXd dir = Eigen::VectorXd::Random(deriv.rows());
	std::cout << "dir: " << dir.transpose() << std::endl;

	Eigen::MatrixXd backupW1 = w1;
	std::vector<std::complex<double>> backupVertVals1 = vertVals1;

	Eigen::MatrixXd backupW2 = w2;
	std::vector<std::complex<double>> backupVertVals2 = vertVals2;

	for (int i = 3; i <= 10; i++)
	{
		double eps = std::pow(0.1, i);
		for (int j = 0; j < 3; j++)
		{
			int baseVid = _baseMesh.faceVertex(faceId, j);

			backupVertVals1[baseVid] = std::complex<double>(vertVals1[baseVid].real() + eps * dir(4 * j), vertVals1[baseVid].imag() + eps * dir(4 * j + 1));
			backupW1(baseVid, 0) = w1(baseVid, 0) + eps * dir(4 * j + 2);
			backupW1(baseVid, 1) = w1(baseVid, 1) + eps * dir(4 * j + 3);

			backupVertVals2[baseVid] = std::complex<double>(vertVals2[baseVid].real() + eps * dir(4 * j + 12), vertVals2[baseVid].imag() + eps * dir(4 * j + 13));
			backupW2(baseVid, 0) = w2(baseVid, 0) + eps * dir(4 * j + 14);
			backupW2(baseVid, 1) = w2(baseVid, 1) + eps * dir(4 * j + 15);
		}
		Eigen::Matrix<std::complex<double>, 24, 1> deriv1;

		std::complex<double> z1 = planeWaveValueDotFromQuad(backupW1, backupW2, backupVertVals1, backupVertVals2, dt, faceId, quadId, &deriv1, NULL);


		std::cout << "eps: " << eps << std::endl;

		std::cout << "value-gradient check: " << (z1 - z) / eps - dir.dot(deriv) << std::endl;

		//		std::cout << "finite difference: " << (z1 - z) / eps << std::endl;
		//		std::cout << "directional derivative: " << dir.dot(deriv) << std::endl;

		std::cout << "gradient-hessian check: " << ((deriv1 - deriv) / eps - hess * dir).norm() << std::endl;
		//		std::cout << "finite difference: " << ((deriv1 - deriv) / eps).real().transpose() << std::endl;
		//		std::cout << "directional derivative: " << (hess * dir).real().transpose() << std::endl;
	}
}

void ComputeZandZdot::testZDotSquarePerface(const Eigen::MatrixXd& w1, const Eigen::MatrixXd& w2,
	const std::vector<std::complex<double>>& vertVals1,
	const std::vector<std::complex<double>>& vertVals2, const double dt,
	int faceId)
{
	Eigen::Matrix<double, 24, 1> deriv;
	Eigen::Matrix<double, 24, 24> hess;

	double e = zDotSquarePerface(w1, w2, vertVals1, vertVals2, dt, faceId, &deriv, &hess);


	Eigen::VectorXd dir = Eigen::VectorXd::Random(deriv.rows());
	std::cout << "dir: " << dir.transpose() << std::endl;

	Eigen::MatrixXd backupW1 = w1;
	std::vector<std::complex<double>> backupVertVals1 = vertVals1;

	Eigen::MatrixXd backupW2 = w2;
	std::vector<std::complex<double>> backupVertVals2 = vertVals2;

	for (int i = 3; i <= 10; i++)
	{
		double eps = std::pow(0.1, i);
		for (int j = 0; j < 3; j++)
		{
			int baseVid = _baseMesh.faceVertex(faceId, j);

			backupVertVals1[baseVid] = std::complex<double>(vertVals1[baseVid].real() + eps * dir(4 * j), vertVals1[baseVid].imag() + eps * dir(4 * j + 1));
			backupW1(baseVid, 0) = w1(baseVid, 0) + eps * dir(4 * j + 2);
			backupW1(baseVid, 1) = w1(baseVid, 1) + eps * dir(4 * j + 3);

			backupVertVals2[baseVid] = std::complex<double>(vertVals2[baseVid].real() + eps * dir(4 * j + 12), vertVals2[baseVid].imag() + eps * dir(4 * j + 13));
			backupW2(baseVid, 0) = w2(baseVid, 0) + eps * dir(4 * j + 14);
			backupW2(baseVid, 1) = w2(baseVid, 1) + eps * dir(4 * j + 15);
		}
		Eigen::Matrix<double, 24, 1> deriv1;

		double e1 = zDotSquarePerface(backupW1, backupW2, backupVertVals1, backupVertVals2, dt, faceId, &deriv1, NULL);


		std::cout << "eps: " << eps << std::endl;
		std::cout << "value-gradient check: " << (e1 - e) / eps - dir.dot(deriv) << std::endl;
		std::cout << "gradient-hessian check: " << ((deriv1 - deriv) / eps - hess * dir).norm() << std::endl;
	}
}

void ComputeZandZdot::testZDotSquareIntegration(const Eigen::MatrixXd& w1, const Eigen::MatrixXd& w2,
	const std::vector<std::complex<double>>& vertVals1,
	const std::vector<std::complex<double>>& vertVals2,
	const double dt)
{
	Eigen::VectorXd deriv;
	std::vector<Eigen::Triplet<double>> T;
	Eigen::SparseMatrix<double> hess;

	double e = zDotSquareIntegration(w1, w2, vertVals1, vertVals2, dt, &deriv, &T, false);
	hess.resize(deriv.rows(), deriv.rows());
	hess.setFromTriplets(T.begin(), T.end());

	Eigen::VectorXd dir = Eigen::VectorXd::Random(deriv.rows());

	Eigen::MatrixXd backupW1 = w1;
	std::vector<std::complex<double>> backupVertVals1 = vertVals1;

	Eigen::MatrixXd backupW2 = w2;
	std::vector<std::complex<double>> backupVertVals2 = vertVals2;

	for (int j = 3; j <= 10; j++)
	{
		double eps = std::pow(0.1, j);

		for (int i = 0; i < _basePos.rows(); i++)
		{
			backupVertVals1[i] = std::complex<double>(vertVals1[i].real() + eps * dir(2 * i), vertVals1[i].imag() + eps * dir(2 * i + 1));
			backupW1(i, 0) = w1(i, 0) + eps * dir(_basePos.rows() * 2 + 2 * i);
			backupW1(i, 1) = w1(i, 1) + eps * dir(_basePos.rows() * 2 + 2 * i + 1);

			backupVertVals2[i] = std::complex<double>(vertVals2[i].real() + eps * dir(2 * i + 4 * _basePos.rows()), vertVals2[i].imag() + eps * dir(2 * i + 1 + 4 * _basePos.rows()));
			backupW2(i, 0) = w2(i, 0) + eps * dir(_basePos.rows() * 6 + 2 * i);
			backupW2(i, 1) = w2(i, 1) + eps * dir(_basePos.rows() * 6 + 2 * i + 1);
		}

		Eigen::VectorXd deriv1;

		double e1 = zDotSquareIntegration(backupW1, backupW2, backupVertVals1, backupVertVals2, dt, &deriv1, NULL);


		std::cout << "eps: " << eps << std::endl;
		std::cout << "value-gradient check: " << (e1 - e) / eps - dir.dot(deriv) << std::endl;
		std::cout << "gradient-hessian check: " << ((deriv1 - deriv) / eps - hess * dir).norm() << std::endl;
	}
}