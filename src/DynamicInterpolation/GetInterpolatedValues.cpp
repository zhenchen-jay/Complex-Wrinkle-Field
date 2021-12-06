#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <iostream>
#include <iomanip>

#ifndef GRAIN_SIZE
#define GRAIN_SIZE 10
#endif

#include "../../include/DynamicInterpolation/GetInterpolatedValues.h"

std::complex<double> GetInterpolatedValues::planeWaveBasis(Eigen::VectorXd p, Eigen::VectorXd pi, Eigen::Vector2d omega, Eigen::VectorXcd *deriv, Eigen::MatrixXcd* hess, std::vector<Eigen::MatrixXcd> *derivHess)
{
	if(deriv)
		deriv->setZero(omega.size());
	if(hess)
		hess->setZero(omega.size(), omega.size());
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

	if(deriv)
	{
		(*deriv) = std::complex<double>(0, 1) * z * pv;
	}
	if(hess)
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

std::complex<double> GetInterpolatedValues::planeWaveValue(const Eigen::MatrixXd &w,
														   const std::vector<std::complex<double>> &vertVals, int vid,
														   Eigen::VectorXcd *deriv, Eigen::MatrixXcd *hess, std::vector<Eigen::MatrixXcd>* derivHess)
{
	int baseFid = _baryCoords[vid].first;
	Eigen::Vector3d baryCoord = _baryCoords[vid].second;
	Eigen::Vector3d weights = _baryWeights[vid].second;

	std::complex<double> z = 0;

	if(deriv)
	{
		deriv->setZero(12);
	}
	if(hess)
	{
		hess->setZero(12, 12);
	}
	if (derivHess)
	{
		derivHess->resize(12);
		for (auto& m : *derivHess)
			m.setZero(12, 12);
	}

	if (vid == 41)
	{
		std::cout << "test error case. " << std::endl;
	}

	for(int j = 0; j < 3; j++)
	{
		int baseVid = _baseMesh.faceVertex(baseFid, j);
		Eigen::Vector2d wi = w.row(baseVid);
		std::complex<double> fi = vertVals[baseVid];
		Eigen::VectorXcd expDeriv;
		Eigen::MatrixXcd expHess;
		std::vector<Eigen::MatrixXcd> expDerivHess;

		std::complex<double> expPart = planeWaveBasis(_upsampledPos.row(vid), _basePos.row(baseVid), wi, (deriv || hess || derivHess) ? &expDeriv : NULL, (hess || derivHess) ? &expHess : NULL, derivHess ? &expDerivHess : NULL);

		z += weights(j) * fi * expPart;

		if(deriv || hess || derivHess)
		{
			Eigen::VectorXcd gradFi(4);
			gradFi << 1, std::complex<double>(0, 1), 0, 0;

			Eigen::VectorXcd fullExpDeriv(4);
			fullExpDeriv << 0, 0, expDeriv(0), expDeriv(1);

			if(deriv)
			{
				(*deriv).segment<4>(4 * j) += weights(j) * (expPart * gradFi + fullExpDeriv * fi);
			}
			if(hess || derivHess)
			{
				Eigen::MatrixXcd fullExpHess(4, 4);
				fullExpHess.setZero();
				fullExpHess.block<2, 2>(2, 2) = expHess;

				if(hess)
					(*hess).block<4, 4>(4 * j, 4 * j) += weights(j) * (gradFi * fullExpDeriv.transpose() + fullExpDeriv * gradFi.transpose() + fi * fullExpHess);
				
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
								(*derivHess)[4 * j + p](4 * j + m, 4 * j + n) += weights(j) * (fullExpHess(p, m) * gradFi(n) + fullExpHess(m, n) * gradFi(p) + fullExpHess(n, p) * gradFi(m) + fi * fullExpDerivHess[p](m, n));
							}
				}
				
			}
		}
	}
	return z;
}

std::complex<double> GetInterpolatedValues::planeWaveValueDot(const Eigen::MatrixXd &w1, const Eigen::MatrixXd &w2,
															  const std::vector<std::complex<double>> &vertVals1,
															  const std::vector<std::complex<double>> &vertVals2,
															  const double dt, int vid, Eigen::VectorXcd *deriv,
															  Eigen::MatrixXcd *hess)
{
	int baseFid = _baryCoords[vid].first;
	Eigen::Vector3d baryCoord = _baryCoords[vid].second;
	Eigen::Vector3d weights = _baryWeights[vid].second;

	std::complex<double> zdot = 0;

	Eigen::VectorXcd zDeriv;
	Eigen::MatrixXcd zHess;
	std::vector<Eigen::MatrixXcd> zDerivHess;

	std::complex<double> z = planeWaveValue(w2, vertVals2, vid, &zDeriv, (deriv || hess) ? &zHess : NULL, hess ? &zDerivHess : NULL);

	Eigen::VectorXd xdot(12);
	xdot.setZero();

	Eigen::MatrixXd gradXdot(12, 24);
	gradXdot.setZero();

	for (int j = 0; j < 3; j++)
	{
		int baseVid = _baseMesh.faceVertex(baseFid, j);
		Eigen::Vector2d wnew = w2.row(baseVid);
		std::complex<double> fnew = vertVals2[baseVid];

		Eigen::Vector2d wold = w1.row(baseVid);
		std::complex<double> fold = vertVals1[baseVid];

		Eigen::Vector4d xjdot;
		xjdot << fnew.real() - fold.real(), fnew.imag() - fold.imag(), wnew(0) - wold(0), wnew(1) - wnew(1);
		xjdot /= dt;

		xdot.segment<4>(4 * j) = xjdot;

		if (deriv || hess)
		{
			gradXdot.block<4, 4>(4 * j, 8 * j).setIdentity();
			gradXdot.block<4, 4>(4 * j, 8 * j) *= -1.0 / dt;
			gradXdot.block<4, 4>(4 * j, 8 * j + 4).setIdentity();
			gradXdot.block<4, 4>(4 * j, 8 * j + 4) *= 1.0 / dt;
		}
	}

	// apply chain rule
	zdot = zDeriv.dot(xdot);

	if (deriv)
		deriv->setZero(24);
	if (hess)
		hess->setZero(24, 24);

	if (deriv || hess)
	{
		if (deriv)
		{
			(*deriv) = zDeriv.transpose() * gradXdot;
			(*deriv).segment<12>(12) += zHess * xdot;
		}

		if (hess)
		{
			for(int m = 0; m < 24; m++)
				for (int n = 0; n < 24; n++)
				{
					if (m >= 12 || n >= 12)
					{
						if (m >= 12)
						{
							(*hess)(m, n) += zHess.row(m - 12).dot(gradXdot.col(n).segment<12>(12));
						}
						if (n >= 12)
						{
							(*hess)(m, n) += zHess.row(n - 12).dot(gradXdot.col(m).segment<12>(12));
						}
						if (m >= 12 && n >= 12)
						{
							for (int p = 0; p < 12; p++)
								(*hess)(m, n) += zDerivHess[p](m - 12, n - 12) * xdot(p);
						}
					}
					
				}
		}
	}

	return zdot;
}

std::vector<std::complex<double>> GetInterpolatedValues::getZValues(const Eigen::MatrixXd &w,
																	const std::vector<std::complex<double>> &vertVals,
																	std::vector<Eigen::VectorXcd>* deriv,
																	std::vector<Eigen::MatrixXcd> *H)
{
	int nupverts = _upsampledPos.rows();
	if(deriv)
		deriv->resize(nupverts);
	if(H)
		H->resize(nupverts);
	std::vector<std::complex<double>> zvals(nupverts);

	auto computeZvals = [&](const tbb::blocked_range<uint32_t>& range){
		for (uint32_t i = range.begin(); i < range.end(); ++i)
		{
			zvals[i] = planeWaveValue(w, vertVals, i, deriv? &(deriv->at(i)) : NULL, H? &(H->at(i)) : NULL, NULL);
		}
	};

	tbb::blocked_range<uint32_t> rangex(0u, (uint32_t)nupverts, GRAIN_SIZE);
	tbb::parallel_for(rangex, computeZvals);

	return zvals;
}

std::vector<std::complex<double>> GetInterpolatedValues::getZDotValues(const Eigen::MatrixXd& w1,
	const Eigen::MatrixXd& w2,
	const std::vector<std::complex<double>>& vertVals1,
	const std::vector<std::complex<double>>& vertVals2,
	const double dt, std::vector<Eigen::VectorXcd>* deriv,
	std::vector<Eigen::MatrixXcd>* H)
{
	int nupverts = _upsampledPos.rows();
	if (deriv)
		deriv->resize(nupverts);
	if (H)
		H->resize(nupverts);
	std::vector<std::complex<double>> zdotvals(nupverts);

	auto computeZDotvals = [&](const tbb::blocked_range<uint32_t>& range) {
		for (uint32_t i = range.begin(); i < range.end(); ++i)
		{
			zdotvals[i] = planeWaveValueDot(w1, w2, vertVals1, vertVals2, dt, i, deriv ? &(deriv->at(i)) : NULL, H ? &(H->at(i)) : NULL);
		}
	};

	tbb::blocked_range<uint32_t> rangex(0u, (uint32_t)nupverts, GRAIN_SIZE);
	tbb::parallel_for(rangex, computeZDotvals);

	return zdotvals;
}

///////////////////////////////////// test functions //////////////////////////////////////
void GetInterpolatedValues::testPlaneWaveBasis(Eigen::VectorXd p, Eigen::VectorXd pi, Eigen::Vector2d omega)
{
	Eigen::VectorXcd deriv;
	Eigen::MatrixXcd hess;
	std::vector<Eigen::MatrixXcd> derivHess;

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

		Eigen::VectorXcd deriv1;
		Eigen::MatrixXcd hess1;
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

void GetInterpolatedValues::testPlaneWaveValue(const Eigen::MatrixXd& w, const std::vector<std::complex<double>>& vertVals, int vid)
{
	Eigen::VectorXcd deriv;
	Eigen::MatrixXcd hess;
	std::vector<Eigen::MatrixXcd> derivHess;
	std::complex<double> z = planeWaveValue(w, vertVals, vid, &deriv, &hess, &derivHess);
	
	Eigen::VectorXd dir = Eigen::VectorXd::Random(deriv.rows());

	Eigen::MatrixXd backupW = w;
	std::vector<std::complex<double>> backupVertVals = vertVals;

	for (int i = 3; i <= 10; i++)
	{
		double eps = std::pow(0.1, i);
		for (int j = 0; j < 3; j++)
		{
			int baseVid = _baseMesh.faceVertex(_baryCoords[vid].first, j);
			backupVertVals[baseVid] = std::complex<double>(vertVals[baseVid].real() + eps * dir(4 * j), vertVals[baseVid].imag() + eps * dir(4 * j + 1));
			backupW(baseVid, 0) = w(baseVid, 0) + eps * dir(4 * j + 2);
			backupW(baseVid, 1) = w(baseVid, 1) + eps * dir(4 * j + 3);
		}
		Eigen::VectorXcd deriv1;
		Eigen::MatrixXcd hess1;
		std::complex<double> z1 = planeWaveValue(backupW, backupVertVals, vid, &deriv1, &hess1, NULL);

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


void GetInterpolatedValues::testPlaneWaveValueDot(const Eigen::MatrixXd& w1, const Eigen::MatrixXd& w2, const std::vector<std::complex<double>>& vertVals1, const std::vector<std::complex<double>>& vertVals2, const double dt, int vid)
{
	Eigen::VectorXcd deriv;
	Eigen::MatrixXcd hess;
	std::complex<double> z = planeWaveValueDot(w1, w2, vertVals1, vertVals2, dt, vid, &deriv, &hess);

	Eigen::VectorXd dir = Eigen::VectorXd::Random(deriv.rows());

	Eigen::MatrixXd backupW1 = w1;
	std::vector<std::complex<double>> backupVertVals1 = vertVals1;

	Eigen::MatrixXd backupW2 = w2;
	std::vector<std::complex<double>> backupVertVals2 = vertVals2;

	for (int i = 3; i <= 10; i++)
	{
		double eps = std::pow(0.1, i);
		for (int j = 0; j < 3; j++)
		{
			int baseVid = _baseMesh.faceVertex(_baryCoords[vid].first, j);

			backupVertVals1[baseVid] = std::complex<double>(vertVals1[baseVid].real() + eps * dir(8 * j), vertVals1[baseVid].imag() + eps * dir(8 * j + 1));
			backupW1(baseVid, 0) = w1(baseVid, 0) + eps * dir(8 * j + 2);
			backupW1(baseVid, 1) = w1(baseVid, 1) + eps * dir(8 * j + 3);

			backupVertVals2[baseVid] = std::complex<double>(vertVals2[baseVid].real() + eps * dir(8 * j + 4), vertVals2[baseVid].imag() + eps * dir(8 * j + 5));
			backupW2(baseVid, 0) = w2(baseVid, 0) + eps * dir(8 * j + 6);
			backupW2(baseVid, 1) = w2(baseVid, 1) + eps * dir(8 * j + 7);
		}
		Eigen::VectorXcd deriv1;
		
		std::complex<double> z1 = planeWaveValueDot(backupW1, backupW2, backupVertVals1, backupVertVals2, dt, vid, &deriv1, NULL);


		std::cout << "eps: " << eps << std::endl;

		std::cout << "value-gradient check: " << (z1 - z) / eps - deriv.dot(dir) << std::endl;
		std::cout << "gradient-hessian check: " << ((deriv1 - deriv) / eps - hess * dir).norm() << std::endl;
	}
}