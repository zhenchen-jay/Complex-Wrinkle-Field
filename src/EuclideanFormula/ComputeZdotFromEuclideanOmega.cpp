#include "../../include/EuclideanFormula/ComputeZdotFromEuclideanOmega.h"
#include <iostream>

using namespace EuclideanFormula;

double ComputeZdotFromEuclideanOmega::computeZdotIntegrationFromQuad(const std::vector<std::complex<double>>& curZvals, const Eigen::MatrixXd& curw, const std::vector<std::complex<double>>& nextZvals, const Eigen::MatrixXd& nextw, int fid, int qid, Eigen::Matrix<double, 30, 1>* deriv, Eigen::Matrix<double, 30, 30>* hess)
{
	double energy = 0;

	Eigen::Vector3d bary(1 - _quadpts[qid].u - _quadpts[qid].v, _quadpts[qid].u, _quadpts[qid].v);
	Eigen::Matrix<double, 3, 3> wcur, wnext;
	std::vector<std::complex<double>> vertZvalcur(3), vertZvalnext(3);

	Eigen::Matrix<std::complex<double>, 15, 1> derivCur, derivNext;
	Eigen::Matrix<std::complex<double>, 15, 15> hessCur, hessNext;

	std::complex<double> zcur = EuclideanFormula::getZvalsFromEuclideanOmega(_pos, _mesh, fid, bary, curZvals, curw, (deriv || hess) ? &derivCur : NULL, hess ? &hessCur : NULL);
	std::complex<double> znext = EuclideanFormula::getZvalsFromEuclideanOmega(_pos, _mesh, fid, bary, nextZvals, nextw, (deriv || hess) ? &derivNext : NULL, hess ? &hessNext : NULL);

	std::complex<double> deltaz = znext - zcur;

	double componentWeights = 0.5 * _faceArea[fid] * _quadpts[qid].weight / (_dt * _dt);
	energy = componentWeights * (deltaz.real() * deltaz.real() + deltaz.imag() * deltaz.imag());

	{
		Eigen::Matrix<std::complex<double>, 30, 1> gradDeltaZ;

		gradDeltaZ.segment<15>(0) = -derivCur;
		gradDeltaZ.segment<15>(15) = derivNext;

		if (deriv)
		{
			(*deriv) = 2 * componentWeights * (deltaz.real() * gradDeltaZ.real() + deltaz.imag() * gradDeltaZ.imag());
		}

		if (hess)
		{
			hess->setZero();
			Eigen::Matrix<std::complex<double>, 30, 30> hessDeltaZ;

			hessDeltaZ.block<15, 15>(0, 0) = -hessCur;
			hessDeltaZ.block<15, 15>(15, 15) = hessNext;

			(*hess) = gradDeltaZ.real() * (gradDeltaZ.real()).transpose() + gradDeltaZ.imag() * gradDeltaZ.imag().transpose();
			(*hess) += deltaz.real() * hessDeltaZ.real() + deltaz.imag() * hessDeltaZ.imag();
			(*hess) *= 2 * componentWeights;

		}

	}


	return energy;
}

double ComputeZdotFromEuclideanOmega::computeZdotIntegrationPerface(const std::vector<std::complex<double>>& curZvals, const Eigen::MatrixXd& curw, const std::vector<std::complex<double>>& nextZvals, const Eigen::MatrixXd& nextw, int fid, Eigen::Matrix<double, 30, 1>* deriv, Eigen::Matrix<double, 30, 30>* hess, bool isProj)
{
	double energy = 0;
	if (deriv)
	{
		deriv->setZero();
	}

	if (hess)
	{
		hess->setZero();
	}

	for (int qid = 0; qid < _quadpts.size(); qid++)
	{
		Eigen::Matrix<double, 30, 1> zdotDeriv;
		Eigen::Matrix<double, 30, 30> zdotHess;
		energy += computeZdotIntegrationFromQuad(curZvals, curw, nextZvals, nextw, fid, qid, deriv ? &zdotDeriv : NULL, hess ? &zdotHess : NULL);

		if (deriv)
		{
			(*deriv) += zdotDeriv;
		}

		if (hess)
		{
			(*hess) += zdotHess;
		}
	}

	if (hess && isProj)
	{
		(*hess) = SPDProjection(*hess);
	}
	return energy;
}


double ComputeZdotFromEuclideanOmega::computeZdotIntegration(const std::vector<std::complex<double>>& curZvals, const Eigen::MatrixXd& curw, const std::vector<std::complex<double>>& nextZvals, const Eigen::MatrixXd& nextw, Eigen::VectorXd* deriv, std::vector<Eigen::Triplet<double>>* hessT, bool isProj)
{
	double energy = 0;

	int nverts = curZvals.size();
	int nedges = curw.rows();
	int nfaces = _mesh.nFaces();

	if (deriv)
		deriv->setZero(10 * nverts);
	if (hessT)
		hessT->clear();

	std::vector<double> energyList(nfaces);
	std::vector<Eigen::Matrix<double, 30, 1>> derivList(nfaces);
	std::vector<Eigen::Matrix<double, 30, 30>> hessList(nfaces);

	auto computeEnergy = [&](const tbb::blocked_range<uint32_t>& range) {
		for (uint32_t i = range.begin(); i < range.end(); ++i)
		{
			energyList[i] = computeZdotIntegrationPerface(curZvals, curw, nextZvals, nextw, i, deriv ? &derivList[i] : NULL, hessT ? &hessList[i] : NULL, isProj);
		}
	};

	tbb::blocked_range<uint32_t> rangex(0u, (uint32_t)nfaces, GRAIN_SIZE);
	tbb::parallel_for(rangex, computeEnergy);

	/*for (uint32_t i = 0; i < nfaces; ++i)
	{
		energyList[i] = computeZdotIntegrationPerface(curZvals, curw, nextZvals, nextw, i, deriv ? &derivList[i] : NULL, hessT ? &hessList[i] : NULL, isProj);
	}*/

	int nOneGroup = 2 * nverts + 3 * nverts;

	for (int i = 0; i < nfaces; i++)
	{
		energy += energyList[i];

		if (deriv)
		{
			for (int j = 0; j < 3; j++)
			{
				int vid = _mesh.faceVertex(i, j);

				for (int k = 0; k < 2; k++)
				{
					(*deriv)(2 * vid + k) += derivList[i](5 * j + k);
					(*deriv)(2 * vid + nOneGroup + k) += derivList[i](5 * j + 15 + k);
				}

				for (int k = 0; k < 3; k++)
				{
					(*deriv)(3 * vid + 2 * nverts + k) += derivList[i](5 * j + 2 + k);
					(*deriv)(3 * vid + nOneGroup + 2 * nverts + k) += derivList[i](5 * j + 17 + k);
				}
			}
		}

		if (hessT)
		{
			for (int j = 0; j < 3; j++)
			{
				int vid = _mesh.faceVertex(i, j);
				for (int k = 0; k < 3; k++)
				{
					int vid1 = _mesh.faceVertex(i, k);
					for (int m = 0; m < 2; m++)
						for (int m1 = 0; m1 < 2; m1++)
						{
							hessT->push_back({ 2 * vid + m, 2 * vid1 + m1, hessList[i](5 * j + m, 5 * k + m1) });
							hessT->push_back({ 2 * vid + m, 2 * vid1 + nOneGroup + m1, hessList[i](5 * j + m, 5 * k + 15 + m1) });
							hessT->push_back({ 2 * vid + nOneGroup + m, 2 * vid1 + m1, hessList[i](5 * j + 15 + m, 5 * k + m1) });
							hessT->push_back({ 2 * vid + nOneGroup + m, 2 * vid1 + nOneGroup + m1, hessList[i](5 * j + 15 + m, 5 * k + 15 + m1) });
						}

					for (int m = 0; m < 3; m++)
						for (int m1 = 0; m1 < 3; m1++)
						{
							hessT->push_back({ 3 * vid + m + 2 * nverts, 3 * vid1 + m1 + 2 * nverts, hessList[i](5 * j + 2 + m, 5 * k + 2 + m1) });
							hessT->push_back({ 3 * vid + m + 2 * nverts, 3 * vid1 + m1 + nOneGroup + 2 * nverts, hessList[i](5 * j + 2 + m, 5 * k + 17 + m1) });
							hessT->push_back({ 3 * vid + m + nOneGroup + 2 * nverts, 3 * vid1 + m1 + 2 * nverts, hessList[i](5 * j + 17 + m, 5 * k + 2 + m1) });
							hessT->push_back({ 3 * vid + m + nOneGroup + 2 * nverts, 3 * vid1 + m1 + nOneGroup + 2 * nverts, hessList[i](5 * j + 17 + m, 5 * k + 17 + m1) });
						}

					for (int m = 0; m < 2; m++)
						for (int m1 = 0; m1 < 3; m1++)
						{
							hessT->push_back({ 2 * vid + m, 3 * vid1 + m1 + 2 * nverts, hessList[i](5 * j + m, 5 * k + 2 + m1) });
							hessT->push_back({ 2 * vid + m, 3 * vid1 + m1 + nOneGroup + 2 * nverts, hessList[i](5 * j + m, 5 * k + 17 + m1) });

							hessT->push_back({ 2 * vid + nOneGroup + m, 3 * vid1 + m1 + 2 * nverts, hessList[i](5 * j + 15 + m, 5 * k + 2 + m1) });
							hessT->push_back({ 2 * vid + nOneGroup + m, 3 * vid1 + m1 + nOneGroup + 2 * nverts, hessList[i](5 * j + 15 + m, 5 * k + 17 + m1) });

							hessT->push_back({ 3 * vid1 + m1 + 2 * nverts, 2 * vid + m, hessList[i](5 * k + 2 + m1, 5 * j + m) });
							hessT->push_back({ 3 * vid1 + m1 + 2 * nverts, 2 * vid + m + nOneGroup, hessList[i](5 * k + 2 + m1, 5 * j + 15 + m) });


							hessT->push_back({ 3 * vid1 + m1 + nOneGroup + 2 * nverts, 2 * vid + m, hessList[i](5 * k + 17 + m1, 5 * j + m) });
							hessT->push_back({ 3 * vid1 + m1 + nOneGroup + 2 * nverts, 2 * vid + m + nOneGroup, hessList[i](5 * k + 17 + m1, 5 * j + 15 + m) });
						}
				}
			}
		}
	}

	return energy;
}

void ComputeZdotFromEuclideanOmega::testZdotIntegrationFromQuad(const std::vector<std::complex<double>>& curZvals,
	const Eigen::MatrixXd& curw,
	const std::vector<std::complex<double>>& nextZvals,
	const Eigen::MatrixXd& nextw, int fid, int qid)
{
	Eigen::Matrix<double, 30, 1> deriv;
	Eigen::Matrix<double, 30, 30> hess;

	double e = computeZdotIntegrationFromQuad(curZvals, curw, nextZvals, nextw, fid, qid, &deriv, &hess);


	Eigen::VectorXd dir = Eigen::VectorXd::Random(deriv.rows());
	std::cout << "dir: " << dir.transpose() << std::endl;

	Eigen::MatrixXd backupW1 = curw;
	std::vector<std::complex<double>> backupVertVals1 = curZvals;

	Eigen::MatrixXd backupW2 = nextw;
	std::vector<std::complex<double>> backupVertVals2 = nextZvals;

	for (int i = 3; i <= 10; i++)
	{
		double eps = std::pow(0.1, i);
		for (int j = 0; j < 3; j++)
		{
			int baseVid = _mesh.faceVertex(fid, j);

			backupVertVals1[baseVid] = std::complex<double>(curZvals[baseVid].real() + eps * dir(5 * j), curZvals[baseVid].imag() + eps * dir(5 * j + 1));
			backupW1(baseVid, 0) = curw(baseVid, 0) + eps * dir(5 * j + 2);
			backupW1(baseVid, 1) = curw(baseVid, 1) + eps * dir(5 * j + 3);
			backupW1(baseVid, 2) = curw(baseVid, 2) + eps * dir(5 * j + 4);

			backupVertVals2[baseVid] = std::complex<double>(nextZvals[baseVid].real() + eps * dir(5 * j + 15), nextZvals[baseVid].imag() + eps * dir(5 * j + 16));
			backupW2(baseVid, 0) = nextw(baseVid, 0) + eps * dir(5 * j + 14);
			backupW2(baseVid, 1) = nextw(baseVid, 1) + eps * dir(5 * j + 15);
			backupW2(baseVid, 2) = nextw(baseVid, 2) + eps * dir(5 * j + 16);
		}
		Eigen::Matrix<double, 30, 1> deriv1;

		double e1 = computeZdotIntegrationFromQuad(backupVertVals1, backupW1, backupVertVals2, backupW2, fid, qid, &deriv1, NULL);


		std::cout << "eps: " << eps << std::endl;
		std::cout << "value-gradient check: " << (e1 - e) / eps - dir.dot(deriv) << std::endl;
		std::cout << "gradient-hessian check: " << ((deriv1 - deriv) / eps - hess * dir).norm() << std::endl;
	}
}

void ComputeZdotFromEuclideanOmega::testZdotIntegrationPerface(const std::vector<std::complex<double>>& curZvals,
	const Eigen::MatrixXd& curw,
	const std::vector<std::complex<double>>& nextZvals,
	const Eigen::MatrixXd& nextw, int fid)
{
	Eigen::Matrix<double, 30, 1> deriv;
	Eigen::Matrix<double, 30, 30> hess;

	double e = computeZdotIntegrationPerface(curZvals, curw, nextZvals, nextw, fid, &deriv, &hess);


	Eigen::VectorXd dir = Eigen::VectorXd::Random(deriv.rows());
	std::cout << "dir: " << dir.transpose() << std::endl;

	Eigen::MatrixXd backupW1 = curw;
	std::vector<std::complex<double>> backupVertVals1 = curZvals;

	Eigen::MatrixXd backupW2 = nextw;
	std::vector<std::complex<double>> backupVertVals2 = nextZvals;

	for (int i = 3; i <= 10; i++)
	{
		double eps = std::pow(0.1, i);
		for (int j = 0; j < 3; j++)
		{
			int baseVid = _mesh.faceVertex(fid, j);

			backupVertVals1[baseVid] = std::complex<double>(curZvals[baseVid].real() + eps * dir(5 * j), curZvals[baseVid].imag() + eps * dir(5 * j + 1));
			backupW1(baseVid, 0) = curw(baseVid, 0) + eps * dir(5 * j + 2);
			backupW1(baseVid, 1) = curw(baseVid, 1) + eps * dir(5 * j + 3);
			backupW1(baseVid, 2) = curw(baseVid, 2) + eps * dir(5 * j + 4);

			backupVertVals2[baseVid] = std::complex<double>(nextZvals[baseVid].real() + eps * dir(5 * j + 15), nextZvals[baseVid].imag() + eps * dir(5 * j + 16));
			backupW2(baseVid, 0) = nextw(baseVid, 0) + eps * dir(5 * j + 14);
			backupW2(baseVid, 1) = nextw(baseVid, 1) + eps * dir(5 * j + 15);
			backupW2(baseVid, 2) = nextw(baseVid, 2) + eps * dir(5 * j + 16);
		}
		Eigen::Matrix<double, 30, 1> deriv1;

		double e1 = computeZdotIntegrationPerface(backupVertVals1, backupW1, backupVertVals2, backupW2, fid, &deriv1, NULL);


		std::cout << "eps: " << eps << std::endl;
		std::cout << "value-gradient check: " << (e1 - e) / eps - dir.dot(deriv) << std::endl;
		std::cout << "gradient-hessian check: " << ((deriv1 - deriv) / eps - hess * dir).norm() << std::endl;
	}
}

void ComputeZdotFromEuclideanOmega::testZdotIntegration(const std::vector<std::complex<double>>& curZvals,
	const Eigen::MatrixXd& curw,
	const std::vector<std::complex<double>>& nextZvals,
	const Eigen::MatrixXd& nextw)
{
	Eigen::VectorXd deriv, deriv1;
	std::vector<Eigen::Triplet<double>> hessT;
	Eigen::SparseMatrix<double> hess;
	double energy = computeZdotIntegration(curZvals, curw, nextZvals, nextw, &deriv, &hessT);
	hess.resize(deriv.rows(), deriv.rows());
	hess.setFromTriplets(hessT.begin(), hessT.end());

	int nverts = curZvals.size();
	int nedges = curw.rows();

	Eigen::VectorXd dir = deriv;
	dir.setRandom();


	auto backupcurZvals = curZvals, backupnextZvals = nextZvals;
	auto backupcurw = curw, backupnextw = nextw;

	for (int i = 3; i < 10; i++)
	{
		double eps = std::pow(0.1, i);

		for (int j = 0; j < nverts; j++)
		{
			backupcurZvals[j] = std::complex<double>(curZvals[j].real() + dir(2 * j) * eps, curZvals[j].imag() + dir(2 * j + 1) * eps);
			backupnextZvals[j] = std::complex<double>(nextZvals[j].real() + dir(2 * j + 5 * nverts) * eps, nextZvals[j].imag() + dir(2 * j + 1 + 5 * nverts) * eps);
		}

		for (int j = 0; j < nverts; j++)
		{
			backupcurw.row(j) = curw.row(j) + eps * dir.segment<3>(2 * nverts + 3 * j).transpose();
			backupnextw.row(j) = nextw.row(j) + eps * dir.segment<3>(2 * nverts + 5 * nverts + 3 * j).transpose();
		}



		double energy1 = computeZdotIntegration(backupcurZvals, backupcurw, backupnextZvals, backupnextw, &deriv1, NULL);

		std::cout << "eps: " << eps << std::endl;
		std::cout << "f-g: " << (energy1 - energy) / eps - dir.dot(deriv) << std::endl;
		std::cout << "g-h: " << ((deriv1 - deriv) / eps - (hess * dir)).norm() << std::endl;
	}
}