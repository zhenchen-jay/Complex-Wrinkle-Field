#include "../../include/EuclideanFormula/InterpolateZvalsFromEuclideanOmega.h"

std::complex<double> EuclideanFormula::planeWaveBasis(Eigen::Vector3d p, Eigen::Vector3d pi, Eigen::Vector3d omega, Eigen::Vector3cd * deriv, Eigen::Matrix3cd * hess)
{
	if (deriv)
		deriv->setZero();
	if (hess)
		hess->setZero();

	if (omega.norm() == 0)
		return 1;
	Eigen::Vector3d pv = p - pi;
	double deltatheta = omega.dot(pv);
	Eigen::Vector3d gradDeltaTheta = pv;

	std::complex<double> z = std::complex<double>(std::cos(deltatheta), std::sin(deltatheta));

	if (deriv)
	{
		(*deriv) = std::complex<double>(0, 1) * z * pv;
	}
	if (hess)
	{
		(*hess) = -z * gradDeltaTheta * gradDeltaTheta.transpose();
	}

	return z;
}

std::complex<double> EuclideanFormula::getZvalsFromEuclideanOmega(const Eigen::MatrixXd& basePos, const MeshConnectivity& baseMesh, const int& faceId, const Eigen::Vector3d& bary, const std::vector<std::complex<double>> vertZvals, const Eigen::MatrixXd& w, Eigen::Matrix<std::complex<double>, 15, 1>* deriv, Eigen::Matrix<std::complex<double>, 15, 15>* hess)
{
	Eigen::Vector3d hatWeight = computeHatWeight(bary(1), bary(2));		// u = alpha1, v = alpha2
	std::complex<double> z = 0;

	if (deriv)
	{
		deriv->setZero();
	}
	if (hess)
	{
		hess->setZero();
	}

	Eigen::Vector3d p = bary(0) * basePos.row(baseMesh.faceVertex(faceId, 0)) + bary(1) * basePos.row(baseMesh.faceVertex(faceId, 1)) + bary(2) * basePos.row(baseMesh.faceVertex(faceId, 2));


	for (int j = 0; j < 3; j++)
	{
		int baseVid = baseMesh.faceVertex(faceId, j);
		Eigen::Vector3d wi = w.row(baseVid);
		std::complex<double> fi = vertZvals[baseVid];
		Eigen::Vector3cd expDeriv;
		Eigen::Matrix3cd expHess;

		std::complex<double> expPart = planeWaveBasis(p, basePos.row(baseVid), wi, (deriv || hess) ? &expDeriv : NULL, hess ? &expHess : NULL);

		z += hatWeight(j) * fi * expPart;

		if (deriv || hess)
		{
			Eigen::Matrix<std::complex<double>, 5, 1> gradFi;
			gradFi << 1, std::complex<double>(0, 1), 0, 0, 0;

			Eigen::Matrix<std::complex<double>, 5, 1> fullExpDeriv;
			fullExpDeriv << 0, 0, expDeriv(0), expDeriv(1), expDeriv(2);

			if (deriv)
			{
				deriv->segment<5>(5 * j) += hatWeight(j) * (expPart * gradFi + fullExpDeriv * fi);
			}
			if (hess)
			{
				Eigen::Matrix<std::complex<double>, 5, 5> fullExpHess;
				fullExpHess.setZero();
				fullExpHess.block<3, 3>(2, 2) = expHess;

				if (hess)
					hess->block<5, 5>(5 * j, 5 * j) += hatWeight(j) * (gradFi * fullExpDeriv.transpose() + fullExpDeriv * gradFi.transpose() + fi * fullExpHess);


			}
		}
	}
	return z;
}

std::vector<std::complex<double>> EuclideanFormula::upsamplingZvals(const Eigen::MatrixXd& triV, const MeshConnectivity& mesh, const std::vector<std::complex<double>>& zvals, const Eigen::MatrixXd& w, const std::vector<std::pair<int, Eigen::Vector3d>>& bary)
{
	int size = bary.size();
	std::vector<std::complex<double>> upzvals(size);
	auto computeZvals = [&](const tbb::blocked_range<uint32_t>& range) {
		for (uint32_t i = range.begin(); i < range.end(); ++i)
		{
			upzvals[i] = getZvalsFromEuclideanOmega(triV, mesh, bary[i].first, bary[i].second, zvals, w, NULL, NULL);
		}
	};

	tbb::blocked_range<uint32_t> rangex(0u, (uint32_t)size, GRAIN_SIZE);
	tbb::parallel_for(rangex, computeZvals);

	return upzvals;
}