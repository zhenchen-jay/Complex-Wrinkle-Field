#include "../include/AmpExtraction.h"
#include "../../include/Optimization/NewtonDescent.h"
#include <igl/cotmatrix.h>

double curlFreeEnergyPerface(const MeshConnectivity& mesh, const Eigen::MatrixXd& w, int faceId)
{
	double E = 0;

	double diff0;
	Eigen::Matrix<double, 3, 1> select0;
	select0.setZero();

	Eigen::Matrix<double, 3, 1> edgews;

	for (int i = 0; i < 3; i++)
	{
		int eid = mesh.faceEdge(faceId, i);
		edgews(i) = w(eid);

		if (mesh.faceVertex(faceId, (i + 1) % 3) == mesh.edgeVertex(eid, 0))
		{
			select0(i) = 1;
		}

		else
		{
			select0(i) = -1;
		}
	}
	diff0 = select0.dot(edgews);

	E = 0.5 * (diff0 * diff0);

	return E;
}

double amplitudeEnergyWithGivenOmegaPerface(const MeshConnectivity& mesh, const Eigen::VectorXd& amp, const Eigen::VectorXd& w, int fid, Eigen::Vector3d* deriv, Eigen::Matrix3d* hess)
{
	double energy = 0;

	double curlSq = curlFreeEnergyPerface(mesh, w, fid);
	Eigen::Vector3d wSq;
	wSq.setZero();


	if (deriv)
		deriv->setZero();
	if (hess)
		hess->setZero();

	for (int i = 0; i < 3; i++)
	{
		int vid = mesh.faceVertex(fid, i);
		energy += 0.5 * amp(vid) * amp(vid) / 3 * curlSq;

		if (deriv)
			(*deriv)(i) += amp(vid) * curlSq / 3;
		if (hess)
			(*hess)(i, i) += curlSq / 3;
	}

	return energy;
}

double amplitudeEnergyWithGivenOmega(const MeshConnectivity& mesh, const Eigen::VectorXd& amp, const Eigen::VectorXd& w,
	Eigen::VectorXd* deriv, std::vector<Eigen::Triplet<double>>* hessT)
{
	double energy = 0;

	int nverts = amp.rows();
	int nfaces = mesh.nFaces();

	std::vector<double> energyList(nfaces);
	std::vector<Eigen::Vector3d> derivList(nfaces);
	std::vector<Eigen::Matrix3d> hessList(nfaces);

	auto computeEnergy = [&](const tbb::blocked_range<uint32_t>& range) {
		for (uint32_t i = range.begin(); i < range.end(); ++i)
		{
			energyList[i] = amplitudeEnergyWithGivenOmegaPerface(mesh, amp, w, i, deriv ? &derivList[i] : NULL, hessT ? &hessList[i] : NULL);
		}
	};

	tbb::blocked_range<uint32_t> rangex(0u, (uint32_t)nfaces, GRAIN_SIZE);
	tbb::parallel_for(rangex, computeEnergy);

	if (deriv)
		deriv->setZero(nverts);
	if (hessT)
		hessT->clear();

	for (int fid = 0; fid < nfaces; fid++)
	{
		energy += energyList[fid];

		if (deriv)
		{
			for (int j = 0; j < 3; j++)
			{
				int vid = mesh.faceVertex(fid, j);
				(*deriv)(vid) += derivList[fid](j);
			}
		}

		if (hessT)
		{
			for (int j = 0; j < 3; j++)
			{
				int vid = mesh.faceVertex(fid, j);
				for (int k = 0; k < 3; k++)
				{
					int vid1 = mesh.faceVertex(fid, k);
					hessT->push_back({ vid, vid1, hessList[fid](j, k) });
				}
			}
		}
	}
	return energy;
}

void ampExtraction(const Eigen::MatrixXd& pos, const MeshConnectivity& mesh, const Eigen::VectorXd& w, const std::vector<std::pair<int, double>>& clampedAmp, Eigen::VectorXd& amp)
{
	int nverts = pos.rows();
	int nfaces = mesh.nFaces();

	Eigen::VectorXd clampedDOFs;
	clampedDOFs.setZero(nverts);

	std::vector<Eigen::Triplet<double>> T;
	// projection matrix
	Eigen::VectorXd freeVids;
	freeVids.setOnes(nverts);

	for (auto& it : clampedAmp)
	{
		clampedDOFs(it.first) = it.second;
		freeVids(it.first) = 0;
	}

	std::vector<int> freeDOFs;
	for (int i = 0; i < nverts; i++)
	{
		if (freeVids(i))
			freeDOFs.push_back(i);
	}

	for (int i = 0; i < freeDOFs.size(); i++)
	{
		T.push_back(Eigen::Triplet<double>(i, freeDOFs[i], 1.0));
	}

	Eigen::SparseMatrix<double> projM(freeDOFs.size(), nverts);
	projM.setFromTriplets(T.begin(), T.end());

	Eigen::SparseMatrix<double> unProjM = projM.transpose();

	auto projVar = [&](const Eigen::VectorXd& fullX)
	{
		Eigen::VectorXd x0 = projM * fullX;
		return x0;
	};

	auto unProjVar = [&](const Eigen::VectorXd& x)
	{
		Eigen::VectorXd fullX = unProjM * x + clampedDOFs;
		return fullX;
	};

	Eigen::SparseMatrix<double> L;
	igl::cotmatrix(pos, mesh.faces(), L);

	auto funVal = [&](const Eigen::VectorXd& x, Eigen::VectorXd* grad, Eigen::SparseMatrix<double>* hess, bool isProj) {
		Eigen::VectorXd deriv, deriv1;
		std::vector<Eigen::Triplet<double>> T;
		Eigen::SparseMatrix<double> H;

		Eigen::VectorXd fullx = unProjVar(x);
		double E = -0.5 * fullx.dot(L * fullx);

		E += amplitudeEnergyWithGivenOmega(mesh, fullx, w, grad ? &deriv1 : NULL, hess ? &T : NULL);

		if (grad)
		{
			deriv = -L * fullx + deriv1;
			(*grad) = projM * deriv;
		}

		if (hess)
		{
			H.resize(fullx.rows(), fullx.rows());
			H.setFromTriplets(T.begin(), T.end());
			(*hess) = projM * (H - L) * unProjM;
		}

		return E;
	};
	auto maxStep = [&](const Eigen::VectorXd& x, const Eigen::VectorXd& dir) {
		return 1.0;
	};

	Eigen::VectorXd x0 = projVar(clampedDOFs);
	OptSolver::newtonSolver(funVal, maxStep, x0, 1000, 1e-6, 1e-10, 1e-15, false);

	Eigen::VectorXd deriv;
	double E = funVal(x0, &deriv, NULL, false);
	std::cout << "terminated with energy : " << E << ", gradient norm : " << deriv.norm() << std::endl << std::endl;
	amp = unProjVar(x0);

}