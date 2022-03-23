#include "../../include/IntrinsicFormula/WrinkleGluingProcess.h"
#include "../../include/Optimization/NewtonDescent.h"
#include <igl/cotmatrix_entries.h>
#include <igl/cotmatrix.h>
#include <igl/doublearea.h>
#include <Eigen/SPQRSupport>

using namespace IntrinsicFormula;

WrinkleGluingProcess::WrinkleGluingProcess(const Eigen::MatrixXd& pos, const MeshConnectivity& mesh, const Eigen::VectorXi& faceFlag, int quadOrd)
{
	_pos = pos;
	_mesh = mesh;
	_faceFlag = faceFlag;
	igl::cotmatrix_entries(pos, mesh.faces(), _cotMatrixEntries);
	igl::doublearea(pos, mesh.faces(), _faceArea);
	_faceArea /= 2.0;
    _quadOrd = quadOrd;

	int nverts = pos.rows();
	int nfaces = mesh.nFaces();
	int nedges = mesh.nEdges();

	_vertFlag.resize(nverts);
	_vertFlag.setConstant(-1);

	_edgeFlag.resize(nedges);
	_edgeFlag.setConstant(-1);

	for (int i = 0; i < nfaces; i++)
	{
		if (faceFlag(i) != -1)
		{
			for(int j = 0; j < 3; j++)
			{
				_vertFlag(mesh.faceVertex(i, j)) = faceFlag(i);
				_edgeFlag(mesh.faceEdge(i, j)) = faceFlag(i);
			}
		}
	}
}

void WrinkleGluingProcess::initialization(const std::vector<std::vector<Eigen::VectorXd>>& refAmpList, std::vector<std::vector<Eigen::MatrixXd>>& refOmegaList)
{
    computeCombinedRefAmpList(refAmpList);
    computeCombinedRefOmegaList(refOmegaList);

    std::vector<std::complex<double>> initZvals;
    std::vector<std::complex<double>> tarZvals;

    int nFrames = _combinedRefAmpList.size() - 2;
    roundVertexZvalsFromHalfEdgeOmegaVertexMag(_mesh, _combinedRefOmegaList[0], _combinedRefAmpList[0], _faceArea, _cotMatrixEntries, _pos.rows(), initZvals);
    roundVertexZvalsFromHalfEdgeOmegaVertexMag(_mesh, _combinedRefOmegaList[nFrames + 1], _combinedRefAmpList[nFrames + 1], _faceArea, _cotMatrixEntries, _pos.rows(), tarZvals);

    _model = IntrinsicKnoppelDrivenFormula(_mesh, _faceArea, _cotMatrixEntries, _combinedRefOmegaList, _combinedRefAmpList, initZvals, tarZvals, _combinedRefOmegaList[0], _combinedRefOmegaList[nFrames + 1], nFrames, 1.0, _quadOrd);
}

void WrinkleGluingProcess::computeCombinedRefAmpList(const std::vector<std::vector<Eigen::VectorXd>>& refAmpList)
{
	int nverts = _pos.rows();
	int nfaces = _mesh.nFaces();
	int nFrames = refAmpList.size();

	_combinedRefAmpList.resize(nFrames);

    double c = std::min(1.0 / (nFrames * nFrames), 1e-3);

	Eigen::SparseMatrix<double> L;
	igl::cotmatrix(_pos, _mesh.faces(), L);
	// form the sparse matrix for interplotate the reference amplitude
	std::vector<Eigen::Triplet<double>> T;
	for (int k = 0; k < L.outerSize(); ++k)
		for (Eigen::SparseMatrix<double>::InnerIterator it(L, k); it; ++it)
		{
			//T.push_back(Eigen::Triplet<double>(it.row(), it.col(), -it.value() + c)); 
			T.push_back(Eigen::Triplet<double>(it.row(), it.col(), -it.value())); 
		}
	// projection matrix
	std::vector<int> fixedVid;
	for (int i = 0; i < nverts; i++)
	{
		if (_vertFlag(i) != -1)
		{
			fixedVid.push_back(i);
		}
	}

	for (int i = 0; i < fixedVid.size(); i++)
	{
		T.push_back(Eigen::Triplet<double>(i + nverts, fixedVid[i], 1.0));
		T.push_back(Eigen::Triplet<double>(fixedVid[i], i + nverts, 1.0));
	}

	Eigen::SparseMatrix<double> ampMat(nverts + fixedVid.size(), nverts + fixedVid.size());
	ampMat.setFromTriplets(T.begin(), T.end());

	//Eigen::SparseQR<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> ampSolver;
	Eigen::SPQR<Eigen::SparseMatrix<double>> ampSolver;
	ampSolver.compute(ampMat);

    Eigen::VectorXd lastAmp = Eigen::VectorXd::Zero(nverts);
    for(int i = 0; i < nverts; i++)
    {
        if (_vertFlag(i) != -1)
            lastAmp(i) = refAmpList[0][_vertFlag[i]](i);
    }

	for (int i = 0; i < nFrames; i++)
	{
		Eigen::VectorXd rhs = Eigen::VectorXd::Zero(nverts + fixedVid.size());
		int id = 0;
		for (int j = 0; j < nverts; j++)
		{
            //rhs(j) = c * lastAmp(j);
			if (_vertFlag(j) != -1)
			{
				rhs(nverts + id) = refAmpList[i][_vertFlag(j)](j);
				id++;
			}
		}
		Eigen::VectorXd sol = ampSolver.solve(rhs);
		std::cout << "\nFrame " << std::to_string(i) << ": residual: " << (ampMat * sol - rhs).norm() << std::endl;
		_combinedRefAmpList[i] = sol.segment(0, nverts);
        lastAmp = _combinedRefAmpList[i];
	}
}

double WrinkleGluingProcess::curlFreeEnergyPerface(const Eigen::MatrixXd& w, int faceId, Eigen::Matrix<double, 6, 1>* deriv, Eigen::Matrix<double, 6, 6>* hess)
{
	double E = 0;

	double diff0, diff1;
	Eigen::Matrix<double, 6, 1> select0, select1;
	select0.setZero();
	select1.setZero();

	Eigen::Matrix<double, 6, 1> edgews;
	
	for (int i = 0; i < 3; i++)
	{
		int eid = _mesh.faceEdge(faceId, i);
		edgews(2 * i) = w(eid, 0);
		edgews(2 * i + 1) = w(eid, 1);

		if (_mesh.faceVertex(faceId, (i + 1) % 3) == _mesh.edgeVertex(eid, 0))
		{
			select0(2 * i) = 1;
			select1(2 * i + 1) = 1;
		}
			
		else
		{
			select0(2 * i + 1) = 1;
			select1(2 * i) = 1;
		}	
	}
	diff0 = select0.dot(edgews);
	diff1 = select1.dot(edgews);

	E = 0.5 * (diff0 * diff0 + diff1 * diff1);
	if (deriv)
	{
		*deriv = (select0 * select0.transpose() + select1 * select1.transpose()) * edgews;
	}
	if (hess)
	{
		*hess = select0 * select0.transpose() + select1 * select1.transpose();
	}

	return E;
}


double WrinkleGluingProcess::curlFreeEnergy(const Eigen::MatrixXd& w, Eigen::VectorXd* deriv, std::vector<Eigen::Triplet<double>> *hessT)
{
	double E = 0;
	int nedges = _mesh.nEdges();
	int nfaces = _mesh.nFaces();

	std::vector<double> energyList(nfaces);
	std::vector<Eigen::Matrix<double, 6, 1>> derivList(nfaces);
	std::vector<Eigen::Matrix<double, 6, 6>> hessList(nfaces);

	auto computeEnergy = [&](const tbb::blocked_range<uint32_t>& range) {
		for (uint32_t i = range.begin(); i < range.end(); ++i)
		{
			energyList[i] = curlFreeEnergyPerface(w, i, deriv ? &derivList[i] : NULL, hessT ? &hessList[i] : NULL);
		}
	};

	tbb::blocked_range<uint32_t> rangex(0u, (uint32_t)nfaces, GRAIN_SIZE);
	tbb::parallel_for(rangex, computeEnergy);

	if (deriv)
		deriv->setZero(2 * nedges);
    if(hessT)
        hessT->clear();

	for (int i = 0; i < nfaces; i++)
	{
		E += energyList[i];

		if (deriv)
		{
			for (int j = 0; j < 3; j++)
			{
				int eid = _mesh.faceEdge(i, j);
				(*deriv)(2 * eid) += derivList[i](2 * j);
				(*deriv)(2 * eid + 1) += derivList[i](2 * j + 1);
			}
		}

		if (hessT)
		{
			for (int j = 0; j < 3; j++)
			{
				int eid = _mesh.faceEdge(i, j);
				for (int k = 0; k < 3; k++)
				{
					int eid1 = _mesh.faceEdge(i, k);
					hessT->push_back({ 2 * eid, 2 * eid1, hessList[i](2 * j, 2 * k) });
					hessT->push_back({ 2 * eid, 2 * eid1 + 1, hessList[i](2 * j, 2 * k + 1) });
					hessT->push_back({ 2 * eid + 1, 2 * eid1, hessList[i](2 * j + 1, 2 * k) });
					hessT->push_back({ 2 * eid + 1, 2 * eid1 + 1, hessList[i](2 * j + 1, 2 * k + 1) });
				}
			}
		}
	}

	return E;
}

void WrinkleGluingProcess::computeCombinedRefOmegaList(const std::vector<std::vector<Eigen::MatrixXd>>& refOmegaList)
{
	int nedges = _mesh.nEdges();
	int nfaces = _mesh.nFaces();
	int nFrames = refOmegaList.size();

    _combinedRefOmegaList.resize(nFrames);

	std::vector<Eigen::Triplet<double>> T;
	// projection matrix
	std::vector<int> freeEid;
	for (int i = 0; i < nedges; i++)
	{
		if (_edgeFlag(i) == -1)
		{
			freeEid.push_back(i);
		}
	}

	for (int i = 0; i < freeEid.size(); i++)
	{
		T.push_back(Eigen::Triplet<double>(2 * i, 2 * freeEid[i], 1.0));
		T.push_back(Eigen::Triplet<double>(2 * i + 1, 2 * freeEid[i] + 1, 1.0));
	}

	Eigen::SparseMatrix<double> projM(2 * freeEid.size(), 2 * nedges);
	projM.setFromTriplets(T.begin(), T.end());

	Eigen::SparseMatrix<double> unProjM = projM.transpose();

	auto projVar = [&](const int frameId)
	{
		Eigen::MatrixXd fullX = Eigen::VectorXd::Zero(nedges * 2);
		for (int i = 0; i < nedges; i++)
		{
			if (_edgeFlag(i) != -1)
			{
				fullX(2 * i) = refOmegaList[frameId][_edgeFlag(i)](i, 0);
				fullX(2 * i + 1) = refOmegaList[frameId][_edgeFlag(i)](i, 1);
			}
		}
		Eigen::VectorXd x0 = projM * fullX;
		return x0;
	};

	auto unProjVar = [&](const Eigen::VectorXd& x, const int frameId)
	{
		Eigen::VectorXd fullX = unProjM * x;

		Eigen::MatrixXd w(nedges, 2);

		for (int i = 0; i < nedges; i++)
		{
			if (_edgeFlag(i) != -1)
			{
				fullX(2 * i) = refOmegaList[frameId][_edgeFlag(i)](i, 0);
				fullX(2 * i + 1) = refOmegaList[frameId][_edgeFlag(i)](i, 1);
			}
			w(i, 0) = fullX(2 * i);
			w(i, 1) = fullX(2 * i + 1);
		}
		return w;
	};
    auto mat2vec = [&](const Eigen::MatrixXd& w)
    {
        Eigen::VectorXd x(2 * w.rows());
        for (int i = 0; i < w.rows(); i++)
        {
            x(2 * i) = w(i, 0);
            x(2 * i + 1) = w(i, 1);
        }
        return x;
    };

    Eigen::MatrixXd prevw(nedges, 2);
    prevw.setZero();
    for (int i = 0; i < nedges; i++)
    {
        if (_edgeFlag(i) != -1)
        {
            prevw(i, 0) = refOmegaList[0][_edgeFlag(i)](i, 0);
            prevw(2 * i + 1) = refOmegaList[0][_edgeFlag(i)](i, 1);
        }
    }
    Eigen::VectorXd prefullx = mat2vec(prevw);



	for (int k = 0; k < nFrames; k++)
	{
		std::cout << "Frame " << std::to_string(k) << ": ";
		auto funVal = [&](const Eigen::VectorXd& x, Eigen::VectorXd* grad, Eigen::SparseMatrix<double>* hess, bool isProj) {
			Eigen::VectorXd deriv;
            std::vector<Eigen::Triplet<double>> T;
			Eigen::SparseMatrix<double> H;
			Eigen::MatrixXd w = unProjVar(x, k);

			double E = curlFreeEnergy(w, grad ? &deriv : NULL, hess ? &T : NULL);
            H.resize(2 * w.rows(), 2 * w.rows());
            H.setFromTriplets(T.begin(), T.end());

            // we need some reg to remove the singularity, where we choose some kinetic energy (||w - prevw||^2), which coeff = 1e-3
            double c = std::min(1.0 / (nFrames * nFrames), 1e-3);
            E += c / 2.0 * (w - prevw).squaredNorm();

			if (grad)
			{
                Eigen::VectorXd fullx = mat2vec(w);
				(*grad) = projM * (deriv + c * (fullx - prefullx));
			}

			if (hess)
			{
                Eigen::SparseMatrix<double> idMat(2 * w.rows(), 2 * w.rows());
                idMat.setIdentity();
				(*hess) = projM * (H + c * idMat) * unProjM;
			}

			return E;
		};
		auto maxStep = [&](const Eigen::VectorXd& x, const Eigen::VectorXd& dir) {
			return 1.0;
		};
        
		Eigen::VectorXd x0 = projVar(k);
		OptSolver::newtonSolver(funVal, maxStep, x0, 1000, 1e-6, 1e-10, 1e-15, false);

		Eigen::VectorXd deriv;
		double E = funVal(x0, &deriv, NULL, false);
		std::cout << "terminated with energy : " << E << ", gradient norm : " << deriv.norm() << std::endl << std::endl;

        prevw = unProjVar(x0, k);
        _combinedRefOmegaList[k] = prevw;
        prefullx = mat2vec(prevw);
	}
}


////////////////////////////////////////////// test functions ///////////////////////////////////////////////////////////////////////////
void WrinkleGluingProcess::testCurlFreeEnergy(const Eigen::MatrixXd &w)
{
    Eigen::VectorXd deriv;
    std::vector<Eigen::Triplet<double>> T;
    Eigen::SparseMatrix<double> hess;
    double E = curlFreeEnergy(w, &deriv, &T);
    hess.resize(2 * w.rows(), 2 * w.rows());
    hess.setFromTriplets(T.begin(), T.end());

    std::cout << "tested curl free energy: " << E << ", gradient norm: " << deriv.norm() << std::endl;

    Eigen::VectorXd dir = deriv;
    dir.setRandom();

    Eigen::VectorXd x(2 * w.rows());
    for(int i = 0; i < w.rows(); i++)
    {
        x(2 * i) = w(i, 0);
        x(2 * i + 1) = w(i, 1);
    }
    for(int i = 3; i < 10; i++)
    {
        double eps = std::pow(0.1, i);
        Eigen::VectorXd deriv1;
        Eigen::MatrixXd w1 = w;
        for(int j = 0; j < w.rows(); j++)
        {
            w1(j, 0) += eps * dir(2 * j);
            w1(j, 1) += eps * dir(2 * j + 1);
        }
        double E1 = curlFreeEnergy(w1, &deriv1, NULL);

        std::cout << "\neps: " << eps << std::endl;
        std::cout << "gradient check: " << std::abs((E1 - E) / eps - dir.dot(deriv)) << std::endl;
        std::cout << "hess check: " << ((deriv1 - deriv) / eps - hess * dir).norm() << std::endl;
    }
}

void WrinkleGluingProcess::testCurlFreeEnergyPerface(const Eigen::MatrixXd &w, int faceId)
{
    Eigen::Matrix<double, 6, 1> deriv;
    Eigen::Matrix<double, 6, 6> hess;
    double E = curlFreeEnergyPerface(w, faceId, &deriv, &hess);
    Eigen::Matrix<double, 6, 1> dir = deriv;
    dir.setRandom();

    std::cout << "tested curl free energy for face: " << faceId << ", energy: " << E << ", gradient norm: " << deriv.norm() << std::endl;

    for(int i = 3; i < 10; i++)
    {
        double eps = std::pow(0.1, i);
        Eigen::MatrixXd w1 = w;
        for(int j = 0; j < 3; j++)
        {
            int eid = _mesh.faceEdge(faceId, j);
            w1(eid, 0) += eps * dir(2 * j);
            w1(eid, 1) += eps * dir(2 * j + 1);
        }
        Eigen::Matrix<double, 6, 1> deriv1;
        double E1 = curlFreeEnergyPerface(w1, faceId, &deriv1, NULL);

        std::cout << "\neps: " << eps << std::endl;
        std::cout << "gradient check: " << std::abs((E1 - E) / eps - dir.dot(deriv)) << std::endl;
        std::cout << "hess check: " << ((deriv1 - deriv) / eps - hess * dir).norm() << std::endl;
    }

}

