#include "../../include/IntrinsicFormula/WrinkleEditingStaticEdgeModel.h"
#include "../../include/Optimization/NewtonDescent.h"
#include "../../include/IntrinsicFormula/KnoppelStripePatternEdgeOmega.h"
#include "../../include/WrinkleFieldsEditor.h"
#include <igl/cotmatrix_entries.h>
#include <igl/cotmatrix.h>
#include <igl/doublearea.h>
#include <igl/boundary_loop.h>
#include <Eigen/SPQRSupport>

using namespace IntrinsicFormula;

WrinkleEditingStaticEdgeModel::WrinkleEditingStaticEdgeModel(const Eigen::MatrixXd& pos, const MeshConnectivity& mesh, const std::vector<VertexOpInfo>& vertexOpts, const Eigen::VectorXi& faceFlag, int quadOrd, double spatialRatio)
{
	_pos = pos;
	_mesh = mesh;
	_faceFlag = faceFlag;
	igl::cotmatrix_entries(pos, mesh.faces(), _cotMatrixEntries);
	igl::doublearea(pos, mesh.faces(), _faceArea);
	_faceArea /= 2.0;
	_quadOrd = quadOrd;
	_spatialRatio = spatialRatio;
	_vertexOpts = vertexOpts;

	int nverts = pos.rows();
	int nfaces = mesh.nFaces();
	int nedges = mesh.nEdges();

	_vertFlag.resize(nverts);
	_vertFlag.setConstant(-1);

	_edgeFlag.resize(nedges);
	_edgeFlag.setConstant(-1);

	_vertArea.setZero(nverts);
	_edgeArea.setZero(nedges);
	
	buildVertexNeighboringInfo(_mesh, _pos.rows(), _vertNeiEdges, _vertNeiFaces);

	_edgeCotCoeffs.setZero(nedges);

	_effectiveVids.clear();
	_effectiveEids.clear();
	_effectiveVids.clear();

	std::vector<int> bnds;
	igl::boundary_loop(_mesh.faces(), bnds);

	std::set<int> edgeset;
	std::set<int> vertset;
	std::set<int> faceset;
	_nInterfaces = 0;

	for (int i = 0; i < nfaces; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			int vid = _mesh.faceVertex(i, j);
			int eid = _mesh.faceEdge(i, j);

			_vertArea(vid) += _faceArea(i) / 3.0;
			

			if (_mesh.edgeFace(eid, 0) == -1 || _mesh.edgeFace(eid, 1) == -1)
			{
				_edgeArea(eid) += _faceArea(i);
			}
			else
				_edgeArea(eid) += _faceArea(i) / 2;

			_edgeCotCoeffs(eid) += _cotMatrixEntries(i, j);

			if (faceFlag(i) != -1)
			{
				_vertFlag(vid) = faceFlag(i);
				_edgeFlag(eid) = faceFlag(i);
			}
			else
			{
				if (faceset.count(i) == 0)
				{
					faceset.insert(i);
					_nInterfaces++;
				}
				
				if (vertset.count(vid) == 0)
					vertset.insert(vid);
				if (edgeset.count(eid) == 0)
					edgeset.insert(eid);
			}
		}
	}
	std::copy(vertset.begin(), vertset.end(), std::back_inserter(_effectiveVids));
	std::copy(edgeset.begin(), edgeset.end(), std::back_inserter(_effectiveEids));
	std::copy(faceset.begin(), faceset.end(), std::back_inserter(_effectiveFids));

	_faceVertMetrics.resize(nfaces);
	for (int i = 0; i < nfaces; i++)
	{
		_faceVertMetrics[i].resize(3);
		for (int j = 0; j < 3; j++)
		{
			int vid = _mesh.faceVertex(i, j);
			int vidj = _mesh.faceVertex(i, (j + 1) % 3);
			int vidk = _mesh.faceVertex(i, (j + 2) % 3);

			Eigen::Vector3d e0 = _pos.row(vidj) - _pos.row(vid);
			Eigen::Vector3d e1 = _pos.row(vidk) - _pos.row(vid);

			Eigen::Matrix2d I;
			I << e0.dot(e0), e0.dot(e1), e1.dot(e0), e1.dot(e1);
			_faceVertMetrics[i][j] = I.inverse();
		}
	}
	std::cout << "number of interfaces: " << _nInterfaces << std::endl;
	std::cout << "min edge area: " << _edgeArea.minCoeff() << ", min vertex area: " << _vertArea.minCoeff() << std::endl;
	std::cout << "max edge area: " << _edgeArea.maxCoeff() << ", max vertex area: " << _vertArea.maxCoeff() << std::endl;

}

void WrinkleEditingStaticEdgeModel::initialization(const Eigen::VectorXd &initAmp, const Eigen::VectorXd &initOmega,
                                                   double numFrames)
{
	
	std::vector<std::complex<double>> initZvals;
    roundZvalsFromEdgeOmegaVertexMag(_mesh, initOmega, initAmp, _faceArea, _cotMatrixEntries, _pos.rows(), initZvals);


    std::vector<Eigen::VectorXd> refOmegaList(numFrames + 2);
    std::vector<Eigen::VectorXd> refAmpList(numFrames + 2);

    refAmpList[0] = initAmp;
    refOmegaList[0]= initOmega;

    double dt = 1.0 / (numFrames + 1);
    for(int i = 1; i <= numFrames + 1; i++)
    {
        std::vector<VertexOpInfo> curVertOpts = _vertexOpts;
        for(int j = 0; j < _vertexOpts.size(); j++)
        {
            if(_vertexOpts[j].vecOptType == None)
                continue;
            double offset = _vertexOpts[j].vecOptType != Enlarge ? 0 : 1;
            double A = _vertexOpts[j].vecOptType != Enlarge ? _vertexOpts[j].vecOptValue : _vertexOpts[j].vecOptValue - 1;

            curVertOpts[j].vecOptValue = offset + A * dt * i;
            curVertOpts[j].vecMagValue = 1 + (_vertexOpts[j].vecMagValue - 1) * dt * i;
            
        }

        WrinkleFieldsEditor::edgeBasedWrinkleEdition(_pos, _mesh, initAmp, initOmega, curVertOpts, refAmpList[i], refOmegaList[i]);
    }

	std::vector<std::complex<double>> tarZvals;

	Eigen::VectorXi bndVertsFlag = _vertFlag;
	for (int i = 0; i < bndVertsFlag.rows(); i++)
	{
		if (_vertFlag(i) != -1)
			bndVertsFlag(i) = 1;
		else
			bndVertsFlag(i) = 0;
	}

	Eigen::VectorXi firstStepFlags = bndVertsFlag;

	for (int i = 0; i < _vertexOpts.size(); i++)
	{
		if(_vertexOpts[i].vecOptType != None)
			firstStepFlags(i) = 0;
	}

	std::cout << "initialize bnd zvals." << std::endl;

	if (!_nInterfaces)
	{
		_combinedRefOmegaList = refOmegaList;
		_combinedRefAmpList = refAmpList;

        roundZvalsFromEdgeOmegaVertexMag(_mesh, refOmegaList[numFrames + 1], refAmpList[numFrames + 1], _faceArea, _cotMatrixEntries, _pos.rows(), tarZvals);
	}
	else
	{
		std::cout << "compute reference omega." << std::endl;
		computeCombinedRefOmegaList(refOmegaList);
		std::cout << "compute reference amplitude." << std::endl;
		computeCombinedRefAmpList(refAmpList, &_combinedRefOmegaList);

		tarZvals = initZvals;
		roundZvalsForSpecificDomainFromEdgeOmegaBndValues(_pos, _mesh, _combinedRefOmegaList[numFrames + 1], firstStepFlags, _faceArea, _cotMatrixEntries, _pos.rows(), tarZvals);

		for (int i = 0; i < tarZvals.size(); i++)
		{
			if (firstStepFlags[i] == 0)
			{
				double arg = std::arg(tarZvals[i]);
				tarZvals[i] = refAmpList[numFrames + 1][i] * std::complex<double>(std::cos(arg), std::sin(arg));
			}

		}

		roundZvalsForSpecificDomainFromEdgeOmegaBndValues(_pos, _mesh, _combinedRefOmegaList[numFrames + 1], bndVertsFlag, _faceArea, _cotMatrixEntries, _pos.rows(), tarZvals);

        for (int i = 0; i < tarZvals.size(); i++)
        {
            if (bndVertsFlag[i] == 0)
            {
                double arg = std::arg(tarZvals[i]);
                tarZvals[i] = refAmpList[numFrames + 1][i] * std::complex<double>(std::cos(arg), std::sin(arg));
            }

        }
	}



	_edgeOmegaList = _combinedRefOmegaList;
	_zvalsList.resize(numFrames + 2);

	_zvalsList[0] = initZvals;
	_zvalsList[numFrames + 1] = tarZvals;

	std::cout << "initialize the intermediate frames." << std::endl;
	for (int i = 1; i <= numFrames; i++)
	{
		 double t = i * dt;

		 _zvalsList[i] = tarZvals;

		 for(int j = 0; j < tarZvals.size(); j++)
		 {
			 _zvalsList[i][j] = (1 - t) * initZvals[j] + t * tarZvals[j];
		 }
	}

	_zdotModel = ComputeZdotFromEdgeOmega(_mesh, _faceArea, _quadOrd, dt);
}

void WrinkleEditingStaticEdgeModel::warmstart()
{
	int nverts = _zvalsList[0].size();
	int nedges = _edgeOmegaList[0].rows();

	int numFrames = _zvalsList.size() - 2;

	int DOFsPerframe = (2 * nverts);
	int DOFs = numFrames * DOFsPerframe;

	auto convertVec2ZList = [&](const Eigen::VectorXd & x)
	{
		for (int i = 0; i < numFrames; i++)
		{
			for (int j = 0; j < nverts; j++)
			{
				_zvalsList[i + 1][j] = std::complex<double>(x(i * DOFsPerframe + 2 * j), x(i * DOFsPerframe + 2 * j + 1));
			}
		}
	};

	auto convertZList2Vec = [&](Eigen::VectorXd& x)
	{
		x.setZero(DOFs);

		for (int i = 0; i < numFrames; i++)
		{
			for (int j = 0; j < nverts; j++)
			{
				x(i * DOFsPerframe + 2 * j) = _zvalsList[i + 1][j].real();
				x(i * DOFsPerframe + 2 * j + 1) = _zvalsList[i + 1][j].imag();
			}
		}
	};

	double dt = 1.0 / (numFrames + 1);

	auto funVal = [&](const Eigen::VectorXd& x, Eigen::VectorXd* grad, Eigen::SparseMatrix<double>* hess, bool isProj)
	{
		convertVec2ZList(x);
		double energy = 0;
		if (grad)
		{
			grad->setZero(DOFs);
		}

		std::vector<Eigen::Triplet<double>> T;
		
		for (int i = 0; i < _zvalsList.size() - 1; i++)
		{
			for (int j = 0; j < nverts; j++)
			{
				Eigen::Vector2d diff;
				double coeff = 1. / (dt * dt) * _vertArea[j];
				diff << (_zvalsList[i + 1][j] - _zvalsList[i][j]).real(), (_zvalsList[i + 1][j] - _zvalsList[i][j]).imag();
				energy += 0.5 * coeff * diff.squaredNorm();

				if (grad)
				{
					if (i == 0)
					{
						grad->segment<2>(i * DOFsPerframe + 2 * j) += coeff * diff;
					}
					else if (i == _zvalsList.size() - 2)
						grad->segment<2>((i - 1) * DOFsPerframe + 2 * j) += -coeff * diff;
					else
					{
						grad->segment<2>((i - 1) * DOFsPerframe + 2 * j) += -coeff * diff;
						grad->segment<2>(i * DOFsPerframe + 2 * j) += coeff * diff;
					}
				}

				if (hess)
				{
					if (i == 0)
					{
						T.push_back({ 2 * j, 2 * j, coeff });
						T.push_back({ 2 * j + 1, 2 * j + 1, coeff });
					}
					else if (i == _zvalsList.size() - 2)
					{
						T.push_back({ (i - 1) * DOFsPerframe + 2 * j, (i - 1) * DOFsPerframe + 2 * j, coeff });
						T.push_back({ (i - 1) * DOFsPerframe + 2 * j + 1, (i - 1) * DOFsPerframe + 2 * j + 1, coeff });
					}
					else
					{
						T.push_back({ (i - 1) * DOFsPerframe + 2 * j, (i - 1) * DOFsPerframe + 2 * j, coeff });
						T.push_back({ i * DOFsPerframe + 2 * j, i * DOFsPerframe + 2 * j, coeff });

						T.push_back({ i * DOFsPerframe + 2 * j, (i - 1) * DOFsPerframe + 2 * j, -coeff });
						T.push_back({ (i - 1) * DOFsPerframe + 2 * j, i * DOFsPerframe + 2 * j, -coeff });

						T.push_back({ (i - 1) * DOFsPerframe + 2 * j + 1, (i - 1) * DOFsPerframe + 2 * j + 1, coeff });
						T.push_back({ i * DOFsPerframe + 2 * j + 1, i * DOFsPerframe + 2 * j + 1, coeff });

						T.push_back({ (i - 1) * DOFsPerframe + 2 * j + 1, i * DOFsPerframe + 2 * j + 1, -coeff });
						T.push_back({ i * DOFsPerframe + 2 * j + 1, (i - 1) * DOFsPerframe + 2 * j + 1, -coeff });
					}

				}
			}
		}
		
		for (int i = 0; i < numFrames; i++)
		{
			int id = i + 1;
			
			double aveAmp = 0;
			for (int j = 0; j < nverts; j++)
			{
				aveAmp += _combinedRefAmpList[id][j] / nverts;
			}

			
			for (int j = 0; j < nverts; j++) {
				double ampSq = _zvalsList[id][j].real() * _zvalsList[id][j].real() +
					_zvalsList[id][j].imag() * _zvalsList[id][j].imag();
				double refAmpSq = _combinedRefAmpList[id][j] * _combinedRefAmpList[id][j];

				energy += _spatialRatio * (ampSq - refAmpSq) * (ampSq - refAmpSq) / (aveAmp * aveAmp);

				if (grad) 
				{
					(*grad)(i * DOFsPerframe + 2 * j) += 2.0 * _spatialRatio / (aveAmp * aveAmp) * (ampSq - refAmpSq) *
						(2.0 * _zvalsList[id][j].real());
					(*grad)(i * DOFsPerframe + 2 * j + 1) += 2.0 * _spatialRatio  / (aveAmp * aveAmp) * (ampSq - refAmpSq) *
						(2.0 * _zvalsList[id][j].imag());
				}

				if (hess) 
				{
					Eigen::Matrix2d tmpHess;
					tmpHess << 2.0 * _zvalsList[id][j].real() * 2.0 * _zvalsList[id][j].real(), 2.0 * _zvalsList[id][j].real() * 2.0 * _zvalsList[id][j].imag(),
						2.0 * _zvalsList[id][j].real() * 2.0 * _zvalsList[id][j].imag(), 2.0 * _zvalsList[id][j].imag() * 2.0 * _zvalsList[id][j].imag();

					tmpHess *= 2.0 * _spatialRatio / (aveAmp * aveAmp);
					tmpHess += 2.0 * _spatialRatio / (aveAmp * aveAmp) * (ampSq - refAmpSq) * (2.0 * Eigen::Matrix2d::Identity());

					if (isProj)
						tmpHess = SPDProjection(tmpHess);

					for (int k = 0; k < 2; k++)
						for (int l = 0; l < 2; l++)
							T.push_back({ i * DOFsPerframe + 2 * j + k, i * DOFsPerframe + 2 * j + l, tmpHess(k, l) });

				}
			}
			

			//std::cout << energy << std::endl;

			// knoppel part
			
			Eigen::VectorXd kDeriv;
			std::vector<Eigen::Triplet<double>> kT;

			Eigen::VectorXd edgeWeight(nedges);
			edgeWeight.setOnes();

			double knoppel = IntrinsicFormula::KnoppelEdgeEnergyGivenMag(_mesh, _combinedRefOmegaList[id], _combinedRefAmpList[id] / aveAmp, edgeWeight, _zvalsList[id], grad ? &kDeriv : NULL, hess ? &kT : NULL);
			energy += _spatialRatio * knoppel;

			if (grad) {
				grad->segment(i * DOFsPerframe, kDeriv.rows()) += _spatialRatio * kDeriv;
			}

			if (hess) {
				for (auto& it : kT) {
					T.push_back({ i * DOFsPerframe + it.row(), i * DOFsPerframe + it.col(), _spatialRatio * it.value() });
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

	};
	auto maxStep = [&](const Eigen::VectorXd& x, const Eigen::VectorXd& dir) {
		return 1.0;
	};

	auto postProcess = [&](Eigen::VectorXd& x)
	{
		//            interpModel.postProcess(x);
	};

	Eigen::VectorXd x;
	convertZList2Vec(x);
	//OptSolver::testFuncGradHessian(funVal, x);
	OptSolver::newtonSolver(funVal, maxStep, x, 1000, 1e-6, 0, 0, true);
	convertVec2ZList(x);
	
}

void WrinkleEditingStaticEdgeModel::convertList2Variable(Eigen::VectorXd& x)
{
	int nverts = _zvalsList[0].size();
	int nedges = _edgeOmegaList[0].rows();

	int numFrames = _zvalsList.size() - 2;

	int DOFsPerframe = (2 * nverts + nedges);

	int DOFs = numFrames * DOFsPerframe;

	x.setZero(DOFs);

	for (int i = 0; i < numFrames; i++)
	{
		for (int j = 0; j < nverts; j++)
		{
			x(i * DOFsPerframe + 2 * j) = _zvalsList[i + 1][j].real();
			x(i * DOFsPerframe + 2 * j + 1) = _zvalsList[i + 1][j].imag();
		}

		for (int j = 0; j < nedges; j++)
		{
			x(i * DOFsPerframe + 2 * nverts + j) = _edgeOmegaList[i + 1](j);
		}
	}
}

void WrinkleEditingStaticEdgeModel::convertVariable2List(const Eigen::VectorXd& x)
{
	int nverts = _zvalsList[0].size();
	int nedges = _edgeOmegaList[0].rows();

	int numFrames = _zvalsList.size() - 2;

	int DOFsPerframe = (2 * nverts + nedges);

	for (int i = 0; i < numFrames; i++)
	{
		for (int j = 0; j < nverts; j++)
		{
			_zvalsList[i + 1][j] = std::complex<double>(x(i * DOFsPerframe + 2 * j), x(i * DOFsPerframe + 2 * j + 1));
		}

		for (int j = 0; j < nedges; j++)
		{
			_edgeOmegaList[i + 1](j) = x(i * DOFsPerframe + 2 * nverts + j);
		}
	}
}

double WrinkleEditingStaticEdgeModel::amplitudeEnergyWithGivenOmegaPerface(const Eigen::VectorXd& amp, const Eigen::VectorXd& w,
	int fid, Eigen::Vector3d* deriv,
	Eigen::Matrix3d* hess)
{
	double energy = 0;

	double curlSq = curlFreeEnergyPerface(w, fid, NULL, NULL);
	Eigen::Vector3d wSq;
	wSq.setZero();


	if (deriv)
		deriv->setZero();
	if (hess)
		hess->setZero();

	for (int i = 0; i < 3; i++)
	{
		int vid = _mesh.faceVertex(fid, i);
		energy += 0.5 * amp(vid) * amp(vid) / 3 * (wSq(i) * _faceArea(fid) + curlSq);

		if (deriv)
			(*deriv)(i) += amp(vid) * (wSq(i) * _faceArea(fid) + curlSq) / 3;
		if (hess)
			(*hess)(i, i) += (wSq(i) * _faceArea(fid) + curlSq) / 3;
	}

	return energy;
}

double WrinkleEditingStaticEdgeModel::amplitudeEnergyWithGivenOmega(const Eigen::VectorXd& amp, const Eigen::VectorXd& w,
	Eigen::VectorXd* deriv,
	std::vector<Eigen::Triplet<double>>* hessT)
{
	double energy = 0;

	int nverts = _pos.rows();
	int nEffectiveFaces = _effectiveFids.size();

	std::vector<double> energyList(nEffectiveFaces);
	std::vector<Eigen::Vector3d> derivList(nEffectiveFaces);
	std::vector<Eigen::Matrix3d> hessList(nEffectiveFaces);

	auto computeEnergy = [&](const tbb::blocked_range<uint32_t>& range) {
		for (uint32_t i = range.begin(); i < range.end(); ++i)
		{
			int fid = _effectiveFids[i];
			energyList[i] = amplitudeEnergyWithGivenOmegaPerface(amp, w, fid, deriv ? &derivList[i] : NULL, hessT ? &hessList[i] : NULL);
		}
	};

	tbb::blocked_range<uint32_t> rangex(0u, (uint32_t)nEffectiveFaces, GRAIN_SIZE);
	tbb::parallel_for(rangex, computeEnergy);

	if (deriv)
		deriv->setZero(nverts);
	if (hessT)
		hessT->clear();

	for (int efid = 0; efid < nEffectiveFaces; efid++)
	{
		energy += energyList[efid];
		int fid = _effectiveFids[efid];

		if (deriv)
		{
			for (int j = 0; j < 3; j++)
			{
				int vid = _mesh.faceVertex(fid, j);
				(*deriv)(vid) += derivList[efid](j);
			}
		}

		if (hessT)
		{
			for (int j = 0; j < 3; j++)
			{
				int vid = _mesh.faceVertex(fid, j);
				for (int k = 0; k < 3; k++)
				{
					int vid1 = _mesh.faceVertex(fid, k);
					hessT->push_back({ vid, vid1, hessList[efid](j, k) });
				}
			}
		}
	}
	return energy;
}

void WrinkleEditingStaticEdgeModel::computeCombinedRefAmpList(const std::vector<Eigen::VectorXd>& refAmpList, std::vector<Eigen::VectorXd>* combinedOmegaList)
{
	int nverts = _pos.rows();
	int nfaces = _mesh.nFaces();
	int nFrames = refAmpList.size();

	_combinedRefAmpList.resize(nFrames);

	double c = std::min(1.0 / (nFrames * nFrames), 1e-3);

	std::vector<Eigen::Triplet<double>> T;
	// projection matrix
	std::vector<int> freeVid;
	for (int i = 0; i < nverts; i++)
	{
		if (_vertFlag(i) == -1)
		{
			freeVid.push_back(i);
		}
	}

	for (int i = 0; i < freeVid.size(); i++)
	{
		T.push_back(Eigen::Triplet<double>(i, freeVid[i], 1.0));
	}

	Eigen::SparseMatrix<double> projM(freeVid.size(), nverts);
	projM.setFromTriplets(T.begin(), T.end());

	Eigen::SparseMatrix<double> unProjM = projM.transpose();

	auto projVar = [&](const int frameId)
	{
		Eigen::MatrixXd fullX = Eigen::VectorXd::Zero(nverts);
		for (int i = 0; i < nverts; i++)
		{
			if (_vertFlag(i) != -1)
			{
				fullX(i) = refAmpList[frameId](i);
			}
		}
		Eigen::VectorXd x0 = projM * fullX;
		return x0;
	};

	auto unProjVar = [&](const Eigen::VectorXd& x, const int frameId)
	{
		Eigen::VectorXd fullX = unProjM * x;

		for (int i = 0; i < nverts; i++)
		{
			if (_vertFlag(i) != -1)
			{
				fullX(i) = refAmpList[frameId](i);
			}
		}
		return fullX;
	};

	Eigen::SparseMatrix<double> L;
	igl::cotmatrix(_pos, _mesh.faces(), L);

    _combinedRefAmpList[0] = refAmpList[0];

    Eigen::VectorXd prevX = refAmpList[0];

    Eigen::SparseMatrix<double> idmat(prevX.rows(), prevX.rows());
    idmat.setIdentity();


	for (int i = 1; i < nFrames; i++)
	{
		std::cout << "Frame " << std::to_string(i) << ": free vertices: " << freeVid.size() << std::endl;;
		auto funVal = [&](const Eigen::VectorXd& x, Eigen::VectorXd* grad, Eigen::SparseMatrix<double>* hess, bool isProj) {
			Eigen::VectorXd deriv, deriv1;
			std::vector<Eigen::Triplet<double>> T;
			Eigen::SparseMatrix<double> H;

			Eigen::VectorXd fullx = unProjVar(x, i);
			double E = -0.5 * fullx.dot(L * fullx) + 0.5 * c * (fullx - prevX).squaredNorm();

			if (combinedOmegaList)
			{
				E += amplitudeEnergyWithGivenOmega(fullx, (*combinedOmegaList)[i], grad ? &deriv1 : NULL, hess ? &T : NULL);
			}

			if (grad)
			{
				deriv = -L * fullx + c * (fullx - prevX);
				if (combinedOmegaList)
					deriv += deriv1;
				(*grad) = projM * deriv;
			}

			if (hess)
			{
				if (combinedOmegaList)
				{
					H.resize(fullx.rows(), fullx.rows());
					H.setFromTriplets(T.begin(), T.end());
					(*hess) = projM * (H - L + c * idmat) * unProjM;
				}

				else
					(*hess) = projM * (-L + c * idmat) * unProjM;

			}

			return E;
		};
		auto maxStep = [&](const Eigen::VectorXd& x, const Eigen::VectorXd& dir) {
			return 1.0;
		};

		Eigen::VectorXd x0 = projVar(i);
		if (_nInterfaces && freeVid.size())
		{
			OptSolver::newtonSolver(funVal, maxStep, x0, 1000, 1e-6, 1e-10, 1e-15, false);

			Eigen::VectorXd deriv;
			double E = funVal(x0, &deriv, NULL, false);
			std::cout << "terminated with energy : " << E << ", gradient norm : " << deriv.norm() << std::endl << std::endl;
		}
			
		_combinedRefAmpList[i] = unProjVar(x0, i);
        prevX = _combinedRefAmpList[i];
	}
}

double WrinkleEditingStaticEdgeModel::curlFreeEnergyPerface(const Eigen::MatrixXd& w, int faceId, Eigen::Matrix<double, 3, 1>* deriv, Eigen::Matrix<double, 3, 3>* hess)
{
	double E = 0;

	double diff0;
	Eigen::Matrix<double, 3, 1> select0;
	select0.setZero();

	Eigen::Matrix<double, 3, 1> edgews;

	for (int i = 0; i < 3; i++)
	{
		int eid = _mesh.faceEdge(faceId, i);
		edgews(i) = w(eid);
		
		if (_mesh.faceVertex(faceId, (i + 1) % 3) == _mesh.edgeVertex(eid, 0))
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
	if (deriv)
	{
		*deriv = select0 * diff0;
	}
	if (hess)
	{
		*hess = select0 * select0.transpose();
	}

	return E;
}


double WrinkleEditingStaticEdgeModel::curlFreeEnergy(const Eigen::MatrixXd& w, Eigen::VectorXd* deriv, std::vector<Eigen::Triplet<double>>* hessT)
{
	double E = 0;
	int nedges = _mesh.nEdges();
	int nEffectiveFaces = _effectiveFids.size();

	std::vector<double> energyList(nEffectiveFaces);
	std::vector<Eigen::Matrix<double, 3, 1>> derivList(nEffectiveFaces);
	std::vector<Eigen::Matrix<double, 3, 3>> hessList(nEffectiveFaces);

	auto computeEnergy = [&](const tbb::blocked_range<uint32_t>& range) {
		for (uint32_t i = range.begin(); i < range.end(); ++i)
		{
			int fid = _effectiveFids[i];
			energyList[i] = curlFreeEnergyPerface(w, fid, deriv ? &derivList[i] : NULL, hessT ? &hessList[i] : NULL);
		}
	};

	tbb::blocked_range<uint32_t> rangex(0u, (uint32_t)nEffectiveFaces, GRAIN_SIZE);
	tbb::parallel_for(rangex, computeEnergy);

	if (deriv)
		deriv->setZero(nedges);
	if (hessT)
		hessT->clear();

	for (int efid = 0; efid < nEffectiveFaces; efid++)
	{
		E += energyList[efid];
		int fid = _effectiveFids[efid];

		if (deriv)
		{
			for (int j = 0; j < 3; j++)
			{
				int eid = _mesh.faceEdge(fid, j);
				(*deriv)(eid) += derivList[efid](j);
			}
		}

		if (hessT)
		{
			for (int j = 0; j < 3; j++)
			{
				int eid = _mesh.faceEdge(fid, j);
				for (int k = 0; k < 3; k++)
				{
					int eid1 = _mesh.faceEdge(fid, k);
					hessT->push_back({ eid, eid1, hessList[efid](j, k) });
				}
			}
		}
	}

	return E;
}


double WrinkleEditingStaticEdgeModel::divFreeEnergyPervertex(const Eigen::MatrixXd& w, int vertId, Eigen::VectorXd* deriv,
	Eigen::MatrixXd* hess)
{
	double energy = 0;
	int neiEdges = _vertNeiEdges[vertId].size();


	Eigen::VectorXd selectedVec0;
	selectedVec0.setZero(neiEdges);
	
	Eigen::VectorXd edgew;
	edgew.setZero(neiEdges);

	for (int i = 0; i < neiEdges; i++)
	{
		int eid = _vertNeiEdges[vertId][i];
		if (_mesh.edgeVertex(eid, 0) == vertId)
		{
			selectedVec0(i) = _edgeCotCoeffs(eid);
		}
		else
		{
			selectedVec0(i) = -_edgeCotCoeffs(eid);
		}

		edgew(i) = w(eid);
	}
	double diff0 = selectedVec0.dot(edgew);

	energy = 0.5 * (diff0 * diff0);
	if (deriv)
	{
		(*deriv) = (diff0 * selectedVec0 );
	}
	if (hess)
	{
		(*hess) = (selectedVec0 * selectedVec0.transpose());
	}

	return energy;
}

double WrinkleEditingStaticEdgeModel::divFreeEnergy(const Eigen::MatrixXd& w, Eigen::VectorXd* deriv,
	std::vector<Eigen::Triplet<double>>* hessT)
{
	double energy = 0;
	int nedges = _mesh.nEdges();
	int nEffectiveVerts = _effectiveVids.size();

	std::vector<double> energyList(nEffectiveVerts);
	std::vector<Eigen::VectorXd> derivList(nEffectiveVerts);
	std::vector<Eigen::MatrixXd> hessList(nEffectiveVerts);

	auto computeEnergy = [&](const tbb::blocked_range<uint32_t>& range) {
		for (uint32_t i = range.begin(); i < range.end(); ++i)
		{
			int vid = _effectiveVids[i];
			energyList[i] = divFreeEnergyPervertex(w, vid, deriv ? &derivList[i] : NULL, hessT ? &hessList[i] : NULL);
		}
	};

	tbb::blocked_range<uint32_t> rangex(0u, (uint32_t)nEffectiveVerts, GRAIN_SIZE);
	tbb::parallel_for(rangex, computeEnergy);

	if (deriv)
		deriv->setZero(nedges);
	if (hessT)
		hessT->clear();

	for (int efid = 0; efid < nEffectiveVerts; efid++)
	{
		int vid = _effectiveVids[efid];
		energy += energyList[efid];

		if (deriv)
		{
			for (int j = 0; j < _vertNeiEdges[vid].size(); j++)
			{
				int eid = _vertNeiEdges[vid][j];
				(*deriv)(eid) += derivList[efid](j);
			}
		}

		if (hessT)
		{
			for (int j = 0; j < _vertNeiEdges[vid].size(); j++)
			{
				int eid = _vertNeiEdges[vid][j];
				for (int k = 0; k < _vertNeiEdges[vid].size(); k++)
				{
					int eid1 = _vertNeiEdges[vid][k];
					hessT->push_back({ eid, eid1, hessList[efid](j, k) });
				}
			}
		}
	}

	return energy;
}

void WrinkleEditingStaticEdgeModel::computeCombinedRefOmegaList(const std::vector<Eigen::VectorXd>& refOmegaList)
{
	int nedges = _mesh.nEdges();
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
		T.push_back(Eigen::Triplet<double>(i, freeEid[i], 1.0));
	}

	Eigen::SparseMatrix<double> projM(freeEid.size(), nedges);
	projM.setFromTriplets(T.begin(), T.end());

	Eigen::SparseMatrix<double> unProjM = projM.transpose();

	auto projVar = [&](const int frameId)
	{
		Eigen::MatrixXd fullX = Eigen::VectorXd::Zero(nedges);
		for (int i = 0; i < nedges; i++)
		{
			if (_edgeFlag(i) != -1)
			{
				fullX(i) = refOmegaList[frameId](i);
			}
		}
		Eigen::VectorXd x0 = projM * fullX;
		return x0;
	};

	auto unProjVar = [&](const Eigen::VectorXd& x, const int frameId)
	{
		Eigen::VectorXd fullX = unProjM * x;

		for (int i = 0; i < nedges; i++)
		{
			if (_edgeFlag(i) != -1)
			{
				fullX(i) = refOmegaList[frameId](i);
			}
			
		}
		return fullX;
	};

	Eigen::VectorXd prevw = refOmegaList[0];
    _combinedRefOmegaList[0] = refOmegaList[0];

	for (int k = 1; k < nFrames; k++)
	{
		std::cout << "Frame " << std::to_string(k) << ": free edges: " << freeEid.size() << std::endl;
		auto funVal = [&](const Eigen::VectorXd& x, Eigen::VectorXd* grad, Eigen::SparseMatrix<double>* hess, bool isProj) {
			Eigen::VectorXd deriv, deriv1;
			std::vector<Eigen::Triplet<double>> T, T1;
			Eigen::SparseMatrix<double> H;
			Eigen::VectorXd w = unProjVar(x, k);

			double E = curlFreeEnergy(w, grad ? &deriv : NULL, hess ? &T : NULL);
			E += divFreeEnergy(w, grad ? &deriv1 : NULL, hess ? &T1 : NULL);

			if (grad)
				deriv += deriv1;
			if (hess)
			{
				std::copy(T1.begin(), T1.end(), std::back_inserter(T));
				H.resize(w.rows(), w.rows());
				H.setFromTriplets(T.begin(), T.end());
			}


			// we need some reg to remove the singularity, where we choose some kinetic energy (||w - prevw||^2), which coeff = 1e-3
			double c = std::min(1.0 / (nFrames * nFrames), 1e-3);
			E += c / 2.0 * (w - prevw).squaredNorm();

			if (grad)
			{
				(*grad) = projM * (deriv + c * (w - prevw));
			}

			if (hess)
			{
				Eigen::SparseMatrix<double> idMat(w.rows(), w.rows());
				idMat.setIdentity();
				(*hess) = projM * (H + c * idMat) * unProjM;
			}

			return E;
		};
		auto maxStep = [&](const Eigen::VectorXd& x, const Eigen::VectorXd& dir) {
			return 1.0;
		};

		Eigen::VectorXd x0 = projVar(k);

		if (_nInterfaces && freeEid.size())
		{
			OptSolver::newtonSolver(funVal, maxStep, x0, 1000, 1e-6, 1e-10, 1e-15, false);

			Eigen::VectorXd deriv;
			double E = funVal(x0, &deriv, NULL, false);
			std::cout << "terminated with energy : " << E << ", gradient norm : " << deriv.norm() << std::endl << std::endl;
		}
			
		prevw = unProjVar(x0, k);
		_combinedRefOmegaList[k] = prevw;
	}
}


double WrinkleEditingStaticEdgeModel::computeEnergy(const Eigen::VectorXd& x, Eigen::VectorXd* deriv, Eigen::SparseMatrix<double>* hess, bool isProj)
{
	int nverts = _zvalsList[0].size();
	int nedges = _edgeOmegaList[0].rows();

	int numFrames = _zvalsList.size() - 2;

	int DOFsPerframe = (2 * nverts + nedges);
	int DOFs = numFrames * DOFsPerframe;

	convertVariable2List(x);

	Eigen::VectorXd curDeriv;
	std::vector<Eigen::Triplet<double>> T, curT;

	double energy = 0;
	if (deriv)
	{
		deriv->setZero(DOFs);
	}

	for (int i = 0; i < _zvalsList.size() - 1; i++)
	{
		energy += _zdotModel.computeZdotIntegration(_zvalsList[i], _edgeOmegaList[i], _zvalsList[i + 1], _edgeOmegaList[i + 1], deriv ? &curDeriv : NULL, hess ? &curT : NULL, isProj);


		if (deriv)
		{

			if (i == 0)
				deriv->segment(0, DOFsPerframe) += curDeriv.segment(DOFsPerframe, DOFsPerframe);
			else if (i == _zvalsList.size() - 2)
				deriv->segment((i - 1) * DOFsPerframe, DOFsPerframe) += curDeriv.segment(0, DOFsPerframe);
			else
			{
				deriv->segment((i - 1) * DOFsPerframe, 2 * DOFsPerframe) += curDeriv;
			}


		}

		if (hess)
		{
			for (auto& it : curT)
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
			curT.clear();
		}
	}

	for (int i = 0; i < numFrames; i++) {
		int id = i + 1;

		// vertex amp diff
		double aveAmp = 0;
		for (int j = 0; j < nverts; j++)
		{
			aveAmp += _combinedRefAmpList[id][j] / nverts;
		}
		for (int j = 0; j < nverts; j++) {
			double ampSq = _zvalsList[id][j].real() * _zvalsList[id][j].real() +
				_zvalsList[id][j].imag() * _zvalsList[id][j].imag();
			double refAmpSq = _combinedRefAmpList[id][j] * _combinedRefAmpList[id][j];

			energy += _spatialRatio * (ampSq - refAmpSq) * (ampSq - refAmpSq) / (aveAmp * aveAmp);

			if (deriv) {
				(*deriv)(i * DOFsPerframe + 2 * j) += 2.0 * _spatialRatio / (aveAmp * aveAmp) * (ampSq - refAmpSq) *
					(2.0 * _zvalsList[id][j].real());
				(*deriv)(i * DOFsPerframe + 2 * j + 1) += 2.0 * _spatialRatio / (aveAmp * aveAmp) * (ampSq - refAmpSq) *
					(2.0 * _zvalsList[id][j].imag());
			}

			if (hess) {
				Eigen::Matrix2d tmpHess;
				tmpHess << 2.0 * _zvalsList[id][j].real() * 2.0 * _zvalsList[id][j].real(), 2.0 * _zvalsList[id][j].real() * 2.0 * _zvalsList[id][j].imag(),
					2.0 * _zvalsList[id][j].real() * 2.0 * _zvalsList[id][j].imag(), 2.0 * _zvalsList[id][j].imag() * 2.0 * _zvalsList[id][j].imag();

				tmpHess *= 2.0 * _spatialRatio / (aveAmp * aveAmp);
				tmpHess += 2.0 * _spatialRatio / (aveAmp * aveAmp) * (ampSq - refAmpSq) * (2.0 * Eigen::Matrix2d::Identity());
				

				if (isProj)
					tmpHess = SPDProjection(tmpHess);

				for (int k = 0; k < 2; k++)
					for (int l = 0; l < 2; l++)
						T.push_back({ i * DOFsPerframe + 2 * j + k, i * DOFsPerframe + 2 * j + l, tmpHess(k, l) });

			}
		}

		// edge omega difference
		for (int j = 0; j < nedges; j++) {
			energy += _spatialRatio * (aveAmp * aveAmp) * (_edgeOmegaList[id](j) - _combinedRefOmegaList[id](j)) * (_edgeOmegaList[id](j) - _combinedRefOmegaList[id](j));

			if (deriv) {
				(*deriv)(i * DOFsPerframe + 2 * nverts + j) += 2 * _spatialRatio * (aveAmp * aveAmp) *
					(_edgeOmegaList[id](j) - _combinedRefOmegaList[id](j));

			}

			if (hess) {
				T.push_back({ i * DOFsPerframe + 2 * nverts + j, i * DOFsPerframe + 2 * nverts + j,
							 2 * _spatialRatio * (aveAmp * aveAmp) });
			}
		}

		// knoppel part
		Eigen::VectorXd kDeriv;
		std::vector<Eigen::Triplet<double>> kT;

		Eigen::VectorXd edgeWeight(nedges);
		edgeWeight.setOnes();

		double knoppel = IntrinsicFormula::KnoppelEdgeEnergyGivenMag(_mesh, _combinedRefOmegaList[id],
			_combinedRefAmpList[id] / aveAmp, edgeWeight,
			_zvalsList[id], deriv ? &kDeriv : NULL,
			hess ? &kT : NULL);
		energy += _spatialRatio * knoppel;

		if (deriv) {
			deriv->segment(i * DOFsPerframe, kDeriv.rows()) += _spatialRatio * kDeriv;
		}

		if (hess) {
			for (auto& it : kT) {
				T.push_back({ i * DOFsPerframe + it.row(), i * DOFsPerframe + it.col(), _spatialRatio * it.value() });
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

////////////////////////////////////////////// test functions ///////////////////////////////////////////////////////////////////////////
void WrinkleEditingStaticEdgeModel::testCurlFreeEnergy(const Eigen::VectorXd& w)
{
	Eigen::VectorXd deriv;
	std::vector<Eigen::Triplet<double>> T;
	Eigen::SparseMatrix<double> hess;
	double E = curlFreeEnergy(w, &deriv, &T);
	hess.resize(w.rows(), w.rows());
	hess.setFromTriplets(T.begin(), T.end());

	std::cout << "tested curl free energy: " << E << ", gradient norm: " << deriv.norm() << std::endl;

	Eigen::VectorXd dir = deriv;
	dir.setRandom();

	for (int i = 3; i < 10; i++)
	{
		double eps = std::pow(0.1, i);
		Eigen::VectorXd deriv1;
		Eigen::VectorXd w1 = w + eps * dir;
		
		double E1 = curlFreeEnergy(w1, &deriv1, NULL);

		std::cout << "\neps: " << eps << std::endl;
		std::cout << "gradient check: " << std::abs((E1 - E) / eps - dir.dot(deriv)) << std::endl;
		std::cout << "hess check: " << ((deriv1 - deriv) / eps - hess * dir).norm() << std::endl;
	}
}

void WrinkleEditingStaticEdgeModel::testCurlFreeEnergyPerface(const Eigen::VectorXd& w, int faceId)
{
	Eigen::Matrix<double, 3, 1> deriv;
	Eigen::Matrix<double, 3, 3> hess;
	double E = curlFreeEnergyPerface(w, faceId, &deriv, &hess);
	Eigen::Matrix<double, 3, 1> dir = deriv;
	dir.setRandom();

	std::cout << "tested curl free energy for face: " << faceId << ", energy: " << E << ", gradient norm: " << deriv.norm() << std::endl;

	for (int i = 3; i < 10; i++)
	{
		double eps = std::pow(0.1, i);
		Eigen::VectorXd w1 = w;
		for (int j = 0; j < 3; j++)
		{
			int eid = _mesh.faceEdge(faceId, j);
			w1(eid) += eps * dir(j);
		}
		Eigen::Matrix<double, 3, 1> deriv1;
		double E1 = curlFreeEnergyPerface(w1, faceId, &deriv1, NULL);

		std::cout << "\neps: " << eps << std::endl;
		std::cout << "gradient check: " << std::abs((E1 - E) / eps - dir.dot(deriv)) << std::endl;
		std::cout << "hess check: " << ((deriv1 - deriv) / eps - hess * dir).norm() << std::endl;
	}

}

void WrinkleEditingStaticEdgeModel::testDivFreeEnergy(const Eigen::VectorXd& w)
{
	Eigen::VectorXd deriv;
	std::vector<Eigen::Triplet<double>> T;
	Eigen::SparseMatrix<double> hess;
	double E = divFreeEnergy(w, &deriv, &T);
	hess.resize(w.rows(), w.rows());
	hess.setFromTriplets(T.begin(), T.end());

	std::cout << "tested div free energy: " << E << ", gradient norm: " << deriv.norm() << std::endl;

	Eigen::VectorXd dir = deriv;
	dir.setRandom();

	for (int i = 3; i < 10; i++)
	{
		double eps = std::pow(0.1, i);
		Eigen::VectorXd deriv1;
		Eigen::VectorXd w1 = w + eps * dir;
		
		double E1 = divFreeEnergy(w1, &deriv1, NULL);

		std::cout << "\neps: " << eps << std::endl;
		std::cout << "gradient check: " << std::abs((E1 - E) / eps - dir.dot(deriv)) << std::endl;
		std::cout << "hess check: " << ((deriv1 - deriv) / eps - hess * dir).norm() << std::endl;
	}
}

void WrinkleEditingStaticEdgeModel::testDivFreeEnergyPervertex(const Eigen::VectorXd& w, int vertId)
{
	Eigen::VectorXd deriv;
	Eigen::MatrixXd hess;
	double E = divFreeEnergyPervertex(w, vertId, &deriv, &hess);
	Eigen::VectorXd dir = deriv;
	dir.setRandom();

	std::cout << "tested div free energy for vertex: " << vertId << ", energy: " << E << ", gradient norm: " << deriv.norm() << std::endl;

	for (int i = 3; i < 10; i++)
	{
		double eps = std::pow(0.1, i);
		Eigen::VectorXd w1 = w;
		for (int j = 0; j < _vertNeiEdges[vertId].size(); j++)
		{
			int eid = _vertNeiEdges[vertId][j];
			w1(eid) += eps * dir(j);
		}
		Eigen::VectorXd deriv1;
		double E1 = divFreeEnergyPervertex(w1, vertId, &deriv1, NULL);

		std::cout << "\neps: " << eps << std::endl;
		std::cout << "gradient check: " << std::abs((E1 - E) / eps - dir.dot(deriv)) << std::endl;
		std::cout << "hess check: " << ((deriv1 - deriv) / eps - hess * dir).norm() << std::endl;
	}

}


void WrinkleEditingStaticEdgeModel::testAmpEnergyWithGivenOmega(const Eigen::VectorXd& amp, const Eigen::VectorXd& w)
{
	Eigen::VectorXd deriv;
	std::vector<Eigen::Triplet<double>> T;
	Eigen::SparseMatrix<double> hess;
	double E = amplitudeEnergyWithGivenOmega(amp, w, &deriv, &T);
	hess.resize(amp.rows(), amp.rows());
	hess.setFromTriplets(T.begin(), T.end());

	std::cout << "tested amp energy: " << E << ", gradient norm: " << deriv.norm() << std::endl;

	Eigen::VectorXd dir = deriv;
	dir.setRandom();


	for (int i = 3; i < 10; i++)
	{
		double eps = std::pow(0.1, i);
		Eigen::VectorXd deriv1;
		Eigen::VectorXd amp1 = amp;
		for (int j = 0; j < amp.rows(); j++)
		{
			amp1(j) += eps * dir(j);
		}
		double E1 = amplitudeEnergyWithGivenOmega(amp1, w, &deriv1, NULL);

		std::cout << "\neps: " << eps << std::endl;
		std::cout << "gradient check: " << std::abs((E1 - E) / eps - dir.dot(deriv)) << std::endl;
		std::cout << "hess check: " << ((deriv1 - deriv) / eps - hess * dir).norm() << std::endl;
	}
}

void WrinkleEditingStaticEdgeModel::testAmpEnergyWithGivenOmegaPerface(const Eigen::VectorXd& amp, const Eigen::VectorXd& w, int faceId)
{
	Eigen::Vector3d deriv;
	Eigen::Matrix3d hess;
	double E = amplitudeEnergyWithGivenOmegaPerface(amp, w, faceId, &deriv, &hess);
	Eigen::Vector3d dir = deriv;
	dir.setRandom();

	std::cout << "tested amp energy for face: " << faceId << ", energy: " << E << ", gradient norm: " << deriv.norm() << std::endl;

	for (int i = 3; i < 10; i++)
	{
		double eps = std::pow(0.1, i);
		Eigen::VectorXd amp1 = amp;
		for (int j = 0; j < 3; j++)
		{
			int vid = _mesh.faceVertex(faceId, j);
			amp1(vid) += eps * dir(j);
		}
		Eigen::Vector3d deriv1;
		double E1 = amplitudeEnergyWithGivenOmegaPerface(amp1, w, faceId, &deriv1, NULL);

		std::cout << "\neps: " << eps << std::endl;
		std::cout << "gradient check: " << std::abs((E1 - E) / eps - dir.dot(deriv)) << std::endl;
		std::cout << "hess check: " << ((deriv1 - deriv) / eps - hess * dir).norm() << std::endl;
	}

}

void WrinkleEditingStaticEdgeModel::testEnergy(Eigen::VectorXd x)
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