#include "../../include/IntrinsicFormula/WrinkleGluingProcess.h"
#include "../../include/Optimization/NewtonDescent.h"
#include <igl/cotmatrix_entries.h>
#include <igl/cotmatrix.h>
#include <igl/doublearea.h>
#include <igl/boundary_loop.h>
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

	_vertArea.setZero(nverts);
	buildVertexNeighboringInfo(_mesh, _pos.rows(), _vertNeiEdges, _vertNeiFaces);

	_edgeCotCoeffs.setZero(nedges);

	_effectiveVids.clear();
	_effectiveEids.clear();
	_effectiveVids.clear();

	std::vector<int> bnds;
	igl::boundary_loop(_mesh.faces(), bnds);

	std::set<int> edgeset;
	std::set<int> vertset;

	for (int i = 0; i < nfaces; i++)
	{
		for(int j = 0; j < 3; j++)
		{
			int vid = _mesh.faceVertex(i, j);
			int eid = _mesh.faceEdge(i, j);

			_vertArea(vid) += _faceArea(i) / _vertNeiFaces.size() / 3.0;
			_edgeCotCoeffs(eid) += _cotMatrixEntries(i, j);

			if (faceFlag(i) != -1)
			{
				_vertFlag(vid) = faceFlag(i);
				_edgeFlag(eid) = faceFlag(i);
			}
			else
			{
				_effectiveFids.push_back(i);
//                if(std::find(bnds.begin(), bnds.end(), vid) == bnds.end() && vertset.count(vid) == 0)     // not on boundary
				if(vertset.count(vid) == 0)
					 vertset.insert(vid);
				if(edgeset.count(eid) == 0)
					edgeset.insert(eid);
			}
		}
	}
	std::copy(vertset.begin(), vertset.end(), std::back_inserter(_effectiveVids));
	std::copy(edgeset.begin(), edgeset.end(), std::back_inserter(_effectiveEids));

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


}

void WrinkleGluingProcess::initialization(const std::vector<std::vector<Eigen::VectorXd>>& refAmpList, std::vector<std::vector<Eigen::MatrixXd>>& refOmegaList)
{
	computeCombinedRefOmegaList(refOmegaList);
	computeCombinedRefAmpList(refAmpList, &_combinedRefOmegaList);

	std::vector<std::complex<double>> initZvals;
	std::vector<std::complex<double>> tarZvals;

	int nFrames = _combinedRefAmpList.size() - 2;
	roundVertexZvalsFromHalfEdgeOmegaVertexMag(_mesh, _combinedRefOmegaList[0], _combinedRefAmpList[0], _faceArea, _cotMatrixEntries, _pos.rows(), initZvals);
	roundVertexZvalsFromHalfEdgeOmegaVertexMag(_mesh, _combinedRefOmegaList[nFrames + 1], _combinedRefAmpList[nFrames + 1], _faceArea, _cotMatrixEntries, _pos.rows(), tarZvals);

	_model = IntrinsicKnoppelDrivenFormula(_mesh, _faceArea, _cotMatrixEntries, _combinedRefOmegaList, _combinedRefAmpList, initZvals, tarZvals, _combinedRefOmegaList[0], _combinedRefOmegaList[nFrames + 1], nFrames, 1.0, _quadOrd,true);
}

double WrinkleGluingProcess::amplitudeEnergyWithGivenOmegaPerface(const Eigen::VectorXd &amp, const Eigen::MatrixXd &w,
																  int fid, Eigen::Vector3d *deriv,
																  Eigen::Matrix3d *hess)
{
	double energy = 0;

	double curlSq = curlFreeEnergyPerface(w, fid, NULL, NULL);
	Eigen::Vector3d wSq;
	wSq.setZero();

	/*for(int i = 0; i < 3; i++)
	{
		Eigen::Vector2d dtheta;

		int eidij = _mesh.faceEdge(fid, (i + 2) % 3);
		int eidik = _mesh.faceEdge(fid, (i + 1) % 3);
		int vid = _mesh.faceVertex(fid, i);
	   

		if (vid == _mesh.edgeVertex(eidij, 0))
		{
			dtheta(0) = w(eidij, 0);
		}
		else
		{
			dtheta(0) = w(eidij, 1);
		}

		if (vid == _mesh.edgeVertex(eidik, 0))
		{
			dtheta(1) = w(eidik, 0);
		}
		else
		{
			dtheta(1) = w(eidik, 1);
		}

		wSq(i) = dtheta.dot(_faceVertMetrics[fid][i] * dtheta);
	}*/

	if (deriv)
		deriv->setZero();
	if (hess)
		hess->setZero();

	for(int i = 0; i < 3; i++)
	{
		int vid = _mesh.faceVertex(fid, i);
		energy += 0.5 * amp(vid) * amp(vid) / 3 * (wSq(i) * _faceArea(fid) + curlSq);

		if(deriv)
			(*deriv)(i) += amp(vid) * (wSq(i) * _faceArea(fid) + curlSq) / 3;
		if(hess)
			(*hess)(i, i) += (wSq(i) * _faceArea(fid) + curlSq) / 3;
	}

	return energy;
}

double WrinkleGluingProcess::amplitudeEnergyWithGivenOmega(const Eigen::VectorXd &amp, const Eigen::MatrixXd &w,
														   Eigen::VectorXd *deriv,
														   std::vector<Eigen::Triplet<double>> *hessT)
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

void WrinkleGluingProcess::computeCombinedRefAmpList(const std::vector<std::vector<Eigen::VectorXd>>& refAmpList, std::vector<Eigen::MatrixXd>* combinedOmegaList)
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
				fullX(i) = refAmpList[frameId][_vertFlag(i)](i);
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
				fullX(i) = refAmpList[frameId][_vertFlag(i)](i);
			}
		}
		return fullX;
	};

	Eigen::SparseMatrix<double> L;
	igl::cotmatrix(_pos, _mesh.faces(), L);


	for (int i = 0; i < nFrames; i++)
	{
		std::cout << "Frame " << std::to_string(i) << ": ";
		auto funVal = [&](const Eigen::VectorXd& x, Eigen::VectorXd* grad, Eigen::SparseMatrix<double>* hess, bool isProj) {
			Eigen::VectorXd deriv, deriv1;
			std::vector<Eigen::Triplet<double>> T;
			Eigen::SparseMatrix<double> H;

			Eigen::VectorXd fullx = unProjVar(x, i);
			double E = -0.5 * fullx.dot(L * fullx);

			if (combinedOmegaList)
			{
				E += amplitudeEnergyWithGivenOmega(fullx, (*combinedOmegaList)[i], grad ? &deriv1 : NULL, hess ? &T : NULL);
			}

			if (grad)
			{
				deriv = -L * fullx;
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
					(*hess) = projM * (H - L) * unProjM;
				}
					
				else
					(*hess) = projM * (-L) * unProjM;
				
			}

			return E;
		};
		auto maxStep = [&](const Eigen::VectorXd& x, const Eigen::VectorXd& dir) {
			return 1.0;
		};

		Eigen::VectorXd x0 = projVar(i);
		//OptSolver::testFuncGradHessian(funVal, x0);
		OptSolver::newtonSolver(funVal, maxStep, x0, 1000, 1e-6, 1e-10, 1e-15, false);

		Eigen::VectorXd deriv;
		double E = funVal(x0, &deriv, NULL, false);
		std::cout << "terminated with energy : " << E << ", gradient norm : " << deriv.norm() << std::endl << std::endl;
		_combinedRefAmpList[i] = unProjVar(x0, i);
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
		*deriv = select0 * diff0 + select1 * diff1;
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
	int nEffectiveFaces = _effectiveFids.size();

	std::vector<double> energyList(nEffectiveFaces);
	std::vector<Eigen::Matrix<double, 6, 1>> derivList(nEffectiveFaces);
	std::vector<Eigen::Matrix<double, 6, 6>> hessList(nEffectiveFaces);

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
		deriv->setZero(2 * nedges);
	if(hessT)
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
				(*deriv)(2 * eid) += derivList[efid](2 * j);
				(*deriv)(2 * eid + 1) += derivList[efid](2 * j + 1);
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
					hessT->push_back({ 2 * eid, 2 * eid1, hessList[efid](2 * j, 2 * k) });
					hessT->push_back({ 2 * eid, 2 * eid1 + 1, hessList[efid](2 * j, 2 * k + 1) });
					hessT->push_back({ 2 * eid + 1, 2 * eid1, hessList[efid](2 * j + 1, 2 * k) });
					hessT->push_back({ 2 * eid + 1, 2 * eid1 + 1, hessList[efid](2 * j + 1, 2 * k + 1) });
				}
			}
		}
	}

	return E;
}


double WrinkleGluingProcess::divFreeEnergyPervertex(const Eigen::MatrixXd &w, int vertId, Eigen::VectorXd *deriv,
													Eigen::MatrixXd *hess)
{
	double energy = 0;
	int neiEdges = _vertNeiEdges[vertId].size();


	Eigen::VectorXd selectedVec0, selectedVec1;
	selectedVec0.setZero(2 * neiEdges);
	selectedVec1.setZero(2 * neiEdges);

	Eigen::VectorXd edgew;
	edgew.setZero(2 * neiEdges);

	for(int i = 0; i < neiEdges; i++)
	{
		int eid = _vertNeiEdges[vertId][i];
		if(_mesh.edgeVertex(eid, 0) == vertId)
		{
			selectedVec0(2 * i) = _edgeCotCoeffs(eid);
			selectedVec1(2 * i + 1) = _edgeCotCoeffs(eid);
		}
		else
		{
			selectedVec0(2 * i + 1) = _edgeCotCoeffs(eid);
			selectedVec1(2 * i) = _edgeCotCoeffs(eid);
		}

		edgew(2 * i) = w(eid, 0);
		edgew(2 * i + 1)  = w(eid, 1);
	}
	double diff0 = selectedVec0.dot(edgew);
	double diff1 = selectedVec1.dot(edgew);

	energy = 0.5 * (diff0 * diff0 + diff1 * diff1);
	if(deriv)
	{
		(*deriv) = (diff0 * selectedVec0 + diff1 * selectedVec1);
	}
	if(hess)
	{
		(*hess) = (selectedVec0 * selectedVec0.transpose() + selectedVec1 * selectedVec1.transpose());
	}

	return energy;
}

double WrinkleGluingProcess::divFreeEnergy(const Eigen::MatrixXd &w, Eigen::VectorXd *deriv,
										   std::vector<Eigen::Triplet<double>> *hessT)
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
		deriv->setZero(2 * nedges);
	if(hessT)
		hessT->clear();

	for(int efid = 0; efid < nEffectiveVerts; efid++)
	{
		int vid = _effectiveVids[efid];
		energy += energyList[efid];

		if(deriv)
		{
			 for(int j = 0; j < _vertNeiEdges[vid].size(); j++)
			 {
				 int eid = _vertNeiEdges[vid][j];
				 (*deriv)(2 * eid) += derivList[efid](2 * j);
				 (*deriv)(2 * eid + 1) += derivList[efid](2 * j + 1);
			 }
		}

		if (hessT)
		{
			for(int j = 0; j < _vertNeiEdges[vid].size(); j++)
			{
				int eid = _vertNeiEdges[vid][j];
				for(int k = 0; k < _vertNeiEdges[vid].size(); k++)
				{
					int eid1 = _vertNeiEdges[vid][k];
					hessT->push_back({ 2 * eid, 2 * eid1, hessList[efid](2 * j, 2 * k) });
					hessT->push_back({ 2 * eid, 2 * eid1 + 1, hessList[efid](2 * j, 2 * k + 1) });
					hessT->push_back({ 2 * eid + 1, 2 * eid1, hessList[efid](2 * j + 1, 2 * k) });
					hessT->push_back({ 2 * eid + 1, 2 * eid1 + 1, hessList[efid](2 * j + 1, 2 * k + 1) });
				}
			}
		}
	}

	return energy;
}

void WrinkleGluingProcess::computeCombinedRefOmegaList(const std::vector<std::vector<Eigen::MatrixXd>>& refOmegaList)
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

	/*
	Eigen::VectorXd x0 = projVar(0);
	Eigen::MatrixXd w0 = unProjVar(x0, 0);

	Eigen::VectorXd deriv, deriv1;
	T.clear();
	std::vector<Eigen::Triplet<double>> T1;
	Eigen::SparseMatrix<double> H;

	double E = curlFreeEnergy(w0, &deriv, &T);
	E += divFreeEnergy(w0, &deriv1, &T1);
	std::copy(T1.begin(), T1.end(), std::back_inserter(T));
	deriv += deriv1;

	H.resize(2 * nedges, 2 * nedges);
	H.setFromTriplets(T.begin(), T.end());

	double c = std::min(1.0 / (nFrames * nFrames), 1e-3);
	Eigen::SparseMatrix<double> idMat(2 * nedges, 2 * nedges);
	idMat.setIdentity();

	Eigen::SparseMatrix<double> reducedH = projM * (H + c * idMat) * unProjM;
	Eigen::CholmodSupernodalLLT<Eigen::SparseMatrix<double>> solver;
	solver.compute(reducedH);

	for (int k = 0; k < nFrames; k++)
	{
		std::cout << "Frame " << std::to_string(k) << ": ";
		Eigen::VectorXd fixedVar(2 * nedges);
		for (int i = 0; i < nedges; i++)
		{
			if (_edgeFlag(i) != -1)
			{
				fixedVar(2 * i) = refOmegaList[k][_edgeFlag(i)](i, 0);
				fixedVar(2 * i + 1) = refOmegaList[k][_edgeFlag(i)](i, 1);
			}
		}

		Eigen::VectorXd rhs = -projM * (H * fixedVar - c * prefullx);
		x0 = solver.solve(rhs);
		std::cout << "sol residual: " << (reducedH * x0 - rhs).norm() << std::endl << std::endl;

		prevw = unProjVar(x0, k);
		_combinedRefOmegaList[k] = prevw;
		prefullx = mat2vec(prevw);
	}
	*/

	for (int k = 0; k < nFrames; k++)
	{
		std::cout << "Frame " << std::to_string(k) << ": ";
		auto funVal = [&](const Eigen::VectorXd& x, Eigen::VectorXd* grad, Eigen::SparseMatrix<double>* hess, bool isProj) {
			Eigen::VectorXd deriv, deriv1;
			std::vector<Eigen::Triplet<double>> T, T1;
			Eigen::SparseMatrix<double> H;
			Eigen::MatrixXd w = unProjVar(x, k);

			double E = curlFreeEnergy(w, grad ? &deriv : NULL, hess ? &T : NULL);
			E += divFreeEnergy(w, grad ? &deriv1 : NULL, hess ? &T1 : NULL);
			
			if(grad)
				deriv += deriv1;
			if (hess)
			{
				std::copy(T1.begin(), T1.end(), std::back_inserter(T));
				H.resize(2 * w.rows(), 2 * w.rows());
				H.setFromTriplets(T.begin(), T.end());
			}
			

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

void WrinkleGluingProcess::testDivFreeEnergy(const Eigen::MatrixXd &w)
{
	Eigen::VectorXd deriv;
	std::vector<Eigen::Triplet<double>> T;
	Eigen::SparseMatrix<double> hess;
	double E = divFreeEnergy(w, &deriv, &T);
	hess.resize(2 * w.rows(), 2 * w.rows());
	hess.setFromTriplets(T.begin(), T.end());

	std::cout << "tested div free energy: " << E << ", gradient norm: " << deriv.norm() << std::endl;

	Eigen::VectorXd dir = deriv;
	dir.setRandom();

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
		double E1 = divFreeEnergy(w1, &deriv1, NULL);

		std::cout << "\neps: " << eps << std::endl;
		std::cout << "gradient check: " << std::abs((E1 - E) / eps - dir.dot(deriv)) << std::endl;
		std::cout << "hess check: " << ((deriv1 - deriv) / eps - hess * dir).norm() << std::endl;
	}
}

void WrinkleGluingProcess::testDivFreeEnergyPervertex(const Eigen::MatrixXd &w, int vertId)
{
	Eigen::VectorXd deriv;
	Eigen::MatrixXd hess;
	double E = divFreeEnergyPervertex(w, vertId, &deriv, &hess);
	Eigen::VectorXd dir = deriv;
	dir.setRandom();

	std::cout << "tested div free energy for vertex: " << vertId << ", energy: " << E << ", gradient norm: " << deriv.norm() << std::endl;

	for(int i = 3; i < 10; i++)
	{
		double eps = std::pow(0.1, i);
		Eigen::MatrixXd w1 = w;
		for(int j = 0; j < _vertNeiEdges[vertId].size(); j++)
		{
			int eid = _vertNeiEdges[vertId][j];
			w1(eid, 0) += eps * dir(2 * j);
			w1(eid, 1) += eps * dir(2 * j + 1);
		}
		Eigen::VectorXd deriv1;
		double E1 = divFreeEnergyPervertex(w1, vertId, &deriv1, NULL);

		std::cout << "\neps: " << eps << std::endl;
		std::cout << "gradient check: " << std::abs((E1 - E) / eps - dir.dot(deriv)) << std::endl;
		std::cout << "hess check: " << ((deriv1 - deriv) / eps - hess * dir).norm() << std::endl;
	}

}


void WrinkleGluingProcess::testAmpEnergyWithGivenOmega(const Eigen::VectorXd& amp, const Eigen::MatrixXd& w)
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

void WrinkleGluingProcess::testAmpEnergyWithGivenOmegaPerface(const Eigen::VectorXd& amp, const Eigen::MatrixXd& w, int faceId)
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