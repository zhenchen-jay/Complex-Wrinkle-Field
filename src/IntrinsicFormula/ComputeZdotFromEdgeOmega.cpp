#include "../../include/IntrinsicFormula/ComputeZdotFromEdgeOmega.h"


double ComputeZdotFromEdgeOmega::computeZdotIntegrationFromQuad(const std::vector<std::complex<double>>& curZvals, const Eigen::VectorXd& curw, const std::vector<std::complex<double>>& nextZvals, const Eigen::VectorXd& nextw, int fid, int qid, Eigen::Matrix<double, 18, 1>* deriv, Eigen::Matrix<double, 18, 18>* hess)
{
	double energy = 0;

	Eigen::Vector3d bary(1 - _quadpts[qid].u - _quadpts[qid].v, _quadpts[qid].u, _quadpts[qid].v), edgewcur, edgewnext;
	std::vector<std::complex<double>> vertZvalcur(3), vertZvalnext(3);

	Eigen::Vector3d wflag(1, 1, 1);

	for (int j = 0; j < 3; j++)
	{
		int vid = _mesh.faceVertex(fid, j);
		int eid = _mesh.faceEdge(fid, j);

		vertZvalcur[j] = curZvals[vid];
		vertZvalnext[j] = nextZvals[vid];
		edgewcur(j) = curw(eid);
		edgewnext(j) = nextw(eid);

		if (_mesh.edgeVertex(eid, 1) == _mesh.faceVertex(fid, (j + 1) % 2))  // defined as mesh.edgeVertex(eid, 1) - mesh.edgeVertex(eid, 0), while we want vid_next_next - vid_next
		{
			wflag(j) = -1;
			edgewcur(j) *= -1;
			edgewnext(j) *= -1;
		}
	}

	Eigen::Matrix<std::complex<double>, 9, 1> derivCur, derivNext;
	Eigen::Matrix<std::complex<double>, 9, 9> hessCur, hessNext;

	std::complex<double> zcur = IntrinsicFormula::getZvalsFromEdgeOmega(bary, vertZvalcur, edgewcur, (deriv || hess) ? &derivCur : NULL, hess ?&hessCur : NULL);
	std::complex<double> znext = IntrinsicFormula::getZvalsFromEdgeOmega(bary, vertZvalnext, edgewnext, (deriv || hess) ? &derivNext : NULL, hess ? &hessNext : NULL);
	std::complex<double> deltaz = znext - zcur;

	double componentWeights = 0.5 * _faceArea[fid] * _quadpts[qid].weight / _dt;
	energy = componentWeights * (deltaz.real() * deltaz.real() + deltaz.imag() * deltaz.imag());

	if (deriv || hess)
	{
		Eigen::Matrix<std::complex<double>, 18, 1> gradDeltaZ;
		gradDeltaZ.segment<9>(0) = derivNext;
		gradDeltaZ.segment<9>(9) = -derivCur;

		for (int j = 0; j < 3; j++)
		{
			gradDeltaZ(6 + j) *= wflag(j);
			gradDeltaZ(15 + j) *= wflag(j);
		}

		if (deriv)
		{
			(*deriv) = 2 * componentWeights * (deltaz.real() * gradDeltaZ.real() + deltaz.imag() * gradDeltaZ.imag());
		}

		if (hess)
		{
			Eigen::Matrix<std::complex<double>, 18, 18> hessDeltaZ;
			hessDeltaZ.block<9, 9>(0, 0) = hessNext;
			hessDeltaZ.block<9, 9>(9, 9) = -hessCur;

			for (int i = 0; i < 3; i++)
			{
				hessDeltaZ.row(6 + i) *= wflag(i);
				hessDeltaZ.row(15 + i) *= wflag(i);
				for (int j = 0; j < 3; j++)
				{
					hessDeltaZ.row(6 + j) *= wflag(j);
					hessDeltaZ.row(15 + j) *= wflag(j);
				}
			}

			(*hess) = gradDeltaZ.real() * (gradDeltaZ.real()).transpose() + gradDeltaZ.imag() * gradDeltaZ.imag().transpose();
			(*hess) += deltaz.real() * hessDeltaZ.real() + deltaz.imag() * hessDeltaZ.imag();
			(*hess) *= 2 * componentWeights;
				
		}

	}


	return energy;
}

double ComputeZdotFromEdgeOmega::computeZdotIntegrationPerface(const std::vector<std::complex<double>>& curZvals, const Eigen::VectorXd& curw, const std::vector<std::complex<double>>& nextZvals, const Eigen::VectorXd& nextw, int fid, Eigen::Matrix<double, 18, 1>* deriv, Eigen::Matrix<double, 18, 18>* hess, bool isProj)
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
		Eigen::Matrix<double, 18, 1> zdotDeriv;
		Eigen::Matrix<double, 18, 18> zdotHess;
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


double ComputeZdotFromEdgeOmega::computeZdotIntegration(const std::vector<std::complex<double>>& curZvals, const Eigen::VectorXd& curw, const std::vector<std::complex<double>>& nextZvals, const Eigen::VectorXd& nextw, Eigen::VectorXd* deriv, std::vector<Eigen::Triplet<double>>* hessT, bool isProj)
{
	double energy = 0;

	int nverts = curZvals.size();
	int nedges = curw.rows();
	int nfaces = _mesh.nFaces();

	if (deriv)
		deriv->setZero(4 * nverts + 2 * nedges);
	if (hessT)
		hessT->clear();

	std::vector<double> energyList(nfaces);
	std::vector<Eigen::Matrix<double, 18, 1>> derivList(nfaces);
	std::vector<Eigen::Matrix<double, 18, 18>> hessList(nfaces);

	auto computeEnergy = [&](const tbb::blocked_range<uint32_t>& range) {
		for (uint32_t i = range.begin(); i < range.end(); ++i)
		{
			energyList[i] = computeZdotIntegrationPerface(curZvals, curw, nextZvals, nextw, i, deriv ? &derivList[i] : NULL, hessT ? &hessList[i] : NULL, isProj);
		}
	};

	tbb::blocked_range<uint32_t> rangex(0u, (uint32_t)nfaces, GRAIN_SIZE);
	tbb::parallel_for(rangex, computeEnergy);

	int nOneGroup = 2 * nverts + nedges;

	for (int i = 0; i < nfaces; i++)
	{
		energy += energyList[i];

		if (deriv)
		{
			for (int j = 0; j < 3; j++)
			{
				int vid = _mesh.faceVertex(i, j);
				int eid = _mesh.faceEdge(i, j);

				(*deriv)(2 * vid) = derivList[i](2 * j);
				(*deriv)(2 * vid + 1) = derivList[i](2 * j + 1);
				(*deriv)(eid + nverts) = derivList[i](6 + j);

				(*deriv)(2 * vid + nOneGroup) = derivList[i](9 + 2 * j);
				(*deriv)(2 * vid + 1 + nOneGroup) = derivList[i](9 + 2 * j + 1);
				(*deriv)(eid + nverts + nOneGroup) = derivList[i](15 + j);
			}
		}

		if (hessT)
		{
			for (int j = 0; j < 3; j++)
			{
				int vid = _mesh.faceVertex(i, j);
				int eid = _mesh.faceEdge(i, j);

				for (int k = 0; k < 3; k++)
				{
					int vid1 = _mesh.faceVertex(i, k);
					int eid1 = _mesh.faceEdge(i, k);

					for (int m1 = 0; m1 < 2; m1++)
					{
						for (int m2 = 0; m2 < 2; m2++)
						{
							hessT->push_back({ 2 * vid + m1, 2 * vid1 + m2,  hessList[i](2 * j + m1, 2 * k + m2) });
							hessT->push_back({ 2 * vid + m1 + nOneGroup, 2 * vid1 + m2 + nOneGroup,  hessList[i](9 + 2 * j + m1, 9 + 2 * k + m2) });
						}
						hessT->push_back({ 2 * vid + m1, eid1 + nverts,  hessList[i](2 * j + m1, 6 + k) });
						hessT->push_back({ 2 * vid + m1 + nOneGroup, eid1 + nverts + nOneGroup,  hessList[i](9 + 2 * j + m1, 15 + k) });

						hessT->push_back({ eid + nverts, 2 * vid1 + m1, hessList[i](6 + j, 2 * k + m1) });
						hessT->push_back({ eid + nverts + nOneGroup, 2 * vid1 + m1 + nOneGroup,  hessList[i](15 + j, 9 + 2 * k + m1) });

					}
						
					
				}

				
			}
		}
	}

	return energy;
}