#include "../../include/IntrinsicFormula/ComputeZdotFromHalfEdgeOmega.h"
#include <iostream>

using namespace IntrinsicFormula;

double ComputeZdotFromHalfEdgeOmega::computeZdotIntegrationFromQuad(const std::vector<std::complex<double>>& curZvals, const Eigen::MatrixXd& curw, const std::vector<std::complex<double>>& nextZvals, const Eigen::MatrixXd& nextw, int fid, int qid, Eigen::Matrix<double, 24, 1>* deriv, Eigen::Matrix<double, 24, 24>* hess)
{
	double energy = 0;

	Eigen::Vector3d bary(1 - _quadpts[qid].u - _quadpts[qid].v, _quadpts[qid].u, _quadpts[qid].v);
	Eigen::Matrix<double, 3,2> edgewcur, edgewnext;
	std::vector<std::complex<double>> vertZvalcur(3), vertZvalnext(3);

	Eigen::Vector3i wflag;
	wflag.setZero();	// 1 means swap

	for (int j = 0; j < 3; j++)
	{
		int vid = _mesh.faceVertex(fid, j);
		int eid = _mesh.faceEdge(fid, j);

		vertZvalcur[j] = curZvals[vid];
		vertZvalnext[j] = nextZvals[vid];
		edgewcur.row(j) = curw.row(eid);
		edgewnext.row(j) = nextw.row(eid);

		if (_mesh.edgeVertex(eid, 1) == _mesh.faceVertex(fid, (j + 1) % 3))  // defined as mesh.edgeVertex(eid, 1) - mesh.edgeVertex(eid, 0), while we want vid_next_next - vid_next
		{
			wflag(j) = 1;
			edgewcur(j, 1) = curw(eid, 0);
			edgewcur(j, 0) = curw(eid, 1);

			edgewnext(j, 1) = nextw(eid, 0);
			edgewnext(j, 0) = nextw(eid, 1);
		}
	}

	Eigen::Matrix<std::complex<double>, 12, 1> derivCur, derivNext;
	Eigen::Matrix<std::complex<double>, 12, 12> hessCur, hessNext;

	std::complex<double> zcur = IntrinsicFormula::getZvalsFromHalfEdgeOmega(bary, vertZvalcur, edgewcur, (deriv || hess) ? &derivCur : NULL, hess ? &hessCur : NULL);
	std::complex<double> znext = IntrinsicFormula::getZvalsFromHalfEdgeOmega(bary, vertZvalnext, edgewnext, (deriv || hess) ? &derivNext : NULL, hess ? &hessNext : NULL);

	std::complex<double> deltaz = znext - zcur;

	double componentWeights = 0.5 * _faceArea[fid] * _quadpts[qid].weight / (_dt * _dt);
	energy = componentWeights * (deltaz.real() * deltaz.real() + deltaz.imag() * deltaz.imag());

	{
		Eigen::Matrix<std::complex<double>, 24, 1> gradDeltaZ;

		gradDeltaZ.segment<12>(0) = -derivCur;
		gradDeltaZ.segment<12>(12) = derivNext;

		
		for (int j = 0; j < 3; j++)
		{
			if (wflag(j) == 1)
			{
				std::complex<double> c = gradDeltaZ(6 + 2 * j);
				gradDeltaZ(6 + 2 * j) = gradDeltaZ(6 + 2 * j + 1);
				gradDeltaZ(6 + 2 * j + 1) = c;

				c = gradDeltaZ(18 + 2 * j);
				gradDeltaZ(18 + 2 * j) = gradDeltaZ(18 + 2 * j + 1);
				gradDeltaZ(18 + 2 * j + 1) = c;
			}
		}

		if (deriv)
		{
			(*deriv) = 2 * componentWeights * (deltaz.real() * gradDeltaZ.real() + deltaz.imag() * gradDeltaZ.imag());
		}

		if (hess)
		{
		    hess->setZero();
			Eigen::Matrix<std::complex<double>, 24, 24> hessDeltaZ;


			hessDeltaZ.block<12, 12>(0, 0) = -hessCur;
			hessDeltaZ.block<12, 12>(12, 12) = hessNext;

			for (int i = 0; i < 3; i++)
			{
				if (wflag(i) == 1)	// do swap
				{
					hessDeltaZ.row(6 + 2 * i).swap(hessDeltaZ.row(6 + 2 * i + 1));
					hessDeltaZ.row(18 + 2 * i).swap(hessDeltaZ.row(18 + 2 * i + 1));
				}
			}
			
			for (int j = 0; j < 3; j++)
			{
				if (wflag(j) == 1) // do swap
				{
					hessDeltaZ.col(6 + 2 * j).swap(hessDeltaZ.col(6 + 2 * j + 1));
					hessDeltaZ.col(18 + 2 * j).swap(hessDeltaZ.col(18 + 2 * j + 1));
				}

			}

			(*hess) = gradDeltaZ.real() * (gradDeltaZ.real()).transpose() + gradDeltaZ.imag() * gradDeltaZ.imag().transpose();
			(*hess) += deltaz.real() * hessDeltaZ.real() + deltaz.imag() * hessDeltaZ.imag();
			(*hess) *= 2 * componentWeights;
				
		}

	}


	return energy;
}

double ComputeZdotFromHalfEdgeOmega::computeZdotIntegrationPerface(const std::vector<std::complex<double>>& curZvals, const Eigen::MatrixXd& curw, const std::vector<std::complex<double>>& nextZvals, const Eigen::MatrixXd& nextw, int fid, Eigen::Matrix<double, 24, 1>* deriv, Eigen::Matrix<double, 24, 24>* hess, bool isProj)
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
		Eigen::Matrix<double, 24, 1> zdotDeriv;
		Eigen::Matrix<double, 24, 24> zdotHess;
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


double ComputeZdotFromHalfEdgeOmega::computeZdotIntegration(const std::vector<std::complex<double>>& curZvals, const Eigen::MatrixXd& curw, const std::vector<std::complex<double>>& nextZvals, const Eigen::MatrixXd& nextw, Eigen::VectorXd* deriv, std::vector<Eigen::Triplet<double>>* hessT, bool isProj)
{
	double energy = 0;

	int nverts = curZvals.size();
	int nedges = curw.rows();
	int nfaces = _mesh.nFaces();

	if (deriv)
		deriv->setZero(4 * nverts + 4 * nedges);
	if (hessT)
		hessT->clear();

	std::vector<double> energyList(nfaces);
	std::vector<Eigen::Matrix<double, 24, 1>> derivList(nfaces);
	std::vector<Eigen::Matrix<double, 24, 24>> hessList(nfaces);

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

	int nOneGroup = 2 * nverts + 2 * nedges;

	for (int i = 0; i < nfaces; i++)
	{
		energy += energyList[i];

		if (deriv)
		{
			for (int j = 0; j < 3; j++)
			{
				int vid = _mesh.faceVertex(i, j);
				int eid = _mesh.faceEdge(i, j);

				(*deriv)(2 * vid) += derivList[i](2 * j);
				(*deriv)(2 * vid + 1) += derivList[i](2 * j + 1);
				(*deriv)(2 * eid + 2 * nverts) += derivList[i](6 + 2 * j);
				(*deriv)(2 * eid + 1 + 2 * nverts) += derivList[i](6 + 2 * j + 1);

				(*deriv)(2 * vid + nOneGroup) += derivList[i](12 + 2 * j);
				(*deriv)(2 * vid + 1 + nOneGroup) += derivList[i](12 + 2 * j + 1);
				(*deriv)(2 * eid + 2 * nverts + nOneGroup) += derivList[i](18 + 2 * j);
				(*deriv)(2 * eid + 1 + 2 * nverts + nOneGroup) += derivList[i](18 + 2 * j + 1);
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
							hessT->push_back({ 2 * vid + m1, 2 * vid1 + m2 + nOneGroup,  hessList[i](2 * j + m1, 12 + 2 * k + m2) });

							hessT->push_back({ 2 * vid + m1 + nOneGroup, 2 * vid1 + m2,  hessList[i](12 + 2 * j + m1, 2 * k + m2) });
							hessT->push_back({ 2 * vid + m1 + nOneGroup, 2 * vid1 + m2 + nOneGroup,  hessList[i](12 + 2 * j + m1, 12 + 2 * k + m2) });

							hessT->push_back({ 2 * eid + m1 + 2 * nverts, 2 * eid1 + m2 + 2 * nverts, hessList[i](6 + 2 * j + m1, 6 + 2 * k + m2) });
							hessT->push_back({ 2 * eid + m1 + 2 * nverts, 2 * eid1 + m2 + 2 * nverts + nOneGroup, hessList[i](6 + 2 * j + m1, 18 + 2 * k + m2) });
							hessT->push_back({ 2 * eid + m1 + 2 * nverts + nOneGroup, 2 * eid1 + m2 + 2 * nverts, hessList[i](18 + 2 * j + m1, 6 + 2 * k + m2) });
							hessT->push_back({ 2 * eid + m1 + 2 * nverts + nOneGroup, 2 * eid1 + m2 + 2 * nverts + nOneGroup, hessList[i](18 + 2 * j + m1, 18 + 2 * k + m2) });

							hessT->push_back({ 2 * vid + m1, 2 * eid1 + m2 + 2 * nverts,  hessList[i](2 * j + m1, 6 + 2 * k + m2) });
							hessT->push_back({ 2 * eid + m1 + 2 * nverts, 2 * vid1 + m2, hessList[i](6 + 2 * j + m1, 2 * k + m2) });
							hessT->push_back({ 2 * vid + m1, 2 * eid1 + m2 + 2 * nverts + nOneGroup, hessList[i](2 * j + m1, 18 + 2 * k + m2) });
							hessT->push_back({ 2 * eid + m1 + 2 * nverts + nOneGroup, 2 * vid1 + m2, hessList[i](18 + 2 * j + m1, 2 * k + m2) });

							hessT->push_back({ 2 * vid + m1 + nOneGroup, 2 * eid1 + m2 + 2 * nverts,  hessList[i](12 + 2 * j + m1, 6 + 2 * k + m2) });
							hessT->push_back({ 2 * eid + m1 + 2 * nverts, 2 * vid1 + m2 + nOneGroup, hessList[i](6 + 2 * j + m1, 12 + 2 * k + m2) });
							hessT->push_back({ 2 * vid + m1 + nOneGroup, 2 * eid1 + m2 + 2 * nverts + nOneGroup,  hessList[i](12 + 2 * j + m1, 18 + 2 * k + m2) });
							hessT->push_back({ 2 * eid + m1 + 2 * nverts + nOneGroup, 2 * vid1 + m2 + nOneGroup,  hessList[i](18 + 2 * j + m1, 12 + 2 * k + m2) });
						}

					}
					
				}

				
			}
		}
	}

	return energy;
}

void ComputeZdotFromHalfEdgeOmega::testZdotIntegrationFromQuad(const std::vector<std::complex<double>> &curZvals,
                                                           const Eigen::MatrixXd&curw,
                                                           const std::vector<std::complex<double>> &nextZvals,
                                                           const Eigen::MatrixXd&nextw, int fid, int qid)
{
    Eigen::Matrix<double, 24, 1> deriv, deriv1;
    Eigen::Matrix<double, 24, 24> hess;
    double energy = computeZdotIntegrationFromQuad(curZvals, curw, nextZvals, nextw, fid, qid, &deriv, &hess);

    auto backupcurZvals = curZvals, backupnextZvals = nextZvals;
    auto backupcurw = curw, backupnextw = nextw;

    auto dir = deriv;
    dir.setRandom();

    for(int i = 3; i < 10; i++)
    {
		double eps = std::pow(0.1, i);

		for (int j = 0; j < 3; j++)
		{
			int vid = _mesh.faceVertex(fid, j);
			int eid = _mesh.faceEdge(fid, j);

			backupcurZvals[vid] = std::complex<double>(curZvals[vid].real() + dir(2 * j) * eps, curZvals[vid].imag() + dir(2 * j + 1) * eps);
			backupnextZvals[vid] = std::complex<double>(nextZvals[vid].real() + dir(2 * j + 12) * eps, nextZvals[vid].imag() + dir(2 * j + 1 + 12) * eps);

			backupcurw(eid, 0) = curw(eid, 0) + eps * dir(2 * j + 6);
			backupcurw(eid, 1) = curw(eid, 1) + eps * dir(2 * j + 1 + 6);

			backupnextw(eid, 0) = nextw(eid, 0) + eps * dir(2 * j + 18);
			backupnextw(eid, 1) = nextw(eid, 1) + eps * dir(2 * j + 1 + 18);
		}

        double energy1 = computeZdotIntegrationFromQuad(backupcurZvals, backupcurw, backupnextZvals, backupnextw, fid, qid, &deriv1, NULL);
        std::cout << "eps: " << eps << std::endl;
        std::cout << "f-g: " << (energy1 - energy) / eps - dir.dot(deriv) << std::endl;
        std::cout << "g-h: " << ((deriv1 - deriv) / eps - hess * dir).norm() << std::endl;
    }
}

void ComputeZdotFromHalfEdgeOmega::testZdotIntegrationPerface(const std::vector<std::complex<double>> &curZvals,
                                                          const Eigen::MatrixXd&curw,
                                                          const std::vector<std::complex<double>> &nextZvals,
                                                          const Eigen::MatrixXd&nextw, int fid)
{
    Eigen::Matrix<double, 24, 1> deriv, deriv1;
    Eigen::Matrix<double, 24, 24> hess;
    double energy = computeZdotIntegrationPerface(curZvals, curw, nextZvals, nextw, fid, &deriv, &hess);


    auto backupcurZvals = curZvals, backupnextZvals = nextZvals;
    auto backupcurw = curw, backupnextw = nextw;

    auto dir = deriv;
    dir.setRandom();

    for(int i = 3; i < 10; i++)
    {
        double eps = std::pow(0.1, i);

        for(int j = 0; j < 3; j++)
        {
            int vid = _mesh.faceVertex(fid, j);
            int eid = _mesh.faceEdge(fid, j);

            backupcurZvals[vid] = std::complex<double>(curZvals[vid].real() + dir(2 * j) * eps, curZvals[vid].imag() + dir(2 * j + 1) * eps);
            backupnextZvals[vid] = std::complex<double>(nextZvals[vid].real() + dir(2 * j + 12) * eps, nextZvals[vid].imag() + dir(2 * j + 1 + 12) * eps);

            backupcurw(eid, 0) = curw(eid, 0) + eps * dir(2 * j + 6);
			backupcurw(eid, 1) = curw(eid, 1) + eps * dir(2 * j + 1 + 6);

            backupnextw(eid, 0) = nextw(eid, 0) + eps * dir(2 * j + 18);
			backupnextw(eid, 1) = nextw(eid, 1) + eps * dir(2 * j + 1 + 18);
        }

        double energy1 = computeZdotIntegrationPerface(backupcurZvals, backupcurw, backupnextZvals, backupnextw, fid, &deriv1, NULL);
        std::cout << "eps: " << eps << std::endl;
        std::cout << "f-g: " << (energy1 - energy) / eps - dir.dot(deriv) << std::endl;
        std::cout << "g-h: " << ((deriv1 - deriv) / eps - hess * dir).norm() << std::endl;
    }
}

void ComputeZdotFromHalfEdgeOmega::testZdotIntegration(const std::vector<std::complex<double>> &curZvals,
                                                   const Eigen::MatrixXd&curw,
                                                   const std::vector<std::complex<double>> &nextZvals,
                                                   const Eigen::MatrixXd&nextw)
{
    Eigen::VectorXd deriv, deriv1;
    std::vector<Eigen::Triplet<double>> hessT;
    Eigen::SparseMatrix<double> hess;
    double energy = computeZdotIntegration(curZvals, curw, nextZvals, nextw, &deriv, &hessT);
    hess.resize(deriv.rows(), deriv.rows());
    hess.setFromTriplets(hessT.begin(), hessT.end());

    int nverts = curZvals.size();
    int nedges = curw.rows();

   /* std::cout << "deriv: \n" << deriv.segment(0, 2 * nverts + nedges) << std::endl;
    std::cout << "hess: \n"  << hess.toDense().block(0, 0, 2 * nverts + nedges, 2 * nverts + nedges) << std::endl;
    std::cout << "eid: " << _mesh.faceEdge(0, 0) << " " << _mesh.faceEdge(0, 1) << " " << _mesh.faceEdge(0, 2) << std::endl;*/

    Eigen::VectorXd dir = deriv;
    dir.setRandom();


    auto backupcurZvals = curZvals, backupnextZvals = nextZvals;
    auto backupcurw = curw, backupnextw = nextw;

    for(int i = 3; i < 10; i++)
    {
        double eps = std::pow(0.1, i);

        for(int j = 0; j < nverts; j++)
        {
            backupcurZvals[j] = std::complex<double>(curZvals[j].real() + dir(2 * j) * eps, curZvals[j].imag() + dir(2 * j + 1) * eps);
            backupnextZvals[j] = std::complex<double>(nextZvals[j].real() + dir(2 * j + 2 * nverts + 2 * nedges) * eps, nextZvals[j].imag() + dir(2 * j + 1 + 2 * nverts + 2 * nedges) * eps);
        }

		for (int j = 0; j < nedges; j++)
		{
			backupcurw.row(j) = curw.row(j) + eps * dir.segment<2>(2 * nverts + 2 * j).transpose();
			backupnextw.row(j) = nextw.row(j) + eps * dir.segment<2>(4 * nverts + 2 * nedges + 2 * j).transpose();
		}

       

        double energy1 = computeZdotIntegration(backupcurZvals, backupcurw, backupnextZvals, backupnextw, &deriv1, NULL);

//        std::cout << "deriv1 - deriv: " << (deriv1 - deriv).transpose() << std::endl;

        std::cout << "eps: " << eps << std::endl;
        std::cout << "f-g: " << (energy1 - energy) / eps - dir.dot(deriv) << std::endl;
        std::cout << "g-h: " << ((deriv1 - deriv)/ eps - (hess * dir)).norm() << std::endl;
    }
}