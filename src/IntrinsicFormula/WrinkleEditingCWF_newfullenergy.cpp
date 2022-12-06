#include "../../include/IntrinsicFormula/WrinkleEditingCWF_newfullenergy.h"
#include "../../include/Optimization/NewtonDescent.h"
#include "../../include/IntrinsicFormula/KnoppelStripePatternEdgeOmega.h"
#include "../../include/WrinkleFieldsEditor.h"
#include <igl/cotmatrix_entries.h>
#include <igl/cotmatrix.h>
#include <igl/doublearea.h>
#include <igl/boundary_loop.h>
#include <Eigen/SPQRSupport>

using namespace IntrinsicFormula;

void WrinkleEditingCWFNewFullEnergy::convertList2Variable(Eigen::VectorXd& x)
{
	int nverts = _pos.rows();
	int nedges = _mesh.nEdges();

	int numFrames = _zvalsList.size() - 2;

	int DOFsPerframe = 2 * nverts + nedges;

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

void WrinkleEditingCWFNewFullEnergy::convertVariable2List(const Eigen::VectorXd& x)
{
	int nverts = _pos.rows();
	int nedges = _mesh.nEdges();

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


double WrinkleEditingCWFNewFullEnergy::spatialKnoppelEnergy(int frameId, Eigen::VectorXd* deriv, std::vector<Eigen::Triplet<double>>* hessT, bool isProj)
{
	double energy = 0;
	int nedges = _mesh.nEdges();
	int nverts = _pos.rows();
	double aveAmp = _refAmpAveList[frameId];
	std::vector<Eigen::Triplet<double>> AT;
	AT.clear();

	int maxFreeVid = 0;

	if (deriv)
		deriv->setZero(2 * nverts + nedges);
	if (hessT)
		hessT->clear();

	for (int eid = 0; eid < nedges; eid++)
	{
		int vid0 = _mesh.edgeVertex(eid, 0);
		int vid1 = _mesh.edgeVertex(eid, 1);

		double r0 = _combinedRefAmpList[frameId](vid0) / aveAmp;
		double r1 = _combinedRefAmpList[frameId](vid1) / aveAmp;

		std::complex<double> expw0 = std::complex<double>(std::cos(_edgeOmegaList[frameId](eid)), std::sin(_edgeOmegaList[frameId](eid)));

		std::complex<double> z0 = _zvalsList[frameId][vid0];
		std::complex<double> z1 = _zvalsList[frameId][vid1];

		double ce = _spatialKnoppelRatio * _edgeArea(eid);

		Eigen::Vector2d v0; v0 << z0.real(), z0.imag();
		Eigen::Vector2d v1; v1 << z1.real(), z1.imag();
		Eigen::Matrix2d rot; rot << std::cos(_edgeOmegaList[frameId](eid)), -std::sin(_edgeOmegaList[frameId](eid)), std::sin(_edgeOmegaList[frameId](eid)), std::cos(_edgeOmegaList[frameId](eid));

		Eigen::Vector2d g = r0 * v1 - r1 * rot * v0;

		energy += 0.5 * g.dot(g) * ce;

		if (deriv || hessT)
		{
			Eigen::Matrix<double, 2, 5> dg;

			Eigen::Vector2d e0, e1;
			e0 << 1, 0; e1 << 0, 1;

			Eigen::Matrix2d drot;
			drot << -std::sin(_edgeOmegaList[frameId](eid)), -std::cos(_edgeOmegaList[frameId](eid)), std::cos(_edgeOmegaList[frameId](eid)), -std::sin(_edgeOmegaList[frameId](eid));

			dg.col(0) = -r1 * rot * e0;
			dg.col(1) = -r1 * rot * e1;
			dg.col(2) = r0 * e0;
			dg.col(3) = r0 * e1;
			dg.col(4) = -r1 * drot * v0;

			if (deriv)
			{
				Eigen::Matrix<double, 5, 1> localGrad = ce * dg.transpose() * g;
				(*deriv)(2 * vid0) += localGrad(0);
				(*deriv)(2 * vid0 + 1) += localGrad(1);
				(*deriv)(2 * vid1) += localGrad(2);
				(*deriv)(2 * vid1 + 1) += localGrad(3);
				(*deriv)(eid + 2 * nverts) += localGrad(4);
			}

			if (hessT)
			{
				Eigen::Matrix<double, 5, 5> localHess = ce * dg.transpose() * dg;

				Eigen::Vector2d hg04, hg14, hg44;
				hg04 = -r1 * drot * e0;
				hg14 = -r1 * drot * e1;
				hg44 = r1 * rot * v0;

				localHess(0, 4) += ce * g.dot(hg04);
				localHess(4, 0) += ce * g.dot(hg04);

				localHess(1, 4) += ce * g.dot(hg14);
				localHess(4, 1) += ce * g.dot(hg14);

				localHess(4, 4) += ce * g.dot(hg44);

				if (isProj)
				{
					localHess = SPDProjection(localHess);
				}

				for (int i = 0; i < 2; i++)
				{
					hessT->push_back(Eigen::Triplet<double>(2 * vid0 + i, 2 * nverts + eid, localHess(i, 4)));
					hessT->push_back(Eigen::Triplet<double>(2 * vid1 + i, 2 * nverts + eid, localHess(2 + i, 4)));

					hessT->push_back(Eigen::Triplet<double>(2 * nverts + eid, 2 * vid0 + i, localHess(4, i)));
					hessT->push_back(Eigen::Triplet<double>(2 * nverts + eid, 2 * vid1 + i, localHess(4, 2 + i)));

					for (int j = 0; j < 2; j++)
					{
						hessT->push_back(Eigen::Triplet<double>(2 * vid0 + i, 2 * vid0 + j, localHess(i, j)));
						hessT->push_back(Eigen::Triplet<double>(2 * vid1 + i, 2 * vid1 + j, localHess(2 + i, 2 + j)));

						hessT->push_back(Eigen::Triplet<double>(2 * vid0 + i, 2 * vid1 + j, localHess(i, 2 + j)));
						hessT->push_back(Eigen::Triplet<double>(2 * vid1 + i, 2 * vid0 + j, localHess(2 + i, j)));
					}
				}

				hessT->push_back(Eigen::Triplet<double>(2 * nverts + eid, 2 * nverts + eid, localHess(4, 4)));
			}

		}
	}

	return energy;
}


double WrinkleEditingCWFNewFullEnergy::computeKineticEnergyPerFaceVertex(int fid, int vInF, int frameId, double dt, Eigen::Matrix<double, 8, 1>* deriv, Eigen::Matrix<double, 8, 8>* hess, bool isProj)
{
	int vid = _mesh.faceVertex(fid, vInF);
	double coeff = _vertWeight(vid) / (dt * dt) * _faceArea[fid] / 3;
	
	coeff *= std::pow(1e-2, 3) / 12; // fake bending stiffness

	int eid0 = _mesh.faceEdge(fid, (vInF + 1) % 3);
	int eid1 = _mesh.faceEdge(fid, (vInF + 2) % 3);
	Eigen::RowVector3d r0 = _pos.row(_mesh.edgeVertex(eid0, 1)) - _pos.row(_mesh.edgeVertex(eid0, 0));
	Eigen::RowVector3d r1 = _pos.row(_mesh.edgeVertex(eid1, 1)) - _pos.row(_mesh.edgeVertex(eid1, 0));

	Eigen::Matrix2d Iinv, I;
	I << r0.dot(r0), r0.dot(r1), r1.dot(r0), r1.dot(r1);
	Iinv = I.inverse();

	Eigen::Vector2d curOmega, nextOmega;
	curOmega << _edgeOmegaList[frameId][eid0], _edgeOmegaList[frameId][eid1];
	nextOmega << _edgeOmegaList[frameId + 1][eid0], _edgeOmegaList[frameId + 1][eid1];
	Eigen::Matrix2d curOmegaMat = curOmega * curOmega.transpose();
	Eigen::Matrix2d nextOmegaMat = nextOmega * nextOmega.transpose();

	Eigen::Vector2d curZVec, nextZVec;
	curZVec << _zvalsList[frameId][vid].real(), _zvalsList[frameId][vid].imag();
	nextZVec << _zvalsList[frameId + 1][vid].real(), _zvalsList[frameId + 1][vid].imag();

	Eigen::Matrix2d diffx, diffy;
	diffx = Iinv * (nextZVec[0] * nextOmegaMat - curZVec[0] * curOmegaMat);
	diffy = Iinv * (nextZVec[1] * nextOmegaMat - curZVec[1] * curOmegaMat);

	double energy = 0.5 * coeff * ((diffx.transpose() * diffx).trace() + (diffy.transpose() * diffy).trace());

	//double energy = coeff * (diffy).trace();

	if (deriv || hess)
	{
		std::vector<Eigen::Matrix2d> gradDiffx(8), gradDiffy(8);
		std::vector<Eigen::Vector2d> gradOmega(2);
		std::vector<Eigen::Matrix2d> gradCurOmegaMat(2), gradNextOmegaMat(2);

		gradOmega[0] << 1, 0;
		gradOmega[1] << 0, 1;
		Eigen::Matrix2d zeroMat = Eigen::Matrix2d::Zero();

		gradCurOmegaMat[0] = gradOmega[0] * curOmega.transpose() + curOmega * gradOmega[0].transpose();
		gradCurOmegaMat[1] = gradOmega[1] * curOmega.transpose() + curOmega * gradOmega[1].transpose();

		gradNextOmegaMat[0] = gradOmega[0] * nextOmega.transpose() + nextOmega * gradOmega[0].transpose();
		gradNextOmegaMat[1] = gradOmega[1] * nextOmega.transpose() + nextOmega * gradOmega[1].transpose();


		// d diffx / d z
		gradDiffx[0] = -Iinv * curOmegaMat;
		gradDiffx[1] = zeroMat;

		// d diffx / d w
		gradDiffx[2] = -curZVec[0] * Iinv * gradCurOmegaMat[0];
		gradDiffx[3] = -curZVec[0] * Iinv * gradCurOmegaMat[1];

		// d diffx / d z
		gradDiffx[4] = Iinv * nextOmegaMat;
		gradDiffx[5] = zeroMat;

		// d diffx / d w
		gradDiffx[6] = nextZVec[0] * Iinv * gradNextOmegaMat[0];
		gradDiffx[7] = nextZVec[0] * Iinv * gradNextOmegaMat[1];


		// d diffy / d z
		gradDiffy[0] = zeroMat;
		gradDiffy[1] = -Iinv * curOmegaMat;
		gradDiffy[4] = zeroMat;
		gradDiffy[5] = Iinv * nextOmegaMat;

		// d diffy / d w
		gradDiffy[2] = -curZVec[1] * Iinv * gradCurOmegaMat[0];
		gradDiffy[3] = -curZVec[1] * Iinv * gradCurOmegaMat[1];
		gradDiffy[6] = nextZVec[1] * Iinv * gradNextOmegaMat[0];
		gradDiffy[7] = nextZVec[1] * Iinv * gradNextOmegaMat[1];

		if (deriv)
		{
			deriv->setZero();
			for (int j = 0; j < 8; j++)
			{
				(*deriv)[j] = coeff * ((gradDiffx[j].transpose() * diffx).trace() + (gradDiffy[j].transpose() * diffy).trace());
				//(*deriv)[j] = coeff * (gradDiffy[j]).trace();
			}
		}

		if (hess)
		{
			hess->setZero();

			// stupid but easy to debug implementation
			std::vector<std::vector<Eigen::Matrix2d>> hDiffx, hDiffy;
			hDiffx.resize(8);
			for (int i = 0; i < 8; i++)
				hDiffx[i].resize(8, Eigen::Matrix2d::Zero());
			hDiffy.resize(8);
			for (int i = 0; i < 8; i++)
				hDiffy[i].resize(8, Eigen::Matrix2d::Zero());
				

			hDiffx[0][2] = -Iinv * gradCurOmegaMat[0];
			hDiffx[0][3] = -Iinv * gradCurOmegaMat[1];

			
			hDiffx[2][0] = -Iinv * gradCurOmegaMat[0];	// gradDiffx[2] = -curZVec[0] * Iinv * gradCurOmegaMat[0];
			hDiffx[2][2] = -curZVec[0] * Iinv * (gradOmega[0] * gradOmega[0].transpose() + gradOmega[0] * gradOmega[0].transpose());
			hDiffx[2][3] = -curZVec[0] * Iinv * (gradOmega[0] * gradOmega[1].transpose() + gradOmega[1] * gradOmega[0].transpose());

			hDiffx[3][0] = -Iinv * gradCurOmegaMat[1];	// gradDiffx[3] = -curZVec[0] * Iinv * gradCurOmegaMat[1];
			hDiffx[3][2] = -curZVec[0] * Iinv * (gradOmega[1] * gradOmega[0].transpose() + gradOmega[1] * gradOmega[0].transpose());	
			hDiffx[3][3] = -curZVec[0] * Iinv * (gradOmega[1] * gradOmega[1].transpose() + gradOmega[1] * gradOmega[1].transpose());

			
			hDiffx[4][6] = Iinv * gradNextOmegaMat[0];  // gradDiffx[4] = Iinv * nextOmegaMat;
			hDiffx[4][7] = Iinv * gradNextOmegaMat[1];												
			
			
			hDiffx[6][4] = Iinv * gradNextOmegaMat[0]; // gradDiffx[6] = nextZVec[0] * Iinv * gradNextOmegaMat[0];
			hDiffx[6][6] = nextZVec[0] * Iinv * (gradOmega[0] * gradOmega[0].transpose() + gradOmega[0] * gradOmega[0].transpose()); // gradDiffx[6] = nextZVec[0] * Iinv * gradNextOmegaMat[0];
			hDiffx[6][7] = nextZVec[0] * Iinv * (gradOmega[0] * gradOmega[1].transpose() + gradOmega[1] * gradOmega[0].transpose());// gradDiffx[6] = nextZVec[0] * Iinv * gradNextOmegaMat[0];

			hDiffx[7][4] = Iinv * gradNextOmegaMat[1]; // gradDiffx[7] = nextZVec[0] * Iinv * gradNextOmegaMat[1];
			hDiffx[7][6] = nextZVec[0] * Iinv * (gradOmega[1] * gradOmega[0].transpose() + gradOmega[1] * gradOmega[0].transpose()); // gradDiffx[7] = nextZVec[0] * Iinv * gradNextOmegaMat[1];
			hDiffx[7][7] = nextZVec[0] * Iinv * (gradOmega[1] * gradOmega[1].transpose() + gradOmega[1] * gradOmega[1].transpose()); // gradDiffx[7] = nextZVec[0] * Iinv * gradNextOmegaMat[1];



			
			hDiffy[1][2] = -Iinv * gradCurOmegaMat[0];	 // gradDiffy[1] = -Iinv * curOmegaMat;
			hDiffy[1][3] = -Iinv * gradCurOmegaMat[1];	 // gradDiffy[1] = -Iinv * curOmegaMat;

			hDiffy[2][1] = -Iinv * gradCurOmegaMat[0]; // gradDiffy[2] = -curZVec[1] * Iinv * gradCurOmegaMat[0];
			hDiffy[2][2] = -curZVec[1] * Iinv * (gradOmega[0] * gradOmega[0].transpose() + gradOmega[0] * gradOmega[0].transpose()); // gradDiffy[2] = -curZVec[1] * Iinv * gradCurOmegaMat[0];
			hDiffy[2][3] = -curZVec[1] * Iinv * (gradOmega[0] * gradOmega[1].transpose() + gradOmega[1] * gradOmega[0].transpose()); // gradDiffy[2] = -curZVec[1] * Iinv * gradCurOmegaMat[0];


			hDiffy[3][1] = -Iinv * gradCurOmegaMat[1]; // gradDiffy[3] = -curZVec[1] * Iinv * gradCurOmegaMat[1];
			hDiffy[3][2] = -curZVec[1] * Iinv * (gradOmega[1] * gradOmega[0].transpose() + gradOmega[1] * gradOmega[0].transpose());
			hDiffy[3][3] = -curZVec[1] * Iinv * (gradOmega[1] * gradOmega[1].transpose() + gradOmega[1] * gradOmega[1].transpose());

			
			hDiffy[5][6] = Iinv * gradNextOmegaMat[0];	// gradDiffy[5] = Iinv * nextOmegaMat;
			hDiffy[5][7] = Iinv * gradNextOmegaMat[1];	// gradDiffy[5] = Iinv * nextOmegaMat;

			hDiffy[6][5] = Iinv * gradNextOmegaMat[0];	// gradDiffy[6] = nextZVec[1] * Iinv * gradNextOmegaMat[0];
			hDiffy[6][6] = nextZVec[1] * Iinv * (gradOmega[0] * gradOmega[0].transpose() + gradOmega[0] * gradOmega[0].transpose()); // gradDiffy[6] = nextZVec[1] * Iinv * gradNextOmegaMat[0];
			hDiffy[6][7] = nextZVec[1] * Iinv * (gradOmega[0] * gradOmega[1].transpose() + gradOmega[1] * gradOmega[0].transpose()); // gradDiffy[6] = nextZVec[1] * Iinv * gradNextOmegaMat[0];

			hDiffy[7][5] = Iinv * gradNextOmegaMat[1]; // gradDiffy[7] = nextZVec[1] * Iinv * gradNextOmegaMat[1];
			hDiffy[7][6] = nextZVec[1] * Iinv * (gradOmega[1] * gradOmega[0].transpose() + gradOmega[1] * gradOmega[0].transpose()); // gradDiffy[7] = nextZVec[1] * Iinv * gradNextOmegaMat[1];
			hDiffy[7][7] = nextZVec[1] * Iinv * (gradOmega[1] * gradOmega[1].transpose() + gradOmega[1] * gradOmega[1].transpose()); // gradDiffy[7] = nextZVec[1] * Iinv * gradNextOmegaMat[1];

			for(int i = 0; i < 8; i++)
				for (int j = 0; j < 8; j++)
				{
					//(*deriv)[j] = coeff * ((gradDiffx[j].transpose() * diffx).trace() + (gradDiffy[j].transpose() * diffy).trace());
					 
					(*hess)(i, j) = coeff * (gradDiffx[i].transpose() * gradDiffx[j] + gradDiffy[i].transpose() * gradDiffy[j]).trace();
					(*hess)(i, j) += coeff * (hDiffx[i][j].transpose() * diffx + hDiffy[i][j].transpose() * diffy).trace();

					//(*hess)(i, j) += coeff * (hDiffy[i][j]).trace();
				}

			if (isProj)
				(*hess) = SPDProjection(*hess);

		}
	}
	return energy;
}

double WrinkleEditingCWFNewFullEnergy::kineticEnergy(int frameId, Eigen::VectorXd* deriv, std::vector<Eigen::Triplet<double>>* hessT, bool isProj)
{
	int nverts = _pos.rows();
	int nedges = _mesh.nEdges();
	int nfaces = _mesh.nFaces();

	double dt = 1. / (_zvalsList.size() - 1);
	double energy = 0;

	int DOFsPerframe = 2 * nverts + nedges;

	if (deriv)
		deriv->setZero(2 * DOFsPerframe);


	for (int fid = 0; fid < nfaces; fid++)
	{
		for (int vInF = 0; vInF < 3; vInF++)
		{
			int vid = _mesh.faceVertex(fid, vInF);
			int eid0 = _mesh.faceEdge(fid, (vInF + 1) % 3);
			int eid1 = _mesh.faceEdge(fid, (vInF + 2) % 3);

			Eigen::Matrix<double, 8, 1> dEnergy;
			Eigen::Matrix<double, 8, 8> hEnergy;
			energy += computeKineticEnergyPerFaceVertex(fid, vInF, frameId, dt, deriv ? &dEnergy : NULL, hessT ? &hEnergy : NULL, isProj);

			if (deriv)
			{

				for (int j = 0; j < 2; j++)
				{
					(*deriv)(2 * vid + j * DOFsPerframe) += dEnergy[0 + 4 * j];
					(*deriv)(2 * vid + 1 + j * DOFsPerframe) += dEnergy[1 + 4 * j];
					(*deriv)(eid0 + 2 * nverts + j * DOFsPerframe) += dEnergy[2 + 4 * j];
					(*deriv)(eid1 + 2 * nverts + j * DOFsPerframe) += dEnergy[3 + 4 * j];
				}
			}

			if (hessT)
			{
				for (int i = 0; i < 2; i++)
				{
					for (int j = 0; j < 2; j++)
					{
						hessT->push_back({ 2 * vid + i, 2 * vid + j, hEnergy(i, j) });
						hessT->push_back({ 2 * vid + i, 2 * vid + j + DOFsPerframe, hEnergy(i, 4 + j) });

						hessT->push_back({ 2 * vid + i + DOFsPerframe, 2 * vid + j, hEnergy(4 + i, j) });
						hessT->push_back({ 2 * vid + i + DOFsPerframe, 2 * vid + j + DOFsPerframe, hEnergy(4 + i, 4 + j) });


						hessT->push_back({ 2 * vid + i + j * DOFsPerframe, 2 * nverts + eid0, hEnergy(4 * j + i, 2)});
						hessT->push_back({ 2 * vid + i + j * DOFsPerframe, 2 * nverts + eid1, hEnergy(4 * j + i, 3) });
						hessT->push_back({ 2 * vid + i + j * DOFsPerframe, 2 * nverts + eid0 + DOFsPerframe, hEnergy(4 * j + i, 4 + 2) });
						hessT->push_back({ 2 * vid + i + j * DOFsPerframe, 2 * nverts + eid1 + DOFsPerframe, hEnergy(4 * j + i, 4 + 3) });

						hessT->push_back({ 2 * nverts + eid0, 2 * vid + i + j * DOFsPerframe, hEnergy(2, 4 * j + i) });
						hessT->push_back({ 2 * nverts + eid1, 2 * vid + i + j * DOFsPerframe, hEnergy(3, 4 * j + i) });
						hessT->push_back({ 2 * nverts + eid0 + DOFsPerframe, 2 * vid + i + j * DOFsPerframe, hEnergy(6, 4 * j + i) });
						hessT->push_back({ 2 * nverts + eid1 + DOFsPerframe, 2 * vid + i + j * DOFsPerframe,  hEnergy(7, 4 * j + i) });

						hessT->push_back({ 2 * nverts + eid0 + i * DOFsPerframe, 2 * nverts + eid0 + j * DOFsPerframe, hEnergy(4 * i + 2, 4 * j + 2) });
						hessT->push_back({ 2 * nverts + eid0 + i * DOFsPerframe, 2 * nverts + eid1 + j * DOFsPerframe, hEnergy(4 * i + 2, 4 * j + 3) });
						hessT->push_back({ 2 * nverts + eid1 + i * DOFsPerframe, 2 * nverts + eid0 + j * DOFsPerframe, hEnergy(4 * i + 3, 4 * j + 2) });
						hessT->push_back({ 2 * nverts + eid1 + i * DOFsPerframe, 2 * nverts + eid1 + j * DOFsPerframe, hEnergy(4 * i + 3, 4 * j + 3) });
					}				
				}
					
			}
		}
	}
	return energy;
}

double WrinkleEditingCWFNewFullEnergy::temporalAmpDifference(int frameId, Eigen::VectorXd* deriv, std::vector<Eigen::Triplet<double>>* hessT, bool isProj)
{
	int nverts = _pos.rows();
	double energy = 0;

	if (deriv)
		deriv->setZero(2 * nverts);
	if (hessT)
		hessT->clear();

	for (int vid = 0; vid < nverts; vid++)
	{
		double ampSq = _zvalsList[frameId][vid].real() * _zvalsList[frameId][vid].real() +
			_zvalsList[frameId][vid].imag() * _zvalsList[frameId][vid].imag();
		double refAmpSq = _combinedRefAmpList[frameId][vid] * _combinedRefAmpList[frameId][vid];
		double ca = _spatialAmpRatio * _vertArea(vid) / (_refAmpAveList[frameId] * _refAmpAveList[frameId]);

		energy += ca * (ampSq - refAmpSq) * (ampSq - refAmpSq);

		if (deriv)
		{
			(*deriv)(2 * vid) += 2.0 * ca * (ampSq - refAmpSq) * (2.0 * _zvalsList[frameId][vid].real());
			(*deriv)(2 * vid + 1) += 2.0 * ca * (ampSq - refAmpSq) * (2.0 * _zvalsList[frameId][vid].imag());
		}

		if (hessT)
		{
			Eigen::Matrix2d tmpHess;
			tmpHess << 
				2.0 * _zvalsList[frameId][vid].real() * 2.0 * _zvalsList[frameId][vid].real(),
				2.0 * _zvalsList[frameId][vid].real() * 2.0 * _zvalsList[frameId][vid].imag(),
				2.0 * _zvalsList[frameId][vid].real() * 2.0 * _zvalsList[frameId][vid].imag(),
				2.0 * _zvalsList[frameId][vid].imag() * 2.0 * _zvalsList[frameId][vid].imag();

			tmpHess *= 2.0 * ca;
			tmpHess += 2.0 * ca * (ampSq - refAmpSq) * (2.0 * Eigen::Matrix2d::Identity());


			if (isProj)
				tmpHess = SPDProjection(tmpHess);

			for (int k = 0; k < 2; k++)
				for (int l = 0; l < 2; l++)
					hessT->push_back({ 2 * vid + k, 2 * vid + l, tmpHess(k, l) });
		}
	}
	return energy;
}

double WrinkleEditingCWFNewFullEnergy::computeEnergy(const Eigen::VectorXd& x, Eigen::VectorXd* deriv, Eigen::SparseMatrix<double>* hess, bool isProj)
{
	int nverts = _pos.rows();
	int nedges = _mesh.nEdges();

	int numFrames = _zvalsList.size() - 2;

	int DOFsPerframe = 2 * nverts + nedges;

	int DOFs = numFrames * DOFsPerframe;

	convertVariable2List(x);

	Eigen::VectorXd curDeriv;
	std::vector<Eigen::Triplet<double>> T, curT;

	double energy = 0;
	if (deriv)
	{
		deriv->setZero(DOFs);
	}
	
	std::vector<Eigen::VectorXd> curKDerivList(numFrames + 1);
	std::vector<std::vector<Eigen::Triplet<double>>> curKTList(numFrames + 1);
	std::vector<double> keList(numFrames + 1);

	auto kineticEnergyPerframe = [&](const tbb::blocked_range<uint32_t>& range) 
	{
		for (uint32_t i = range.begin(); i < range.end(); ++i)
		{
			keList[i] = kineticEnergy(i, deriv ? &curKDerivList[i] : NULL, hess ? &curKTList[i] : NULL, isProj);
		}
	};

	tbb::blocked_range<uint32_t> rangex(0u, (uint32_t)numFrames + 1, GRAIN_SIZE);
	tbb::parallel_for(rangex, kineticEnergyPerframe);


	for (int i = 0; i < _zvalsList.size() - 1; i++)
	{
		energy += keList[i];

		if (deriv)
		{
			if (i == 0)
				deriv->segment(0, DOFsPerframe) += curKDerivList[i].segment(DOFsPerframe, DOFsPerframe);
			else if (i == _zvalsList.size() - 2)
				deriv->segment((i - 1) * DOFsPerframe, DOFsPerframe) += curKDerivList[i].segment(0, DOFsPerframe);
			else
			{
				deriv->segment((i - 1) * DOFsPerframe, 2 * DOFsPerframe) += curKDerivList[i];
			}
		}

		if (hess)
		{
			for (auto& it : curKTList[i])
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
		}
	}

	std::vector<Eigen::VectorXd> ampDerivList(numFrames), knoppelDerivList(numFrames);
	std::vector<std::vector<Eigen::Triplet<double>>> ampTList(numFrames), knoppelTList(numFrames);
	std::vector<double> ampEnergyList(numFrames), knoppelEnergyList(numFrames);

	auto otherEnergiesPerframe = [&](const tbb::blocked_range<uint32_t>& range)
	{
		for (uint32_t i = range.begin(); i < range.end(); ++i)
		{
			ampEnergyList[i] = temporalAmpDifference(i + 1, deriv ? &ampDerivList[i] : NULL, hess ? &ampTList[i] : NULL, isProj);
			knoppelEnergyList[i] = spatialKnoppelEnergy(i + 1, deriv ? &knoppelDerivList[i] : NULL, hess ? &knoppelTList[i] : NULL, isProj);
		}
	};

	tbb::blocked_range<uint32_t> rangex1(0u, (uint32_t)numFrames, GRAIN_SIZE);
	tbb::parallel_for(rangex1, otherEnergiesPerframe);
	

	for (int i = 0; i < numFrames; i++)
	{
		// energy += ampEnergyList[i];
		energy += knoppelEnergyList[i];

		if (deriv) 
		{
			deriv->segment(i * DOFsPerframe, DOFsPerframe) += knoppelDerivList[i];
			deriv->segment(i * DOFsPerframe, 2 * nverts) += ampDerivList[i];
		}

		if (hess) 
		{
			for (auto& it : ampTList[i])
			{
				T.push_back({ i * DOFsPerframe + it.row(), i * DOFsPerframe + it.col(), it.value() });
			}
			for (auto& it : knoppelTList[i])
			{
				T.push_back({ i * DOFsPerframe + it.row(), i * DOFsPerframe + it.col(), it.value() });
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

void WrinkleEditingCWFNewFullEnergy::solveIntermeditateFrames(Eigen::VectorXd& x, int numIter, double gradTol, double xTol, double fTol, bool isdisplayInfo, std::string workingFolder)
{
	std::cout << "New CWF model: " << std::endl;
	auto funVal = [&](const Eigen::VectorXd& x, Eigen::VectorXd* grad, Eigen::SparseMatrix<double>* hess, bool isProj) {
		Eigen::VectorXd deriv;
		Eigen::SparseMatrix<double> H;
		double E = computeEnergy(x, grad ? &deriv : NULL, hess ? &H : NULL, isProj);

		if (grad)
		{
			(*grad) = deriv;
		}

		if (hess)
		{
			(*hess) = H;
		}

		return E;
	};
	auto maxStep = [&](const Eigen::VectorXd& x, const Eigen::VectorXd& dir) {
		return 1.0;
	};

	auto getVecNorm = [&](const Eigen::VectorXd& x, double& znorm, double& wnorm) {
		getComponentNorm(x, znorm, wnorm);
	};
	auto saveTmpRes = [&](const Eigen::VectorXd& x, std::string* folder)
	{
		save(x, folder);
	};

	OptSolver::testFuncGradHessian(funVal, x);

	auto x0 = x;
	Eigen::VectorXd grad;
	Eigen::SparseMatrix<double> hess;
	double f0 = funVal(x0, &grad, &hess, false);
	std::cout << "initial f: " << f0 << ", grad norm: " << grad.norm() << ", hess norm: " << hess.norm() << std::endl;
	OptSolver::newtonSolver(funVal, maxStep, x, 100, gradTol, std::max(1e-16, xTol), std::max(1e-16, fTol), true, getVecNorm, &workingFolder, saveTmpRes);
	std::cout << "before optimization: " << x0.norm() << ", after optimization: " << x.norm() << std::endl;
	std::cout << "solve finished." << std::endl;

	int numFrames = _zvalsList.size() - 2;

	convertVariable2List(x);

	Eigen::VectorXd curDeriv;
	std::vector<Eigen::Triplet<double>> T, curT;

	std::vector<double> keList(numFrames + 1);

	auto kineticEnergyPerframe = [&](const tbb::blocked_range<uint32_t>& range) 
	{
		for (uint32_t i = range.begin(); i < range.end(); ++i)
		{
			keList[i] = kineticEnergy(i, NULL, NULL, false);
		}
	};

	tbb::blocked_range<uint32_t> rangex(0u, (uint32_t)numFrames + 1, GRAIN_SIZE);
	tbb::parallel_for(rangex, kineticEnergyPerframe);

	for(int i = 0; i < numFrames + 1; i++)
	{
		std::cout << "frame: " << i << ", kinetic: " << keList[i] << std::endl;
	}

}

void WrinkleEditingCWFNewFullEnergy::testKineticEnergyPerFaceVertex(int fid, int vInF, int frameId, double dt)
{
	Eigen::Matrix<double, 8, 1> deriv, deriv1;
	Eigen::Matrix<double, 8, 8> hess;

	double e0 = computeKineticEnergyPerFaceVertex(fid, vInF, frameId, dt, &deriv, &hess, false);

	int vid = _mesh.faceVertex(fid, vInF);
	int eid0 = _mesh.faceEdge(fid, (vInF + 1) % 3);
	int eid1 = _mesh.faceEdge(fid, (vInF + 2) % 3);

	std::complex<double> backupcurZvals = _zvalsList[frameId][vid];
	std::complex<double> backupnextZvals = _zvalsList[frameId + 1][vid];

	double backupCurW0 = _edgeOmegaList[frameId][eid0];
	double backupCurW1 = _edgeOmegaList[frameId][eid1];
	double backupNextW0 = _edgeOmegaList[frameId + 1][eid0];
	double backupNextW1 = _edgeOmegaList[frameId + 1][eid1];

	Eigen::Matrix<double, 8, 1> dir;
	dir.setRandom();

	for (int i = 3; i < 9; i++)
	{
		double eps = std::pow(0.1, i);
		_zvalsList[frameId][vid] = std::complex<double>(backupcurZvals.real() + eps * dir(0), backupcurZvals.imag() + eps * dir(1));
		_edgeOmegaList[frameId][eid0] = backupCurW0 + eps * dir(2);
		_edgeOmegaList[frameId][eid1] = backupCurW1 + eps * dir(3);

		_zvalsList[frameId + 1][vid] = std::complex<double>(backupnextZvals.real() + eps * dir(4), backupnextZvals.imag() + eps * dir(5));
		_edgeOmegaList[frameId + 1][eid0] = backupNextW0 + eps * dir(6);
		_edgeOmegaList[frameId + 1][eid1] = backupNextW1 + eps * dir(7);


		double e1 = computeKineticEnergyPerFaceVertex(fid, vInF, frameId, 1, &deriv1, NULL, false);
		std::cout << "eps: " << eps << std::endl;
		std::cout << "energy-gradient: " << (e1 - e0) / eps - deriv.dot(dir) << std::endl;
		std::cout << "gradient-hess: " << ((deriv1 - deriv) / eps - hess * dir).norm() << std::endl;

	}


}