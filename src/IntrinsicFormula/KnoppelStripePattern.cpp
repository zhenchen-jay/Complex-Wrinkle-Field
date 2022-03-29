#include "../../include/IntrinsicFormula/KnoppelStripePattern.h"
#include "../../include/IntrinsicFormula/AmpSolver.h"
#include "../../include/Optimization/NewtonDescent.h"
#include <SymGEigsShiftSolver.h>
#include <MatOp/SparseCholesky.h>
#include <Eigen/CholmodSupport>
#include <MatOp/SparseSymShiftSolve.h>
#include <iostream>

using namespace IntrinsicFormula;

void IntrinsicFormula::computeMatrixA(const MeshConnectivity &mesh, const Eigen::MatrixXd &halfEdgeW,
									 const Eigen::VectorXd &faceArea, const Eigen::MatrixXd &cotEntries,
									 const int nverts, Eigen::SparseMatrix<double> &A)
{
	std::vector<Eigen::Triplet<double>> AT;
	int nfaces = mesh.nFaces();
	int nedges = mesh.nEdges();

	Eigen::VectorXd halfEdgeWeight(nedges);
	halfEdgeWeight.setConstant(1.0);

//    for (int i = 0; i < nfaces; i++)  // form mass matrix
//    {
//        for (int j = 0; j < 3; j++)
//        {
//            int eid = mesh.faceEdge(i, j);
//            halfEdgeWeight(eid) += cotEntries(i, j);
//        }
//    }
	for(int i = 0; i < nedges; i++)
	{
		int vid0 = mesh.edgeVertex(i, 0);
		int vid1 = mesh.edgeVertex(i, 1);

		AT.push_back({2 * vid0, 2 * vid0, 2 * halfEdgeWeight(i)});
		AT.push_back({2 * vid0 + 1, 2 * vid0 + 1, 2 * halfEdgeWeight(i)});

		AT.push_back({2 * vid1, 2 * vid1, 2 * halfEdgeWeight(i)});
		AT.push_back({2 * vid1 + 1, 2 * vid1 + 1, 2 * halfEdgeWeight(i)});

		std::complex<double> expw0 = std::complex<double>(std::cos(halfEdgeW(i, 0)), std::sin(halfEdgeW(i, 0)));
		std::complex<double> expw1 = std::complex<double>(std::cos(halfEdgeW(i, 1)), std::sin(halfEdgeW(i, 1)));

		AT.push_back({2 * vid0, 2 * vid1, -halfEdgeWeight(i) * (expw0.real() + expw1.real())});
		AT.push_back({2 * vid0 + 1, 2 * vid1, -halfEdgeWeight(i) * (-expw0.imag() + expw1.imag())});
		AT.push_back({2 * vid0, 2 * vid1 + 1, -halfEdgeWeight(i) * (expw0.imag() - expw1.imag())});
		AT.push_back({2 * vid0 + 1, 2 * vid1 + 1, -halfEdgeWeight(i) * (expw0.real() + expw1.real())});

		AT.push_back({ 2 * vid1, 2 * vid0, -halfEdgeWeight(i) * (expw0.real() + expw1.real()) });
		AT.push_back({ 2 * vid1, 2 * vid0 + 1, -halfEdgeWeight(i) * (-expw0.imag() + expw1.imag()) });
		AT.push_back({ 2 * vid1 + 1, 2 * vid0, -halfEdgeWeight(i) * (expw0.imag() - expw1.imag()) });
		AT.push_back({ 2 * vid1 + 1, 2 * vid0 + 1, -halfEdgeWeight(i) * (expw0.real() + expw1.real()) });

	}
	A.resize(2 * nverts, 2 * nverts);
	A.setFromTriplets(AT.begin(), AT.end());
}

void IntrinsicFormula::computeMatrixAGivenMag(const MeshConnectivity &mesh, const Eigen::MatrixXd &halfEdgeW, const Eigen::VectorXd& vertAmp, const Eigen::VectorXd& faceArea, const Eigen::MatrixXd& cotEntries, const int nverts, Eigen::SparseMatrix<double>& A) {
	std::vector<Eigen::Triplet<double>> AT;
	int nfaces = mesh.nFaces();
	int nedges = mesh.nEdges();

	Eigen::VectorXd halfEdgeWeight(nedges);
	halfEdgeWeight.setConstant(1.0);

//    for (int i = 0; i < nfaces; i++)  // form mass matrix
//    {
//        for (int j = 0; j < 3; j++)
//        {
//            int eid = mesh.faceEdge(i, j);
//            halfEdgeWeight(eid) += cotEntries(i, j);
//        }
//    }

	for (int i = 0; i < nedges; i++) {
		int vid0 = mesh.edgeVertex(i, 0);
		int vid1 = mesh.edgeVertex(i, 1);

		double r0 = vertAmp(vid0);
		double r1 = vertAmp(vid1);

		std::complex<double> expw0 = std::complex<double>(std::cos(halfEdgeW(i, 0)), std::sin(halfEdgeW(i, 0)));
		std::complex<double> expw1 = std::complex<double>(std::cos(halfEdgeW(i, 1)), std::sin(halfEdgeW(i, 1)));


		AT.push_back({2 * vid0, 2 * vid0, 2 * r1 * r1 * halfEdgeWeight(i)});
		AT.push_back({2 * vid0 + 1, 2 * vid0 + 1, 2 * r1 * r1 * halfEdgeWeight(i)});

		AT.push_back({2 * vid1, 2 * vid1, 2 * r0 * r0 * halfEdgeWeight(i)});
		AT.push_back({2 * vid1 + 1, 2 * vid1 + 1, 2 * r0 * r0 * halfEdgeWeight(i)});


		AT.push_back({2 * vid0, 2 * vid1, -halfEdgeWeight(i) * (expw0.real() + expw1.real()) * r0 * r1});
		AT.push_back({2 * vid0 + 1, 2 * vid1, -halfEdgeWeight(i) * (-expw0.imag() + expw1.imag()) * r0 * r1});
		AT.push_back({2 * vid0, 2 * vid1 + 1, -halfEdgeWeight(i) * (expw0.imag() - expw1.imag()) * r0 * r1});
		AT.push_back({2 * vid0 + 1, 2 * vid1 + 1, -halfEdgeWeight(i) * (expw0.real() + expw1.real()) * r0 * r1});

		AT.push_back({2 * vid1, 2 * vid0, -halfEdgeWeight(i) * (expw0.real() + expw1.real()) * r0 * r1});
		AT.push_back({2 * vid1, 2 * vid0 + 1, -halfEdgeWeight(i) * (-expw0.imag() + expw1.imag()) * r0 * r1});
		AT.push_back({2 * vid1 + 1, 2 * vid0, -halfEdgeWeight(i) * (expw0.imag() - expw1.imag()) * r0 * r1});
		AT.push_back({2 * vid1 + 1, 2 * vid0 + 1, -halfEdgeWeight(i) * (expw0.real() + expw1.real()) * r0 * r1});
	}
	A.resize(2 * nverts, 2 * nverts);
	A.setFromTriplets(AT.begin(), AT.end());
}

double IntrinsicFormula::KnoppelEnergy(const MeshConnectivity& mesh, const Eigen::MatrixXd& halfEdgeW, const Eigen::VectorXd& faceArea, const Eigen::MatrixXd& cotEntries, const std::vector<std::complex<double>>& zvals, Eigen::VectorXd* deriv, std::vector<Eigen::Triplet<double>>* hess)
{
	std::vector<Eigen::Triplet<double>> AT;
	int nfaces = mesh.nFaces();
	int nedges = mesh.nEdges();
	int nverts = zvals.size();

	Eigen::VectorXd halfEdgeWeight(nedges);
	halfEdgeWeight.setConstant(1.0);

//    for (int i = 0; i < nfaces; i++)  // form mass matrix
//    {
//        for (int j = 0; j < 3; j++)
//        {
//            int eid = mesh.faceEdge(i, j);
//            halfEdgeWeight(eid) += cotEntries(i, j);
//        }
//    }
	double energy = 0;
	
	for (int i = 0; i < nedges; i++)
	{
		int vid0 = mesh.edgeVertex(i, 0);
		int vid1 = mesh.edgeVertex(i, 1);

		std::complex<double> expw0 = std::complex<double>(std::cos(halfEdgeW(i, 0)), std::sin(halfEdgeW(i, 0)));
		std::complex<double> expw1 = std::complex<double>(std::cos(halfEdgeW(i, 1)), std::sin(halfEdgeW(i, 1)));

		std::complex<double> z0 = zvals[vid0];
		std::complex<double> z1 = zvals[vid1];


		energy += 0.5 * (norm((z0 * expw0 - z1)) + norm((z1 * expw1 - z0))) * halfEdgeWeight(i);

		if (deriv || hess)
		{
			AT.push_back({ 2 * vid0, 2 * vid0, 2 * halfEdgeWeight(i) });
			AT.push_back({ 2 * vid0 + 1, 2 * vid0 + 1, 2 * halfEdgeWeight(i) });

			AT.push_back({ 2 * vid1, 2 * vid1, 2 * halfEdgeWeight(i) });
			AT.push_back({ 2 * vid1 + 1, 2 * vid1 + 1, 2 * halfEdgeWeight(i) });


			AT.push_back({ 2 * vid0, 2 * vid1, -halfEdgeWeight(i) * (expw0.real() + expw1.real()) });
			AT.push_back({ 2 * vid0 + 1, 2 * vid1, -halfEdgeWeight(i) * (-expw0.imag() + expw1.imag()) });
			AT.push_back({ 2 * vid0, 2 * vid1 + 1, -halfEdgeWeight(i) * (expw0.imag() - expw1.imag()) });
			AT.push_back({ 2 * vid0 + 1, 2 * vid1 + 1, -halfEdgeWeight(i) * (expw0.real() + expw1.real()) });

			AT.push_back({ 2 * vid1, 2 * vid0, -halfEdgeWeight(i) * (expw0.real() + expw1.real()) });
			AT.push_back({ 2 * vid1, 2 * vid0 + 1, -halfEdgeWeight(i) * (-expw0.imag() + expw1.imag()) });
			AT.push_back({ 2 * vid1 + 1, 2 * vid0, -halfEdgeWeight(i) * (expw0.imag() - expw1.imag()) });
			AT.push_back({ 2 * vid1 + 1, 2 * vid0 + 1, -halfEdgeWeight(i) * (expw0.real() + expw1.real()) });
		}
	}

	if (deriv || hess)
	{
		Eigen::SparseMatrix<double> A;

		A.resize(2 * nverts, 2 * nverts);
		A.setFromTriplets(AT.begin(), AT.end());

		if (deriv)
		{
			Eigen::VectorXd fvals(2 * nverts);
			for (int i = 0; i < nverts; i++)
			{
				fvals(2 * i) = zvals[i].real();
				fvals(2 * i + 1) = zvals[i].imag();
			}
			(*deriv) = A * fvals;
		}
		   
		if (hess)
			(*hess) = AT;
	}

	return energy;
}

double IntrinsicFormula::KnoppelEnergyGivenMag(const MeshConnectivity& mesh, const Eigen::MatrixXd& halfEdgeW, const Eigen::VectorXd& vertAmp, const Eigen::VectorXd& faceArea, const Eigen::MatrixXd& cotEntries, const std::vector<std::complex<double>>& zvals, Eigen::VectorXd* deriv, std::vector<Eigen::Triplet<double>>* hess)
{
	std::vector<Eigen::Triplet<double>> AT;
	int nfaces = mesh.nFaces();
	int nedges = mesh.nEdges();
	int nverts = vertAmp.size();

	Eigen::VectorXd halfEdgeWeight(nedges);
	halfEdgeWeight.setConstant(1.0);

//    for (int i = 0; i < nfaces; i++)  // form mass matrix
//    {
//        for (int j = 0; j < 3; j++)
//        {
//            int eid = mesh.faceEdge(i, j);
//            halfEdgeWeight(eid) += cotEntries(i, j);
//        }
//    }
	double energy = 0;

	for (int i = 0; i < nedges; i++)
	{
		int vid0 = mesh.edgeVertex(i, 0);
		int vid1 = mesh.edgeVertex(i, 1);

		double r0 = vertAmp(vid0);
		double r1 = vertAmp(vid1);

		std::complex<double> expw0 = std::complex<double>(std::cos(halfEdgeW(i, 0)), std::sin(halfEdgeW(i, 0)));
		std::complex<double> expw1 = std::complex<double>(std::cos(halfEdgeW(i, 1)), std::sin(halfEdgeW(i, 1)));

		std::complex<double> z0 = zvals[vid0];
		std::complex<double> z1 = zvals[vid1];


		energy += 0.5 * (norm((r1 * z0 * expw0 - r0 * z1)) + norm((r0 * z1 * expw1 - r1 * z0))) * halfEdgeWeight(i);

		if (deriv || hess)
		{
			AT.push_back({ 2 * vid0, 2 * vid0, 2 * r1 * r1 * halfEdgeWeight(i) });
			AT.push_back({ 2 * vid0 + 1, 2 * vid0 + 1, 2 * r1 * r1 * halfEdgeWeight(i) });

			AT.push_back({ 2 * vid1, 2 * vid1, 2 * r0 * r0 * halfEdgeWeight(i) });
			AT.push_back({ 2 * vid1 + 1, 2 * vid1 + 1, 2 * r0 * r0 * halfEdgeWeight(i) });


			AT.push_back({ 2 * vid0, 2 * vid1, -halfEdgeWeight(i) * (expw0.real() + expw1.real()) * r0 * r1 });
			AT.push_back({ 2 * vid0 + 1, 2 * vid1, -halfEdgeWeight(i) * (-expw0.imag() + expw1.imag()) * r0 * r1 });
			AT.push_back({ 2 * vid0, 2 * vid1 + 1, -halfEdgeWeight(i) * (expw0.imag() - expw1.imag()) * r0 * r1 });
			AT.push_back({ 2 * vid0 + 1, 2 * vid1 + 1, -halfEdgeWeight(i) * (expw0.real() + expw1.real()) * r0 * r1 });

			AT.push_back({ 2 * vid1, 2 * vid0, -halfEdgeWeight(i) * (expw0.real() + expw1.real()) * r0 * r1 });
			AT.push_back({ 2 * vid1, 2 * vid0 + 1, -halfEdgeWeight(i) * (-expw0.imag() + expw1.imag()) * r0 * r1 });
			AT.push_back({ 2 * vid1 + 1, 2 * vid0, -halfEdgeWeight(i) * (expw0.imag() - expw1.imag()) * r0 * r1 });
			AT.push_back({ 2 * vid1 + 1, 2 * vid0 + 1, -halfEdgeWeight(i) * (expw0.real() + expw1.real()) * r0 * r1 });
		}
	}

	if (deriv || hess)
	{
		Eigen::SparseMatrix<double> A;

		A.resize(2 * nverts, 2 * nverts);
		A.setFromTriplets(AT.begin(), AT.end());

		// check whether A is PD


		if (deriv)
		{
			Eigen::VectorXd fvals(2 * nverts);
			for (int i = 0; i < nverts; i++)
			{
				fvals(2 * i) = zvals[i].real();
				fvals(2 * i + 1) = zvals[i].imag();
			}
			(*deriv) = A * fvals;
		}

		if (hess)
			(*hess) = AT;
	}

	return energy;
}

double IntrinsicFormula::KnoppelEnergyFor2DVertexOmegaPerEdge(const Eigen::MatrixXd& pos, const MeshConnectivity& mesh, const Eigen::VectorXd& faceArea, const Eigen::MatrixXd& cotEntries, const std::vector<std::complex<double>>& zvals, const Eigen::MatrixXd& vertexOmega, const double edgeWeight, int eid, Eigen::Matrix<double, 8, 1>* deriv, Eigen::Matrix<double, 8, 8>* hess, bool isProj)
{
	int vid0 = mesh.edgeVertex(eid, 0);
	int vid1 = mesh.edgeVertex(eid, 1);

	Eigen::Vector2d e = (pos.row(vid1) - pos.row(vid0)).segment<2>(0);

	double w01 = vertexOmega.row(vid0).dot(e);
	double w10 = -vertexOmega.row(vid1).dot(e);

	double sin0 = std::sin(w01), cos0 = std::cos(w01);
	double sin1 = std::sin(w10), cos1 = std::cos(w10);


	Eigen::Vector2d z0(zvals[vid0].real(), zvals[vid0].imag());
	Eigen::Vector2d z1(zvals[vid1].real(), zvals[vid1].imag());

	Eigen::Matrix2d mat0, mat1;
	mat0 << cos0, -sin0, sin0, cos0;
	mat1 << cos1, -sin1, sin1, cos1;

	Eigen::Vector2d f0, f1;
	f0 = mat0 * z0 - z1;
	f1 = mat1 * z1 - z0;

	double energy = 0.5 * (f0.dot(f0) + f1.dot(f1)) * edgeWeight;

	if (deriv)
		deriv->setZero();

	if (deriv || hess)
	{
		Eigen::Vector2d e0, e1;
		e0 << 1, 0;
		e1 << 0, 1;

		Eigen::Matrix2d dmat0, dmat1;
		dmat0 << -sin0, -cos0, cos0, -sin0;
		dmat1 << -sin1, -cos1, cos1, -sin1;

		Eigen::Matrix<double, 2, 8> gradF0, gradF1;
		gradF0.setZero();
		gradF1.setZero();

		gradF0.col(0) = mat0 * e0;
		gradF0.col(1) = mat0 * e1;
		gradF0.col(2) = -e0;
		gradF0.col(3) = -e1;

		gradF0.col(4) = dmat0 * z0 * e(0);
		gradF0.col(5) = dmat0 * z0 * e(1);

		gradF1.col(0) = -e0;
		gradF1.col(1) = -e1;
		gradF1.col(2) = mat1 * e0;
		gradF1.col(3) = mat1 * e1;

		gradF1.col(6) = -dmat1 * z1 * e(0);
		gradF1.col(7) = -dmat1 * z1 * e(1);

		if (deriv)
			*deriv = (f0.transpose() * gradF0 + f1.transpose() * gradF1) * edgeWeight;

		if (hess)
		{
			*hess = (gradF0.transpose() * gradF0 + gradF1.transpose() * gradF1) * edgeWeight;

			
			(*hess)(0, 4) += edgeWeight * (f0(0) * -sin0 * e(0) + f0(1) * cos0 * e(0));
			(*hess)(0, 5) += edgeWeight * (f0(0) * -sin0 * e(1) + f0(1) * cos0 * e(1));

			(*hess)(1, 4) += edgeWeight * (f0(0) * -cos0 * e(0) + f0(1) * -sin0 * e(0));
			(*hess)(1, 5) += edgeWeight * (f0(0) * -cos0 * e(1) + f0(1) * -sin0 * e(1));

			(*hess)(2, 6) += edgeWeight * (f1(0) * sin1 * e(0) - f1(1) * cos1 * e(0));
			(*hess)(2, 7) += edgeWeight * (f1(0) * sin1 * e(1) - f1(1) * cos1 * e(1));

			(*hess)(3, 6) += edgeWeight * (f1(0) * cos1 * e(0) + f1(1) * sin1 * e(0));
			(*hess)(3, 7) += edgeWeight * (f1(0) * cos1 * e(1) + f1(1) * sin1 * e(1));

			(*hess)(4, 0) = (*hess)(0, 4);
			(*hess)(4, 1) = (*hess)(1, 4);
			(*hess)(4, 4) += edgeWeight * (f0(0) * (-z0(0) * cos0 + z0(1) * sin0) + f0(1) * (-z0(0) * sin0 - z0(1) * cos0)) * e(0) * e(0);
			(*hess)(4, 5) += edgeWeight * (f0(0) * (-z0(0) * cos0 + z0(1) * sin0) + f0(1) * (-z0(0) * sin0 - z0(1) * cos0)) * e(0) * e(1);


			(*hess)(5, 0) = (*hess)(0, 5);
			(*hess)(5, 1) = (*hess)(1, 5);
			(*hess)(5, 4) += edgeWeight * (f0(0) * (-z0(0) * cos0 + z0(1) * sin0) + f0(1) * (-z0(0) * sin0 - z0(1) * cos0)) * e(1) * e(0);
			(*hess)(5, 5) += edgeWeight * (f0(0) * (-z0(0) * cos0 + z0(1) * sin0) + f0(1) * (-z0(0) * sin0 - z0(1) * cos0)) * e(1) * e(1);


			(*hess)(6, 2) = (*hess)(2, 6);
			(*hess)(6, 3) = (*hess)(3, 6);
			(*hess)(6, 6) += edgeWeight * (f1(0) * (-z1(0) * cos1 + z1(1) * sin1) + f1(1) * (-z1(0) * sin1 - z1(1) * cos1)) * e(0) * e(0);
			(*hess)(6, 7) += edgeWeight * (f1(0) * (-z1(0) * cos1 + z1(1) * sin1) + f1(1) * (-z1(0) * sin1 - z1(1) * cos1)) * e(0) * e(1);


			(*hess)(7, 2) = (*hess)(2, 7);
			(*hess)(7, 3) = (*hess)(3, 7);
			(*hess)(7, 6) += edgeWeight * (f1(0) * (-z1(0) * cos1 + z1(1) * sin1) + f1(1) * (-z1(0) * sin1 - z1(1) * cos1)) * e(1) * e(0);
			(*hess)(7, 7) += edgeWeight * (f1(0) * (-z1(0) * cos1 + z1(1) * sin1) + f1(1) * (-z1(0) * sin1 - z1(1) * cos1)) * e(1) * e(1);
			

			if (isProj)
				(*hess) = SPDProjection(*hess);

		}

	}

	return energy;
}

double IntrinsicFormula::KnoppelEnergyFor2DVertexOmega(const Eigen::MatrixXd& pos, const MeshConnectivity& mesh, const Eigen::VectorXd& faceArea, const Eigen::MatrixXd& cotEntries, const std::vector<std::complex<double>>& zvals, const Eigen::MatrixXd& vertexOmega, Eigen::VectorXd* deriv, std::vector<Eigen::Triplet<double>>* hess, bool isProj)
{
	std::vector<Eigen::Triplet<double>> AT;
	int nfaces = mesh.nFaces();
	int nedges = mesh.nEdges();
	int nverts = zvals.size();

	Eigen::VectorXd edgeWeight(nedges);
	edgeWeight.setZero();

	for (int i = 0; i < nfaces; i++)  // form mass matrix
	{
		for (int j = 0; j < 3; j++)
		{
			int eid = mesh.faceEdge(i, j);
			edgeWeight(eid) += cotEntries(i, j);
		}
	}
	double energy = 0;
	edgeWeight.setConstant(1.0);

	if (deriv)
	{
		deriv->setZero(4 * nverts);
	}
	
	for (int i = 0; i < nedges; i++)
	{
		Eigen::Matrix<double, 8, 1> edgeDeriv;
		Eigen::Matrix<double, 8, 8> edgeHess;

		energy += KnoppelEnergyFor2DVertexOmegaPerEdge(pos, mesh, faceArea, cotEntries, zvals, vertexOmega, edgeWeight(i), i, deriv ? &edgeDeriv : NULL, hess ? &edgeHess : NULL, isProj);


		int vid0 = mesh.edgeVertex(i, 0);
		int vid1 = mesh.edgeVertex(i, 1);

		if (deriv)
		{
			deriv->segment<2>(2 * vid0) += edgeDeriv.segment<2>(0);
			deriv->segment<2>(2 * vid1) += edgeDeriv.segment<2>(2);

			deriv->segment<2>(2 * vid0 + 2 * nverts) += edgeDeriv.segment<2>(4);
			deriv->segment<2>(2 * vid1 + 2 * nverts) += edgeDeriv.segment<2>(6);

		}

		if (hess)
		{
			for(int m = 0; m < 2; m++)
				for (int n = 0; n < 2; n++)
				{
					hess->push_back({ 2 * vid0 + m , 2 * vid0 + n, edgeHess(m, n) });
					hess->push_back({ 2 * vid0 + m , 2 * vid1 + n, edgeHess(m, 2 + n) });
					hess->push_back({ 2 * vid0 + m , 2 * vid0 + 2 * nverts + n, edgeHess(m, 4 + n) });
					hess->push_back({ 2 * vid0 + m , 2 * vid1 + 2 * nverts + n, edgeHess(m, 6 + n) });


					hess->push_back({ 2 * vid1 + m , 2 * vid0 + n, edgeHess(2 + m, n) });
					hess->push_back({ 2 * vid1 + m , 2 * vid1 + n, edgeHess(2 + m, 2 + n) });
					hess->push_back({ 2 * vid1 + m , 2 * vid0 + 2 * nverts + n, edgeHess(2 + m, 4 + n) });
					hess->push_back({ 2 * vid1 + m , 2 * vid1 + 2 * nverts + n, edgeHess(2 + m, 6 + n) });

					hess->push_back({ 2 * vid0 + 2 * nverts + m , 2 * vid0 + n, edgeHess(4 + m, n) });
					hess->push_back({ 2 * vid0 + 2 * nverts + m , 2 * vid1 + n, edgeHess(4 + m, 2 + n) });
					hess->push_back({ 2 * vid0 + 2 * nverts + m , 2 * vid0 + 2 * nverts + n, edgeHess(4 + m, 4 + n) });
					hess->push_back({ 2 * vid0 + 2 * nverts + m , 2 * vid1 + 2 * nverts + n, edgeHess(4 + m, 6 + n) });

					hess->push_back({ 2 * vid1 + 2 * nverts + m , 2 * vid0 + n, edgeHess(6 + m, n) });
					hess->push_back({ 2 * vid1 + 2 * nverts + m , 2 * vid1 + n, edgeHess(6 + m, 2 + n) });
					hess->push_back({ 2 * vid1 + 2 * nverts + m , 2 * vid0 + 2 * nverts + n, edgeHess(6 + m, 4 + n) });
					hess->push_back({ 2 * vid1 + 2 * nverts + m , 2 * vid1 + 2 * nverts + n, edgeHess(6 + m, 6 + n) });

				}
			
		}
	}
	return energy;
}

void IntrinsicFormula::roundVertexZvalsFromHalfEdgeOmega(const MeshConnectivity &mesh, const Eigen::MatrixXd &halfEdgeW,
														 const Eigen::VectorXd &faceArea,
														 const Eigen::MatrixXd &cotEntries, const int nverts,
														 std::vector<std::complex<double>> &zvals)
{
	std::vector<Eigen::Triplet<double>> BT;
	int nfaces = mesh.nFaces();
	int nedges = mesh.nEdges();
	
	for(int i = 0; i < nfaces; i++)  // form mass matrix
	{
		for(int j =0; j < 3; j++)
		{
			int vid = mesh.faceVertex(i, j);
			BT.push_back({2 * vid, 2 * vid, faceArea(i) / 3.0});
			BT.push_back({2 * vid + 1, 2 * vid + 1, faceArea(i) / 3.0});
		}
	}
	Eigen::SparseMatrix<double> A;
	computeMatrixA(mesh, halfEdgeW, faceArea, cotEntries, nverts, A);

	Eigen::SparseMatrix<double> B(2 * nverts, 2 * nverts);
	B.setFromTriplets(BT.begin(), BT.end());
   /* std::cout << A.toDense() << std::endl;
	std::cout << B.toDense() << std::endl;*/

	Spectra::SymShiftInvert<double> op(A, B);
	Spectra::SparseSymMatProd<double> Bop(B);
	Spectra::SymGEigsShiftSolver<Spectra::SymShiftInvert<double>, Spectra::SparseSymMatProd<double>, Spectra::GEigsMode::ShiftInvert> geigs(op, Bop, 1, 6, -1e-6);
	geigs.init();
	int nconv = geigs.compute(Spectra::SortRule::LargestMagn, 1e6);

	Eigen::VectorXd evalues;
	Eigen::MatrixXd evecs;

	evalues = geigs.eigenvalues();
	evecs = geigs.eigenvectors();
	if (nconv != 1 || geigs.info() != Spectra::CompInfo::Successful)
	{
		std::cout << "Eigensolver failed to converge!!" << std::endl;
	}

	std::cout << "Eigenvalue is " << evalues[0] << std::endl;

	zvals.clear();
	for(int i = 0; i < nverts; i++)
	{
		zvals.push_back(std::complex<double>(evecs(2 * i, 0), evecs(2 * i + 1, 0)));
	}
}

void IntrinsicFormula::roundVertexZvalsFromHalfEdgeOmegaVertexMag(const MeshConnectivity &mesh, const Eigen::MatrixXd& halfEdgeW, const Eigen::VectorXd& vertAmp, const Eigen::VectorXd& faceArea, const Eigen::MatrixXd& cotEntries, const int nverts, std::vector<std::complex<double>>& zvals)
{
	std::vector<Eigen::Triplet<double>> BT;
	int nfaces = mesh.nFaces();
	int nedges = mesh.nEdges();

	for(int i = 0; i < nfaces; i++)  // form mass matrix
	{
		for(int j =0; j < 3; j++)
		{
			int vid = mesh.faceVertex(i, j);
			BT.push_back({2 * vid, 2 * vid, faceArea(i) / 3.0});
			BT.push_back({2 * vid + 1, 2 * vid + 1, faceArea(i) / 3.0});
		}
	}
	Eigen::SparseMatrix<double> A;
	computeMatrixAGivenMag(mesh, halfEdgeW, vertAmp, faceArea, cotEntries, nverts, A);
	//computeMatrixA(mesh, halfEdgeW, faceArea, cotEntries, nverts, A);

	Eigen::CholmodSupernodalLLT<Eigen::SparseMatrix<double>> solver;
	Eigen::SparseMatrix<double> I = A;
	I.setIdentity();
	double eps = 1e-16;
	auto tmpA = A + eps * I;
	solver.compute(tmpA);
	while(solver.info() != Eigen::Success)
	{
		std::cout << "matrix is not PD after adding "<< eps << " * I" << std::endl;
		solver.compute(tmpA);
		eps *= 2;
	}

	Eigen::SparseMatrix<double> B(2 * nverts, 2 * nverts);
	B.setFromTriplets(BT.begin(), BT.end());
	/* std::cout << A.toDense() << std::endl;
	 std::cout << B.toDense() << std::endl;*/

	Spectra::SymShiftInvert<double> op(A, B);
	Spectra::SparseSymMatProd<double> Bop(B);
	Spectra::SymGEigsShiftSolver<Spectra::SymShiftInvert<double>, Spectra::SparseSymMatProd<double>, Spectra::GEigsMode::ShiftInvert> geigs(op, Bop, 1, 6, -2 * eps);
	geigs.init();
	int nconv = geigs.compute(Spectra::SortRule::LargestMagn, 1e6);

	Eigen::VectorXd evalues;
	Eigen::MatrixXd evecs;

	evalues = geigs.eigenvalues();
	evecs = geigs.eigenvectors();
	if (nconv != 1 || geigs.info() != Spectra::CompInfo::Successful)
	{
		std::cout << "Eigensolver failed to converge!!" << std::endl;
		exit(1);
	}

	std::cout << "Eigenvalue is " << evalues[0] << std::endl;

	zvals.clear();
	for(int i = 0; i < nverts; i++)
	{
		std::complex<double> z = std::complex<double>(evecs(2 * i, 0), evecs(2 * i + 1, 0));
//        z *= vertAmp(i);
		z *= vertAmp(i) / std::abs(z);
		zvals.push_back(z);
	}
}

void IntrinsicFormula::roundZvalsForSpecificDomainWithGivenMag(const MeshConnectivity& mesh, const Eigen::MatrixXd& halfEdgeW, const Eigen::VectorXd& vertAmp, const Eigen::VectorXi& vertFlags, const Eigen::VectorXd& faceArea, const Eigen::MatrixXd& cotEntries, const int nverts, std::vector<std::complex<double>>& zvals)
{
	std::vector<Eigen::Triplet<double>> PT;
	int nDOFs = 0;
	for (int i = 0; i < vertFlags.size(); i++)
	{
		if (vertFlags(i) == 1)
		{
			PT.push_back({ 2 * nDOFs, 2 * i, 1.0 });
			PT.push_back({ 2 * nDOFs + 1, 2 * i + 1, 1.0 });
			nDOFs += 2;
		}
	}
	Eigen::SparseMatrix<double> projM;
	projM.resize(nDOFs, 2 * nverts);
	projM.setFromTriplets(PT.begin(), PT.end());

	std::vector<Eigen::Triplet<double>> BT;
	int nfaces = mesh.nFaces();
	int nedges = mesh.nEdges();

	for (int i = 0; i < nfaces; i++)  // form mass matrix
	{
		for (int j = 0; j < 3; j++)
		{
			int vid = mesh.faceVertex(i, j);
			BT.push_back({ 2 * vid, 2 * vid, faceArea(i) / 3.0 });
			BT.push_back({ 2 * vid + 1, 2 * vid + 1, faceArea(i) / 3.0 });
		}
	}
	Eigen::SparseMatrix<double> A;
	computeMatrixAGivenMag(mesh, halfEdgeW, vertAmp, faceArea, cotEntries, nverts, A);

	Eigen::SparseMatrix<double> B(2 * nverts, 2 * nverts);
	B.setFromTriplets(BT.begin(), BT.end());
	
	B = projM * B * projM.transpose();
	A = projM * A * projM.transpose();

	Spectra::SymShiftInvert<double> op(A, B);
	Spectra::SparseSymMatProd<double> Bop(B);
	Spectra::SymGEigsShiftSolver<Spectra::SymShiftInvert<double>, Spectra::SparseSymMatProd<double>, Spectra::GEigsMode::ShiftInvert> geigs(op, Bop, 1, 6, -1e-6);
	geigs.init();
	int nconv = geigs.compute(Spectra::SortRule::LargestMagn, 1e6);

	Eigen::VectorXd evalues;
	Eigen::MatrixXd evecs;

	evalues = geigs.eigenvalues();
	evecs = geigs.eigenvectors();
	if (nconv != 1 || geigs.info() != Spectra::CompInfo::Successful)
	{
		std::cout << "Eigensolver failed to converge!!" << std::endl;
	}

	std::cout << "Eigenvalue is " << evalues[0] << std::endl;

	zvals.clear();
	Eigen::VectorXd fullVar = projM.transpose() * evecs;
	for (int i = 0; i < nverts; i++)
	{
		std::complex<double> z = std::complex<double>(fullVar(2 * i, 0), fullVar(2 * i + 1, 0));
		z *= vertAmp(i) / std::abs(z);
		zvals.push_back(z);
	}
}

void IntrinsicFormula::roundZvalsForSpecificDomainWithBndValues(const MeshConnectivity& mesh, const Eigen::MatrixXd& halfEdgeW, const Eigen::VectorXi& vertFlags, const Eigen::VectorXd& faceArea, const Eigen::MatrixXd& cotEntries, const int nverts, std::vector<std::complex<double>>& vertZvals)
{
	Eigen::VectorXd clampedVals(2 * nverts);
	clampedVals.setZero();

	std::vector<Eigen::Triplet<double>> PT;
	int nDOFs = 0;
	for (int i = 0; i < nverts; i++)
	{
		if (vertFlags(i) == 0)	// free variables
		{
			PT.push_back({ 2 * nDOFs, 2 * i, 1.0 });
			PT.push_back({ 2 * nDOFs + 1, 2 * i + 1, 1.0 });
			nDOFs += 2;
		}
		else
		{
			clampedVals(2 * i) = vertZvals[i].real();
			clampedVals(2 * i + 1) = vertZvals[i].imag();
		}
	}

	Eigen::SparseMatrix<double> projM, unProjM;
	projM.resize(nDOFs, 2 * nverts);
	projM.setFromTriplets(PT.begin(), PT.end());

	unProjM = projM.transpose();

	auto zList2Vec = [&](const std::vector<std::complex<double>>& zvals)
	{
		Eigen::VectorXd zvec(2 * zvals.size());
		for (int i = 0; i < zvals.size(); i++)
		{
			zvec(2 * i) = zvals[i].real();
			zvec(2 * i + 1) = zvals[i].imag();
		}
		return zvec;
	};

	auto vec2zList = [&](const Eigen::VectorXd& zvec)
	{
		std::vector<std::complex<double>> zList;
		for (int i = 0; i < zvec.size() / 2; i++)
		{
			zList.push_back(std::complex<double>(zvec(2 * i), zvec(2 * i + 1)));
		}
		return zList;
	};

	auto projVar = [&](const std::vector<std::complex<double>>& zvals)
	{
		Eigen::VectorXd zvec = zList2Vec(zvals);
		return projM * zvec;
	};

	auto unprojVar = [&](const Eigen::VectorXd& zvec, const Eigen::VectorXd& clampedZvecs)
	{
		Eigen::VectorXd fullZvec = unProjM * zvec + clampedZvecs;
		return vec2zList(fullZvec);
	};

	auto funVal = [&](const Eigen::VectorXd& x, Eigen::VectorXd* grad, Eigen::SparseMatrix<double>* hess, bool isProj) 
	{
		std::vector<std::complex<double>> zList = unprojVar(x, clampedVals);
		Eigen::VectorXd deriv;
		std::vector<Eigen::Triplet<double>> T;
		Eigen::SparseMatrix<double> H;
		double E = KnoppelEnergy(mesh, halfEdgeW, faceArea, cotEntries, zList, grad ? &deriv : NULL, hess ? &T : NULL);

		if (grad)
		{
			(*grad) = projM * deriv;
		}
		if (hess)
		{
			H.resize(2 * nverts, 2 * nverts);
			H.setFromTriplets(T.begin(), T.end());
			(*hess) = projM * H * unProjM;
		}

		return E;
	};

	auto maxStep = [&](const Eigen::VectorXd& x, const Eigen::VectorXd& dir) {
		return 1.0;
	};

	Eigen::VectorXd x0 = projVar(vertZvals);
	OptSolver::newtonSolver(funVal, maxStep, x0, 1000, 1e-6, 1e-10, 1e-15, false);

	Eigen::VectorXd deriv;
	double E = funVal(x0, &deriv, NULL, false);
	std::cout << "terminated with energy : " << E << ", gradient norm : " << deriv.norm() << std::endl << std::endl;
	vertZvals = unprojVar(x0, clampedVals);
}


double IntrinsicFormula::lArg(const long &n, const Eigen::Vector3d &bary)
{
	double larg = 0;
	double ti = bary(0), tj = bary(1), tk = bary(2);
	if(tk <= ti && tk <= tj)
		larg = M_PI / 3 * n * (1 + (tj - ti) / (1 - 3 * tk));
	else if (ti <= tj && ti <= tk)
		larg = M_PI / 3 * n * (3 + (tk - tj) / (1 - 3 * ti));
	else
		larg = M_PI / 3 * n * (5 + (ti - tk) / (1 - 3 * tj));
	return larg;
}

void IntrinsicFormula::getUpsamplingTheta(const MeshConnectivity &mesh, const Eigen::MatrixXd &halfEdgeW,
										  const std::vector<std::complex<double>> &zvals,
										  const std::vector<std::pair<int, Eigen::Vector3d>> &bary, Eigen::VectorXd& upTheta)
{
	Eigen::VectorXd edgeW = (halfEdgeW.col(0) - halfEdgeW.col(1)) / 2;
	int upsize = bary.size();
	upTheta.setZero(upsize);

	for(int i = 0; i < bary.size(); i++)
	{
		int fid = bary[i].first;
		double omegaIJ = edgeW(mesh.faceEdge(fid, 2));
		double omegaJK = edgeW(mesh.faceEdge(fid, 0));
		double omegaKI = edgeW(mesh.faceEdge(fid, 1));

		double cIJ = mesh.faceVertex(fid, 0) == mesh.edgeVertex(mesh.faceEdge(fid, 2), 0) ? 1 : -1;
		double cJK = mesh.faceVertex(fid, 1) == mesh.edgeVertex(mesh.faceEdge(fid, 0), 0) ? 1 : -1;
		double cKI = mesh.faceVertex(fid, 2) == mesh.edgeVertex(mesh.faceEdge(fid, 1), 0) ? 1 : -1;

		omegaIJ *= cIJ;
		omegaJK *= cJK;
		omegaKI *= cKI;

		std::complex<double> rij( std::cos(omegaIJ), std::sin(omegaIJ) );
		std::complex<double> rjk( std::cos(omegaJK), std::sin(omegaJK) );
		std::complex<double> rki( std::cos(omegaKI), std::sin(omegaKI) );

		std::complex<double> psiI = zvals[mesh.faceVertex(fid, 0)];
		std::complex<double> psiJ = zvals[mesh.faceVertex(fid, 1)];
		std::complex<double> psiK = zvals[mesh.faceVertex(fid, 2)];


		double alphaI = std::arg(psiI);
		double alphaJ = alphaI + omegaIJ - std::arg(rij*psiI/psiJ); //fmodPI((varphiI + omegaIJ) - varphiJ); // could do this in terms of angles instead of complex numbers...
		double alphaK = alphaJ + omegaJK - std::arg(rjk*psiJ/psiK); //fmodPI((varphiJ + omegaJK) - varphiK); // mostly a matter of taste---possibly a matter of performance?
		double alphaL = alphaK + omegaKI - std::arg(rki*psiK/psiI); //fmodPI((varphiK + omegaKI) - varphiI);

		// adjust triangles containing zeros
		long n = std::lround((alphaL-alphaI)/(2.*M_PI));
		alphaJ -= 2.*M_PI*n/3.;
		alphaK -= 4.*M_PI*n/3.;

		double theta = lArg(n, bary[i].second);
		upTheta(i) = theta + bary[i].second(0) * alphaI + bary[i].second(1) * alphaJ + bary[i].second(2) * alphaK;
	}
}

void IntrinsicFormula::testRoundingEnergy(const MeshConnectivity &mesh, const Eigen::MatrixXd &halfEdgeW,
										  const Eigen::VectorXd &faceArea, const Eigen::MatrixXd &cotEntries,
										  const int nverts, std::vector<std::complex<double>> zvals)
{
	Eigen::SparseMatrix<double> A;
	computeMatrixA(mesh, halfEdgeW, faceArea, cotEntries, nverts, A);
	int nfaces = mesh.nFaces();
	int nedges = mesh.nEdges();

	Eigen::VectorXd halfEdgeWeight(nedges);
	halfEdgeWeight.setZero();

	for(int i = 0; i < nfaces; i++)  // form mass matrix
	{
		for(int j =0; j < 3; j++)
		{
			int eid = mesh.faceEdge(i, j);
			halfEdgeWeight(eid) += cotEntries(i, j);
		}
	}
	double energy = 0;
	for(int i = 0; i < nedges; i++)
	{
		int vid0 = mesh.edgeVertex(i, 0);
		int vid1 = mesh.edgeVertex(i, 1);

		std::complex<double> expw0 = std::complex<double>(std::cos(halfEdgeW(i, 0)), std::sin(halfEdgeW(i, 0)));
		std::complex<double> expw1 = std::complex<double>(std::cos(halfEdgeW(i, 1)), std::sin(halfEdgeW(i, 1)));

		Eigen::Matrix2d tmpMat;

		tmpMat(0, 0) = expw0.real();
		tmpMat(0, 1) = expw0.imag();
		tmpMat(1, 0) = -expw0.imag();
		tmpMat(1, 1) = expw0.real();
		Eigen::Vector2d aibi(zvals[vid0].real(), zvals[vid0].imag());
		Eigen::Vector2d ajbj(zvals[vid1].real(), zvals[vid1].imag());

		double part1 = 2 * aibi.dot(tmpMat * ajbj) + aibi.squaredNorm() + ajbj.squaredNorm();
		double part2 = ((zvals[vid0] * expw0 - zvals[vid1])).real() * ((zvals[vid0] * expw0 - zvals[vid1])).real() + ((zvals[vid0] * expw0 - zvals[vid1])).imag() * ((zvals[vid0] * expw0 - zvals[vid1])).imag();

		energy += 0.5 * (norm((zvals[vid0] * expw0 - zvals[vid1])) + norm((zvals[vid1] * expw1 - zvals[vid0]))) * halfEdgeWeight(i);

	}

	Eigen::VectorXd x(2 * nverts);
	for(int i = 0; i < nverts; i++)
	{
		x(2 * i) = zvals[i].real();
		x(2 * i + 1) = zvals[i].imag();
	}
}

void IntrinsicFormula::testKnoppelEnergyFor2DVertexOmegaPerEdge(const Eigen::MatrixXd& pos, const MeshConnectivity& mesh, const Eigen::VectorXd& faceArea, const Eigen::MatrixXd& cotEntries, const std::vector<std::complex<double>>& zvals, const Eigen::MatrixXd& vertexOmega, const double edgeWeight, int eid)
{
	int nverts = pos.rows();
	auto zbackup = zvals;
	auto wbackup = vertexOmega;

	Eigen::Matrix<double, 8, 1> deriv;
	Eigen::Matrix<double, 8, 8> hess;
   
	double e = KnoppelEnergyFor2DVertexOmegaPerEdge(pos, mesh, faceArea, cotEntries, zvals, vertexOmega, edgeWeight, eid, &deriv, &hess, false);
	std::cout << "energy: " << e << std::endl;

	int vid0 = mesh.edgeVertex(eid, 0);
	int vid1 = mesh.edgeVertex(eid, 1);

	Eigen::Vector2d edge = (pos.row(vid1) - pos.row(vid0)).segment<2>(0);
	Eigen::Vector2d z0(zvals[vid0].real(), zvals[vid0].imag());
	Eigen::Vector2d z1(zvals[vid1].real(), zvals[vid1].imag());

	std::cout << "(a0, b0): " << z0.transpose() << std::endl;
	std::cout << "(a1, b1): " << z1.transpose() << std::endl;
	std::cout << "e: " << edge.transpose() << std::endl;
	std::cout << "w0: " << vertexOmega.row(vid0) << std::endl;
	std::cout << "w1: " << vertexOmega.row(vid1) << std::endl;

	std::cout << "hess: \n" << hess << std::endl;
   

	Eigen::Matrix<double, 8, 1> dir = deriv;
	dir.setRandom();

   

	for (int i = 3; i < 9; i++)
	{
		double eps = std::pow(0.1, i);

		for (int j = 0; j < 2; j++)
		{
			int v = mesh.edgeVertex(eid, j);
			zbackup[v] = std::complex<double>(zvals[v].real() + dir(2 * j) * eps, zvals[v].imag() + dir(1 + 2 * j) * eps);
			wbackup(v, 0) = vertexOmega(v, 0) + eps * dir(4 + 2 * j);
			wbackup(v, 1) = vertexOmega(v, 1) + eps * dir(5 + 2 * j);
		}

		Eigen::Matrix<double, 8, 1> deriv1;
		double e1 = KnoppelEnergyFor2DVertexOmegaPerEdge(pos, mesh, faceArea, cotEntries, zbackup, wbackup, edgeWeight, eid, &deriv1, NULL, false);

		std::cout << "eps: " << eps << std::endl;
		std::cout << "value-gradient check: " << (e1 - e) / eps - dir.dot(deriv) << std::endl;
		std::cout << "gradient-hessian check: " << ((deriv1 - deriv) / eps - hess * dir).norm() << std::endl;
	}
}


void IntrinsicFormula::testKnoppelEnergyFor2DVertexOmega(const Eigen::MatrixXd& pos, const MeshConnectivity& mesh, const Eigen::VectorXd& faceArea, const Eigen::MatrixXd& cotEntries, const std::vector<std::complex<double>>& zvals, const Eigen::MatrixXd& vertexOmega)
{
	Eigen::VectorXd deriv;
	Eigen::SparseMatrix<double> hess;
	std::vector<Eigen::Triplet<double>> hessT;

	double e = KnoppelEnergyFor2DVertexOmega(pos, mesh, faceArea, cotEntries, zvals, vertexOmega, &deriv, &hessT, false);
	std::cout << "energy: " << e << std::endl;
	hess.resize(deriv.rows(), deriv.rows());
	hess.setFromTriplets(hessT.begin(), hessT.end());

	Eigen::VectorXd dir = deriv;
	dir.setRandom();
	
	int nverts = pos.rows();
	auto zbackup = zvals;
	auto wbackup = vertexOmega;

	for (int i = 3; i < 9; i++)
	{
		double eps = std::pow(0.1, i);

		for (int v = 0; v < nverts; v++)
		{
			zbackup[v] = std::complex<double>(zvals[v].real() + dir(2 * v) * eps, zvals[v].imag() + dir(2 * v + 1) * eps);
			wbackup(v, 0) = vertexOmega(v, 0) + eps * dir(2 * v + 2 * nverts);
			wbackup(v, 1) = vertexOmega(v, 1) + eps * dir(2 * v + 2 * nverts + 1);
		}

		Eigen::VectorXd deriv1;
		double e1 = KnoppelEnergyFor2DVertexOmega(pos, mesh, faceArea, cotEntries, zbackup, wbackup, &deriv1, NULL, false);

		std::cout << "eps: " << eps << std::endl;
		std::cout << "value-gradient check: " << (e1 - e) / eps - dir.dot(deriv) << std::endl;
		std::cout << "gradient-hessian check: " << ((deriv1 - deriv) / eps - hess * dir).norm() << std::endl;
	}
}