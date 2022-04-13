#include "../../include/IntrinsicFormula/KnoppelStripePatternEdgeOmega.h"
#include "../../include/Optimization/NewtonDescent.h"
#include <igl/cotmatrix.h>
#include <SymGEigsShiftSolver.h>
#include <MatOp/SparseCholesky.h>
#include <Eigen/CholmodSupport>
#include <MatOp/SparseSymShiftSolve.h>
#include <iostream>

using namespace IntrinsicFormula;

void IntrinsicFormula::computeEdgeMatrix(const MeshConnectivity &mesh, const Eigen::VectorXd& edgeW,
									 const Eigen::VectorXd &faceArea, const Eigen::MatrixXd &cotEntries,
									 const int nverts, Eigen::SparseMatrix<double> &A)
{
	std::vector<Eigen::Triplet<double>> AT;
	int nfaces = mesh.nFaces();
	int nedges = mesh.nEdges();

	Eigen::VectorXd edgeWeight(nedges);
	edgeWeight.setConstant(1.0);

//    for (int i = 0; i < nfaces; i++)  // form mass matrix
//    {
//        for (int j = 0; j < 3; j++)
//        {
//            int eid = mesh.faceEdge(i, j);
//            edgeWeight(eid) += cotEntries(i, j);
//        }
//    }
	for(int i = 0; i < nedges; i++)
	{
		int vid0 = mesh.edgeVertex(i, 0);
		int vid1 = mesh.edgeVertex(i, 1);

		AT.push_back({2 * vid0, 2 * vid0, edgeWeight(i)});
		AT.push_back({2 * vid0 + 1, 2 * vid0 + 1, edgeWeight(i)});

		AT.push_back({2 * vid1, 2 * vid1, edgeWeight(i)});
		AT.push_back({2 * vid1 + 1, 2 * vid1 + 1, edgeWeight(i)});

		std::complex<double> expw0 = std::complex<double>(std::cos(edgeW(i)), std::sin(edgeW(i)));

		AT.push_back({2 * vid0, 2 * vid1, -edgeWeight(i) * (expw0.real())});
		AT.push_back({2 * vid0 + 1, 2 * vid1, -edgeWeight(i) * (-expw0.imag())});
		AT.push_back({2 * vid0, 2 * vid1 + 1, -edgeWeight(i) * (expw0.imag())});
		AT.push_back({2 * vid0 + 1, 2 * vid1 + 1, -edgeWeight(i) * (expw0.real())});

		AT.push_back({ 2 * vid1, 2 * vid0, -edgeWeight(i) * (expw0.real()) });
		AT.push_back({ 2 * vid1, 2 * vid0 + 1, -edgeWeight(i) * (-expw0.imag()) });
		AT.push_back({ 2 * vid1 + 1, 2 * vid0, -edgeWeight(i) * (expw0.imag()) });
		AT.push_back({ 2 * vid1 + 1, 2 * vid0 + 1, -edgeWeight(i) * (expw0.real()) });

	}
	A.resize(2 * nverts, 2 * nverts);
	A.setFromTriplets(AT.begin(), AT.end());
}

void IntrinsicFormula::computeEdgeMatrixGivenMag(const MeshConnectivity &mesh, const Eigen::VectorXd& edgeW, const Eigen::VectorXd& vertAmp, const Eigen::VectorXd& faceArea, const Eigen::MatrixXd& cotEntries, const int nverts, Eigen::SparseMatrix<double>& A) {
	std::vector<Eigen::Triplet<double>> AT;
	int nfaces = mesh.nFaces();
	int nedges = mesh.nEdges();

	Eigen::VectorXd edgeWeight(nedges);
	edgeWeight.setConstant(1.0);

//    for (int i = 0; i < nfaces; i++)  // form mass matrix
//    {
//        for (int j = 0; j < 3; j++)
//        {
//            int eid = mesh.faceEdge(i, j);
//            edgeWeight(eid) += cotEntries(i, j);
//        }
//    }

	for (int i = 0; i < nedges; i++) {
		int vid0 = mesh.edgeVertex(i, 0);
		int vid1 = mesh.edgeVertex(i, 1);

		double r0 = vertAmp(vid0);
		double r1 = vertAmp(vid1);

		std::complex<double> expw0 = std::complex<double>(std::cos(edgeW(i)), std::sin(edgeW(i)));


		AT.push_back({2 * vid0, 2 * vid0, r1 * r1 * edgeWeight(i)});
		AT.push_back({2 * vid0 + 1, 2 * vid0 + 1, r1 * r1 * edgeWeight(i)});

		AT.push_back({2 * vid1, 2 * vid1, r0 * r0 * edgeWeight(i)});
		AT.push_back({2 * vid1 + 1, 2 * vid1 + 1, r0 * r0 * edgeWeight(i)});


		AT.push_back({2 * vid0, 2 * vid1, -edgeWeight(i) * (expw0.real()) * r0 * r1});
		AT.push_back({2 * vid0 + 1, 2 * vid1, -edgeWeight(i) * (-expw0.imag()) * r0 * r1});
		AT.push_back({2 * vid0, 2 * vid1 + 1, -edgeWeight(i) * (expw0.imag()) * r0 * r1});
		AT.push_back({2 * vid0 + 1, 2 * vid1 + 1, -edgeWeight(i) * (expw0.real()) * r0 * r1});

		AT.push_back({2 * vid1, 2 * vid0, -edgeWeight(i) * (expw0.real()) * r0 * r1});
		AT.push_back({2 * vid1, 2 * vid0 + 1, -edgeWeight(i) * (-expw0.imag()) * r0 * r1});
		AT.push_back({2 * vid1 + 1, 2 * vid0, -edgeWeight(i) * (expw0.imag()) * r0 * r1});
		AT.push_back({2 * vid1 + 1, 2 * vid0 + 1, -edgeWeight(i) * (expw0.real()) * r0 * r1});
	}
	A.resize(2 * nverts, 2 * nverts);
	A.setFromTriplets(AT.begin(), AT.end());
}

double IntrinsicFormula::KnoppelEdgeEnergy(const MeshConnectivity& mesh, const Eigen::VectorXd& edgeW, const Eigen::VectorXd& faceArea, const Eigen::MatrixXd& cotEntries, const std::vector<std::complex<double>>& zvals, Eigen::VectorXd* deriv, std::vector<Eigen::Triplet<double>>* hess)
{
	std::vector<Eigen::Triplet<double>> AT;
	int nfaces = mesh.nFaces();
	int nedges = mesh.nEdges();
	int nverts = zvals.size();

	Eigen::VectorXd edgeWeight(nedges);
	edgeWeight.setConstant(1.0);

//    for (int i = 0; i < nfaces; i++)  // form mass matrix
//    {
//        for (int j = 0; j < 3; j++)
//        {
//            int eid = mesh.faceEdge(i, j);
//            edgeWeight(eid) += cotEntries(i, j);
//        }
//    }
	double energy = 0;
	
	for (int i = 0; i < nedges; i++)
	{
		int vid0 = mesh.edgeVertex(i, 0);
		int vid1 = mesh.edgeVertex(i, 1);

		std::complex<double> expw0 = std::complex<double>(std::cos(edgeW(i)), std::sin(edgeW(i)));

		std::complex<double> z0 = zvals[vid0];
		std::complex<double> z1 = zvals[vid1];

		energy += 0.5 * norm((z0 * expw0 - z1)) * edgeWeight(i);

		if (deriv || hess)
		{
			AT.push_back({ 2 * vid0, 2 * vid0, edgeWeight(i) });
			AT.push_back({ 2 * vid0 + 1, 2 * vid0 + 1, edgeWeight(i) });

			AT.push_back({ 2 * vid1, 2 * vid1, edgeWeight(i) });
			AT.push_back({ 2 * vid1 + 1, 2 * vid1 + 1, edgeWeight(i) });


			AT.push_back({ 2 * vid0, 2 * vid1, -edgeWeight(i) * (expw0.real()) });
			AT.push_back({ 2 * vid0 + 1, 2 * vid1, -edgeWeight(i) * (-expw0.imag()) });
			AT.push_back({ 2 * vid0, 2 * vid1 + 1, -edgeWeight(i) * (expw0.imag()) });
			AT.push_back({ 2 * vid0 + 1, 2 * vid1 + 1, -edgeWeight(i) * (expw0.real()) });

			AT.push_back({ 2 * vid1, 2 * vid0, -edgeWeight(i) * (expw0.real()) });
			AT.push_back({ 2 * vid1, 2 * vid0 + 1, -edgeWeight(i) * (-expw0.imag()) });
			AT.push_back({ 2 * vid1 + 1, 2 * vid0, -edgeWeight(i) * (expw0.imag()) });
			AT.push_back({ 2 * vid1 + 1, 2 * vid0 + 1, -edgeWeight(i) * (expw0.real()) });
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

double IntrinsicFormula::KnoppelEdgeEnergyGivenMag(const MeshConnectivity& mesh, const Eigen::VectorXd& edgeW, const Eigen::VectorXd& vertAmp, const Eigen::VectorXd& edgeWeight, const std::vector<std::complex<double>>& zvals, Eigen::VectorXd* deriv, std::vector<Eigen::Triplet<double>>* hess)
{
	std::vector<Eigen::Triplet<double>> AT;
	int nfaces = mesh.nFaces();
	int nedges = mesh.nEdges();
	int nverts = vertAmp.size();

	double energy = 0;

	for (int i = 0; i < nedges; i++)
	{
		int vid0 = mesh.edgeVertex(i, 0);
		int vid1 = mesh.edgeVertex(i, 1);

		double r0 = vertAmp(vid0);
		double r1 = vertAmp(vid1);

		std::complex<double> expw0 = std::complex<double>(std::cos(edgeW(i, 0)), std::sin(edgeW(i, 0)));

		std::complex<double> z0 = zvals[vid0];
		std::complex<double> z1 = zvals[vid1];


		energy += 0.5 * norm((r1 * z0 * expw0 - r0 * z1)) * edgeWeight(i);

		if (deriv || hess)
		{
			AT.push_back({ 2 * vid0, 2 * vid0, r1 * r1 * edgeWeight(i) });
			AT.push_back({ 2 * vid0 + 1, 2 * vid0 + 1, r1 * r1 * edgeWeight(i) });

			AT.push_back({ 2 * vid1, 2 * vid1, r0 * r0 * edgeWeight(i) });
			AT.push_back({ 2 * vid1 + 1, 2 * vid1 + 1, r0 * r0 * edgeWeight(i) });


			AT.push_back({ 2 * vid0, 2 * vid1, -edgeWeight(i) * (expw0.real()) * r0 * r1 });
			AT.push_back({ 2 * vid0 + 1, 2 * vid1, -edgeWeight(i) * (-expw0.imag()) * r0 * r1 });
			AT.push_back({ 2 * vid0, 2 * vid1 + 1, -edgeWeight(i) * (expw0.imag()) * r0 * r1 });
			AT.push_back({ 2 * vid0 + 1, 2 * vid1 + 1, -edgeWeight(i) * (expw0.real()) * r0 * r1 });

			AT.push_back({ 2 * vid1, 2 * vid0, -edgeWeight(i) * (expw0.real()) * r0 * r1 });
			AT.push_back({ 2 * vid1, 2 * vid0 + 1, -edgeWeight(i) * (-expw0.imag()) * r0 * r1 });
			AT.push_back({ 2 * vid1 + 1, 2 * vid0, -edgeWeight(i) * (expw0.imag()) * r0 * r1 });
			AT.push_back({ 2 * vid1 + 1, 2 * vid0 + 1, -edgeWeight(i) * (expw0.real()) * r0 * r1 });
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

void IntrinsicFormula::roundZvalsFromEdgeOmega(const MeshConnectivity &mesh, const Eigen::VectorXd& edgeW,
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
	computeEdgeMatrix(mesh, edgeW, faceArea, cotEntries, nverts, A);

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

void IntrinsicFormula::roundZvalsFromEdgeOmegaVertexMag(const MeshConnectivity &mesh, const Eigen::VectorXd& edgeW, const Eigen::VectorXd& vertAmp, const Eigen::VectorXd& faceArea, const Eigen::MatrixXd& cotEntries, const int nverts, std::vector<std::complex<double>>& zvals)
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
	computeEdgeMatrixGivenMag(mesh, edgeW, vertAmp, faceArea, cotEntries, nverts, A);

	Eigen::CholmodSupernodalLLT<Eigen::SparseMatrix<double>> solver;
	Eigen::SparseMatrix<double> I = A;
	I.setIdentity();
	double eps = 1e-16;
    Eigen::SparseMatrix<double> tmpA = A + eps * I;
	solver.compute(tmpA);
	while(solver.info() != Eigen::Success)
	{
		std::cout << "matrix is not PD after adding "<< eps << " * I" << std::endl;
		solver.compute(tmpA);
		eps *= 2;
        tmpA = A + eps * I;
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
		z = vertAmp(i) * std::complex<double>(std::cos(std::arg(z)), std::sin(std::arg(z)));
		zvals.push_back(z);
	}
}

void IntrinsicFormula::roundZvalsForSpecificDomainFromEdgeOmegaGivenMag(const MeshConnectivity& mesh, const Eigen::VectorXd& edgeW, const Eigen::VectorXd& vertAmp, const Eigen::VectorXi& vertFlags, const Eigen::VectorXd& faceArea, const Eigen::MatrixXd& cotEntries, const int nverts, std::vector<std::complex<double>>& zvals)
{
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
	computeEdgeMatrixGivenMag(mesh, edgeW, vertAmp, faceArea, cotEntries, nverts, A);
//    computeMatrixA(mesh, edgeW, faceArea, cotEntries, nverts, A);

	Eigen::SparseMatrix<double> B(2 * nverts, 2 * nverts);
	B.setFromTriplets(BT.begin(), BT.end());

    Eigen::CholmodSupernodalLLT<Eigen::SparseMatrix<double>> solver;
    Eigen::SparseMatrix<double> I = A;
    I.setIdentity();
    double eps = 1e-16;
    Eigen::SparseMatrix<double> tmpA = A + eps * I;
    solver.compute(tmpA);
    while(solver.info() != Eigen::Success)
    {
        std::cout << "matrix is not PD after adding "<< eps << " * I" << std::endl;
        solver.compute(tmpA);
        eps *= 2;
        tmpA = A + eps * I;
    }

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
	}

	std::cout << "Eigenvalue is " << evalues[0] << std::endl;

	zvals.clear();
    Eigen::VectorXd fullVar = evecs;
//	Eigen::VectorXd fullVar = projM.transpose() * evecs;
	for (int i = 0; i < nverts; i++)
	{
		std::complex<double> z = std::complex<double>(fullVar(2 * i, 0), fullVar(2 * i + 1, 0));

        if (vertFlags(i) == 1)
        {
            z *= vertAmp(i) / std::abs(z);
        }

		zvals.push_back(z);
	}
}

void IntrinsicFormula::roundZvalsForSpecificDomainFromEdgeOmegaBndValues(const Eigen::MatrixXd& pos, const MeshConnectivity& mesh, const Eigen::VectorXd& edgeW, const Eigen::VectorXi& vertFlags, const Eigen::VectorXd& faceArea, const Eigen::MatrixXd& cotEntries, const int nverts, std::vector<std::complex<double>>& vertZvals)
{
	Eigen::VectorXd clampedVals(2 * nverts);
	clampedVals.setZero();

	std::vector<Eigen::Triplet<double>> T;
	int nDOFs = 0;
	for (int i = 0; i < nverts; i++)
	{
		if (vertFlags(i) == 0)	// free variables
		{
			T.push_back({ 2 * nDOFs, 2 * i, 1.0 });
			T.push_back({ 2 * nDOFs + 1, 2 * i + 1, 1.0 });

			nDOFs += 1;
		}
		else
		{
			clampedVals(2 * i) = vertZvals[i].real();
			clampedVals(2 * i + 1) = vertZvals[i].imag();
		}
	}
	if (nDOFs == 0)
		return;

	Eigen::SparseMatrix<double> PT(clampedVals.rows(), 2 * nDOFs + 1), P(2 * nDOFs + 1, clampedVals.rows());
	for (int i = 0; i < clampedVals.rows(); i++)
	{
		T.push_back(Eigen::Triplet<double>(2 * nDOFs, i, clampedVals(i)));
	}
	
	P.setFromTriplets(T.begin(), T.end());
	PT = P.transpose();

	Eigen::SparseMatrix<double> A;
	computeEdgeMatrix(mesh, edgeW, faceArea, cotEntries, nverts, A);

	A = P * A * PT;

	Eigen::CholmodSupernodalLLT<Eigen::SparseMatrix<double>> solver;
	Eigen::SparseMatrix<double> I = A;
	I.setIdentity();
	double eps = 1e-16;
	Eigen::SparseMatrix<double> tmpA = A + eps * I;
	solver.compute(tmpA);
	while (solver.info() != Eigen::Success)
	{
		std::cout << "matrix is not PD after adding " << eps << " * I" << std::endl;
		solver.compute(tmpA);
		eps *= 2;
		tmpA = A + eps * I;
	}

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

	Eigen::SparseMatrix<double> B(2 * nverts, 2 * nverts);
	B.setFromTriplets(BT.begin(), BT.end());

	B = P * B * PT;

	std::cout << "eval compute begins." << std::endl;
	std::cout << "matrix A size: " << A.rows() << " " << A.cols() << std::endl;
	std::cout << "matrix B size: " << B.rows() << " " << B.cols() << std::endl;
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

	std::cout << "Eigenvalue is " << evalues[0] << ", scale value: " << evecs(2 * nDOFs) << std::endl;

	vertZvals.clear();
	Eigen::VectorXd fullVar = PT * evecs / evecs(2 * nDOFs);

	for (int i = 0; i < nverts; i++)
	{
		std::complex<double> z = std::complex<double>(fullVar(2 * i, 0), fullVar(2 * i + 1, 0));
		vertZvals.push_back(z);
	}

}


static double lArg(const long &n, const Eigen::Vector3d &bary)
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

void IntrinsicFormula::getUpsamplingThetaFromEdgeOmega(const MeshConnectivity &mesh, const Eigen::VectorXd& edgeW,
										  const std::vector<std::complex<double>> &zvals,
										  const std::vector<std::pair<int, Eigen::Vector3d>> &bary, Eigen::VectorXd& upTheta)
{
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

