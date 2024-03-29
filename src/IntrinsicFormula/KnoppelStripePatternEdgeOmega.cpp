#include "../../include/IntrinsicFormula/KnoppelStripePatternEdgeOmega.h"
#include "../../include/Optimization/NewtonDescent.h"
#include <igl/cotmatrix.h>
#include <SymGEigsShiftSolver.h>
#include <MatOp/SparseCholesky.h>
#include <Eigen/CholmodSupport>
#include <MatOp/SparseSymShiftSolve.h>
#include <iostream>

using namespace IntrinsicFormula;

void IntrinsicFormula::computeEdgeMatrix(const MeshConnectivity &mesh, const Eigen::VectorXd& edgeW, const Eigen::VectorXd& edgeWeight,
									 const int nverts, Eigen::SparseMatrix<double> &A)
{
	std::vector<Eigen::Triplet<double>> AT;
	int nfaces = mesh.nFaces();
	int nedges = mesh.nEdges();

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

void IntrinsicFormula::computeEdgeMatrixGivenMag(const MeshConnectivity &mesh, const Eigen::VectorXd& edgeW, const Eigen::VectorXd& vertAmp, const Eigen::VectorXd& edgeWeight, const int nverts, Eigen::SparseMatrix<double>& A) {
	std::vector<Eigen::Triplet<double>> AT;
	int nfaces = mesh.nFaces();
	int nedges = mesh.nEdges();

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

double IntrinsicFormula::KnoppelEdgeEnergy(const MeshConnectivity& mesh, const Eigen::VectorXd& edgeW, const Eigen::VectorXd& edgeWeight, const std::vector<std::complex<double>>& zvals, Eigen::VectorXd* deriv, std::vector<Eigen::Triplet<double>>* hess)
{
	std::vector<Eigen::Triplet<double>> AT;
	int nfaces = mesh.nFaces();
	int nedges = mesh.nEdges();
	int nverts = zvals.size();

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
	const Eigen::VectorXd& edgeWeight, const Eigen::VectorXd& vertArea, const int nverts, std::vector<std::complex<double>> &zvals)
{
	std::vector<Eigen::Triplet<double>> BT;
	int nfaces = mesh.nFaces();
	int nedges = mesh.nEdges();

	for (int i = 0; i < nverts; i++)
	{
		BT.push_back({ 2 * i, 2 * i, vertArea(i) });
		BT.push_back({ 2 * i + 1, 2 * i + 1, vertArea(i) });
	}
	
	
	Eigen::SparseMatrix<double> A;
	computeEdgeMatrix(mesh, edgeW, edgeWeight, nverts, A);

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

void IntrinsicFormula::roundZvalsFromEdgeOmegaVertexMag(const MeshConnectivity &mesh, const Eigen::VectorXd& edgeW, const Eigen::VectorXd& vertAmp, const Eigen::VectorXd& edgeWeight, const Eigen::VectorXd& vertArea, const int nverts, std::vector<std::complex<double>>& zvals)
{
	std::vector<Eigen::Triplet<double>> BT;
	int nfaces = mesh.nFaces();
	int nedges = mesh.nEdges();

	for (int i = 0; i < nverts; i++)
	{
		BT.push_back({ 2 * i, 2 * i, vertArea(i) });
		BT.push_back({ 2 * i + 1, 2 * i + 1, vertArea(i) });
	}

	Eigen::SparseMatrix<double> A;
	computeEdgeMatrixGivenMag(mesh, edgeW, vertAmp, edgeWeight, nverts, A);

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
	//B.setIdentity();

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

void IntrinsicFormula::roundZvalsForSpecificDomainFromEdgeOmegaBndValuesDirectly(const MeshConnectivity& mesh, const Eigen::VectorXd& edgeW, const Eigen::VectorXi& vertFlags, const Eigen::VectorXd& edgeWeight, const Eigen::VectorXd& vertArea, const int nverts, std::vector<std::complex<double>>& vertZvals, Eigen::VectorXd* vertAmp)
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
	Eigen::SparseMatrix<double> projM, unProjM;
	projM.resize(2 * nDOFs, 2 * nverts);
	projM.setFromTriplets(PT.begin(), PT.end());

	unProjM = projM.transpose();

	auto zList2CoordVec = [&](const std::vector<std::complex<double>>& zvals, Eigen::VectorXd& xvec, Eigen::VectorXd& yvec)
	{
		xvec.setZero(zvals.size());
		yvec.setZero(zvals.size());

		for (int i = 0; i < zvals.size(); i++)
		{
			xvec(i) = zvals[i].real();
			yvec(i) = zvals[i].imag();
		}
	};

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
		Eigen::VectorXd fullZvec = unProjM * zvec + clampedVals;
		return vec2zList(fullZvec);
	};

	auto funVal = [&](const Eigen::VectorXd& x, Eigen::VectorXd* grad, Eigen::SparseMatrix<double>* hess, bool isProj)
	{
		std::vector<std::complex<double>> zList = unprojVar(x, clampedVals);
		Eigen::VectorXd deriv;
		std::vector<Eigen::Triplet<double>> T;
		Eigen::SparseMatrix<double> H;
		double E = 0;
		if(vertAmp)
			E = KnoppelEdgeEnergyGivenMag(mesh, edgeW, *vertAmp, edgeWeight, zList, grad ? &deriv : NULL, hess ? &T : NULL);
		else
			E = KnoppelEdgeEnergy(mesh, edgeW, edgeWeight, zList, grad ? &deriv : NULL, hess ? &T : NULL);

		Eigen::VectorXd fullx = zList2Vec(zList);

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
	OptSolver::newtonSolver(funVal, maxStep, x0, 1000, 1e-6, 1e-10, 1e-15, true);

	Eigen::VectorXd deriv;
	double E = funVal(x0, &deriv, NULL, false);
	std::cout << "terminated with energy : " << E << ", gradient norm : " << deriv.norm() << std::endl << std::endl;
	vertZvals = unprojVar(x0, clampedVals);
}


void IntrinsicFormula::roundZvalsForSpecificDomainFromEdgeOmegaBndValues(const MeshConnectivity& mesh, const Eigen::VectorXd& edgeW,  const Eigen::VectorXi& vertFlags, const Eigen::VectorXd& edgeWeight, const Eigen::VectorXd& vertArea, const int nverts, std::vector<std::complex<double>>& vertZvals, Eigen::VectorXd* vertAmp)
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

	std::cout << "num of DOFs: " << nDOFs << std::endl;

	Eigen::SparseMatrix<double> PT, P;

	if (clampedVals.norm())
	{
		PT.resize(clampedVals.rows(), 2 * nDOFs + 1), P.resize(2 * nDOFs + 1, clampedVals.rows());
		for (int i = 0; i < clampedVals.rows(); i++)
		{
			T.push_back(Eigen::Triplet<double>(2 * nDOFs, i, clampedVals(i)));
		}
	}
	else
	{
		PT.resize(clampedVals.rows(), 2 * nDOFs), P.resize(2 * nDOFs, clampedVals.rows());
	}
	
	
	P.setFromTriplets(T.begin(), T.end());
	PT = P.transpose();

	Eigen::SparseMatrix<double> A;
	if (vertAmp)
		computeEdgeMatrixGivenMag(mesh, edgeW, *vertAmp, edgeWeight, nverts, A);
	else
		computeEdgeMatrix(mesh, edgeW, edgeWeight, nverts, A);

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

	for (int i = 0; i < nverts; i++)
	{
		BT.push_back({ 2 * i, 2 * i, vertArea(i) });
		BT.push_back({ 2 * i + 1, 2 * i + 1, vertArea(i) });
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

	std::cout << "Eigenvalue is " << evalues[0];

	Eigen::VectorXd fullVar;

	if (clampedVals.norm())
	{
		std::cout << ", scale value: " << evecs(2 * nDOFs) << std::endl;
		fullVar = PT * evecs / evecs(2 * nDOFs);
	}
		
	else
	{
		std::cout << std::endl;
		fullVar = PT * evecs;
	}
		

	vertZvals.clear();
	 

	for (int i = 0; i < nverts; i++)
	{
		std::complex<double> z = std::complex<double>(fullVar(2 * i, 0), fullVar(2 * i + 1, 0));
		vertZvals.push_back(z);
	}

}

void IntrinsicFormula::roundZvalsWithTheGivenReference(const MeshConnectivity& mesh, const Eigen::VectorXd& edgeW, const Eigen::VectorXd& vertAmp, const std::vector<std::complex<double>>& refZvals, const Eigen::VectorXd& vertWeight, const Eigen::VectorXd& edgeWeight, const Eigen::VectorXd& vertArea, const int nverts, std::vector<std::complex<double>>& vertZvals)
{
	auto zList2CoordVec = [&](const std::vector<std::complex<double>>& zvals, Eigen::VectorXd& xvec, Eigen::VectorXd& yvec)
	{
		xvec.setZero(zvals.size());
		yvec.setZero(zvals.size());

		for (int i = 0; i < zvals.size(); i++)
		{
			xvec(i) = zvals[i].real();
			yvec(i) = zvals[i].imag();
		}
	};

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

	
	auto funVal = [&](const Eigen::VectorXd& x, Eigen::VectorXd* grad, Eigen::SparseMatrix<double>* hess, bool isProj)
	{
		std::vector<std::complex<double>> zList = vec2zList(x);
		Eigen::VectorXd deriv;
		std::vector<Eigen::Triplet<double>> T;
		Eigen::SparseMatrix<double> H;
		double E = KnoppelEdgeEnergyGivenMag(mesh, edgeW, vertAmp, edgeWeight, zList, grad ? &deriv : NULL, hess ? &T : NULL);

		for (int i = 0; i < zList.size(); i++)
		{
			E += 0.5 * vertWeight(i) * std::abs(zList[i] - refZvals[i]) * std::abs(zList[i] - refZvals[i]) * vertArea(i);
		}

		if (grad)
		{
			(*grad) = deriv;
			for (int i = 0; i < zList.size(); i++)
			{
				(*grad)(2 * i) += vertWeight(i) * vertArea(i) * (zList[i] - refZvals[i]).real();
				(*grad)(2 * i + 1) += vertWeight(i) * vertArea(i) * (zList[i] - refZvals[i]).imag();
			}
		}
		if (hess)
		{
			for (int i = 0; i < zList.size(); i++)
			{
				T.push_back( { 2 * i, 2 * i, vertWeight(i)* vertArea(i) });
				T.push_back({ 2 * i + 1, 2 * i + 1, vertWeight(i) * vertArea(i) });
			}
			hess->resize(2 * nverts, 2 * nverts);
			hess->setFromTriplets(T.begin(), T.end());
		}

		return E;
	};

	auto maxStep = [&](const Eigen::VectorXd& x, const Eigen::VectorXd& dir) {
		return 1.0;
	};

	Eigen::VectorXd x0 = zList2Vec(refZvals);
	OptSolver::newtonSolver(funVal, maxStep, x0, 1000, 1e-6, 1e-10, 1e-15, true);

	Eigen::VectorXd deriv;
	double E = funVal(x0, &deriv, NULL, false);
	std::cout << "terminated with energy : " << E << ", gradient norm : " << deriv.norm() << std::endl << std::endl;
	vertZvals = vec2zList(x0);

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

