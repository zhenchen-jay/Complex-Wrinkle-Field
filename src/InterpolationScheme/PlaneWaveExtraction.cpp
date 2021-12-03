#include "../../include/InterpolationScheme/PlaneWaveExtraction.h"

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <iostream>
#include <nasoq_eigen.h>

#ifndef GRAIN_SIZE
#define GRAIN_SIZE 10
#endif


double PlaneWaveExtraction::faceFieldDifferencePerEdge(const Eigen::MatrixXd faceFields, int eid,
													   Eigen::Vector4d *deriv, Eigen::Matrix4d *hess)
{
	if(deriv)
		deriv->setZero();
	if(hess)
		hess->setZero();

	int f0 = _mesh.edgeFace(eid, 0);
	int f1 = _mesh.edgeFace(eid, 1);
	if(f0 == -1 || f1 == -1)
		return 0;

	double energy = 0.5 * ((faceFields(f0, 0) - faceFields(f1, 0)) * (faceFields(f0, 0) - faceFields(f1, 0)) + (faceFields(f0, 1) - faceFields(f1, 1)) * (faceFields(f0, 1) - faceFields(f1, 1)) );

	if(deriv)
	{
		(*deriv)(0) = faceFields(f0, 0) - faceFields(f1, 0);
		(*deriv)(1) = faceFields(f0, 1) - faceFields(f1, 1);

		(*deriv)(2) = faceFields(f1, 0) - faceFields(f0, 0);
		(*deriv)(3) = faceFields(f1, 1) - faceFields(f0, 1);
	}

	if(hess)
	{
		hess->row(0) << 1, 0, -1, 0;
		hess->row(1) << 0, 1, 0, -1;
		hess->row(2) << -1, 0, 1, 0;
		hess->row(3) << 0, -1, 0, 1;
	}

	return energy;
	
}

double PlaneWaveExtraction::faceFieldDifference(const Eigen::MatrixXd faceFields, Eigen::VectorXd *deriv,
												Eigen::SparseMatrix<double> *hess)
{
	int nfaces = _mesh.nFaces();
	int nedges = _mesh.nEdges();

	if(deriv)
		deriv->setZero(2 * nfaces);

	std::vector<double> energyList(nedges);
	std::vector<Eigen::Vector4d> derivList(nedges);
	std::vector<Eigen::Matrix4d> hessList(nedges);

	auto computeEnergy = [&](const tbb::blocked_range<uint32_t>& range){
		for (uint32_t i = range.begin(); i < range.end(); ++i)
		{
			Eigen::Matrix4d hessMat;
			energyList[i] = faceFieldDifferencePerEdge(faceFields, i, deriv ? &(derivList[i]) : NULL, hess ? &(hessList[i]) : NULL);
		}
	};
	tbb::blocked_range<uint32_t> rangex(0u, (uint32_t)_mesh.nEdges(), GRAIN_SIZE);
	tbb::parallel_for(rangex, computeEnergy);

	double energy = 0;
	std::vector<Eigen::Triplet<double>> T;

	for(int i = 0; i < _mesh.nEdges(); i++)
	{
		int f0 = _mesh.edgeFace(i, 0);
		int f1 = _mesh.edgeFace(i, 1);
		if(f0 == -1 || f1 == -1)
			continue;
		energy += energyList[i];
		if(deriv)
		{
			(*deriv)(2 * f0) += derivList[i](0);
			(*deriv)(2 * f0 + 1) += derivList[i](1);
			(*deriv)(2 * f1) += derivList[i](2);
			(*deriv)(2 * f1 + 1) += derivList[i](3);
		}

		if(hess)
		{
			for(int j = 0; j < 2; j++)
				for(int k = 0; k < 2; k++)
				{
					T.push_back({2 * f0 + j, 2 * f0 + k, hessList[i](j, k)});
					T.push_back({2 * f0 + j, 2 * f1 + k, hessList[i](j, 2 + k)});

					T.push_back({2 * f1 + j, 2 * f0 + k, hessList[i](2 + j, k)});
					T.push_back({2 * f1 + j, 2 * f1 + k, hessList[i](2 + j, 2 + k)});
				}

		}
	}
	return energy;
}

double PlaneWaveExtraction::optEnergyPerEdge(const Eigen::VectorXd &x, int eid, Eigen::Matrix<double, 6, 1> *deriv,
											 Eigen::Matrix<double, 6, 6> *hess)
{
	if(deriv)
		deriv->setZero();
	if(hess)
		hess->setZero();

	int f0 = _mesh.edgeFace(eid, 0);
	int f1 = _mesh.edgeFace(eid, 1);
	if(f0 == -1 || f1 == -1)
		return 0;

	Eigen::VectorXd w1, w2, diffw;
   
	w1 = _inputFields.row(_mesh.faceVertex(f0, 0)) * x(3 * f0) + _inputFields.row(_mesh.faceVertex(f0, 1)) * x(3 * f0 + 1) + _inputFields.row(_mesh.faceVertex(f0, 2)) * x(3 * f0 + 2);
	w2 = _inputFields.row(_mesh.faceVertex(f1, 0)) * x(3 * f1) + _inputFields.row(_mesh.faceVertex(f1, 1)) * x(3 * f1 + 1) + _inputFields.row(_mesh.faceVertex(f1, 2)) * x(3 * f1 + 2);

	diffw = w1 - w2;

	double energy = 0.5 * diffw.squaredNorm();

	if(deriv || hess)
	{
		Eigen::Matrix<double, 2, 6> gradDiffw;

		for(int i = 0; i < 3; i++)
		{
			gradDiffw.col(i) << _inputFields(_mesh.faceVertex(f0, i), 0), _inputFields(_mesh.faceVertex(f0, i), 1);
			gradDiffw.col(3 + i) << -_inputFields(_mesh.faceVertex(f1, i), 0), -_inputFields(_mesh.faceVertex(f1, i), 1);
		}
		if(deriv)
			(*deriv) = gradDiffw.transpose() * diffw.segment<2>(0);
		if (hess)
		{
			(*hess) = gradDiffw.transpose() * gradDiffw;
		}
		   
	}

	return energy;
}

double PlaneWaveExtraction::optEnergy(const Eigen::VectorXd &x, Eigen::VectorXd *deriv,
									  Eigen::SparseMatrix<double> *hess)
{
	if(x.rows() != _mesh.nFaces() * 3)
	{
		std::cerr << "In function optEnergy, variable size doesn't match the face number * 3!" << std::endl;
		exit(1);
	}

	if(deriv)
		deriv->setZero(x.size());

	int nedges = _mesh.nEdges();

	std::vector<double> energyList(nedges);
	std::vector<Eigen::Matrix<double, 6, 1>> derivList(nedges);
	std::vector<Eigen::Matrix<double, 6, 6>> hessList(nedges);

	auto computeEnergy = [&](const tbb::blocked_range<uint32_t>& range){
		for (uint32_t i = range.begin(); i < range.end(); ++i)
		{
			Eigen::Matrix4d hessMat;
			energyList[i] = optEnergyPerEdge(x, i, deriv ? &(derivList[i]) : NULL, hess ? &(hessList[i]) : NULL);
		}
	};
	tbb::blocked_range<uint32_t> rangex(0u, (uint32_t)nedges, GRAIN_SIZE);
	tbb::parallel_for(rangex, computeEnergy);

	double energy = 0;
	std::vector<Eigen::Triplet<double>> T;

	for(int i = 0; i < nedges; i++)
	{
		int f0 = _mesh.edgeFace(i, 0);
		int f1 = _mesh.edgeFace(i, 1);
		if(f0 == -1 || f1 == -1)
			continue;
		energy += energyList[i];
		if(deriv)
		{
			for(int j = 0; j < 3; j++)
			{
				(*deriv)(3 * f0 + j) += derivList[i](j);
				(*deriv)(3 * f1 + j) += derivList[i](3 + j);
			}
		}

		if(hess)
		{
			for(int j = 0; j < 3; j++)
				for(int k = 0; k < 3; k++)
				{
					T.push_back({3 * f0 + j, 3 * f0 + k, hessList[i](j, k)});
					T.push_back({3 * f0 + j, 3 * f1 + k, hessList[i](j, 3 + k)});

					T.push_back({3 * f1 + j, 3 * f0 + k, hessList[i](3 + j, k)});
					T.push_back({3 * f1 + j, 3 * f1 + k, hessList[i](3 + j, 3 + k)});
				}

		}
	}
	if (hess)
	{
		hess->resize(x.size(), x.size());
		hess->setFromTriplets(T.begin(), T.end());
	}
	return energy;

}

bool PlaneWaveExtraction::extractPlaneWave(Eigen::MatrixXd &planeFields)
{
	Eigen::SparseMatrix<double> H;

	int nfaces = _mesh.nFaces();
	int nedges = _mesh.nEdges();

	Eigen::VectorXd x(3 * nfaces);
	x.setConstant(1.0/3);


	double energy = optEnergy(x, NULL, &H);     // the optimization function is actually 0.5 x^T H x
	std::cout << "error (should be 0): " << energy - 0.5 * x.dot(H*x) << std::endl;

	// add some reg to H PD
	Eigen::SparseMatrix<double> I(3 * nfaces, 3 * nfaces);
	I.setIdentity();

	// build equality constraints
	int row = 0;
	std::vector<Eigen::Triplet<double>> T;
	for(int i = 0; i < nfaces; i++)
	{
		T.push_back({row, 3 * i, 1.0});
		T.push_back({row, 3 * i + 1, 1.0});
		T.push_back({row, 3 * i + 2, 1.0});

		row++;
	}

	for(int i = 0; i < nedges; i++)
	{
		int f0 = _mesh.edgeFace(i, 0);
		int f1 = _mesh.edgeFace(i, 1);
		if(f0 == -1 || f1 == -1)
			continue;

		Eigen::VectorXd edge = _pos.row(_mesh.edgeVertex(i, 0)) - _pos.row(_mesh.edgeVertex(i, 1));
		for(int j = 0; j < 3; j++)
		{
			int v0 = _mesh.faceVertex(f0, j);
			int v1 = _mesh.faceVertex(f1, j);
			double c0 = edge.segment<2>(0).dot(_inputFields.row(v0).segment<2>(0));
			double c1 = -edge.segment<2>(0).dot(_inputFields.row(v1).segment<2>(0));

			T.push_back({row, 3 * f0 + j, c0});
			T.push_back({row, 3 * f1 + j, c1});
		}
		row++;
	}

	Eigen::SparseMatrix<double> Aeq(row, 3 * nfaces);
	Aeq.setFromTriplets(T.begin(), T.end());

	Eigen::VectorXd beq = Eigen::VectorXd::Zero(row);
	beq.segment(0, nfaces).setConstant(1.0);

	// inequality constraint
	Eigen::SparseMatrix<double> Aineq = -I;
	Eigen::VectorXd bineq(3 * nfaces);
	bineq.setZero();

	Eigen::VectorXd y, z;

	

    nasoq::QPSettings qpsettings;
	qpsettings.eps = 1e-8;
	getNumIter(qpsettings.eps, qpsettings.inner_iter_ref, qpsettings.outer_iter_ref);
	qpsettings.nasoq_variant = "PREDET";
	qpsettings.diag_perturb = 1e-10;

	int converged = nasoq::quadprog(H.triangularView<Eigen::Lower>(), Eigen::VectorXd::Zero(3 * nfaces), Aeq, beq, Aineq, Eigen::VectorXd::Zero(3 * nfaces), x, y, z, &qpsettings);

	if (converged == nasoq::nasoq_status::Optimal)
	{
		std::cout << "reach the optimal!" << std::endl;
	}
	else
	{
		if (converged == nasoq::nasoq_status::Inaccurate)
			std::cout << "result may be inaccurate, only primal-feasibility is satisfied." << std::endl;
		else if (converged == nasoq::nasoq_status::Infeasible)
			std::cout << "infeasible, the problem is unbounded" << std::endl;
		else
			std::cout << "NotConverged" << std::endl;
	}

	planeFields.setZero(nfaces, 2);
	for (int i = 0; i < nfaces; i++)
	{
		planeFields.row(i) = x(3 * i) * _inputFields.row(_mesh.faceVertex(i, 0)) + x(3 * i + 1) * _inputFields.row(_mesh.faceVertex(i, 1)) + x(3 * i + 2) * _inputFields.row(_mesh.faceVertex(i, 2));
	}
	return true;
}

void PlaneWaveExtraction::testFaceFieldDifferencePerEdge(Eigen::MatrixXd faceField, int eid)
{
	int f0 = _mesh.edgeFace(eid, 0);
	int f1 = _mesh.edgeFace(eid, 1);

	if (f0 == -1 || f1 == -1)
	{
		std::cout << "bnd edges!" << std::endl;
		return;
	}

	Eigen::Vector4d deriv, deriv1;
	Eigen::Matrix4d hess;
	double energy = faceFieldDifferencePerEdge(faceField, eid, &deriv, &hess);

	Eigen::Vector4d dir;
	dir.setRandom();
	Eigen::MatrixXd backupField = faceField;
	for(int i = 3; i < 10; i++)
	{
		double eps = std::pow(10, -i);
		faceField(f0, 0) = backupField(f0, 0) + eps * dir(0);
		faceField(f0, 1) = backupField(f0, 1) + eps * dir(1);
		faceField(f1, 0) = backupField(f1, 0) + eps * dir(2);
		faceField(f1, 1) = backupField(f1, 1) + eps * dir(3);

		double energy1 = faceFieldDifferencePerEdge(faceField, eid, &deriv1, NULL);

		std::cout << "eps: " << eps << std::endl;
		std::cout << "value-gradient check: " << (energy1 - energy)/eps - deriv.dot(dir) << std::endl;
		std::cout << "gradient-hessian check: " << ((deriv1 - deriv) / eps - hess * dir).norm() << std::endl;
	}
}

void PlaneWaveExtraction::testFaceFieldDifference(Eigen::MatrixXd faceFields)
{
	Eigen::VectorXd deriv, deriv1;
	Eigen::SparseMatrix<double> hess;
	double energy = faceFieldDifference(faceFields, &deriv, &hess);

	Eigen::VectorXd dir = deriv;
	dir.setRandom();

	Eigen::MatrixXd backupFields = faceFields;
	for(int i = 3; i < 10; i++)
	{
		double eps = std::pow(10, -i);
		for(int i = 0; i < faceFields.rows(); i++)
		{
			faceFields(i, 0) = backupFields(i, 0) + eps * dir(2 * i);
			faceFields(i, 1) = backupFields(i, 1) + eps * dir(2 * i + 1);
		}

		double energy1 = faceFieldDifference(faceFields, &deriv1, NULL);

		std::cout << "eps: " << eps << std::endl;
		std::cout << "value-gradient check: " << (energy1 - energy)/ eps - deriv.dot(dir) << std::endl;
		std::cout << "gradient-hessian check: " << ((deriv1 - deriv)/ eps - hess*dir).norm() << std::endl;
	}
}

void PlaneWaveExtraction::testOptEnergyPerEdge(Eigen::VectorXd x, int eid)
{
	int f0 = _mesh.edgeFace(eid, 0);
	int f1 = _mesh.edgeFace(eid, 1);

	if(f0 == -1 || f1 == -1)
	{
		std::cout << "bnd edges!" << std::endl;
		return;
	}

	Eigen::Matrix<double, 6, 1> deriv, deriv1, dir;
	Eigen::Matrix<double, 6, 6> hess;
	dir.setRandom();

	double energy = optEnergyPerEdge(x, eid, &deriv, &hess);
	
	Eigen::VectorXd backupX = x;

	for(int i = 3; i < 10; i++)
	{
		double eps = std::pow(10, -i);
		for(int j = 0; j < 3; j++)
		{
			x(3 * f0 + j) = backupX(3 * f0 + j) + eps * dir(j);
			x(3 * f1 + j) = backupX(3 * f1 + j) + eps * dir(3 + j);
		}

		double energy1 = optEnergyPerEdge(x,eid, &deriv1, NULL);
		std::cout << "eps: " << eps << std::endl;
		std::cout << "value-gradient check: " << (energy1 - energy)/ eps - deriv.dot(dir) << std::endl;
		Eigen::VectorXd dirGrad = hess * dir;
		std::cout << "gradient-hessian check: " << ((deriv1 - deriv)/ eps - dirGrad).norm() << std::endl;

	}

}

void PlaneWaveExtraction::testOptEnergy(Eigen::VectorXd x)
{
	Eigen::VectorXd deriv, deriv1, backupX, dir;
	Eigen::SparseMatrix<double> hess, hess1;

	double energy = optEnergy(x, &deriv, &hess);
	dir = deriv;
	dir.setRandom();

	backupX = x;

	for(int i = 3; i < 10; i++)
	{
		double eps = std::pow(10, -i);
		x = backupX + eps * dir;

		double energy1 = optEnergy(x, &deriv1, NULL);

		std::cout << "eps: " << eps << std::endl;
		std::cout << "value-gradient check: " << (energy1 - energy)/ eps - deriv.dot(dir) << std::endl;
		std::cout << "gradient-hessian check: " << ((deriv1 - deriv)/ eps - hess*dir).norm() << std::endl;
	}
	backupX.setRandom();
	double energy1 = optEnergy(backupX, &deriv1, &hess1);

	std::cout << "difference in x: " << (x - backupX).norm() << std::endl;
	std::cout << "difference in hess: " << (hess - hess1).norm() << std::endl;
	std::cout << "energy difference: " << 0.5 * backupX.dot(hess1 * backupX) - energy1 << std::endl;

}