#include <Eigen/CholmodSupport>

#include "../../include/Optimization/LineSearch.h"
#include "../../include/Optimization/NewtonDescent.h"

void OptSolver::newtonSolver(std::function<double(const Eigen::VectorXd&, Eigen::VectorXd*, Eigen::SparseMatrix<double>*, bool)> objFunc, std::function<double(const Eigen::VectorXd&, const Eigen::VectorXd&)> findMaxStep, Eigen::VectorXd& x0, int numIter, double gradTol, double xTol, double fTol, bool disPlayInfo)
{
	const int DIM = x0.rows();
	Eigen::VectorXd grad = Eigen::VectorXd::Zero(DIM);
	Eigen::SparseMatrix<double> hessian;

	Eigen::VectorXd neggrad, delta_x;
	double maxStepSize = 1.0;
	double reg = 1e-8;

//	bool isProj = true;

    bool isProj = false;
	int i = 0;
	for (; i < numIter; i++)
	{
		if(disPlayInfo)
			std::cout << "\niter: " << i << std::endl;
		double f = objFunc(x0, &grad, &hessian, isProj);

		Eigen::SparseMatrix<double> H = hessian;
		Eigen::SparseMatrix<double> I(DIM, DIM);
		I.setIdentity();
		hessian = H;
		Eigen::CholmodSupernodalLLT<Eigen::SparseMatrix<double>> solver(hessian);

//		Eigen::SimplicialLLT<Eigen::SparseMatrix<double> > solver(hessian);


		while (solver.info() != Eigen::Success)
		{
			if (disPlayInfo)
			{
				if (isProj)
					std::cout << "some small perturb is needed to remove round-off error, current reg = " << reg << std::endl;
				else
					std::cout << "Matrix is not positive definite, current reg = " << reg << std::endl;
			}
				
			hessian = H + reg * I;
			solver.compute(hessian);
			reg = std::max(2 * reg, 1e-16);
		}

		neggrad = -grad;
		delta_x = solver.solve(neggrad);

		maxStepSize = findMaxStep(x0, delta_x);

		double rate = LineSearch::backtrackingArmijo(x0, grad, delta_x, objFunc, maxStepSize);

		if (!isProj)
		{
			reg *= 0.5;
			reg = std::max(reg, 1e-16);
		}
		
		x0 = x0 + rate * delta_x;

		double fnew = objFunc(x0, &grad, NULL, isProj);
		if (disPlayInfo)
		{
			std::cout << "line search rate : " << rate << ", actual hessian : " << !isProj << ", reg = " << reg << std::endl;
			std::cout << "f_old: " << f << ", f_new: " << fnew << ", grad norm: " << grad.norm() << ",  " << grad.segment(0, delta_x.size() / 2).norm() << ", " << grad.segment(delta_x.size() / 2, delta_x.size() / 2).norm() << ", delta x: " << rate * delta_x.norm() << " , z change: " << rate * delta_x.segment(0, delta_x.size() / 2).norm() << ", w change: " << rate * delta_x.segment(delta_x.size() / 2, delta_x.size() / 2).norm() << ", delta_f: " << f - fnew << std::endl;
		}
		
		if (f - fnew < 1e-5 || rate * delta_x.norm() < 1e-5 || grad.norm() < 1e-4)
			isProj = false;


		if (rate < 1e-8)
		{
			std::cout << "terminate with small line search rate (<1e-8): L2-norm = " << grad.norm() << std::endl;
			return;
		}

		if (grad.norm() < gradTol)
		{
			std::cout << "terminate with gradient L2-norm = " << grad.norm() << std::endl;
			return;
		}
			
		if (rate * delta_x.norm() < xTol)
		{
			std::cout << "terminate with small variable change, gradient L2-norm = " << grad.norm() << std::endl;
			return;
		}
			
		if (f - fnew < fTol)
		{ 
			std::cout << "terminate with small energy change, gradient L2-norm = " << grad.norm() << std::endl;
			return;
		}
	}
	if (i >= numIter)
		std::cout << "terminate with reaching the maximum iteration, with gradient L2-norm = " << grad.norm() << std::endl;
		
}


void OptSolver::testFuncGradHessian(std::function<double(const Eigen::VectorXd&, Eigen::VectorXd*, Eigen::SparseMatrix<double>*, bool)> objFunc, const Eigen::VectorXd& x0)
{
	Eigen::VectorXd dir = x0;
	dir(0) = 0;
	dir.setRandom();

	Eigen::VectorXd grad;
	Eigen::SparseMatrix<double> H;

	double f = objFunc(x0, &grad, &H, false);
	std::cout << "f: " << f << std::endl;

	for (int i = 3; i < 10; i++)
	{
		double eps = std::pow(0.1, i);
		Eigen::VectorXd x = x0 + eps * dir;
		Eigen::VectorXd grad1;
		double f1 = objFunc(x, &grad1, NULL, false);

		std::cout << "\neps: " << eps << std::endl;
		std::cout << "energy-gradient: " << (f1 - f) / eps - grad.dot(dir) << std::endl;
		std::cout << "gradient-hessian: " << ((grad1 - grad) / eps - H * dir).norm() << std::endl;

//		std::cout << "gradient-difference: \n" << (grad1 - grad) / eps << std::endl;
//		std::cout << "direction-hessian: \n" << H * dir << std::endl;
	}
}

