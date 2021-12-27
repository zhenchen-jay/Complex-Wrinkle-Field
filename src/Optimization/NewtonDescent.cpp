#include <Eigen/CholmodSupport>
#include <fstream>
#include <iomanip>
#include "../../include/Optimization/LineSearch.h"
#include "../../include/Optimization/NewtonDescent.h"
#include "../../include/timer.h"

void OptSolver::newtonSolver(std::function<double(const Eigen::VectorXd&, Eigen::VectorXd*, Eigen::SparseMatrix<double>*, bool)> objFunc, std::function<double(const Eigen::VectorXd&, const Eigen::VectorXd&)> findMaxStep, Eigen::VectorXd& x0, int numIter, double gradTol, double xTol, double fTol, bool disPlayInfo, std::function<void(const Eigen::VectorXd&, double&, double&)> getNormFunc, std::string* savingFolder)
{
	const int DIM = x0.rows();
	Eigen::VectorXd grad = Eigen::VectorXd::Zero(DIM);
	Eigen::SparseMatrix<double> hessian;

	Eigen::VectorXd neggrad, delta_x;
	double maxStepSize = 1.0;
	double reg = 1e-8;

	bool isProj = true;
    Timer<std::chrono::high_resolution_clock> totalTimer;
    double totalAssemblingTime = 0;
    double totalSolvingTime = 0;
    double totalLineSearchTime = 0;

    totalTimer.start();
	std::ofstream optInfo;
	if (savingFolder)
	{
		optInfo = std::ofstream((*savingFolder) + "_optInfo.txt");
		optInfo << "Newton solver with termination criterion: " << std::endl;
		std::cout << "gradient tol: " << gradTol << ", function update tol: " << fTol << ", variable update tol: " << xTol << ", maximum iteration: " << numIter << std::endl << std::endl;
	}
    // bool isProj = false;
	int i = 0;
	for (; i < numIter; i++)
	{
		if(disPlayInfo)
			std::cout << "\niter: " << i << std::endl;
		if(savingFolder)
			optInfo << "\niter: " << i << std::endl;
        Timer<std::chrono::high_resolution_clock> localTimer;
        localTimer.start();
		double f = objFunc(x0, &grad, &hessian, isProj);
        localTimer.stop();
        double localAssTime = localTimer.elapsed<std::chrono::milliseconds>() * 1e-3;
        totalAssemblingTime += localAssTime;

        localTimer.start();
		Eigen::SparseMatrix<double> H = hessian;
		Eigen::SparseMatrix<double> I(DIM, DIM);
		I.setIdentity();
		hessian = H;
		std::cout << "num of nonzeros: " << hessian.nonZeros() << ", rows: " << hessian.rows() << ", cols: " << hessian.cols() << std::endl;
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

        localTimer.stop();
        double localSolvingTime = localTimer.elapsed<std::chrono::milliseconds>() * 1e-3;
        totalSolvingTime += localSolvingTime;


		maxStepSize = findMaxStep(x0, delta_x);

        localTimer.start();
		double rate = LineSearch::backtrackingArmijo(x0, grad, delta_x, objFunc, maxStepSize);
        localTimer.stop();
        double localLinesearchTime = localTimer.elapsed<std::chrono::milliseconds>() * 1e-3;
        totalLineSearchTime += localLinesearchTime;

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
			std::cout << "f_old: " << f << ", f_new: " << fnew << ", grad norm: " << grad.norm() << ", delta x: " << rate * delta_x.norm() << ", delta_f: " << f - fnew << std::endl;
			if (getNormFunc)
			{
				double gradz, gradw;
				getNormFunc(grad, gradz, gradw);

				double updatez, updatew;
				getNormFunc(rate * delta_x, updatez, updatew);
				std::cout << "z grad: " << gradz << ", w grad: " << gradw << ", z change: " << updatez << ", w change: " << updatew  << std::endl;
			}
            std::cout << "timing info (in total seconds): " << std::endl;
            std::cout << "assembling took: " << totalAssemblingTime << ", LLT solver took: "  << totalSolvingTime << ", line search took: " << totalLineSearchTime << std::endl;
		}

		if (savingFolder)
		{
			optInfo << "line search rate : " << rate << ", actual hessian : " << !isProj << ", reg = " << reg << std::endl;
			optInfo << "f_old: " << f << ", f_new: " << fnew << ", grad norm: " << grad.norm() << ", delta x: " << rate * delta_x.norm() << ", delta_f: " << f - fnew << std::endl;
			if (getNormFunc)
			{
				double gradz, gradw;
				getNormFunc(grad, gradz, gradw);

				double updatez, updatew;
				getNormFunc(rate * delta_x, updatez, updatew);
				optInfo << "z grad: " << gradz << ", w grad: " << gradw << ", z change: " << updatez << ", w change: " << updatew << std::endl;
			}
			optInfo << "timing info (in total seconds): " << std::endl;
			optInfo << "assembling took: " << totalAssemblingTime << ", LLT solver took: " << totalSolvingTime << ", line search took: " << totalLineSearchTime << std::endl;

			if (i % 100 == 0)
			{
				std::string fileName = (*savingFolder) + "intermediate.txt";
				std::ofstream ofs(fileName);
				if (ofs)
				{
					ofs << std::setprecision(std::numeric_limits<long double>::digits10 + 1) << x0 << std::endl;
				}
			}
		}
		
		if ((f - fnew) / f < 1e-5 || rate * delta_x.norm() < 1e-5 || grad.norm() < 1e-4)
			isProj = false;

		if (rate < 1e-8)
		{
			std::cout << "terminate with small line search rate (<1e-8): L2-norm = " << grad.norm() << std::endl;
			break;
		}

		if (grad.norm() < gradTol)
		{
			std::cout << "terminate with gradient L2-norm = " << grad.norm() << std::endl;
			break;
		}
			
		if (rate * delta_x.norm() < xTol)
		{
			std::cout << "terminate with small variable change, gradient L2-norm = " << grad.norm() << std::endl;
			break;
		}
			
		if (f - fnew < fTol)
		{ 
			std::cout << "terminate with small energy change, gradient L2-norm = " << grad.norm() << std::endl;
			break;
		}
	}
	if (i >= numIter)
		std::cout << "terminate with reaching the maximum iteration, with gradient L2-norm = " << grad.norm() << std::endl;

    totalTimer.stop();
    if(disPlayInfo)
    {
        std::cout << "total time costed (s): " << totalTimer.elapsed<std::chrono::milliseconds>() * 1e-3 << ", within that, assembling took: " << totalAssemblingTime << ", LLT solver took: "  << totalSolvingTime << ", line search took: " << totalLineSearchTime << std::endl;
    }

	if (savingFolder)
	{
		optInfo << "total time costed (s): " << totalTimer.elapsed<std::chrono::milliseconds>() * 1e-3 << ", within that, assembling took: " << totalAssemblingTime << ", LLT solver took: " << totalSolvingTime << ", line search took: " << totalLineSearchTime << std::endl;

		std::string fileName = (*savingFolder) + "final_res.txt";
		std::ofstream ofs(fileName);
		if (ofs)
		{
			ofs << std::setprecision(std::numeric_limits<long double>::digits10 + 1) << x0 << std::endl;
		}


	}
		
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

