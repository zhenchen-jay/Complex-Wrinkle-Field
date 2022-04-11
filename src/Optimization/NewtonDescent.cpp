#include <Eigen/CholmodSupport>
#include <fstream>
#include <iomanip>
#include "../../include/Optimization/LineSearch.h"
#include "../../include/Optimization/NewtonDescent.h"
#include "../../include/timer.h"
// #include "SuiteSparse_config.h"

void OptSolver::newtonSolver(std::function<double(const Eigen::VectorXd&, Eigen::VectorXd*, Eigen::SparseMatrix<double>*, bool)> objFunc, std::function<double(const Eigen::VectorXd&, const Eigen::VectorXd&)> findMaxStep, Eigen::VectorXd& x0, int numIter, double gradTol, double xTol, double fTol, bool disPlayInfo, std::function<void(const Eigen::VectorXd&, double&, double&)> getNormFunc, std::string* savingFolder, std::function<void(Eigen::VectorXd&)> postProcess)
{
	const int DIM = x0.rows();
    Eigen::VectorXd randomVec = x0;
    randomVec.setRandom();
    x0 += 1e-6 * randomVec;
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
		optInfo = std::ofstream((*savingFolder) + "optInfo.txt");
		optInfo << "Newton solver with termination criterion: " << std::endl;
		std::cout << "gradient tol: " << gradTol << ", function update tol: " << fTol << ", variable update tol: " << xTol << ", maximum iteration: " << numIter << std::endl << std::endl;
	}
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
		std::cout << "num of nonzeros: " << H.nonZeros() << ", rows: " << H.rows() << ", cols: " << H.cols() << std::endl;
		Eigen::CholmodSupernodalLLT<Eigen::SparseMatrix<double>> solver(H);

		// Eigen::CholmodSimplicialLLT<Eigen::SparseMatrix<double> > solver(H);


		while (solver.info() != Eigen::Success)
		{
			if (disPlayInfo)
			{
				if (isProj)
					std::cout << "some small perturb is needed to remove round-off error, current reg = " << reg << std::endl;
				else
					std::cout << "Matrix is not positive definite, current reg = " << reg << std::endl;
			}
				
			H = hessian + reg * I;
			solver.compute(H);
			reg = std::max(2 * reg, 1e-16);

            if(reg > 1e4)
            {
                std::cout << "reg is too large, use SPD hessian instead." << std::endl;
                reg = 1e-6;
                isProj = true;
                f = objFunc(x0, &grad, &hessian, isProj);
            }
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

        if(postProcess)
        {
            postProcess(x0);
            if (disPlayInfo)
            {
                double tmpfnew = objFunc(x0, &grad, NULL, isProj);
                std::cout << "after post process, f_new: " << tmpfnew << ", grad norm: " << grad.norm() << std::endl;
            }

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

		}
		
		if ((f - fnew) / f < 1e-5 || delta_x.norm() < 1e-5 || grad.norm() < 1e-4)
		{
			isProj = false;
			// double f = objFunc(x0, &grad, &hessian, isProj);
			// H = hessian;
			// solver.compute(hessian);
			// double tmpReg = 1e-8;
			// while (solver.info() != Eigen::Success)
			// {	
			// 	hessian = H + tmpReg * I;
			// 	solver.compute(hessian);
			// 	tmpReg = std::max(2 * tmpReg, 1e-16);
			// }
			// if(tmpReg > 1e-5)
			// {
			// 	std::cout << "hessian is far away from PD, stick with PD hessian." << std::endl;
			// 	isProj = true;
			// }

		}
			

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

    double f = objFunc(x0, &grad, &hessian, false);
    Eigen::SparseMatrix<double> I(DIM, DIM), H = hessian;
    I.setIdentity();
    Eigen::CholmodSupernodalLLT<Eigen::SparseMatrix<double>> solver(H);

    reg = 1e-10;
    while (solver.info() != Eigen::Success)
    {
        if (disPlayInfo)
        {
            if (isProj)
                std::cout << "some small perturb is needed to remove round-off error, current reg = " << reg << std::endl;
            else
                std::cout << "Matrix is not positive definite, current reg = " << reg << std::endl;
        }

        H = hessian + reg * I;
        solver.compute(H);
        reg = std::max(2 * reg, 1e-16);
    }
    std::cout << "end up with energy: " << f << ", gradient: " << grad.norm() << std::endl;
    std::cout << "mininal evalues of hessian is between: " << reg / 2 << ", " << reg << std::endl;

    totalTimer.stop();
    if(disPlayInfo)
    {
        std::cout << "total time costed (s): " << totalTimer.elapsed<std::chrono::milliseconds>() * 1e-3 << ", within that, assembling took: " << totalAssemblingTime << ", LLT solver took: "  << totalSolvingTime << ", line search took: " << totalLineSearchTime << std::endl;
    }
		
}


