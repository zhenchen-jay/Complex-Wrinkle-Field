#include <Eigen/CholmodSupport>
#include <iomanip>
#include <iostream>
#include "../../include/Optimization/LineSearch.h"
#include "../../include/Optimization/LBFGSSolver.h"
#include "../../include/timer.h"

void OptSolver::lbfgsSolver(std::function<double(const Eigen::VectorXd&, Eigen::VectorXd*, Eigen::SparseMatrix<double>*, bool)> objFunc, std::function<double(const Eigen::VectorXd&, const Eigen::VectorXd&)> findMaxStep, Eigen::VectorXd& x0, int numIter, double gradTol, double xTol, double fTol, bool displayInfo, std::function<void(const Eigen::VectorXd&, double&, double&)> getNormFunc)
{
	std::cout << "LBFGS-Solver" << std::endl;
	const size_t m = 10;
	const size_t DIM = x0.rows();
	Eigen::MatrixXd sVector = Eigen::MatrixXd::Zero(DIM, m);
	Eigen::MatrixXd yVector = Eigen::MatrixXd::Zero(DIM, m);
	Eigen::VectorXd alpha = Eigen::VectorXd::Zero(m);
	Eigen::VectorXd grad(DIM), q(DIM), grad_old(DIM), s(DIM), y(DIM);
	double f = objFunc(x0, &grad, NULL, false);
	Eigen::VectorXd x_old = x0;

	if (displayInfo)
	{
		std::cout << "start energy: " << f << std::endl;
		std::cout << "start gradient norm: " << grad.template lpNorm<Eigen::Infinity>() << std::endl;
	}


	size_t iter = 0, globIter = 0;
	double H0k = 1;
	do {
		const double relativeEpsilon = static_cast<double>(0.0001) * std::max<double>(static_cast<double>(1.0), x0.norm());

		if (grad.norm() < gradTol)
		{
			std::cout << "gradient is too small, ||g||_2: " << grad.norm() << std::endl;
			break;
		}
		//Algorithm 7.4 (L-BFGS two-loop recursion)
		q = grad;
		const int k = std::min<double>(m, iter);

		// for i = k − 1, k − 2, . . . , k − m§
		for (int i = k - 1; i >= 0; i--) {
			// alpha_i <- rho_i*s_i^T*q
			const double rho = 1.0 / static_cast<Eigen::VectorXd>(sVector.col(i))
				.dot(static_cast<Eigen::VectorXd>(yVector.col(i)));
			alpha(i) = rho * static_cast<Eigen::VectorXd>(sVector.col(i)).dot(q);
			// q <- q - alpha_i*y_i
			q = q - alpha(i) * yVector.col(i);
		}

		// r <- H_k^0*q
		q = H0k * q;
		//for i k − m, k − m + 1, . . . , k − 1
		for (int i = 0; i < k; i++) {
			// beta <- rho_i * y_i^T * r
			const double rho = 1.0 / static_cast<Eigen::VectorXd>(sVector.col(i))
				.dot(static_cast<Eigen::VectorXd>(yVector.col(i)));
			const double beta = rho * static_cast<Eigen::VectorXd>(yVector.col(i)).dot(q);
			// r <- r + s_i * ( alpha_i - beta)
			q = q + sVector.col(i) * (alpha(i) - beta);
		}
		// stop with result "H_k*f_f'=q"
		double alpha_init = findMaxStep(x0, -q);
		double rate = LineSearch::backtrackingArmijo(x0, grad, -q, objFunc, alpha_init);


		// update guess  
		x0 = x0 - rate * q;

		grad_old = grad;
		double fold = f;
		f = objFunc(x0, &grad, NULL, false);

		s = x0 - x_old;
		y = grad - grad_old;

		// update the history
		if (iter < m) {
			sVector.col(iter) = s;
			yVector.col(iter) = y;
		}
		else {

			sVector.leftCols(m - 1) = sVector.rightCols(m - 1).eval();
			sVector.rightCols(1) = s;
			yVector.leftCols(m - 1) = yVector.rightCols(m - 1).eval();
			yVector.rightCols(1) = y;
		}
		// update the scaling factor
		H0k = y.dot(s) / static_cast<double>(y.dot(y));

		x_old = x0;
		if (displayInfo)
		{
			std::cout << std::endl << "iter: " << globIter << ", linesearch rate: " << rate << std::endl;
			std::cout << std::setprecision(10) << "fold = " << fold << ", f = " << f << ", ||grad|| " << grad.norm() << ", delta x: " << rate * q.norm() << ", delta_f: " << fold - f << std::endl;
			if (getNormFunc)
			{
				double gradz, gradw;
				getNormFunc(grad, gradz, gradw);

				double updatez, updatew;
				getNormFunc(rate * q, updatez, updatew);
				std::cout << "z grad: " << gradz << ", w grad: " << gradw << ", z change: " << updatez << ", w change: " << updatew << std::endl;
			}
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

		if (rate * q.norm() < xTol)
		{
			std::cout << "terminate with small variable change, gradient L2-norm = " << grad.norm() << std::endl;
			break;
		}

		if (fold - f < fTol)
		{
			std::cout << "terminate with small energy change, gradient L2-norm = " << grad.norm() << std::endl;
			break;
		}
		
		iter++;
		globIter++;
	} while (globIter < numIter);

}