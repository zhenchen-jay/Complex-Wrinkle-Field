#pragma once
#include "CommonFunctions.h"


namespace OptSolver
{
	void augmentedLagrangianSolver(
		std::function<double(const Eigen::VectorXd&, Eigen::VectorXd*, Eigen::SparseMatrix<double>*, bool)> objFunc, 
		std::function<double(const Eigen::VectorXd&, const Eigen::VectorXd&)> findMaxStep, 
		std::function<double(const Eigen::VectorXd&, const Eigen::VectorXd&, Eigen::VectorXd*, Eigen::SparseMatrix<double>*, bool)> constraintsFunc, 
		std::function<double(const Eigen::VectorXd&, const Eigen::VectorXd&, Eigen::VectorXd*, Eigen::SparseMatrix<double>*, bool)> penaltyFunc,
		Eigen::VectorXd& x0, Eigen::VectorXd& lambda, Eigen::VectorXd& mu,
		int numIter = 1000, double gradTol = 1e-6, double xTol = 1e-8, double cTol = 1e-4, bool displayInfo = false, std::function<void(const Eigen::VectorXd&, double&, double&)> getNormFunc = NULL, std::string* savingFolder = NULL, std::function<void(Eigen::VectorXd&)> postProcess = NULL);
}


