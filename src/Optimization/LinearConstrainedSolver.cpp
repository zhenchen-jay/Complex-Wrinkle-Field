#include <Eigen/CholmodSupport>
//#include <nasoq_eigen.h>
#include "../../include/Optimization/LineSearch.h"
#include "../../include/Optimization/LinearConstrainedSolver.h"

//static void getNumIter(double acc_thresh, int& inner_iter_ref, int& outer_iter_ref)
//{
//    if (acc_thresh >= 1e-3)
//    {
//        inner_iter_ref = outer_iter_ref = 0;
//    }
//    else if (acc_thresh < 1e-3 && acc_thresh >= 1e-6)
//    {
//        inner_iter_ref = outer_iter_ref = 1;
//    }
//    // if (acc_thresh >= 1e-3)
//    // {
//    //     inner_iter_ref = outer_iter_ref = 1;
//    // }
//    else if (acc_thresh < 1e-6 && acc_thresh >= 1e-10)
//    {
//        inner_iter_ref = outer_iter_ref = 2;
//    }
//    else if (acc_thresh < 1e-10 && acc_thresh >= 1e-13)
//    {
//        inner_iter_ref = outer_iter_ref = 3;
//    }
//    else
//    {
//        inner_iter_ref = outer_iter_ref = 9;
//    }
//}
//
//
void OptSolver::linearConstSolver(std::function<double(const Eigen::VectorXd&, Eigen::VectorXd*, Eigen::SparseMatrix<double>*, bool)> objFunc, std::function<double(const Eigen::VectorXd&, const Eigen::VectorXd&)> findMaxStep, const Eigen::SparseMatrix<double>& Aeq, const Eigen::VectorXd& beq, const Eigen::SparseMatrix<double>& Aineq, const Eigen::VectorXd& bineq, Eigen::VectorXd& x0, int numIter, double gradTol, double xTol, double fTol, bool disPlayInfo, std::function<void(const Eigen::VectorXd&, double&, double&)> getNormFunc)
{}
//    const int DIM = x0.rows();
//    Eigen::VectorXd grad = Eigen::VectorXd::Zero(DIM);
//    Eigen::SparseMatrix<double> hessian;
//
//    Eigen::VectorXd neggrad, delta_x;
//    double maxStepSize = 1.0;
//    double reg = 1e-8;
//
//    bool isProj = true;
//
//    // bool isProj = false;
//    int i = 0;
//    for (; i < numIter; i++)
//    {
//        if(disPlayInfo)
//            std::cout << "\niter: " << i << std::endl;
//        double f = objFunc(x0, &grad, &hessian, isProj);
//
//        Eigen::SparseMatrix<double> H = hessian;
//        Eigen::SparseMatrix<double> I(DIM, DIM);
//        I.setIdentity();
//        hessian = H;
//        Eigen::CholmodSupernodalLLT<Eigen::SparseMatrix<double>> solver(hessian);
//
//
//
//        while (solver.info() != Eigen::Success)
//        {
//            if (disPlayInfo)
//            {
//                if (isProj)
//                    std::cout << "some small perturb is needed to remove round-off error, current reg = " << reg << std::endl;
//                else
//                    std::cout << "Matrix is not positive definite, current reg = " << reg << std::endl;
//            }
//
//            hessian = H + reg * I;
//            solver.compute(hessian);
//            reg = std::max(2 * reg, 1e-16);
//        }
//
//        // apply nasoq to find a descent direction
//        Eigen::VectorXd y, z;
//        nasoq::QPSettings qpsettings;
//        qpsettings.eps = 1e-8;
//        getNumIter(qpsettings.eps, qpsettings.inner_iter_ref, qpsettings.outer_iter_ref);
//        qpsettings.nasoq_variant = "PREDET";
//        qpsettings.diag_perturb = 1e-10;
//
//         int converged = nasoq::quadprog(hessian.triangularView<Eigen::Lower>(), grad, Aeq, beq, Aineq, bineq, delta_x, y, z, &qpsettings);
//
//         while (converged != nasoq::nasoq_status::Optimal)
//         {
//             if(reg > 100 * hessian.norm())
//             {
//                 std::cout << "reg is larger than 100 * ||H||, but nasoq still failed!" << std::endl;
//                 if (converged == nasoq::nasoq_status::Inaccurate)
//                     std::cout << "result may be inaccurate, only primal-feasibility is satisfied." << std::endl;
//                 else if (converged == nasoq::nasoq_status::Infeasible)
//                     std::cout << "infeasible, the problem is unbounded" << std::endl;
//                 else
//                     std::cout << "NotConverged" << std::endl;
//                 return;
//             }
//         	std::cout << "nasoq solver failed!, try to increase the reg: " << reg << std::endl;
//         	reg = std::max(2 * reg, 1e-16);
//         	hessian = H + reg * I;
//         	converged = nasoq::quadprog(hessian.triangularView<Eigen::Lower>(), grad, Aeq, beq, Aineq, bineq, delta_x, y, z, &qpsettings);
//         }
//
//        maxStepSize = findMaxStep(x0, delta_x);
//
//        double rate = LineSearch::backtrackingArmijo(x0, grad, delta_x, objFunc, maxStepSize);
//
//        if (!isProj)
//        {
//            reg *= 0.5;
//            reg = std::max(reg, 1e-16);
//        }
//
//        x0 = x0 + rate * delta_x;
//
//        double fnew = objFunc(x0, &grad, NULL, isProj);
//        if (disPlayInfo)
//        {
//            std::cout << "line search rate : " << rate << ", actual hessian : " << !isProj << ", reg = " << reg << std::endl;
//            std::cout << "f_old: " << f << ", f_new: " << fnew << ", grad norm: " << grad.norm() << ", delta x: " << rate * delta_x.norm() << ", delta_f: " << f - fnew << std::endl;
//            if (getNormFunc)
//            {
//                double gradz, gradw;
//                getNormFunc(grad, gradz, gradw);
//
//                double updatez, updatew;
//                getNormFunc(rate * delta_x, updatez, updatew);
//                std::cout << "z grad: " << gradz << ", w grad: " << gradw << ", z change: " << updatez << ", w change: " << updatew  << std::endl;
//            }
//        }
//
//        if ((f - fnew) / f < 1e-5 || rate * delta_x.norm() < 1e-5 || grad.norm() < 1e-4)
//            isProj = false;
//
//
//        if (rate < 1e-8)
//        {
//            std::cout << "terminate with small line search rate (<1e-8): L2-norm = " << grad.norm() << std::endl;
//            return;
//        }
//
//        if (grad.norm() < gradTol)
//        {
//            std::cout << "terminate with gradient L2-norm = " << grad.norm() << std::endl;
//            return;
//        }
//
//        if (rate * delta_x.norm() < xTol)
//        {
//            std::cout << "terminate with small variable change, gradient L2-norm = " << grad.norm() << std::endl;
//            return;
//        }
//
//        if (f - fnew < fTol)
//        {
//            std::cout << "terminate with small energy change, gradient L2-norm = " << grad.norm() << std::endl;
//            return;
//        }
//    }
//    if (i >= numIter)
//        std::cout << "terminate with reaching the maximum iteration, with gradient L2-norm = " << grad.norm() << std::endl;
//
//}