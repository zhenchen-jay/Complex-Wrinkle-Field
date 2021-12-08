#include "../../include/DynamicInterpolation/TimeIntegratedFrames.h"
#include "../../include/Optimization/NewtonDescent.h"

void TimeIntegratedFrames::updateWZList()
{
    int nverts = _basePos.rows();
    Eigen::MatrixXd w(nverts, 2);
    std::vector<std::complex<double>> z(nverts);

    for(int i =0; i < nverts; i++)
    {
        z[i] = std::complex<double>(_curX(2 * i), _curX(2 * i + 1));
        w.row(i) << _curX(2 * i + 2 * nverts), _curX(2 * i + 2 * nverts + 1);
    }
    _wList.push_back(w);
    _vertValsList.push_back(z);
}

double TimeIntegratedFrames::computeEnergy(const Eigen::VectorXd &x, Eigen::VectorXd *deriv,
                                           Eigen::SparseMatrix<double> *hess, bool isProj)
{
     double energy = 0;
     Eigen::VectorXd xtilde = _curX + _dt * _curV;

     double K = 0.1;
     energy = 0.5 * (x - xtilde).dot(x - xtilde) + 0.5 * K * (x - _tarX).dot(x - _tarX);

     if(deriv)
     {
         *deriv = x - xtilde + K * (x - _tarX);
     }
     if(hess)
     {
         hess->resize(x.rows(), x.rows());
         hess->setIdentity();
         *hess *= (1 + K);
     }
     return energy;
}

void TimeIntegratedFrames::solveInterpFrames()
{
    Eigen::VectorXd x = _initX;

    auto funVal = [&](const Eigen::VectorXd& x, Eigen::VectorXd* grad, Eigen::SparseMatrix<double>* hess, bool isProj) {
        Eigen::VectorXd deriv;
        Eigen::SparseMatrix<double> H;
        double E = computeEnergy(x, grad ? &deriv : NULL, hess ? &H : NULL, isProj);

        if (grad)
        {
            (*grad) = deriv;
        }

        if (hess)
        {
            (*hess) = H;
        }

        return E;
    };

    auto maxStep = [&](const Eigen::VectorXd& x, const Eigen::VectorXd& dir) {
        return 1.0;
    };

    for(int i = 0; i < _numFrames; i++)
    {
        OptSolver::newtonSolver(funVal, maxStep, x, 1000, 1e-6, 0, 0, false);
        _curV = (x - _curX) / _dt;
        _curX = x;
        updateWZList();
    }
}