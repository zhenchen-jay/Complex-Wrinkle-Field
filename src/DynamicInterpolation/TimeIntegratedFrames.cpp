#include "../../include/DynamicInterpolation/TimeIntegratedFrames.h"
#include "../../include/Optimization/NewtonDescent.h"
#include "../../include/IntrinsicFormula/KnoppelStripePattern.h"


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
    int nverts = x.size() / 4;
     double energy = 0;
     Eigen::VectorXd xtilde = _curX + _dt * _curV;

     double p = 0;
     Eigen::VectorXd pDeriv;
     Eigen::SparseMatrix<double> pHess;
     std::vector<Eigen::Triplet<double>> pT;

     std::vector<std::complex<double>> z(nverts);
     for (int i = 0; i < nverts; i++)
     {
         z[i] = std::complex<double>(x(2 * i), x(2 * i + 1));
     }

     if (_knopType == W_WTar)
     {
         if(deriv)
            pDeriv.setZero(x.rows());

         for (int i = 0; i < nverts; i++)
         {
             p += (x(2 * i + 2 * nverts) - _wTar(i, 0)) * (x(2 * i + 2 * nverts) - _wTar(i, 0)) + (x(2 * i + 2 * nverts + 1) - _wTar(i, 1)) * (x(2 * i + 2 * nverts + 1) - _wTar(i, 1));
             if (deriv)
             {
                 pDeriv(2 * nverts + 2 * i) = 2 * (x(2 * i + 2 * nverts) - _wTar(i, 0));
                 pDeriv(2 * nverts + 2 * i + 1) = 2 * (x(2 * i + 2 * nverts + 1) - _wTar(i, 1));
             }

             if (hess)
             {
                 pT.push_back({ 2 * nverts + 2 * i , 2 * nverts + 2 * i, 2.0 });
                 pT.push_back({ 2 * nverts + 2 * i + 1, 2 * nverts + 2 * i + 1, 2.0 });
             }

         }
     }
     else if (_knopType == Z_WTar || _knopType == WZ_Tar)
     {
        
         if (deriv)
             pDeriv.setZero(x.rows());

         Eigen::VectorXd sDeriv;
         p = IntrinsicFormula::KnoppelEnergy(_baseMesh, _halfEdgewTar, _faceArea, _cotEntries, z, deriv ? &sDeriv : NULL, hess ? &pT : NULL);

         if (deriv)
            pDeriv.segment(0, 2 * nverts) = sDeriv;

         if (_knopType == WZ_Tar)
         {
             for (int i = 0; i < nverts; i++)
             {
                 p += (x(2 * i + 2 * nverts) - _wTar(i, 0)) * (x(2 * i + 2 * nverts) - _wTar(i, 0)) + (x(2 * i + 2 * nverts + 1) - _wTar(i, 1)) * (x(2 * i + 2 * nverts + 1) - _wTar(i, 1));
                 if (deriv)
                 {
                     pDeriv(2 * nverts + 2 * i) += 2 * (x(2 * i + 2 * nverts) - _wTar(i, 0));
                     pDeriv(2 * nverts + 2 * i + 1) += 2 * (x(2 * i + 2 * nverts + 1) - _wTar(i, 1));
                 }

                 if (hess)
                 {
                     pT.push_back({ 2 * nverts + 2 * i , 2 * nverts + 2 * i, 2.0 });
                     pT.push_back({ 2 * nverts + 2 * i + 1, 2 * nverts + 2 * i + 1, 2.0 });
                 }

             }
         }
     }
     else
     {
         Eigen::MatrixXd w(nverts, 2);
         for (int i = 0; i < nverts; i++)
         {
             w.row(i) << x(2 * i + 2 * nverts), x(2 * i + 2 * nverts + 1);
         }
         p = IntrinsicFormula::KnoppelEnergyFor2DVertexOmega(_basePos, _baseMesh, _faceArea, _cotEntries, z, w, deriv ? &pDeriv : NULL, hess ? &pT : NULL, isProj);
     }
     

     Eigen::VectorXd penaltyDeriv;
     std::vector<Eigen::Triplet<double>> penaltyT;
     Eigen::SparseMatrix<double> penaltyHess;

     double penalty = unitMagEnergy(z, deriv ? &penaltyDeriv : NULL, hess ? &penaltyT : NULL, isProj);

     if (_useInertial)
         energy = 0.5 * (x - xtilde).dot(x - xtilde) + _K * _dt * _dt * p + _unitPenaltyCoeff * _dt * _dt * penalty;
     else
         energy = _K * p + _unitPenaltyCoeff * penalty;

     if(deriv)
     {
         if (_useInertial)
         {
             *deriv = x - xtilde;
             (*deriv) += _K * _dt * _dt * pDeriv;
             deriv->segment(0, 2 * nverts) = _unitPenaltyCoeff * _dt * _dt * penaltyDeriv;
         }
         else
         {
             deriv->setZero(x.rows());
             (*deriv) = _K * pDeriv;
             deriv->segment(0, 2 * nverts) = _unitPenaltyCoeff * penaltyDeriv;
         }
     }
     /*if(hess)
     {
         pHess.resize(x.rows(), x.rows());
         pHess.setFromTriplets(pT.begin(), pT.end());

         penaltyHess.resize(x.rows(), x.rows());
         penaltyHess.setFromTriplets(penaltyT.begin(), penaltyT.end());

         if (_useInertial)
         {
             hess->resize(x.rows(), x.rows());
             hess->setIdentity();

             (*hess) += _K * _dt * _dt * pHess + _unitPenaltyCoeff * _dt * _dt * penaltyHess;
         }
         else
             *hess = _K * pHess + _unitPenaltyCoeff * penaltyHess;
         
     }*/
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
    _wTar.setZero(x.size() / 4, 3);
    for(int i = 0; i < _numFrames; i++)
    {
        double t = (i + 1) * _dt;
        _wTar.block(0, 0, _wTar.rows(), 2) = (1 - t) * _w0 + t * _w1;
        _halfEdgewTar = vertexVec2IntrinsicHalfEdgeVec(_wTar, _basePos, _baseMesh);

        OptSolver::newtonSolver(funVal, maxStep, x, 1000, 1e-6, 0, 0, true);
        _curV = (x - _curX) / _dt;
        _curX = x;
        updateWZList();
    }
}


void TimeIntegratedFrames::testEnergy(Eigen::VectorXd x)
{
    Eigen::VectorXd deriv;
    Eigen::SparseMatrix<double> hess;

    double e = computeEnergy(x, &deriv, &hess, false);
    std::cout << "energy: " << e << std::endl;

    Eigen::VectorXd dir = deriv;
    dir.setRandom();

    for (int i = 3; i < 9; i++)
    {
        double eps = std::pow(0.1, i);

        Eigen::VectorXd deriv1;
        double e1 = computeEnergy(x + eps * dir, &deriv1, NULL, false);

        std::cout << "eps: " << eps << std::endl;
        std::cout << "value-gradient check: " << (e1 - e) / eps - dir.dot(deriv) << std::endl;
        std::cout << "gradient-hessian check: " << ((deriv1 - deriv) / eps - hess * dir).norm() << std::endl;
    }
}