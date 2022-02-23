#include "../../include/DynamicInterpolation/TimeIntegratedStrainDrivenModel.h"
#include "../../include/Optimization/NewtonDescent.h"
#include "../../include/IntrinsicFormula/KnoppelStripePattern.h"


void TimeIntegratedStrainDrivenModel::updateWZList()
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


double TimeIntegratedStrainDrivenModel::computeSpatialEnergyPerVertex(int vertId, const std::vector<std::complex<double>> &zvals, const Eigen::MatrixXd &w, Eigen::Vector4d *deriv, Eigen::Matrix4d *hess, bool isProj)
{
    double amp = std::abs(zvals[vertId]);
    Eigen::Vector2d omega = w.row(vertId);

    Eigen::Vector2d compress = _curCompDir.row(vertId);
    Eigen::Vector2d omegaTar = _curWTar.row(vertId);
    double ampTar = _curAmpTar(vertId);

    double energy = 0;

    if(_modelFlag == 0)
    {
        // fake stretching
        Eigen::Vector2d diff = amp * omega - compress;
        energy = 0.5 * diff.dot(diff);

        // fake bending to prevent zero amp
        energy += _fakeThickness * 0.5 * amp * omega.dot(omega);

        if(deriv || hess)
        {
            Eigen::Vector2d gradAmp;
            Eigen::Matrix2d gradOmega = Eigen::Matrix2d::Identity();

            gradAmp << zvals[vertId].real() / amp, zvals[vertId].imag() / amp;
            Eigen::Matrix<double, 2, 4> gradDiff;
            gradDiff.block<2, 2>(0, 0) = omega * gradAmp.transpose();
            gradDiff.block<2, 2>(0, 2) = amp * gradOmega;

            if(deriv)
            {
                // fake stretching
                (*deriv) = gradDiff.transpose() * diff;

                // fake bending
                deriv->segment<2>(0) += _fakeThickness * 0.5 * omega.dot(omega) * gradAmp;
                deriv->segment<2>(2) += _fakeThickness * amp * omega.transpose() * gradOmega;
            }


            if(hess)
            {
                // fake stretching
                (*hess) = gradDiff.transpose() * gradDiff;
                std::vector<Eigen::Matrix4d> hessDiff(2, Eigen::Matrix4d::Zero());

                Eigen::Matrix2d hessAmp;
                double ampCube = std::pow(amp, 3);
                hessAmp << 1 / amp - zvals[vertId].real() * zvals[vertId].real() / ampCube, -zvals[vertId].real() * zvals[vertId].imag() / ampCube, -zvals[vertId].real() * zvals[vertId].imag() / ampCube,  1 / amp - zvals[vertId].imag() * zvals[vertId].imag() / ampCube;

                hessDiff[0].block<2, 2>(0, 0) = omega(0) * hessAmp;
                hessDiff[0].block<2, 2>(0, 2) = gradAmp * gradOmega.col(0).transpose();
                hessDiff[0].block<2, 2>(2, 0) = gradOmega.col(0) * gradAmp.transpose();

                hessDiff[1].block<2, 2>(0, 0) = omega(1) * hessAmp;
                hessDiff[1].block<2, 2>(0, 2) = gradAmp * gradOmega.col(1).transpose();
                hessDiff[1].block<2, 2>(2, 0) = gradOmega.col(1) * gradAmp.transpose();

                (*hess) += diff(0) * hessDiff[0] + diff(1) * hessDiff[1];

                // fake bending
                hess->block<2, 2>(0, 0) += _fakeThickness * 0.5 * omega.dot(omega) * hessAmp;
                hess->block<2, 2>(0, 2) += _fakeThickness * gradAmp * (omega.transpose() * gradOmega);
                hess->block<2, 2>(2, 0) += _fakeThickness * (omega.transpose() * gradOmega).transpose() * gradAmp.transpose();
                hess->block<2, 2>(2, 2) += _fakeThickness * amp * (gradOmega.transpose() * gradOmega);

                if(isProj)
                    (*hess) = SPDProjection((*hess));

            }
        }
    }
    else
    {
        double ampSq = amp * amp;
        double ampTarSq = ampTar * ampTar;
        energy = 0.5 * (ampSq - ampTarSq) * (ampSq - ampTarSq) + 0.5 * _fakeThickness * (omega - omegaTar).dot(omega - omegaTar);
        // we use amp^2 to avoid zero amp issue, use 1/_fakeThickness to enlarge the influence of amplitude part
        if(deriv)
            deriv->setZero();
        if(hess)
            hess->setZero();

        if(deriv || hess)
        {
            Eigen::Vector2d gradAmpSq;
            Eigen::Matrix2d gradOmega = Eigen::Matrix2d::Identity();

            gradAmpSq << 2.0 * zvals[vertId].real(), 2.0 * zvals[vertId].imag();

            if(deriv)
            {
                deriv->segment<2>(0) = (ampSq - ampTarSq) * gradAmpSq;
                deriv->segment<2>(2) = _fakeThickness * (omega - omegaTar).transpose() * gradOmega;
            }

            if(hess)
            {
                Eigen::Matrix2d hessAmpSq = Eigen::Matrix2d::Identity() * 2;
                hess->block<2, 2>(0, 0) = ((ampSq - ampTarSq) * hessAmpSq + gradAmpSq * gradAmpSq.transpose());
                if(isProj)
                    hess->block<2, 2>(0, 0) = SPDProjection(hess->block<2, 2>(0, 0));
                hess->block<2, 2>(2, 2) = _fakeThickness * gradOmega.transpose() * gradOmega;
            }
        }
    }
    return energy;
}

double TimeIntegratedStrainDrivenModel::computeSpatialEnergy(const std::vector<std::complex<double>> &zvals, const Eigen::MatrixXd &w, Eigen::VectorXd *deriv,
                                                       std::vector<Eigen::Triplet<double>> *hessT, bool isProj){
    int nverts = _basePos.rows();
    std::vector<double> energyList(nverts);
    std::vector<Eigen::Vector4d> derivList(nverts);
    std::vector<Eigen::Matrix4d> hessList(nverts);

    auto computeEnergyPerVert = [&](const tbb::blocked_range<uint32_t>& range) {
        for (uint32_t i = range.begin(); i < range.end(); ++i)
        {
            energyList[i] = computeSpatialEnergyPerVertex(i, zvals, w,deriv ?  &(derivList[i]) : NULL, hessT? &(hessList[i]) : NULL, isProj);
        }
    };

    tbb::blocked_range<uint32_t> rangex(0u, (uint32_t)nverts, GRAIN_SIZE);
    tbb::parallel_for(rangex, computeEnergyPerVert);

    if(deriv)
        deriv->resize(4 * nverts);

    if(hessT)
        hessT->clear();

    double energy = 0;

    for(int i = 0; i < nverts; i++)
    {
        energy += energyList[i];
        if(deriv)
        {
            deriv->segment<2>(2 * i) = derivList[i].segment<2>(0);
            deriv->segment<2>(2 * i + 2 * nverts) = derivList[i].segment<2>(2);
        }
        if(hessT)
        {
            for(int j = 0; j < 2; j++)
                for(int k = 0; k < 2; k++)
                {
                    hessT->push_back({2 * i + j, 2 * i + k , hessList[i](j, k)});
                    hessT->push_back({2 * i + j + 2 * nverts, 2 * i + k , hessList[i](2 + j, k)});
                    hessT->push_back({2 * i + j, 2 * i + k+ 2 * nverts , hessList[i](j, 2 + k)});
                    hessT->push_back({2 * i + j + 2 * nverts, 2 * i + k + 2 * nverts, hessList[i](2 + j, 2 + k)});
                }

        }
    }
    return energy;
}

double TimeIntegratedStrainDrivenModel::computeEnergy(const Eigen::VectorXd &x, Eigen::VectorXd *deriv,
                                           Eigen::SparseMatrix<double> *hess, bool isProj) {
    int nverts = x.size() / 4;
    double energy = 0;
    Eigen::VectorXd xtilde = _curX + _dt * _curV;

    double p = 0;
    Eigen::VectorXd pDeriv;
    Eigen::SparseMatrix<double> pHess;
    std::vector<Eigen::Triplet<double>> pT;

    std::vector<std::complex<double>> z(nverts);
    for (int i = 0; i < nverts; i++) {
        z[i] = std::complex<double>(x(2 * i), x(2 * i + 1));
    }

    Eigen::MatrixXd w(nverts, 2);
    for (int i = 0; i < nverts; i++) {
        w.row(i) << x(2 * i + 2 * nverts), x(2 * i + 2 * nverts + 1);
    }

    p = computeSpatialEnergy(z, w, deriv ? &pDeriv : NULL, hess ? &pT : NULL, isProj);
    if (_useInertial)
        energy = 0.5 * (x - xtilde).dot(x - xtilde) + _spatialRate * _dt * _dt * p;
    else
        energy = _spatialRate * p;

    if (deriv) {
        if (_useInertial) {
            *deriv = x - xtilde + _spatialRate * _dt * _dt * pDeriv;
        } else {
            deriv->setZero(x.rows());
            (*deriv) = _spatialRate * pDeriv;
        }
    }
    if (hess) {
        pHess.resize(x.rows(), x.rows());
        pHess.setFromTriplets(pT.begin(), pT.end());

        if (_useInertial) {
            hess->resize(x.rows(), x.rows());
            hess->setIdentity();

            (*hess) += _spatialRate * _dt * _dt * pHess;
        } else
            *hess = _spatialRate * pHess;

    }
    return energy;
}

void TimeIntegratedStrainDrivenModel::solveInterpFrames()
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
        _curCompDir = _compressDir[i + 1];
        _curWTar = _wTarList[i + 1];
        _curAmpTar = _aTarList[i + 1];

        OptSolver::newtonSolver(funVal, maxStep, x, 1000, 1e-6, 0, 0, true);
        _curV = (x - _curX) / _dt;
        _curX = x;
        updateWZList();
    }
}


void TimeIntegratedStrainDrivenModel::testEnergy(Eigen::VectorXd x)
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