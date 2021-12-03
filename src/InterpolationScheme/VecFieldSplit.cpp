#include "../../include/InterpolationScheme/VecFieldSplit.h"
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <iostream>

#ifndef GRAIN_SIZE
#define GRAIN_SIZE 10
#endif

double VecFieldsSplit::planeWaveSmoothnessPerface(const Eigen::MatrixXd vecFields, int faceId, Eigen::VectorXd *deriv,
                                                 Eigen::MatrixXd *hess)
{
    double energy = 0;
    int dim = vecFields.cols();

    if(deriv)
        deriv->setZero(3 * dim);
    if(hess)
        hess->setZero(3 * dim, 3 * dim);

    for(int j = 0; j < 3; j++)
    {
        int vid = _mesh.faceVertex(faceId, j);
        int vid1 = _mesh.faceVertex(faceId, (j+1)%3);

        Eigen::VectorXd diff = vecFields.row(vid) - vecFields.row(vid1);
        energy += 0.5 * diff.squaredNorm();

        if(deriv || hess)
        {
            Eigen::MatrixXd gradDiff = Eigen::MatrixXd::Zero(dim, 3 * dim);
            gradDiff.block(0, j*dim, dim, dim).setIdentity();
            gradDiff.block(0, ((j + 1)%3)*dim, dim, dim).setIdentity();
            gradDiff.block(0, ((j + 1)%3)*dim, dim, dim) *= -1;

            if(deriv)
                (*deriv) += gradDiff.transpose() * diff;

            if(hess)
                (*hess) += gradDiff.transpose() * gradDiff;
        }
    }
    return energy;
}

double VecFieldsSplit::planeWaveSmoothness(const Eigen::MatrixXd vecFields, Eigen::VectorXd *deriv,
                                          Eigen::SparseMatrix<double> *hess)
{
    int nfaces = _mesh.nFaces();
    int nverts = _pos.rows();

    int dim = vecFields.cols();

    if(deriv)
        deriv->setZero(dim * nverts);

    std::vector<double> energyList(nfaces);
    std::vector<Eigen::VectorXd> derivList(nfaces);
    std::vector<Eigen::MatrixXd> hessList(nfaces);

    auto computeEnergy = [&](const tbb::blocked_range<uint32_t>& range){
        for (uint32_t i = range.begin(); i < range.end(); ++i)
        {
            energyList[i] = planeWaveSmoothnessPerface(vecFields, i, deriv ? &(derivList[i]) : NULL, hess ? &(hessList[i]) : NULL);
        }
    };
    tbb::blocked_range<uint32_t> rangex(0u, (uint32_t)nfaces, GRAIN_SIZE);
    tbb::parallel_for(rangex, computeEnergy);

    double energy = 0;
    std::vector<Eigen::Triplet<double>> T;

    for(int i = 0; i < nfaces; i++)
    {
        energy += energyList[i];
        for(int j = 0; j < 3; j++)
        {
            int vid = _mesh.faceVertex(i, j);
            if(deriv)
            {
                for(int k = 0; k < dim; k++)
                {
                    (*deriv)(dim * vid + k) += derivList[i](dim * j + k);
                }
            }
            if(hess)
            {
                for(int k = 0; k < 3; k++)
                {
                    int vid1 = _mesh.faceVertex(i, k);
                    for(int m = 0; m < dim; m++)
                        for(int n = 0; n < dim; n++)
                        {
                            T.push_back({dim * vid + m, dim * vid1 + n, hessList[i](dim * j + m, dim * k + n)});
                        }
                }
            }
        }
    }
    if(hess)
    {
        hess->resize(dim * nverts, dim * nverts);
        hess->setFromTriplets(T.begin(), T.end());
    }
    return energy;
}

double VecFieldsSplit::whirlpoolSmoothnessPerface(const Eigen::MatrixXd vecFields, int faceId, Eigen::VectorXd *deriv,
                                                 Eigen::MatrixXd *hess, bool isPorj)
{
    // assume dim = 2
    double energy = 0;
    int dim = vecFields.cols();

    if(deriv)
        deriv->setZero(3 * dim);
    if(hess)
        hess->setZero(3 * dim, 3 * dim);

    std::vector<Eigen::Vector2d> centers;
    std::vector<Eigen::MatrixXd> gradCenters;
    std::vector<Eigen::MatrixXd> hessCentersFirstCoord;
    std::vector<Eigen::MatrixXd> hessCentersSecondCoord;

    for(int i = 0; i < 3; i++)
    {
        int vid = _mesh.faceVertex(faceId, i);
        Eigen::Vector2d pi = _pos.row(vid).segment<2>(0);
        Eigen::Vector2d wPerp;
        wPerp << -vecFields(vid, 1), vecFields(vid, 0);
        double sqNorm = vecFields(vid, 0) * vecFields(vid, 0) + vecFields(vid, 1) * vecFields(vid, 1);

        Eigen::Vector2d p0 = pi + wPerp / sqNorm;
        centers.push_back(p0);

        if(deriv || hess)
        {
            Eigen::MatrixXd gradP0 = Eigen::MatrixXd::Zero(2, 3 * dim);
            gradP0.col(dim * i) = Eigen::Vector2d(0, 1) / sqNorm -  2 * vecFields(vid, 0) / (sqNorm * sqNorm) * wPerp;
            gradP0.col(dim * i + 1) = Eigen::Vector2d(-1, 0) / sqNorm -  2 * vecFields(vid, 1) / (sqNorm * sqNorm) * wPerp;

            gradCenters.push_back(gradP0);

            if(hess)
            {
                Eigen::MatrixXd hess0 = Eigen::MatrixXd::Zero(3 * dim, 3 * dim);
                Eigen::MatrixXd hess1 = Eigen::MatrixXd::Zero(3 * dim, 3 * dim);

                double x = vecFields(vid, 0);
                double y = vecFields(vid, 1);

                hess0(dim * i, dim * i) = -8 * x * x * y / (sqNorm * sqNorm * sqNorm) + 2 * y / (sqNorm * sqNorm);
                hess0(dim * i, dim * i + 1) = -8 * x * y * y / (sqNorm * sqNorm * sqNorm) + 2 * x / (sqNorm * sqNorm);
                hess0(dim * i + 1, dim * i) = hess0(dim * i, dim * i + 1);
                hess0(dim * i + 1, dim * i + 1) = -8 * y * y * y / (sqNorm * sqNorm * sqNorm) + 6 * y / (sqNorm * sqNorm);


                hess1(dim * i, dim * i) = 8 * x * x * x / (sqNorm * sqNorm * sqNorm) - 6 * x / (sqNorm * sqNorm);
                hess1(dim * i, dim * i + 1) = 8 * x * x * y / (sqNorm * sqNorm * sqNorm) - 2 * y / (sqNorm * sqNorm);
                hess1(dim * i + 1, dim * i) = hess1(dim * i, dim * i + 1);
                hess1(dim * i + 1, dim * i + 1) = 8 * x * y * y / (sqNorm * sqNorm * sqNorm) - 2 * x / (sqNorm * sqNorm);


                hessCentersFirstCoord.push_back(hess0);
                hessCentersSecondCoord.push_back(hess1);
            }
        }
    }

    for(int i = 0; i < 3; i++)
    {
        Eigen::Vector2d diff = centers[i] - centers[(i+1)%3];
        energy += 0.5 * diff.squaredNorm();
        if(deriv)
        {
            (*deriv) += (gradCenters[i] - gradCenters[(i + 1)%3]).transpose() * diff;
        }
        if(hess)
        {
            (*hess) += (gradCenters[i] - gradCenters[(i + 1)%3]).transpose() * (gradCenters[i] - gradCenters[(i + 1)%3]);
            (*hess) += diff(0) * (hessCentersFirstCoord[i] - hessCentersFirstCoord[(i+1)%3]) + diff(1) * (hessCentersSecondCoord[i] - hessCentersSecondCoord[(i+1)%3]);
        }
    }
    if(hess && isPorj)
        (*hess) = SPDProjection((*hess));
    return energy;
}

double VecFieldsSplit::whirlpoolSmoothness(const Eigen::MatrixXd vecFields, Eigen::VectorXd *deriv,
                                          Eigen::SparseMatrix<double> *hess, bool isProj)
{
    int nfaces = _mesh.nFaces();
    int nverts = _pos.rows();

    int dim = vecFields.cols();

    if(deriv)
        deriv->setZero(dim * nverts);

    std::vector<double> energyList(nfaces);
    std::vector<Eigen::VectorXd> derivList(nfaces);
    std::vector<Eigen::MatrixXd> hessList(nfaces);

    auto computeEnergy = [&](const tbb::blocked_range<uint32_t>& range){
        for (uint32_t i = range.begin(); i < range.end(); ++i)
        {
            energyList[i] = whirlpoolSmoothnessPerface(vecFields, i, deriv ? &(derivList[i]) : NULL, hess ? &(hessList[i]) : NULL, isProj);
        }
    };
    tbb::blocked_range<uint32_t> rangex(0u, (uint32_t)nfaces, GRAIN_SIZE);
    tbb::parallel_for(rangex, computeEnergy);

    double energy = 0;
    std::vector<Eigen::Triplet<double>> T;

    for(int i = 0; i < nfaces; i++)
    {
        energy += energyList[i];
        for(int j = 0; j < 3; j++)
        {
            int vid = _mesh.faceVertex(i, j);
            if(deriv)
            {
                for(int k = 0; k < dim; k++)
                {
                    (*deriv)(dim * vid + k) += derivList[i](dim * j + k);
                }
            }
            if(hess)
            {
                for(int k = 0; k < 3; k++)
                {
                    int vid1 = _mesh.faceVertex(i, k);
                    for(int m = 0; m < dim; m++)
                        for(int n = 0; n < dim; n++)
                        {
                            T.push_back({dim * vid + m, dim * vid1 + n, hessList[i](dim * j + m, dim * k + n)});
                        }
                }
            }
        }
    }
    if(hess)
    {
        hess->resize(dim * nverts, dim * nverts);
        hess->setFromTriplets(T.begin(), T.end());
    }

    return energy;
}

double VecFieldsSplit::optEnergy(const Eigen::VectorXd &x, Eigen::VectorXd *deriv, Eigen::SparseMatrix<double> *hess, bool isProj)
{
    // step 1: convert x into vector fields
    Eigen::MatrixXd vecFields(_pos.rows(), 2);
    for(int i = 0; i < _pos.rows(); i++)
        vecFields.row(i) = x.segment<2>(2 * i).transpose();

    Eigen::VectorXd whirlPoolDeriv, planeDeriv;
    Eigen::SparseMatrix<double> whirlPoolHess, planeHess;
    double energy = whirlpoolSmoothness(_inputFields - vecFields, deriv ? & whirlPoolDeriv : NULL, hess ? &whirlPoolHess : NULL, isProj);
    double energy1 = planeWaveSmoothness(vecFields, deriv? &planeDeriv : NULL, hess? &planeHess : NULL);

    if(deriv)
        (*deriv) = -whirlPoolDeriv + planeDeriv;
    if(hess)
        (*hess) = whirlPoolHess + planeHess;

    return energy + energy1;

}

//////////////////////////////////////////////////// test functions ////////////////////////////////////////////////////////////////////
void VecFieldsSplit::testPlaneWaveSmoothnessPerface(Eigen::MatrixXd vecFields, int faceId)
{
    Eigen::VectorXd deriv;
    Eigen::MatrixXd hess;
    double energy = planeWaveSmoothnessPerface(vecFields, faceId, &deriv, &hess);
    Eigen::VectorXd dir;
    dir = deriv;
    dir.setRandom();
    Eigen::MatrixXd backupVec = vecFields;
    for(int i = 3; i < 9; i++)
    {
        double eps = std::pow(10, -i);
        for(int j = 0; j < 3; j++)
        {
            int vid = _mesh.faceVertex(faceId, j);
            backupVec(vid, 0) = vecFields(vid, 0) + eps * dir(2 * j);
            backupVec(vid, 1) = vecFields(vid, 1) + eps * dir(2 * j + 1);
        }

        Eigen::VectorXd deriv1;
        double energy1 = planeWaveSmoothnessPerface(backupVec, faceId, &deriv1, NULL);
        std::cout << "eps: " << eps << std::endl;
        std::cout << "value-gradient check: " << (energy1 - energy)/eps - deriv.dot(dir) << std::endl;
        std::cout << "gradient-hessian check: " << ((deriv1 - deriv) / eps - hess * dir).norm() << std::endl;
    }
}

void VecFieldsSplit::testWhirlpoolSmoothnessPerface(Eigen::MatrixXd vecFields, int faceId)
{
    std::cout << "test per face whirl pool" << std::endl;
    Eigen::VectorXd deriv;
    Eigen::MatrixXd hess;
    double energy = whirlpoolSmoothnessPerface(vecFields, faceId, &deriv, &hess);
    Eigen::VectorXd dir;
    dir = deriv;
    dir.setRandom();
    Eigen::MatrixXd backupVec = vecFields;
    for(int i = 3; i < 9; i++)
    {
        double eps = std::pow(10, -i);
        for(int j = 0; j < 3; j++)
        {
            int vid = _mesh.faceVertex(faceId, j);
            backupVec(vid, 0) = vecFields(vid, 0) + eps * dir(2 * j);
            backupVec(vid, 1) = vecFields(vid, 1) + eps * dir(2 * j + 1);
        }

        Eigen::VectorXd deriv1;
        double energy1 = whirlpoolSmoothnessPerface(backupVec, faceId, &deriv1, NULL);
        std::cout << "eps: " << eps << std::endl;
        std::cout << "value-gradient check: " << (energy1 - energy)/eps - deriv.dot(dir) << std::endl;
        std::cout << "gradient-hessian check: " << ((deriv1 - deriv) / eps - hess * dir).norm() << std::endl;
    }
}

void VecFieldsSplit::testPlaneWaveSmoothness(Eigen::MatrixXd vecFields)
{
    Eigen::VectorXd deriv;
    Eigen::SparseMatrix<double> hess;
    double energy = planeWaveSmoothness(vecFields, &deriv, &hess);
    Eigen::VectorXd dir;
    dir = deriv;
    dir.setRandom();
    Eigen::MatrixXd backupVec = vecFields;
    for(int i = 3; i < 9; i++)
    {
        double eps = std::pow(10, -i);
        for(int j = 0; j < _pos.rows(); j++)
        {
            backupVec(j, 0) = vecFields(j, 0) + eps * dir(2 * j);
            backupVec(j, 1) = vecFields(j, 1) + eps * dir(2 * j + 1);
        }


        Eigen::VectorXd deriv1;
        double energy1 = planeWaveSmoothness(backupVec, &deriv1, NULL);

        std::cout << "eps: " << eps << std::endl;
        std::cout << "value-gradient check: " << (energy1 - energy)/eps - deriv.dot(dir) << std::endl;
        std::cout << "gradient-hessian check: " << ((deriv1 - deriv) / eps - hess * dir).norm() << std::endl;
    }
}

void VecFieldsSplit::testWhirlpoolSmoothness(Eigen::MatrixXd vecFields)
{
    Eigen::VectorXd deriv;
    Eigen::SparseMatrix<double> hess;
    double energy = whirlpoolSmoothness(vecFields, &deriv, &hess);
    Eigen::VectorXd dir;
    dir = deriv;
    dir.setRandom();
    Eigen::MatrixXd backupVec = vecFields;
    for(int i = 3; i < 9; i++)
    {
        double eps = std::pow(10, -i);
        for(int j = 0; j < _pos.rows(); j++)
        {
            backupVec(j, 0) = vecFields(j, 0) + eps * dir(2 * j);
            backupVec(j, 1) = vecFields(j, 1) + eps * dir(2 * j + 1);
        }


        Eigen::VectorXd deriv1;
        double energy1 = whirlpoolSmoothness(backupVec, &deriv1, NULL);

        std::cout << "eps: " << eps << std::endl;
        std::cout << "value-gradient check: " << (energy1 - energy)/eps - deriv.dot(dir) << std::endl;
        std::cout << "gradient-hessian check: " << ((deriv1 - deriv) / eps - hess * dir).norm() << std::endl;
    }
}

void VecFieldsSplit::testOptEnergy(Eigen::VectorXd x)
{
    Eigen::VectorXd deriv;
    Eigen::SparseMatrix<double> hess;
    double energy = optEnergy(x, &deriv, &hess);
    Eigen::VectorXd dir;
    dir = deriv;
    dir.setRandom();
    Eigen::VectorXd backupX = x;
    for(int i = 3; i < 9; i++)
    {
        double eps = std::pow(10, -i);
        backupX = x + eps * dir;

        Eigen::VectorXd deriv1;
        double energy1 = optEnergy(backupX, &deriv1, NULL);

        std::cout << "eps: " << eps << std::endl;
        std::cout << "value-gradient check: " << (energy1 - energy)/eps - deriv.dot(dir) << std::endl;
        std::cout << "gradient-hessian check: " << ((deriv1 - deriv) / eps - hess * dir).norm() << std::endl;
    }
}