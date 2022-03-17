#include "../../include/IntrinsicFormula/IntrinsicKnoppelDrivenFormula.h"
#include "../../include/json.hpp"
#include <iostream>
#include <fstream>

#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>
#include <igl/doublearea.h>

using namespace IntrinsicFormula;

void IntrinsicKnoppelDrivenFormula::convertList2Variable(Eigen::VectorXd& x)
{
    int nverts = _zvalsList[0].size();
    int nedges = _edgeOmegaList[0].rows();

    int numFrames = _zvalsList.size() - 2;

    int DOFsPerframe = (2 * nverts + 2 * nedges);

    int DOFs = numFrames * DOFsPerframe;

    x.setZero(DOFs);

    for (int i = 0; i < numFrames; i++)
    {
        for (int j = 0; j < nverts; j++)
        {
            x(i * DOFsPerframe + 2 * j) = _zvalsList[i + 1][j].real();
            x(i * DOFsPerframe + 2 * j + 1) = _zvalsList[i + 1][j].imag();
        }

        for (int j = 0; j < nedges; j++)
        {
            x(i * DOFsPerframe + 2 * nverts + 2 * j) = _edgeOmegaList[i + 1](j, 0);
            x(i * DOFsPerframe + 2 * nverts + 2 * j + 1) = _edgeOmegaList[i + 1](j, 1);
        }
    }
}

void IntrinsicKnoppelDrivenFormula::convertVariable2List(const Eigen::VectorXd& x)
{
    int nverts = _zvalsList[0].size();
    int nedges = _edgeOmegaList[0].rows();

    int numFrames = _zvalsList.size() - 2;
    int DOFsPerframe = (2 * nverts + 2 * nedges);

    for (int i = 0; i < numFrames; i++)
    {
        for (int j = 0; j < nverts; j++)
        {
            _zvalsList[i + 1][j] = std::complex<double>(x(i * DOFsPerframe + 2 * j), x(i * DOFsPerframe + 2 * j + 1));
        }

        for (int j = 0; j < nedges; j++)
        {
            _edgeOmegaList[i + 1](j, 0) = x(i * DOFsPerframe + 2 * nverts + 2 * j);
            _edgeOmegaList[i + 1](j, 1) = x(i * DOFsPerframe + 2 * nverts + 2 * j + 1);
        }
    }
}

double IntrinsicKnoppelDrivenFormula::computeEnergy(const Eigen::VectorXd& x, Eigen::VectorXd* deriv, Eigen::SparseMatrix<double>* hess, bool isProj)
{
    int nverts = _zvalsList[0].size();
    int nedges = _edgeOmegaList[0].rows();

    int numFrames = _zvalsList.size() - 2;
    int DOFsPerframe = (2 * nverts + 2 * nedges);
    int DOFs = numFrames * DOFsPerframe;

    convertVariable2List(x);

    Eigen::VectorXd curDeriv;
    std::vector<Eigen::Triplet<double>> T, curT;

    double energy = 0;
    if (deriv)
    {
        deriv->setZero(DOFs);
    }

    for (int i = 0; i < _zvalsList.size() - 1; i++)
    {
        energy += _zdotModel.computeZdotIntegration(_zvalsList[i], _edgeOmegaList[i], _zvalsList[i+1], _edgeOmegaList[i+1], deriv ? &curDeriv : NULL, hess ? &curT : NULL, isProj);


        if (deriv)
        {
            if (i == 0)
                deriv->segment(0, DOFsPerframe) += curDeriv.segment(DOFsPerframe, DOFsPerframe);
            else if (i == _zvalsList.size() - 2)
                deriv->segment((i - 1) * DOFsPerframe, DOFsPerframe) += curDeriv.segment(0, DOFsPerframe);
            else
            {
                deriv->segment((i - 1) * DOFsPerframe, 2 * DOFsPerframe) += curDeriv;
            }

        }

        if (hess)
        {
            for (auto& it : curT)
            {
                if (i == 0)
                {
                    if (it.row() >= DOFsPerframe && it.col() >= DOFsPerframe)
                        T.push_back({ it.row() - DOFsPerframe, it.col() - DOFsPerframe, it.value() });
                }
                else if (i == _zvalsList.size() - 2)
                {
                    if (it.row() < DOFsPerframe && it.col() < DOFsPerframe)
                        T.push_back({ it.row() + (i - 1) * DOFsPerframe, it.col() + (i - 1) * DOFsPerframe, it.value() });
                }
                else
                {
                    T.push_back({ it.row() + (i - 1) * DOFsPerframe, it.col() + (i - 1) * DOFsPerframe, it.value() });
                }

            }
            curT.clear();
        }
    }

    for(int i = 0; i < numFrames; i++) {

        // vertex amp diff
        double aveAmp = 0;
        for (int j = 0; j < nverts; j++) {
            aveAmp += _refAmpList[i + 1][j] / nverts;
        }
        for (int j = 0; j < nverts; j++) {
            double ampSq = _zvalsList[i + 1][j].real() * _zvalsList[i + 1][j].real() +
                _zvalsList[i + 1][j].imag() * _zvalsList[i + 1][j].imag();
            double refAmpSq = _refAmpList[i + 1][j] * _refAmpList[i + 1][j];

            energy += _spatialRatio * (ampSq - refAmpSq) * (ampSq - refAmpSq) / (aveAmp * aveAmp);

            if (deriv) {
                (*deriv)(i * DOFsPerframe + 2 * j) += 2.0 * _spatialRatio / (aveAmp * aveAmp) * (ampSq - refAmpSq) *
                    (2.0 * _zvalsList[i + 1][j].real());
                (*deriv)(i * DOFsPerframe + 2 * j + 1) += 2.0 * _spatialRatio / (aveAmp * aveAmp) * (ampSq - refAmpSq) *
                    (2.0 * _zvalsList[i + 1][j].imag());
            }

            if (hess) {
                Eigen::Matrix2d tmpHess;
                tmpHess << 2.0 * _zvalsList[i + 1][j].real() * 2.0 * _zvalsList[i + 1][j].real(), 2.0 * _zvalsList[i +
                    1][j].real() *
                    2.0 * _zvalsList[i +
                    1][j].imag(),
                    2.0 * _zvalsList[i + 1][j].real() * 2.0 * _zvalsList[i + 1][j].imag(), 2.0 * _zvalsList[i +
                    1][j].imag() *
                    2.0 * _zvalsList[i +
                    1][j].imag();

                tmpHess *= 2.0 * _spatialRatio / (aveAmp * aveAmp);
                tmpHess += 2.0 * _spatialRatio / (aveAmp * aveAmp) * (ampSq - refAmpSq) *
                    (2.0 * Eigen::Matrix2d::Identity());

                if (isProj)
                    tmpHess = SPDProjection(tmpHess);

                for (int k = 0; k < 2; k++)
                    for (int l = 0; l < 2; l++)
                        T.push_back({ i * DOFsPerframe + 2 * j + k, i * DOFsPerframe + 2 * j + l, tmpHess(k, l) });

            }
        }

        // edge omega difference
        for (int j = 0; j < nedges; j++) {
            energy += _spatialRatio * (aveAmp * aveAmp) * (_edgeOmegaList[i + 1] - _refEdgeOmegaList[i + 1]).row(j).dot(
                    (_edgeOmegaList[i + 1] - _refEdgeOmegaList[i + 1]).row(j));

            if (deriv) {
                (*deriv)(i * DOFsPerframe + 2 * nverts + 2 * j) += 2 * _spatialRatio * (aveAmp * aveAmp) *
                                                                   (_edgeOmegaList[i + 1] - _refEdgeOmegaList[i + 1])(j,
                                                                                                                      0);
                (*deriv)(i * DOFsPerframe + 2 * nverts + 2 * j + 1) += 2 * _spatialRatio * (aveAmp * aveAmp) *
                                                                       (_edgeOmegaList[i + 1] -
                                                                        _refEdgeOmegaList[i + 1])(j, 1);
            }

            if (hess) {
                T.push_back({i * DOFsPerframe + 2 * nverts + 2 * j, i * DOFsPerframe + 2 * nverts + 2 * j,
                             2 * _spatialRatio * (aveAmp * aveAmp)});
                T.push_back({i * DOFsPerframe + 2 * nverts + 2 * j + 1, i * DOFsPerframe + 2 * nverts + 2 * j + 1,
                             2 * _spatialRatio * (aveAmp * aveAmp)});
            }
        }

        // knoppel part
        Eigen::VectorXd kDeriv;
        std::vector<Eigen::Triplet<double>> kT;

        double knoppel = IntrinsicFormula::KnoppelEnergyGivenMag(_mesh, _refEdgeOmegaList[i + 1],
                                                                 _refAmpList[i + 1] / aveAmp, _faceArea, _cotEntries,
                                                                 _zvalsList[i + 1], deriv ? &kDeriv : NULL,
                                                                 hess ? &kT : NULL);
        energy += _spatialRatio * knoppel;

        if (deriv) {
            deriv->segment(i * DOFsPerframe, kDeriv.rows()) += _spatialRatio * kDeriv;
        }

        if (hess) {
            for (auto &it: kT) {
                T.push_back({i * DOFsPerframe + it.row(), i * DOFsPerframe + it.col(), _spatialRatio * it.value()});
            }
        }
    }


    if (hess)
    {
        //std::cout << "num of triplets: " << T.size() << std::endl;
        hess->resize(DOFs, DOFs);
        hess->setFromTriplets(T.begin(), T.end());
    }
    return energy;
}


void IntrinsicKnoppelDrivenFormula::testEnergy(Eigen::VectorXd x)
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