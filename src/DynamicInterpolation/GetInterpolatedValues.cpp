#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

#ifndef GRAIN_SIZE
#define GRAIN_SIZE 10
#endif

#include "../../include/DynamicInterpolation/GetInterpolatedValues.h"

std::complex<double> GetInterpolatedValues::planeWaveBasis(Eigen::VectorXd p, Eigen::VectorXd v, Eigen::Vector2d omega, Eigen::VectorXcd *deriv, Eigen::MatrixXcd* hess)
{
    if(deriv)
        deriv->setZero(omega.size());
    if(hess)
        hess->setZero(omega.size(), omega.size());
    if (omega.norm() == 0)
        return 1;
    Eigen::Vector2d pv = (p - v).segment<2>(0);
    double deltatheta = omega.dot(pv);
    Eigen::Vector2d gradDeltaTheta = pv;

    if(deriv)
    {
        (*deriv) = std::complex<double>(-std::sin(deltatheta), std::cos(deltatheta)) * pv;
    }
    if(hess)
    {
        (*hess) = -std::complex<double>(std::cos(deltatheta), std::sin(deltatheta)) * gradDeltaTheta * gradDeltaTheta.transpose();
    }

    return std::complex<double>(std::cos(deltatheta), std::sin(deltatheta));
}

std::complex<double> GetInterpolatedValues::planeWaveValue(const Eigen::MatrixXd &w,
                                                           const std::vector<std::complex<double>> &vertVals, int vid,
                                                           Eigen::VectorXcd *deriv, Eigen::MatrixXcd *hess, bool isProj)
{
    int baseFid = _baryCoords[vid].first;
    Eigen::Vector3d baryCoord = _baryCoords[vid].second;
    Eigen::Vector3d weights = computeWeight(baryCoord);

    std::complex<double> z = 0;

    if(deriv)
    {
        deriv->setZero(12);
    }
    if(hess)
    {
        hess->setZero(12, 12);
    }

    for(int j = 0; j < 3; j++)
    {
        int baseVid = _baseMesh.faceVertex(baseFid, j);
        Eigen::Vector2d wi = w.row(baseVid);
        std::complex<double> fi = vertVals[baseVid];
        Eigen::VectorXcd expDeriv;
        Eigen::MatrixXcd expHess;
        std::complex<double> expPart = planeWaveBasis(_basePos.row(baseVid), _upsampledPos.row(vid), w, deriv ? &expDeriv : NULL, hess ? &expHess : NULL);

        z += weights(j) * fi * expPart;

        if(deriv || hess)
        {
            Eigen::VectorXcd gradFi(4);
            gradFi << 1, std::complex<double>(0, 1), 0, 0;

            Eigen::VectorXcd fullExpDeriv(4);
            fullExpDeriv << 0, 0, expDeriv(0), expDeriv(1);

            Eigen::MatrixXcd fullExpHess(4, 4);
            fullExpHess.setZero();
            fullExpHess.block<2, 2>(2, 2) = expHess;

            if(deriv)
            {
                (*deriv).segment<4>(4 * j) += weights(j) * (expPart * gradFi + fullExpDeriv * fi);
            }
            if(hess)
            {
                (*hess).block<4, 4>(4 * j, 4 * j) += weights(j) * (gradFi * fullExpDeriv.transpose() + fullExpDeriv * gradFi.transpose() + fi * fullExpHess);
            }
        }
    }
    return z;
}

std::complex<double> GetInterpolatedValues::planeWaveValueDot(const Eigen::MatrixXd &w1, const Eigen::MatrixXd &w2,
                                                              const std::vector<std::complex<double>> &vertVals1,
                                                              const std::vector<std::complex<double>> &vertVals2,
                                                              const double dt, int vid, Eigen::VectorXcd *deriv,
                                                              Eigen::MatrixXcd *hess, bool isProj)
{

}

std::vector<std::complex<double>> GetInterpolatedValues::getZValues(const Eigen::MatrixXd &w,
                                                                    const std::vector<std::complex<double>> &vertVals,
                                                                    std::vector<Eigen::VectorXcd>* deriv,
                                                                    std::vector<Eigen::MatrixXcd> *H, bool isProj)
{
    int nupverts = _upsampledPos.rows();
    if(deriv)
        deriv->resize(nupverts);
    if(H)
        H->resize(nupverts);
    std::vector<std::complex<double>> zvals(nupverts);

    auto computeZvals = [&](const tbb::blocked_range<uint32_t>& range){
        for (uint32_t i = range.begin(); i < range.end(); ++i)
        {
            zvals[i] = planeWaveValue(w, vertVals, i, deriv? &(deriv->at(i)) : NULL, H? &(H->at(i)) : NULL, isProj);
        }
    };

    tbb::blocked_range<uint32_t> rangex(0u, (uint32_t)nupverts, GRAIN_SIZE);
    tbb::parallel_for(rangex, computeZvals);

    return zvals;
}