#pragma once
#include "CommonTools.h"

/*
 * In this file, I implemented several interpolation schemes based on so called Side-vertex interpolation (see reference "The Side-Vertex Method for Interpolation in Triangles")
 */


/*
 * The initial side-vertex interpolation, where the F(P) = \sum [(1 - bi) F(Si) - bi F(Pi)],
 * where
 *          P = \sum bi Pi,
 *          Si = bj / (bj + bk) Pj + bk / (bj + bk) Pk
 *          F(Si) = bj / (bj + bk) F(Pj) + bk / (bj + bk) Pk
 */
template <typename Scalar>
Scalar linearSideVertexInterpolation(const std::vector<Scalar>& vertVal, const Eigen::Vector3d& bary)
{
     Scalar F = 0;
     for(int i = 0; i < 3; i++)
     {
         Scalar Fsi = 0;
         int j = (1 + i) % 3;
         int k = (2 + i) % 3;
         double sum = bary[j] + bary[k];
         if(sum == 0)
         {
             F += bary[i] * vertVal[i];    // bary[j] + bary[k] = sum = 1 - bary[i]
             continue;
         }
         else
            Fsi = bary[j] / sum * vertVal[j] + bary[k] / sum * vertVal[k];

         F += (1 - bary[i]) * Fsi + bary[i] * vertVal[i];
     }
     return F;
}

template<typename Scalar>
void HermiteInterpolation(const Scalar f0, const Scalar f1, const Scalar df0, const Scalar df1, double t, Scalar &f, Scalar *df = NULL)
{
    auto h = [](double t)
    {
        return t * t * (3 - 2 * t);
    };

    auto dh = [](double t)
    {
        return 6 * (t - t * t);
    };


    auto hbar = [](double t)
    {
        return t * t * (t - 1);
    };

    auto dhbar = [](double t)
    {
        return 3 * t * t - 2 * t;
    };

    f = h(t) * f1 + hbar(t) * df1 + h(1 - t) * f0 - hbar(1 - t) * df0;
    if(df)
    {
        (*df) = dh(t) * f1 + dhbar(t) * df1 - dh(1 - t) * f0 + dhbar(1 - t) * df0;
    }
}

/*
 * The initial side-vertex interpolation, where the F(P) = \sum [(1 - bi) F(Si) - bi F(Pi)],
 * where
 *          P = \sum bi Pi,
 *          Si = bj / (bj*bk) Pj + bk / (bj * bk) Pk
 *          F(Si) is computed using Hermite interpolation
 */
template <typename Scalar>
Scalar cubicSideVertexInterpolation(const std::vector<Scalar>& vertVal, const std::vector<Eigen::Matrix<Scalar, 3, 1>>& verDeriv, const std::vector<Eigen::Vector3d>& tri, const Eigen::Vector3d& bary)
{
    Scalar F = 0;
    for(int i = 0; i < 3; i++)
    {
        Scalar Fsi = 0;
        int j = (1 + i) % 3;
        int k = (2 + i) % 3;
        double sum = bary[j] + bary[k];
        if(sum == 0)
        {
            F += bary[i] * vertVal[i];    // bary[j] + bary[k] = sum = 1 - bary[i]
            continue;
        }
        else
        {
            Scalar f0 = vertVal[j];
            Scalar f1 = vertVal[k];

            Eigen::Vector3d e = tri[k] - tri[j];
            if(!e.norm())
            {
                F += bary[i] * vertVal[i];
                continue;
            }
            e = e / e.norm();
            Scalar df0 = verDeriv[j].dot(e);
            Scalar df1 = verDeriv[k].dot(e);
            double t = bary[k] / sum;

            HermiteInterpolation(f0, f1, df0, df1, t, Fsi);
        }

        F += (1 - bary[i]) * Fsi + bary[i] * vertVal[i];
    }
    return F;
}


/*
 * The modified side-vertex interpolation, proposed in the paper "Water Wave Animation via Wavefront Parameter Interpolation"
 * where
 *          D[F] = (b_1^2 b_2^2 D_0[F] + b_2^2 b_0^2 D_1[F] + b_0^2 b_1^2 D_2[F]) / sum
 *          sum = b_1^2 b_2^2 + b_2^2 b_0^2 + b_0^2 b_1^2
 */
template <typename Scalar>
Scalar WojtanSideVertexInterpolation(const std::vector<Scalar>& vertVal, const std::vector<Eigen::Matrix<Scalar, 3, 1>>& verDeriv, const std::vector<Eigen::Vector3d>& tri, const Eigen::Vector3d& bary)
{
    double sum = 0;
    for(int i = 0; i < 3; i++)
        sum += bary[i] * bary[i] * bary[(i + 1) % 3] * bary[(i + 1) % 3];

    Eigen::Vector3d faceNormal;
    Eigen::Vector3d e0 = tri[1] - tri[0];
    Eigen::Vector3d e1 = tri[2] - tri[0];
    faceNormal = e0.cross(e1);
    if(!faceNormal.norm()){
        return 0; // degenerate triangles. No action is needed
    }
    faceNormal /= faceNormal.norm();

    std::vector<Eigen::Matrix<Scalar, 3, 1>> projFsDeriv(3, Eigen::Matrix<Scalar, 3, 1>::Zero());
    std::vector<Scalar> Fs(3, 0);
    for(int i = 0; i < 3; i++)
    {
        int j = (1 + i) % 3;
        int k = (2 + i) % 3;
        Eigen::Vector3d e = tri[k] - tri[j];
        if(e.norm())
        {
            Scalar f0 = vertVal[j];
            Scalar f1 = vertVal[k];

            e = e / e.norm();
            Scalar df0 = verDeriv[j].dot(e);
            Scalar df1 = verDeriv[k].dot(e);

            Eigen::VectorXd binormal = e.cross(faceNormal);

            double t = bary[k] / sum;

            Scalar dftan, dfnormal;
            Scalar Fsi = 0;
            HermiteInterpolation(f0, f1, df0, df1, t, Fsi, &dftan);

            Scalar nf0 = verDeriv[j].dot(binormal);
            Scalar nf1 = verDeriv[k].dot(binormal);

            dfnormal = 0;
            HermiteInterpolation(nf0, nf1, 0, 0, t, dfnormal);

            projFsDeriv[i] = dfnormal * binormal + dftan * e;
            Fs[i] = Fsi;
        }
    }

    auto h = [](double t)
    {
        return t * t * (3 - 2 * t);
    };

    Scalar DF = 0;
    for(int i = 0; i < 3; i++)
    {
        double wi = bary[(i + 1) % 3] * bary[(i + 2) % 3];
        wi = wi * wi / sum;

        int j = (1 + i) % 3;
        int k = (2 + i) % 3;

        // compute Di = Pi + Bi
        // for Pi
        Scalar Pi = 0;
        Pi += h(bary[i]) * vertVal[i];

        Scalar dFdej = 0, dFdek = 0;

        for(int n = 0; n < 3; n++)
        {
            dFdej += (tri[k][n] - tri[i][n]) * verDeriv[i][n];
            dFdek += (tri[j][n] - tri[i][n]) * verDeriv[i][n];
        }

        Pi += bary[i] * bary[i] * (bary[j] * dFdek + bary[k] * dFdej);

        // for Bi
        Scalar Bi = 0;
        Scalar dFsdej = 0, dFsdek = 0;
        for(int n = 0; n < 3; n++)
        {
            dFsdej += (tri[k][n] - tri[i][n]) * projFsDeriv[i][n];
            dFsdek += (tri[j][n] - tri[i][n]) * projFsDeriv[i][n];
        }
        Bi = h(1 - bary[i]) * Fs[i] - bary[i] * (1 - bary[i]) * (bary[j] * dFsdek + bary[k] * dFsdej);

        DF += wi * (Pi + Bi);
    }
    return DF;
}
