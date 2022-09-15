#pragma once
#include "CommonInterpFunc.h"

/*
 * In this file, I implemented several intrinsic interpolation schemes based on so called Side-vertex interpolation (see reference "The Side-Vertex Method for Interpolation in Triangles")
 */


/*
 * The initial side-vertex interpolation, where the F(P) = \sum [(1 - bi) F(Si) - bi F(Pi)],
 * where
 *          P = \sum bi Pi,
 *          Si = bj / (bj + bk) Pj + bk / (bj + bk) Pk
 *          F(Si) = bj / (bj + bk) F(Pj) + bk / (bj + bk) Pk
 */
template <typename Scalar>
Scalar intrinsicLinearSideVertexInterpolation(const std::vector<Scalar>& vertVal, const Eigen::Vector3d& bary)          // no difference between this and linearSideVertexInterpolation implemented in the SideVertexInterplation.h
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


/*
 * The initial side-vertex interpolation, where the F(P) = \sum [(1 - bi) F(Si) - bi F(Pi)],
 * where
 *          P = \sum bi Pi,
 *          Si = bj / (bj*bk) Pj + bk / (bj * bk) Pk
 *          F(Si) is computed using Hermite interpolation
 */

/*
 * edgeDeriv[i] is the one form on the edge i (V_{i+1}, V_{i+2}), which can be viewed as grad_F.dot(e_i). e_i = V_{i + 2} - V_{i + 1}
 */
template <typename Scalar>
Scalar intrinsicCubicSideVertexInterpolation(const std::vector<Scalar>& vertVal, const std::vector<Scalar>& edgeDeriv, const std::vector<Eigen::Vector3d>& tri, const Eigen::Vector3d& bary)
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
            Scalar df0 = edgeDeriv[i];
            Scalar df1 = edgeDeriv[i];
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

/*
 * edgeDeriv[i] is the one form on the edge i (V_{i+1}, V_{i+2}), which can be viewed as grad_F.dot(e_i). e_i = V_{i + 2} - V_{i + 1}
 */
template <typename Scalar>
Scalar WojtanSideVertexInterpolation(const std::vector<Scalar>& vertVal, const std::vector<Scalar>& edgeDeriv, const std::vector<Eigen::Vector3d>& tri, const Eigen::Vector3d& bary)
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
            Scalar df0 = edgeDeriv[i];
            Scalar df1 = edgeDeriv[i];

            Eigen::VectorXd binormal = e.cross(faceNormal);

            double t = bary[k] / sum;

            Scalar dftan, dfnormal;
            Scalar Fsi = 0;
            HermiteInterpolation(f0, f1, df0, df1, t, Fsi, &dftan);

            // for the normal part, it is kinda tricky
            Eigen::Vector3d ei = e;
            Eigen::Vector3d ek = tri[j] - tri[i];
            Eigen::Vector3d ej = tri[i] - tri[k];

            // for nf0: nf0 = grad_F.dot(bn) = grad_F.dot(s * ek + t * ei) = s * wk + t * wi
            Eigen::Matrix2d Mj, Mk;
            Eigen::Vector2d rj, rk;
            Eigen::Vector2d sj, sk;

            Mj << ei.dot(ei), ei.dot(ek), ek.dot(ei), ek.dot(ek);
            Mk << ei.dot(ei), ei.dot(ej), ej.dot(ei), ej.dot(ej);

            rj << binormal.dot(ei), binormal.dot(ek);
            rk << binormal.dot(ei), binormal.dot(ej);

            sj = Mj.inverse() * rj;
            sk = Mk.inverse() * rk;


            Scalar nf0 = sj[0] * edgeDeriv[i] + sj[1] * edgeDeriv[k];
            Scalar nf1 = sk[0] * edgeDeriv[i] + sk[1] * edgeDeriv[j];

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

        Scalar dFdej = -edgeDeriv[j], dFdek = edgeDeriv[k];

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
