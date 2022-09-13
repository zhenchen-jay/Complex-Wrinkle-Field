#pragma once
#include "CommonTools.h"

/*
 * In this file, I implemented several interpolation schemes based on so called Side-vertex interpolation (see reference "The Side-Vertex Method for Interpolation in Triangles")
 */


/*
 * The initial side-vertex interpolation, where the F(P) = \sum [(1 - bi) F(Si) - bi F(Pi)],
 * where
 *          P = \sum bi Pi,
 *          Si = bj / (bj*bk) Pj + bk / (bj * bk) Pk
 *          F(Si) = bj / (bj * bk) F(Pj) + bk / (bj * bk) Pk
 */
template <typename Scalar>
Scalar linearSideVertexInterpolation(std::vector<Scalar> vertVal, Eigen::Vector3d bary)
{

}

/*
 * The initial side-vertex interpolation, where the F(P) = \sum [(1 - bi) F(Si) - bi F(Pi)],
 * where
 *          P = \sum bi Pi,
 *          Si = bj / (bj*bk) Pj + bk / (bj * bk) Pk
 *          F(Si) is computed using Hermite interpolation
 */
template <typename Scalar>
Scalar cubicSideVertexInterpolation(std::vector<Scalar> vertVal, std::vector<Eigen::Matrix<Scalar, 3, 1>> verDeriv, Eigen::Vector3d bary)
{

}


/*
 * The modified side-vertex interpolation, proposed in the paper "Water Wave Animation via Wavefront Parameter Interpolation"
 * where
 *          D[F] = (b_1^2 b_2^2 D_0[F] + b_2^2 b_0^2 D_1[F] + b_0^2 b_1^2 D_2[F]) / sum
 *          sum = b_1^2 b_2^2 + b_2^2 b_0^2 + b_0^2 b_1^2
 */
template <typename Scalar>
Scalar WojtanSideVertexInterpolation(std::vector<Scalar> vertVal, std::vector<Eigen::Matrix<Scalar, 3, 1>> verDeriv, Eigen::Vector3d bary)
{

}
