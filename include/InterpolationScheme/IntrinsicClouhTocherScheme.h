#pragma once
#include "CommonInterpFunc.h"

/*
 * In this file, I implemented intrinsic interpolation schemes based on 9-parameter Clouth-Tocher interpolation (see reference "Triangular Bernstein-Bezier patches")
 */

static int factorial(const int n)
{
	int f = 1;
	for (int i = 1; i <= n; i++)
		f *= i;
	return f;
};

template <typename Scalar>
Scalar intrinsicClouthTocherInterpolation(const std::vector<Scalar>& vertVal, const std::vector<Scalar>& edgeDeriv, const std::vector<Eigen::Vector3d>& tri, const Eigen::Vector3d& bary)
{
	double zeroSum = 0;
	for (int i = 0; i < 3; i++)
		zeroSum += bary[i] * bary[i] * bary[(i + 1) % 3] * bary[(i + 1) % 3];
	if (zeroSum < 1e-15) // two numerical zeros in the bary
	{
		int flag = 0;
		double max = bary[flag];
		for (int i = 1; i < 3; i++)
		{
			if (max < bary[i])
			{
				max = bary[i];
				flag = i;
			}
		}
		return vertVal[flag];
	}

	Scalar bijk[4][4][4];

	bijk[3][0][0] = vertVal[0];
	bijk[0][3][0] = vertVal[1];
	bijk[0][0][3] = vertVal[2];

	for (int i = 1; i < 3; i++)
	{
		// e0
		Scalar f0 = vertVal[1];
		Scalar f1 = vertVal[2];
		Scalar df0 = edgeDeriv[0];
		Scalar df1 = edgeDeriv[0];
		double t = i / 3.0;

		HermiteInterpolation1D(f0, f1, df0, df1, t, bijk[i][3 - i][0]);
	   
		// e1
		f0 = vertVal[2];
		f1 = vertVal[0];
		df0 = edgeDeriv[1];
		df1 = edgeDeriv[1];
		t = i / 3.0;

		HermiteInterpolation1D(f0, f1, df0, df1, t, bijk[i][0][3 - i]);

		// e2
		f0 = vertVal[0];
		f1 = vertVal[1];
		df0 = edgeDeriv[2];
		df1 = edgeDeriv[2];
		t = i / 3.0;

		HermiteInterpolation1D(f0, f1, df0, df1, t, bijk[0][i][3 - i]);
		
	}

	bijk[1][1][1] = 0;
	for (int i = 1; i < 3; i++)
	{
		bijk[1][1][1] += 0.25 * (bijk[0][i][3 - i] + bijk[i][0][3 - i] + bijk[i][3 - i][0]);
	}
	bijk[1][1][1] -= 1.0 / 6 * (bijk[3][0][0] + bijk[0][3][0] + bijk[0][0][3]);

	Scalar F = 0;
	for(int i = 0; i <= 3; i++)
		for(int j = 0; j <= 3 - i; j++)
		{   
			int k = 3 - i - j;
			F += factorial(3) / factorial(i) / factorial(j) / factorial(k) * bijk[i][j][k];
		}
	return F;
}



