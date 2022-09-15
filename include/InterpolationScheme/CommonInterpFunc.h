#pragma once
#include "../CommonTools.h"

template<typename Scalar>
void LinearInterpolation1D(const Scalar f0, const Scalar f1, double t, Scalar &f)
{
     f = (1 - t) * f0 + t * f1;
}

template<typename Scalar>
void HermiteInterpolation1D(const Scalar f0, const Scalar f1, const Scalar df0, const Scalar df1, double t, Scalar &f, Scalar *df = NULL)
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