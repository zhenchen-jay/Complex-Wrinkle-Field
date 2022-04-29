#pragma once
#include "ComplexLoop.h"

class ComplexLoopAmpPhase : public ComplexLoop
{
public:
    void virtual Subdivide(const Eigen::VectorXd& omega, const std::vector<std::complex<double>>& zvals, Eigen::VectorXd& omegaNew, std::vector<std::complex<double>>& upZvals, int level) override;
};

/*
* Apply loop mask for amp, and phase.
*/