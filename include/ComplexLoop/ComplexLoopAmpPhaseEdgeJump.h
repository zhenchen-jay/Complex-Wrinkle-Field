#pragma once
#include "ComplexLoop.h"

class ComplexLoopAmpPhaseEdgeJump : public ComplexLoop	
{
public:
    void virtual Subdivide(const Eigen::VectorXd& omega, const std::vector<std::complex<double>>& zvals, Eigen::VectorXd& omegaNew, std::vector<std::complex<double>>& upZvals, int level) override;

public:
    void BuildComplexS0(SparseMatrixX& A, SparseMatrixX& B) const;

private:
    void _AssembleVertEvenInterior(int vi, TripletInserter outV, TripletInserter outE) const;
    void _AssembleVertEvenBoundary(int vi, TripletInserter outV, TripletInserter outE) const;
    void _AssembleVertOddInterior(int edge, TripletInserter outV, TripletInserter outE) const;
    void _AssembleVertOddBoundary(int edge, TripletInserter outV, TripletInserter outE) const;
};

/*
* Apply loop mask for amp, and phase, then correct the phase by the edge jump (encoded in the omega). This works fine when the edge omega is locally integrable. For non-integrable omegas, you will get super strange results
*/