#include "../../include/IntrinsicFormula/AmpSolver.h"
#include <igl/cotmatrix_entries.h>
#include <igl/massmatrix.h>
#include <Eigen/Sparse>
#include <SymGEigsShiftSolver.h>
#include <MatOp/SparseCholesky.h>
#include <MatOp/SparseSymShiftSolve.h>
#include <cassert>

// Baseline implementation; can be used if Spectra not available

/*
static void inversePowerIteration(const Eigen::SparseMatrix<double>& A, const Eigen::SparseMatrix<double>& B, Eigen::VectorXd& sol)
{
    sol.resize(A.rows());
    sol.setRandom();
    double eps = 1e-6;
    Eigen::SparseMatrix<double> M = A + eps * B;
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double> > solver(M);
    sol.setRandom();
    for (int i = 0; i < 10000; i++)
    {
        sol = solver.solve(B * sol);
        sol /= std::sqrt(sol.transpose() * B * sol);
    }  
}
*/

void ampSolver(const Eigen::MatrixXd& V, const MeshConnectivity& mesh, const Eigen::MatrixXd& omegas, Eigen::VectorXd& amplitudes)
{
    Eigen::MatrixXd C;
    igl::cotmatrix_entries(V, mesh.faces(), C);

    int nedges = mesh.nEdges();
    Eigen::VectorXd onestar(nedges);
    
    for (int i = 0; i < nedges; i++)
    {
        onestar[i] = 0;
        for (int j = 0; j < 2; j++)
        {
            int fidx = mesh.edgeFace(i, j);
            if (fidx != -1)
            {
                int opp = mesh.oppositeVertexIndex(i, j);
                onestar[i] += C(fidx, opp);
            }
        }
    }

    std::vector<Eigen::Triplet<double> > Mcoeffs;
    for (int i = 0; i < nedges; i++)
    {
        int v0 = mesh.edgeVertex(i, 0);
        int v1 = mesh.edgeVertex(i, 1);
        
        // Laplacian coeffs
        Mcoeffs.push_back({ v0, v0, onestar[i] });
        Mcoeffs.push_back({ v1, v1, onestar[i] });
        Mcoeffs.push_back({ v0, v1, -onestar[i] });
        Mcoeffs.push_back({ v1, v0, -onestar[i] });

        // frequency coeffs
        // frequency coeffs
        double freqweight1 = 0.5 * omegas(i, 0) * omegas(i, 0);
        double freqweight2 = 0.5 * omegas(i, 1) * omegas(i, 1);
        Mcoeffs.push_back({ v0, v0, freqweight1 * onestar[i] });
        Mcoeffs.push_back({ v1, v1, freqweight2 * onestar[i] });
    }

    int nverts = V.rows();
    Eigen::SparseMatrix<double> M(nverts, nverts);
    M.setFromTriplets(Mcoeffs.begin(), Mcoeffs.end());

    Eigen::SparseMatrix<double> massM;
    igl::massmatrix(V, mesh.faces(), igl::MASSMATRIX_TYPE_BARYCENTRIC, massM);

    Spectra::SymShiftInvert<double> op(M, massM);
    Spectra::SparseSymMatProd<double> Bop(massM);
    Spectra::SymGEigsShiftSolver<Spectra::SymShiftInvert<double>, Spectra::SparseSymMatProd<double>, Spectra::GEigsMode::ShiftInvert> geigs(op, Bop, 1, 6, -1e-6);
    geigs.init();
    int nconv = geigs.compute(Spectra::SortRule::LargestMagn, 1e6);

    Eigen::VectorXd evalues;
    Eigen::MatrixXd evecs;

    evalues = geigs.eigenvalues();
    evecs = geigs.eigenvectors();
    if (nconv != 1 || geigs.info() != Spectra::CompInfo::Successful)
    {
        std::cout << "Eigensolver failed to converge!!" << std::endl;
    }
    amplitudes = evecs.col(0);
    std::cout << "Eigenvalue is " << evalues[0] << std::endl;

    //inversePowerIteration(M, massM, amplitudes);    

    // try to fix sign
    int posvotes = 0;
    int negvotes = 0;
    for (int i = 0; i < nverts; i++)
    {
        if (amplitudes[i] < 0)
            negvotes++;
        else
            posvotes++;
    }
    if (negvotes > posvotes)
        amplitudes *= -1;
}