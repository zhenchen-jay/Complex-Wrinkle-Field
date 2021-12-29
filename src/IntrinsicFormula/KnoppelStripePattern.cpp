#include "../../include/IntrinsicFormula/KnoppelStripePattern.h"
#include "../../include/IntrinsicFormula/AmpSolver.h"
#include <SymGEigsShiftSolver.h>
#include <MatOp/SparseCholesky.h>
#include <MatOp/SparseSymShiftSolve.h>
#include <iostream>

using namespace IntrinsicFormula;

void IntrinsicFormula::computeMatrixA(const MeshConnectivity &mesh, const Eigen::MatrixXd &halfEdgeW,
                                     const Eigen::VectorXd &faceArea, const Eigen::MatrixXd &cotEntries,
                                     const int nverts, Eigen::SparseMatrix<double> &A)
{
    std::vector<Eigen::Triplet<double>> AT;
    int nfaces = mesh.nFaces();
    int nedges = mesh.nEdges();

    Eigen::VectorXd halfEdgeWeight(nedges);
    halfEdgeWeight.setZero();

    for(int i = 0; i < nfaces; i++)  // form mass matrix
    {
        for(int j =0; j < 3; j++)
        {
            int eid = mesh.faceEdge(i, j);
            halfEdgeWeight(eid) += cotEntries(i, j);
        }
    }
    double energy = 0;
    for(int i = 0; i < nedges; i++)
    {
        int vid0 = mesh.edgeVertex(i, 0);
        int vid1 = mesh.edgeVertex(i, 1);

        AT.push_back({2 * vid0, 2 * vid0, 2 * halfEdgeWeight(i)});
        AT.push_back({2 * vid0 + 1, 2 * vid0 + 1, 2 * halfEdgeWeight(i)});

        AT.push_back({2 * vid1, 2 * vid1, 2 * halfEdgeWeight(i)});
        AT.push_back({2 * vid1 + 1, 2 * vid1 + 1, 2 * halfEdgeWeight(i)});

        std::complex<double> expw0 = std::complex<double>(std::cos(halfEdgeW(i, 0)), std::sin(halfEdgeW(i, 0)));
        std::complex<double> expw1 = std::complex<double>(std::cos(halfEdgeW(i, 1)), std::sin(halfEdgeW(i, 1)));

        AT.push_back({2 * vid0, 2 * vid1, -halfEdgeWeight(i) * (expw0.real() + expw1.real())});
        AT.push_back({2 * vid0 + 1, 2 * vid1, -halfEdgeWeight(i) * (-expw0.imag() + expw1.imag())});
        AT.push_back({2 * vid0, 2 * vid1 + 1, -halfEdgeWeight(i) * (expw0.imag() - expw1.imag())});
        AT.push_back({2 * vid0 + 1, 2 * vid1 + 1, -halfEdgeWeight(i) * (expw0.real() + expw1.real())});

        AT.push_back({ 2 * vid1, 2 * vid0, -halfEdgeWeight(i) * (expw0.real() + expw1.real()) });
        AT.push_back({ 2 * vid1, 2 * vid0 + 1, -halfEdgeWeight(i) * (-expw0.imag() + expw1.imag()) });
        AT.push_back({ 2 * vid1 + 1, 2 * vid0, -halfEdgeWeight(i) * (expw0.imag() - expw1.imag()) });
        AT.push_back({ 2 * vid1 + 1, 2 * vid0 + 1, -halfEdgeWeight(i) * (expw0.real() + expw1.real()) });

    }
    A.resize(2 * nverts, 2 * nverts);
    A.setFromTriplets(AT.begin(), AT.end());
}

void IntrinsicFormula::roundVertexZvalsFromHalfEdgeOmega(const MeshConnectivity &mesh, const Eigen::MatrixXd &halfEdgeW,
                                                         const Eigen::VectorXd &faceArea,
                                                         const Eigen::MatrixXd &cotEntries, const int nverts,
                                                         std::vector<std::complex<double>> &zvals)
{
    std::vector<Eigen::Triplet<double>> BT;
    int nfaces = mesh.nFaces();
    int nedges = mesh.nEdges();
    
    for(int i = 0; i < nfaces; i++)  // form mass matrix
    {
        for(int j =0; j < 3; j++)
        {
            int vid = mesh.faceVertex(i, j);
            BT.push_back({2 * vid, 2 * vid, faceArea(i) / 3.0});
            BT.push_back({2 * vid + 1, 2 * vid + 1, faceArea(i) / 3.0});
        }
    }
    Eigen::SparseMatrix<double> A;
    computeMatrixA(mesh, halfEdgeW, faceArea, cotEntries, nverts, A);

    Eigen::SparseMatrix<double> B(2 * nverts, 2 * nverts);
    B.setFromTriplets(BT.begin(), BT.end());
   /* std::cout << A.toDense() << std::endl;
    std::cout << B.toDense() << std::endl;*/

    Spectra::SymShiftInvert<double> op(A, B);
    Spectra::SparseSymMatProd<double> Bop(B);
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

    std::cout << "Eigenvalue is " << evalues[0] << std::endl;

    zvals.clear();
    for(int i = 0; i < nverts; i++)
    {
        zvals.push_back(std::complex<double>(evecs(2 * i, 0), evecs(2 * i + 1, 0)));
    }
}

double IntrinsicFormula::lArg(const long &n, const Eigen::Vector3d &bary)
{
    double larg = 0;
    double ti = bary(0), tj = bary(1), tk = bary(2);
    if(tk <= ti && tk <= tj)
        larg = M_PI / 3 * n * (1 + (tj - ti) / (1 - 3 * tk));
    else if (ti <= tj && ti <= tk)
        larg = M_PI / 3 * n * (3 + (tk - tj) / (1 - 3 * ti));
    else
        larg = M_PI / 3 * n * (5 + (ti - tk) / (1 - 3 * tj));
    return larg;
}

void IntrinsicFormula::getUpsamplingTheta(const MeshConnectivity &mesh, const Eigen::MatrixXd &halfEdgeW,
                                          const std::vector<std::complex<double>> &zvals,
                                          const std::vector<std::pair<int, Eigen::Vector3d>> &bary, Eigen::VectorXd upTheta)
{
    Eigen::VectorXd edgeW = (halfEdgeW.col(0) - halfEdgeW.col(1)) / 2;
    int upsize = bary.size();
    upTheta.setZero(upsize);

    for(int i = 0; i < bary.size(); i++)
    {
        int fid = bary[i].first;
        double omegaIJ = edgeW(mesh.faceEdge(fid, 2));
        double omegaJK = edgeW(mesh.faceEdge(fid, 0));
        double omegaKI = edgeW(mesh.faceEdge(fid, 1));

        double cIJ = mesh.faceVertex(fid, 0) == mesh.edgeVertex(mesh.faceEdge(fid, 2), 0) ? 1 : -1;
        double cJK = mesh.faceVertex(fid, 1) == mesh.edgeVertex(mesh.faceEdge(fid, 0), 0) ? 1 : -1;
        double cKI = mesh.faceVertex(fid, 2) == mesh.edgeVertex(mesh.faceEdge(fid, 1), 0) ? 1 : -1;

        omegaIJ *= cIJ;
        omegaJK *= cJK;
        omegaKI *= cKI;

        std::complex<double> rij( std::cos(omegaIJ), std::sin(omegaIJ) );
        std::complex<double> rjk( std::cos(omegaJK), std::sin(omegaJK) );
        std::complex<double> rki( std::cos(omegaKI), std::sin(omegaKI) );

        std::complex<double> psiI = zvals[mesh.faceVertex(fid, 0)];
        std::complex<double> psiJ = zvals[mesh.faceVertex(fid, 1)];
        std::complex<double> psiK = zvals[mesh.faceVertex(fid, 2)];


        double alphaI = std::arg(psiI);
        double alphaJ = alphaI + omegaIJ - std::arg(rij*psiI/psiJ); //fmodPI((varphiI + omegaIJ) - varphiJ); // could do this in terms of angles instead of complex numbers...
        double alphaK = alphaJ + omegaJK - std::arg(rjk*psiJ/psiK); //fmodPI((varphiJ + omegaJK) - varphiK); // mostly a matter of taste---possibly a matter of performance?
        double alphaL = alphaK + omegaKI - std::arg(rki*psiK/psiI); //fmodPI((varphiK + omegaKI) - varphiI);

        // adjust triangles containing zeros
        long n = std::lround((alphaL-alphaI)/(2.*M_PI));
        alphaJ -= 2.*M_PI*n/3.;
        alphaK -= 4.*M_PI*n/3.;

        double theta = lArg(n, bary[i].second);
        upTheta(i) = theta + bary[i].second(0) * alphaI + bary[i].second(1) * alphaJ + bary[i].second(2) * alphaK;
    }
}

void IntrinsicFormula::testRoundingEnergy(const MeshConnectivity &mesh, const Eigen::MatrixXd &halfEdgeW,
                                          const Eigen::VectorXd &faceArea, const Eigen::MatrixXd &cotEntries,
                                          const int nverts, std::vector<std::complex<double>> zvals)
{
    Eigen::SparseMatrix<double> A;
    computeMatrixA(mesh, halfEdgeW, faceArea, cotEntries, nverts, A);
    int nfaces = mesh.nFaces();
    int nedges = mesh.nEdges();

    Eigen::VectorXd halfEdgeWeight(nedges);
    halfEdgeWeight.setZero();

    for(int i = 0; i < nfaces; i++)  // form mass matrix
    {
        for(int j =0; j < 3; j++)
        {
            int eid = mesh.faceEdge(i, j);
            halfEdgeWeight(eid) += cotEntries(i, j);
        }
    }
    double energy = 0;
    for(int i = 0; i < nedges; i++)
    {
        int vid0 = mesh.edgeVertex(i, 0);
        int vid1 = mesh.edgeVertex(i, 1);

        std::complex<double> expw0 = std::complex<double>(std::cos(halfEdgeW(i, 0)), std::sin(halfEdgeW(i, 0)));
        std::complex<double> expw1 = std::complex<double>(std::cos(halfEdgeW(i, 1)), std::sin(halfEdgeW(i, 1)));

        Eigen::Matrix2d tmpMat;

        tmpMat(0, 0) = expw0.real();
        tmpMat(0, 1) = expw0.imag();
        tmpMat(1, 0) = -expw0.imag();
        tmpMat(1, 1) = expw0.real();
        Eigen::Vector2d aibi(zvals[vid0].real(), zvals[vid0].imag());
        Eigen::Vector2d ajbj(zvals[vid1].real(), zvals[vid1].imag());

        double part1 = 2 * aibi.dot(tmpMat * ajbj) + aibi.squaredNorm() + ajbj.squaredNorm();
        double part2 = ((zvals[vid0] * expw0 - zvals[vid1])).real() * ((zvals[vid0] * expw0 - zvals[vid1])).real() + ((zvals[vid0] * expw0 - zvals[vid1])).imag() * ((zvals[vid0] * expw0 - zvals[vid1])).imag();

        energy += 0.5 * (norm((zvals[vid0] * expw0 - zvals[vid1])) + norm((zvals[vid1] * expw1 - zvals[vid0]))) * halfEdgeWeight(i);

    }

    Eigen::VectorXd x(2 * nverts);
    for(int i = 0; i < nverts; i++)
    {
        x(2 * i) = zvals[i].real();
        x(2 * i + 1) = zvals[i].imag();
    }
}