#pragma once
#include "../../dep/SecStencils/Mesh.h"

class ComplexLoopNew	// We modify the Loop.h
{
protected:
    Mesh const* _meshPtr;

public:
    ComplexLoopNew() : _meshPtr(0) { }

    ~ComplexLoopNew() { }

    inline Mesh const* GetMesh() const { return _meshPtr; }
    inline void  SetMesh(Mesh const* ptr) { _meshPtr = ptr; }
    inline void  SetMesh(const Mesh& mesh) { SetMesh(&mesh); }

    void BuildS0(SparseMatrixX& A) const;
    void BuildS1(SparseMatrixX& A) const;

    void GetSubdividedEdges(std::vector< std::vector<int> >& edgeToVert) const;
    void GetSubdividedFaces(std::vector< std::vector<int> >& faceToVert);


    bool IsVertRegular(int vert) const;
    bool AreIrregularVertsIsolated() const;

private:
    int _GetVertVertIndex(int vert) const;
    int _GetEdgeVertIndex(int edge) const;
    int _GetFaceVertIndex(int face) const;

    int _GetEdgeEdgeIndex(int edge, int vertInEdge) const;
    int _GetFaceEdgeIndex(int face, int edgeInFace) const;

    int _GetCentralFaceIndex(int face) const;
    int _GetCornerFaceIndex(int face, int vertInFace) const;

    Scalar _GetAlpha(int vert) const;
    Scalar _GetBeta(int vert) const;

    void _AssembleVertEvenInterior(int vi, TripletInserter out) const;
    void _AssembleVertEvenBoundary(int vi, TripletInserter out) const;
    void _AssembleVertOddInterior(int edge, TripletInserter out) const;
    void _AssembleVertOddBoundary(int edge, TripletInserter out) const;

    void _AssembleVertEvenInterior(int vi, TripletInserter outV, TripletInserter outE) const;
    void _AssembleVertEvenBoundary(int vi, TripletInserter outV, TripletInserter outE) const;
    void _AssembleVertOddInterior(int edge, TripletInserter outV, TripletInserter outE) const;
    void _AssembleVertOddBoundary(int edge, TripletInserter outV, TripletInserter outE) const;

    void _AssembleEdgeEvenInterior(int edge, int vertInEdge, TripletInserter out) const;
    void _AssembleEdgeEvenBoundary(int edge, int vertInEdge, TripletInserter out) const;
    void _AssembleEdgeEvenPartialBoundary(int edge, int vertInEdge, TripletInserter out) const;
    void _AssembleEdgeOdd(int face, int edgeInFace, TripletInserter out) const;

    void _InsertEdgeEdgeValue(int row, int col, int vert, int rSign, Scalar val, TripletInserter out) const;
    void _InsertEdgeFaceValue(int row, int face, int vert, int rSign, Scalar val, TripletInserter out) const;
};

std::complex<double> interpZ(const std::vector<std::complex<double>>& zList, const std::vector<Eigen::Vector3d>& gradThetaList, std::vector<double>& coords, const std::vector<Eigen::Vector3d>& pList, Eigen::Vector3cd* gradZ = NULL);

std::complex<double> computeZandGradZ(const Mesh& mesh, const Eigen::VectorXd& omega, const std::vector<std::complex<double>>& zvals, int fid, const Eigen::Vector3d& bary, Eigen::Vector3cd* gradz);

void updateLoopedZvals(const Mesh& mesh, const Eigen::VectorXd& omega, const std::vector<std::complex<double>>& zvals, std::vector<std::complex<double>>& upZvals, std::vector<Eigen::Vector3cd> * upgradZvals = NULL);

void SubdivideNew(const Mesh& mesh, const Eigen::VectorXd& omega, const std::vector<std::complex<double>>& zvals, Eigen::VectorXd& omegaNew, std::vector<std::complex<double>>& upZvals, int level, Mesh& meshNew);