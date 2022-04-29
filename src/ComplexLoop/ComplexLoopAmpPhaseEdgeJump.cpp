#include "../../include/ComplexLoop/ComplexLoopAmpPhaseEdgeJump.h"
#include <iostream>
#include <cassert>
#include <memory>

void ComplexLoopAmpPhaseEdgeJump::BuildComplexS0(SparseMatrixX& A, SparseMatrixX& B) const
{
	assert(_mesh.IsTriangulated());

	std::vector<TripletX> tripletsV, tripletsE;
	int V = _mesh.GetVertCount();
	int E = _mesh.GetEdgeCount();

	// Even verts
	for (int vi = 0; vi < V; ++vi)
	{
		if (_mesh.IsVertBoundary(vi))
			_AssembleVertEvenBoundary(vi, std::back_inserter(tripletsV), std::back_inserter(tripletsE));
		else
			_AssembleVertEvenInterior(vi, std::back_inserter(tripletsV), std::back_inserter(tripletsE));
	}

	// Odd verts
	for (int edge = 0; edge < E; ++edge)
	{
		if (_mesh.IsEdgeBoundary(edge))
			_AssembleVertOddBoundary(edge, std::back_inserter(tripletsV), std::back_inserter(tripletsE));
		else
			_AssembleVertOddInterior(edge, std::back_inserter(tripletsV), std::back_inserter(tripletsE));
	}

	A.resize(V + E, V);
	A.setFromTriplets(tripletsV.begin(), tripletsV.end());

	B.resize(V + E, E);
	B.setFromTriplets(tripletsE.begin(), tripletsE.end());
}

void ComplexLoopAmpPhaseEdgeJump::_AssembleVertEvenInterior(int vi, TripletInserter outV, TripletInserter outE) const
{
	Scalar alpha = _GetAlpha(vi);
	int row = _GetVertVertIndex(vi);

	const std::vector<int>& edges = _mesh.GetVertEdges(vi);
	for (int k = 0; k < edges.size(); ++k)
	{
		int edge = edges[k];
		int viInEdge = _mesh.GetVertIndexInEdge(edge, vi);
		int vj = _mesh.GetEdgeVerts(edge)[(viInEdge + 1) % 2];

		int sign = 1;
		if (viInEdge == 1)
			sign *= -1;
		*outE++ = TripletX(row, edge, sign * alpha);
	}
	
	*outV++ = TripletX(row, vi, 1.);
}


void ComplexLoopAmpPhaseEdgeJump::_AssembleVertEvenBoundary(int vi, TripletInserter outV, TripletInserter outE) const
{
	int row = _GetVertVertIndex(vi);
	if (!_isFixBnd)
	{
		std::vector<int> boundary(2);
		boundary[0] = _mesh.GetVertEdges(vi).front();
		boundary[1] = _mesh.GetVertEdges(vi).back();

		for (int j = 0; j < boundary.size(); ++j)
		{
			int edge = boundary[j];
			assert(_mesh.IsEdgeBoundary(edge));
			int viInEdge = _mesh.GetVertIndexInEdge(edge, vi);

			int sign = 1;
			if (viInEdge == 1)
				sign *= -1;

			*outE++ = TripletX(row, edge, sign * 0.125);
		}
	}
	
	*outV++ = TripletX(row, vi, 1.);
}

void ComplexLoopAmpPhaseEdgeJump::_AssembleVertOddInterior(int edge, TripletInserter outV, TripletInserter outE) const
{
	int row = _GetEdgeVertIndex(edge);
	for (int j = 0; j < 2; ++j)
	{
		int face = _mesh.GetEdgeFaces(edge)[j];
		int offset = _mesh.GetEdgeIndexInFace(face, edge);

		int vi = _mesh.GetFaceVerts(face)[(offset + 0) % 3];
		int vj = _mesh.GetFaceVerts(face)[(offset + 1) % 3];
		int vk = _mesh.GetFaceVerts(face)[(offset + 2) % 3];

		int ei = edge;
		int ej = _mesh.GetFaceEdges(face)[(offset + 1) % 3];
		int ek = _mesh.GetFaceEdges(face)[(offset + 2) % 3];

		int sign0 = 1, sign1 = 1;
		if (_mesh.GetVertIndexInEdge(ek, vi) == 1)
			sign0 *= -1;
		if (_mesh.GetVertIndexInEdge(ej, vj) == 1)
			sign1 *= -1;

		*outE++ = TripletX(row, ek, sign0 * 1. / 16);
		*outE++ = TripletX(row, ej, sign1 * 1. / 16);
	}

	*outE++ = TripletX(row, edge, 1. / 2);
	*outV++ = TripletX(row, _mesh.GetEdgeVerts(edge)[0], 1.);
}

void ComplexLoopAmpPhaseEdgeJump::_AssembleVertOddBoundary(int edge, TripletInserter outV, TripletInserter outE) const
{
	int row = _GetEdgeVertIndex(edge);

	*outE++ = TripletX(row, edge, 1. / 2.);
	*outV++ = TripletX(row, _mesh.GetEdgeVerts(edge)[0], 1.);
}

void ComplexLoopAmpPhaseEdgeJump::Subdivide(const Eigen::VectorXd& omega, const std::vector<std::complex<double>>& zvals, Eigen::VectorXd& omegaNew, std::vector<std::complex<double>>& upZvals, int level)
{
	int nverts = _mesh.GetVertCount();

	Eigen::VectorXd amp(nverts);
	Eigen::VectorXd theta(nverts);
	omegaNew = omega;

	for (int i = 0; i < nverts; i++)
	{
		amp(i) = std::abs(zvals[i]);
		theta(i) = std::arg(zvals[i]);
	}

	MatrixX X;
	_mesh.GetPos(X);
	omegaNew = omega;

	for (int l = 0; l < level; ++l)
	{
		SparseMatrixX tmpS0, tmpS1, tmpSV, tmpSE;
		BuildS0(tmpS0);
		BuildS1(tmpS1);
		BuildComplexS0(tmpSV, tmpSE);

		X = tmpS0 * X;
		amp = tmpS0 * amp;

		theta = tmpSV * theta + tmpSE * omegaNew;

		omegaNew = tmpS1 * omegaNew;

		std::vector<Vector3> points;
		ConvertToVector3(X, points);

		std::vector< std::vector<int> > edgeToVert;
		GetSubdividedEdges(edgeToVert);

		std::vector< std::vector<int> > faceToVert;
		GetSubdividedFaces(faceToVert);

		_mesh.Populate(points, faceToVert, edgeToVert);

	}

	int nupverts = _mesh.GetVertCount();
	upZvals.resize(nupverts);

	for (int i = 0; i < nupverts; i++)
	{
		upZvals[i] = amp(i)*std::complex<double>(std::cos(theta(i)), std::sin(theta(i)));
	}

}