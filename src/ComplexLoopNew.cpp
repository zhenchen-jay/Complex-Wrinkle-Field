#include "../include/ComplexLoopNew.h"
#include "../include/CommonTools.h"
#include <iostream>
#include <cassert>
#include <memory>

bool
ComplexLoopNew::IsVertRegular(int vert) const
{
	assert(_meshPtr);
	if (_meshPtr->IsVertBoundary(vert)) return true;
	return (_meshPtr->GetVertEdges(vert).size() == 6);
}

bool
ComplexLoopNew::AreIrregularVertsIsolated() const
{
	assert(_meshPtr);
	for (int edge = 0; edge < _meshPtr->GetEdgeCount(); ++edge)
	{
		const std::vector<int>& eVerts = _meshPtr->GetEdgeVerts(edge);
		if (IsVertRegular(eVerts[0])) continue;
		if (IsVertRegular(eVerts[1])) continue;
		return false;
	}
	return true;
}

int
ComplexLoopNew::_GetVertVertIndex(int vert) const
{
	return vert;
}

int
ComplexLoopNew::_GetEdgeVertIndex(int edge) const
{
	assert(_meshPtr);
	return _meshPtr->GetVertCount() + edge;
}

int
ComplexLoopNew::_GetFaceVertIndex(int /*face*/) const
{
	assert(false);
	return -1;
}

int
ComplexLoopNew::_GetEdgeEdgeIndex(int edge, int vertInEdge) const
{
	return 2 * edge + vertInEdge;
}

int
ComplexLoopNew::_GetFaceEdgeIndex(int face, int edgeInFace) const
{
	assert(_meshPtr);
	assert(_meshPtr->IsTriangulated());
	return 2 * _meshPtr->GetEdgeCount() + 3 * face + edgeInFace;
}

int
ComplexLoopNew::_GetCentralFaceIndex(int face) const
{
	assert(_meshPtr);
	assert(_meshPtr->IsTriangulated());
	return 4 * face + 3;
}

int
ComplexLoopNew::_GetCornerFaceIndex(int face, int vertInFace) const
{
	assert(_meshPtr);
	assert(_meshPtr->IsTriangulated());
	return 4 * face + vertInFace;
}

void
ComplexLoopNew::GetSubdividedEdges(std::vector< std::vector<int> >& edgeToVert) const
{
	assert(_meshPtr);
	assert(_meshPtr->IsTriangulated());

	int E = _meshPtr->GetEdgeCount();
	int F = _meshPtr->GetFaceCount();
	edgeToVert.resize(2 * E + 3 * F);

	for (int edge = 0; edge < E; ++edge)
	{
		const std::vector<int>& eVerts = _meshPtr->GetEdgeVerts(edge);
		for (int i = 0; i < 2; ++i)
		{
			int v0 = _GetVertVertIndex(eVerts[i]);
			int v1 = _GetEdgeVertIndex(edge);
			if (v0 > v1) std::swap(v0, v1);

			int index = _GetEdgeEdgeIndex(edge, i);
			edgeToVert[index].push_back(v0);
			edgeToVert[index].push_back(v1);
		}
	}

	for (int face = 0; face < F; ++face)
	{
		const std::vector<int>& fEdges = _meshPtr->GetFaceEdges(face);
		for (int i = 0; i < 3; ++i)
		{
			int v0 = _GetEdgeVertIndex(fEdges[i]);
			int v1 = _GetEdgeVertIndex(fEdges[(i + 1) % 3]);
			if (v0 > v1) std::swap(v0, v1);

			int index = _GetFaceEdgeIndex(face, i);
			edgeToVert[index].push_back(v0);
			edgeToVert[index].push_back(v1);
		}
	}
}

void
ComplexLoopNew::GetSubdividedFaces(std::vector< std::vector<int> >& faceToVert)
{
	assert(_meshPtr);
	assert(_meshPtr->IsTriangulated());

	int V = _meshPtr->GetVertCount();
	int F = _meshPtr->GetFaceCount();
	faceToVert.resize(4 * F);

	std::vector<int> faceFlagsNew;
	faceFlagsNew.resize(4 * F, 0);  

	for (int face = 0; face < _meshPtr->GetFaceCount(); ++face)
	{
		int central = _GetCentralFaceIndex(face);
		const std::vector<int>& fVerts = _meshPtr->GetFaceVerts(face);
		const std::vector<int>& fEdges = _meshPtr->GetFaceEdges(face);
		for (int j = 0; j < 3; ++j)
		{
			// Corner face
			int index = _GetCornerFaceIndex(face, j);
			faceToVert[index].push_back(_GetVertVertIndex(fVerts[j]));
			faceToVert[index].push_back(_GetEdgeVertIndex(fEdges[j]));
			faceToVert[index].push_back(_GetEdgeVertIndex(fEdges[(j + 2) % 3]));
			// Central face
			faceToVert[central].push_back(_GetEdgeVertIndex(fEdges[j]));
		}
	}
}

Scalar
ComplexLoopNew::_GetAlpha(int vert) const
{
	assert(_meshPtr);
	assert(!_meshPtr->IsVertBoundary(vert));
	const std::vector<int>& vEdges = _meshPtr->GetVertEdges(vert);

	// Fig5 left [Wang et al. 2006]
	Scalar alpha = 0.375;
	if (vEdges.size() == 3) alpha /= 2;
	else                    alpha /= vEdges.size();
	return alpha;
}

Scalar
ComplexLoopNew::_GetBeta(int vert) const
{
	assert(_meshPtr);
	assert(!_meshPtr->IsVertBoundary(vert));
	const std::vector<int>& vFaces = _meshPtr->GetVertFaces(vert);

	// Fig5 right [Wang et al. 2006] 
	Scalar beta = 0.;
	if (vFaces.size() >= 6) beta = 0.25;
	else if (vFaces.size() == 5) beta = 0.25 - 0.0625 * std::pow(std::sin(0.4 * M_PI), 2);
	else if (vFaces.size() == 4) beta = 0.125;
	else if (vFaces.size() == 3) beta = 0.25 / 3.;
	else assert(false);
	return beta;
}

void
ComplexLoopNew::BuildS0(SparseMatrixX& A) const
{
	assert(_meshPtr);
	assert(_meshPtr->IsTriangulated());

	std::vector<TripletX> triplets;
	int V = _meshPtr->GetVertCount();
	int E = _meshPtr->GetEdgeCount();

	// Even verts
	for (int vi = 0; vi < V; ++vi)
	{
		if (_meshPtr->IsVertBoundary(vi))
			_AssembleVertEvenBoundary(vi, std::back_inserter(triplets));
		else
			_AssembleVertEvenInterior(vi, std::back_inserter(triplets));
	}

	// Odd verts
	for (int edge = 0; edge < E; ++edge)
	{
		if (_meshPtr->IsEdgeBoundary(edge))
			_AssembleVertOddBoundary(edge, std::back_inserter(triplets));
		else
			_AssembleVertOddInterior(edge, std::back_inserter(triplets));
	}

	A.resize(V + E, V);
	A.setFromTriplets(triplets.begin(), triplets.end());
}

void
ComplexLoopNew::_AssembleVertEvenInterior(int vi, TripletInserter out) const
{
	// Fig5 left [Wang et al. 2006]   
	Scalar alpha = _GetAlpha(vi);
	int row = _GetVertVertIndex(vi);
	const std::vector<int>& edges = _meshPtr->GetVertEdges(vi);
	for (int k = 0; k < edges.size(); ++k)
	{
		int edge = edges[k];
		int viInEdge = _meshPtr->GetVertIndexInEdge(edge, vi);
		int vj = _meshPtr->GetEdgeVerts(edge)[(viInEdge + 1) % 2];
		*out++ = TripletX(row, vj, alpha);
	}
	*out++ = TripletX(row, vi, 1. - alpha * edges.size());
}

void
ComplexLoopNew::_AssembleVertEvenBoundary(int vi, TripletInserter out) const
{
	// Fig8 mid-top [Wang et al. 2006] 
	std::vector<int> boundary(2);
	boundary[0] = _meshPtr->GetVertEdges(vi).front();
	boundary[1] = _meshPtr->GetVertEdges(vi).back();

	int row = _GetVertVertIndex(vi);
	for (int j = 0; j < boundary.size(); ++j)
	{
		int edge = boundary[j];
		assert(_meshPtr->IsEdgeBoundary(edge));
		int viInEdge = _meshPtr->GetVertIndexInEdge(edge, vi);
		int vj = _meshPtr->GetEdgeVerts(edge)[(viInEdge + 1) % 2];
		*out++ = TripletX(row, vj, 0.125);
	}
	*out++ = TripletX(row, vi, 0.75);
}

void
ComplexLoopNew::_AssembleVertOddInterior(int edge, TripletInserter out) const
{
	// Fig4 left-bot [Wang et al. 2006]
	for (int j = 0; j < 2; ++j)
	{
		int face = _meshPtr->GetEdgeFaces(edge)[j];
		int offset = _meshPtr->GetEdgeIndexInFace(face, edge);

		int vi = _meshPtr->GetFaceVerts(face)[(offset + 0) % 3];
		int vj = _meshPtr->GetFaceVerts(face)[(offset + 1) % 3];
		int vk = _meshPtr->GetFaceVerts(face)[(offset + 2) % 3];

		int row = _GetEdgeVertIndex(edge);
		*out++ = TripletX(row, vi, 0.1875);
		*out++ = TripletX(row, vj, 0.1875);
		*out++ = TripletX(row, vk, 0.125);
	}
}

void
ComplexLoopNew::_AssembleVertOddBoundary(int edge, TripletInserter out) const
{
	// Fig8 mid-bot [Wang et al. 2006]
	int row = _GetEdgeVertIndex(edge);
	for (int j = 0; j < 2; ++j)
	{
		int vj = _meshPtr->GetEdgeVerts(edge)[j];
		*out++ = TripletX(row, vj, 0.5);
	}
}

void ComplexLoopNew::_AssembleVertEvenInterior(int vi, TripletInserter outV, TripletInserter outE) const
{
	Scalar alpha = _GetAlpha(vi);
	int row = _GetVertVertIndex(vi);
	const std::vector<int>& edges = _meshPtr->GetVertEdges(vi);
	for (int k = 0; k < edges.size(); ++k)
	{
		int edge = edges[k];
		int viInEdge = _meshPtr->GetVertIndexInEdge(edge, vi);
		int vj = _meshPtr->GetEdgeVerts(edge)[(viInEdge + 1) % 2];

		int sign = 1;
		if (viInEdge == 1)
			sign *= -1;
		*outE++ = TripletX(row, edge, sign * alpha);
	}
	*outV++ = TripletX(row, vi, 1.);
}


void ComplexLoopNew::_AssembleVertEvenBoundary(int vi, TripletInserter outV, TripletInserter outE) const
{
	std::vector<int> boundary(2);
	boundary[0] = _meshPtr->GetVertEdges(vi).front();
	boundary[1] = _meshPtr->GetVertEdges(vi).back();

	int row = _GetVertVertIndex(vi);
	for (int j = 0; j < boundary.size(); ++j)
	{
		int edge = boundary[j];
		assert(_meshPtr->IsEdgeBoundary(edge));
		int viInEdge = _meshPtr->GetVertIndexInEdge(edge, vi);

		int sign = 1;
		if (viInEdge == 1)
			sign *= -1;

		*outE++ = TripletX(row, edge, sign * 0.125);
	}
	*outV++ = TripletX(row, vi, 1.);
}

void ComplexLoopNew::_AssembleVertOddInterior(int edge, TripletInserter outV, TripletInserter outE) const
{
	int row = _GetEdgeVertIndex(edge);
	for (int j = 0; j < 2; ++j)
	{
		int face = _meshPtr->GetEdgeFaces(edge)[j];
		int offset = _meshPtr->GetEdgeIndexInFace(face, edge);

		int vi = _meshPtr->GetFaceVerts(face)[(offset + 0) % 3];
		int vj = _meshPtr->GetFaceVerts(face)[(offset + 1) % 3];
		int vk = _meshPtr->GetFaceVerts(face)[(offset + 2) % 3];

		int ei = edge;
		int ej = _meshPtr->GetFaceEdges(face)[(offset + 1) % 3];
		int ek = _meshPtr->GetFaceEdges(face)[(offset + 2) % 3];

		int sign0 = 1, sign1 = 1;
		if (_meshPtr->GetVertIndexInEdge(ek, vi) == 1)
			sign0 *= -1;
		if (_meshPtr->GetVertIndexInEdge(ej, vj) == 1)
			sign1 *= -1;

		*outE++ = TripletX(row, ek, sign0 * 1. / 16);
		*outE++ = TripletX(row, ej, sign1 * 1. / 16);
	}

	*outE++ = TripletX(row, edge, 1. / 2);
	*outV++ = TripletX(row, _meshPtr->GetEdgeVerts(edge)[0], 1.);
}

void ComplexLoopNew::_AssembleVertOddBoundary(int edge, TripletInserter outV, TripletInserter outE) const
{
	int row = _GetEdgeVertIndex(edge);

	*outE++ = TripletX(row, edge, 1. / 2.);
	*outV++ = TripletX(row, _meshPtr->GetEdgeVerts(edge)[0], 1.);
}


void
ComplexLoopNew::BuildS1(SparseMatrixX& A) const
{
	assert(_meshPtr);
	assert(_meshPtr->IsTriangulated());

	std::vector<TripletX> triplets;
	int E = _meshPtr->GetEdgeCount();
	int F = _meshPtr->GetFaceCount();

	for (int edge = 0; edge < E; ++edge)
	{
		if (_meshPtr->IsEdgeBoundary(edge))
		{
			_AssembleEdgeEvenBoundary(edge, 0, std::back_inserter(triplets));
			_AssembleEdgeEvenBoundary(edge, 1, std::back_inserter(triplets));
		}
		else
		{
			const std::vector<int>& eVerts = _meshPtr->GetEdgeVerts(edge);
			for (int i = 0; i < eVerts.size(); ++i)
			{
				if (_meshPtr->IsVertBoundary(eVerts[i]))
					_AssembleEdgeEvenPartialBoundary(edge, i, std::back_inserter(triplets));
				else
					_AssembleEdgeEvenInterior(edge, i, std::back_inserter(triplets));
			}
		}
	}

	for (int face = 0; face < F; ++face)
	{
		_AssembleEdgeOdd(face, 0, std::back_inserter(triplets));
		_AssembleEdgeOdd(face, 1, std::back_inserter(triplets));
		_AssembleEdgeOdd(face, 2, std::back_inserter(triplets));
	}

	A.resize(2 * E + 3 * F, E);
	A.setFromTriplets(triplets.begin(), triplets.end());
}

void
ComplexLoopNew::_AssembleEdgeEvenBoundary(int edge, int vertInEdge, TripletInserter out) const
{
	int row = _GetEdgeEdgeIndex(edge, vertInEdge);
	int vert = _meshPtr->GetEdgeVerts(edge)[vertInEdge];
	int rSign = (_GetVertVertIndex(vert) < _GetEdgeVertIndex(edge)) ? 1 : -1;

	int nEdge = _meshPtr->GetVertEdges(vert).front();
	int vertInNedge = _meshPtr->GetVertIndexInEdge(nEdge, vert);
	int nSign = _meshPtr->GetVertSignInEdge(nEdge, vertInNedge);

	int pEdge = _meshPtr->GetVertEdges(vert).back();
	int vertInPedge = _meshPtr->GetVertIndexInEdge(pEdge, vert);
	int pSign = _meshPtr->GetVertSignInEdge(pEdge, vertInPedge);

	assert(edge == nEdge || edge == pEdge);

	if (edge == nEdge)
	{
		// Symmetric case of Fig8 right in [Wang et al. 2006]
		*out++ = TripletX(row, nEdge, (nSign == rSign) ? -0.375 : 0.375);
		*out++ = TripletX(row, pEdge, (pSign == rSign) ? 0.125 : -0.125);
	}
	else
	{
		// Fig8 right in [Wang et al. 2006]
		*out++ = TripletX(row, pEdge, (pSign == rSign) ? -0.375 : 0.375);
		*out++ = TripletX(row, nEdge, (nSign == rSign) ? 0.125 : -0.125);
	}
}

void
ComplexLoopNew::_InsertEdgeEdgeValue(int row, int col, int vert, int rSign, Scalar val, TripletInserter out) const
{
	// Handy function that sets the sign of val for an edge col incident to vert.
	int vertInCol = _meshPtr->GetVertIndexInEdge(col, vert);
	int sign = _meshPtr->GetVertSignInEdge(col, vertInCol);
	*out++ = TripletX(row, col, (sign == rSign) ? -val : val);
}

void
ComplexLoopNew::_InsertEdgeFaceValue(int row, int face, int vert, int rSign, Scalar val, TripletInserter out) const
{
	// Handy function that sets the sign of val for an edge col incident to face.
	int vertInFace = _meshPtr->GetVertIndexInFace(face, vert);
	int colInFace = (vertInFace + 1) % 3;
	int col = _meshPtr->GetFaceEdges(face)[colInFace];
	int sign = _meshPtr->GetEdgeSignInFace(face, colInFace);
	*out++ = TripletX(row, col, (sign == rSign) ? val : -val);
}

void
ComplexLoopNew::_AssembleEdgeEvenPartialBoundary(int edge, int vertInEdge, TripletInserter out) const
{
	int row = _GetEdgeEdgeIndex(edge, vertInEdge);
	int vert = _meshPtr->GetEdgeVerts(edge)[vertInEdge];
	int rSign = (_GetVertVertIndex(vert) < _GetEdgeVertIndex(edge)) ? 1 : -1;

	const std::vector<int>& vEdges = _meshPtr->GetVertEdges(vert);
	const int edgeCount = vEdges.size();

	const std::vector<int>& vFaces = _meshPtr->GetVertFaces(vert);
	const int faceCount = vFaces.size();

	int edgeInVert = _meshPtr->GetEdgeIndexInVert(vert, edge);
	assert(edgeInVert != edgeCount - 1);
	assert(edgeInVert != 0);

	std::vector< std::pair<int, Scalar> > eValues;
	std::vector< std::pair<int, Scalar> > fValues;

	assert(faceCount > 1);

	if (faceCount == 2)
	{
		// Case not covered in [Wang et al. 2006]
		eValues.push_back(std::make_pair(vEdges[0], 0.15625 / 3));
		eValues.push_back(std::make_pair(vEdges[1], 0.8125 / 3));
		eValues.push_back(std::make_pair(vEdges[2], 0.15625 / 3));

		fValues.push_back(std::make_pair(vFaces[0], 0.15625 / 3));
		fValues.push_back(std::make_pair(vFaces[1], -0.15625 / 3));
	}
	else if (faceCount == 3 && edgeInVert == 1)
	{
		// Fig10 mid-left in [Wang et al. 2006]
		eValues.push_back(std::make_pair(vEdges[0], 0.15625 / 3));
		eValues.push_back(std::make_pair(vEdges[1], 0.3125));
		eValues.push_back(std::make_pair(vEdges[2], 0.09375));
		eValues.push_back(std::make_pair(vEdges[3], -0.25 / 3));

		fValues.push_back(std::make_pair(vFaces[0], 0.15625 / 3));
		fValues.push_back(std::make_pair(vFaces[1], -0.03125 / 3));
		fValues.push_back(std::make_pair(vFaces[2], -0.125 / 3));
	}
	else if (faceCount == 3 && edgeInVert == 2)
	{
		// Symmetric case of Fig10 mid-left in [Wang et al. 2006]
		eValues.push_back(std::make_pair(vEdges[3], 0.15625 / 3));
		eValues.push_back(std::make_pair(vEdges[2], 0.3125));
		eValues.push_back(std::make_pair(vEdges[1], 0.09375));
		eValues.push_back(std::make_pair(vEdges[0], -0.25 / 3));

		fValues.push_back(std::make_pair(vFaces[2], -0.15625 / 3));
		fValues.push_back(std::make_pair(vFaces[1], 0.03125 / 3));
		fValues.push_back(std::make_pair(vFaces[0], 0.125 / 3));
	}
	else if (faceCount == 4 && edgeInVert == 2)
	{
		// Fig10 mid-right in [Wang et al. 2006]
		eValues.push_back(std::make_pair(vEdges[0], -0.09375));
		eValues.push_back(std::make_pair(vEdges[1], 0.125));
		eValues.push_back(std::make_pair(vEdges[2], 0.3125)); // typo in [Wang et al. 2006]
		eValues.push_back(std::make_pair(vEdges[3], 0.125));
		eValues.push_back(std::make_pair(vEdges[4], -0.09375));

		fValues.push_back(std::make_pair(vFaces[0], 0.03125));
		fValues.push_back(std::make_pair(vFaces[1], 0.03125));
		fValues.push_back(std::make_pair(vFaces[2], -0.03125));
		fValues.push_back(std::make_pair(vFaces[3], -0.03125));
	}
	else if (edgeInVert == 1)
	{
		// Fig.10 mid-mid in [Wang et al. 2006]
		eValues.push_back(std::make_pair(vEdges[0], 0.15625 / 3));
		eValues.push_back(std::make_pair(vEdges[1], 0.3125)); // typo in [Wang et al. 2006]
		eValues.push_back(std::make_pair(vEdges[2], 0.09375));
		eValues.push_back(std::make_pair(vEdges[3], 0.125 / 3));
		eValues.push_back(std::make_pair(vEdges.back(), -0.125));

		fValues.push_back(std::make_pair(vFaces[0], 0.15625 / 3));
		fValues.push_back(std::make_pair(vFaces[1], -0.03125 / 3));
		fValues.push_back(std::make_pair(vFaces[2], -0.125 / 3));
	}
	else if (edgeInVert == edgeCount - 2)
	{
		// Symmetric case of Fig.10 mid-mid in [Wang et al. 2006]
		eValues.push_back(std::make_pair(vEdges[edgeCount - 1], 0.15625 / 3));
		eValues.push_back(std::make_pair(vEdges[edgeCount - 2], 0.3125)); // typo in [Wang et al. 2006]
		eValues.push_back(std::make_pair(vEdges[edgeCount - 3], 0.09375));
		eValues.push_back(std::make_pair(vEdges[edgeCount - 4], 0.125 / 3));
		eValues.push_back(std::make_pair(vEdges.front(), -0.125));

		fValues.push_back(std::make_pair(vFaces[faceCount - 1], -0.15625 / 3));
		fValues.push_back(std::make_pair(vFaces[faceCount - 2], 0.03125 / 3));
		fValues.push_back(std::make_pair(vFaces[faceCount - 3], 0.125 / 3));
	}
	else if (edgeInVert == 2)
	{
		// Fig10 bot-left in [Wang et al. 2006]
		eValues.push_back(std::make_pair(vEdges[0], -0.09375));
		eValues.push_back(std::make_pair(vEdges[1], 0.125));
		eValues.push_back(std::make_pair(vEdges[2], 0.3125)); // typo in [Wang et al. 2006]
		eValues.push_back(std::make_pair(vEdges[3], 0.125));
		eValues.push_back(std::make_pair(vEdges[4], 0.03125));
		eValues.push_back(std::make_pair(vEdges[edgeCount - 1], -0.125));

		fValues.push_back(std::make_pair(vFaces[0], 0.03125));
		fValues.push_back(std::make_pair(vFaces[1], 0.03125));
		fValues.push_back(std::make_pair(vFaces[2], -0.03125));
		fValues.push_back(std::make_pair(vFaces[3], -0.03125));
	}
	else if (edgeInVert == edgeCount - 3)
	{
		// Symmetric case of Fig10 bot-left in [Wang et al. 2006]
		eValues.push_back(std::make_pair(vEdges[edgeCount - 1], -0.09375));
		eValues.push_back(std::make_pair(vEdges[edgeCount - 2], 0.125));
		eValues.push_back(std::make_pair(vEdges[edgeCount - 3], 0.3125)); // typo in [Wang et al. 2006]
		eValues.push_back(std::make_pair(vEdges[edgeCount - 4], 0.125));
		eValues.push_back(std::make_pair(vEdges[edgeCount - 5], 0.03125));
		eValues.push_back(std::make_pair(vEdges[0], -0.125));

		fValues.push_back(std::make_pair(vFaces[faceCount - 1], -0.03125));
		fValues.push_back(std::make_pair(vFaces[faceCount - 2], -0.03125));
		fValues.push_back(std::make_pair(vFaces[faceCount - 3], 0.03125));
		fValues.push_back(std::make_pair(vFaces[faceCount - 4], 0.03125));
	}
	else
	{
		// Fig10 bot-mid in [Wang et al. 2006]
		eValues.push_back(std::make_pair(vEdges[edgeInVert - 2], 0.03125));
		eValues.push_back(std::make_pair(vEdges[edgeInVert - 1], 0.125));
		eValues.push_back(std::make_pair(vEdges[edgeInVert], 0.3125)); // typo in [Wang et al. 2006]
		eValues.push_back(std::make_pair(vEdges[edgeInVert + 1], 0.125));
		eValues.push_back(std::make_pair(vEdges[edgeInVert + 2], 0.03125));
		eValues.push_back(std::make_pair(vEdges.front(), -0.125));
		eValues.push_back(std::make_pair(vEdges.back(), -0.125));

		fValues.push_back(std::make_pair(vFaces[edgeInVert - 2], 0.03125));
		fValues.push_back(std::make_pair(vFaces[edgeInVert - 1], 0.03125));
		fValues.push_back(std::make_pair(vFaces[edgeInVert], -0.03125));
		fValues.push_back(std::make_pair(vFaces[edgeInVert + 1], -0.03125));
	}

	for (size_t i = 0; i < eValues.size(); ++i)
	{
		_InsertEdgeEdgeValue(row, eValues[i].first, vert, rSign, eValues[i].second, out);
	}

	for (size_t i = 0; i < fValues.size(); ++i)
	{
		_InsertEdgeFaceValue(row, fValues[i].first, vert, rSign, fValues[i].second, out);
	}
}

void
ComplexLoopNew::_AssembleEdgeEvenInterior(int edge, int vertInEdge, TripletInserter out) const
{
	int vert = _meshPtr->GetEdgeVerts(edge)[vertInEdge];
	int edgeInVert = _meshPtr->GetEdgeIndexInVert(vert, edge);

	int row = _GetEdgeEdgeIndex(edge, vertInEdge);
	int rSign = (_GetVertVertIndex(vert) < _GetEdgeVertIndex(edge)) ? 1 : -1;

	const std::vector<int>& vEdges = _meshPtr->GetVertEdges(vert);
	const std::vector<int>& vFaces = _meshPtr->GetVertFaces(vert);

	std::vector< std::pair<int, Scalar> > eValues;
	std::vector< std::pair<int, Scalar> > fValues;

	const int count = vEdges.size();
	assert(count == vFaces.size());

	Scalar alpha = _GetAlpha(vert);
	Scalar beta = _GetBeta(vert);

	if (count == 3)
	{
		// Fig6 bot [Wang et al. 2006]
		eValues.push_back(std::make_pair(vEdges[(edgeInVert + 0) % count], 0.375 - alpha - 0.25 * beta));
		eValues.push_back(std::make_pair(vEdges[(edgeInVert + 1) % count], 0.125 * (1. + beta) - alpha));
		eValues.push_back(std::make_pair(vEdges[(edgeInVert + 2) % count], 0.125 * (1. + beta) - alpha));
	}
	else if (count == 4)
	{
		// Case not covered in [Wang et al. 2006]
		eValues.push_back(std::make_pair(vEdges[(edgeInVert + 0) % count], 0.375 - alpha - 0.25 * beta));
		eValues.push_back(std::make_pair(vEdges[(edgeInVert + 1) % count], 0.125 - alpha));
		eValues.push_back(std::make_pair(vEdges[(edgeInVert + 2) % count], 0.25 * beta - alpha));
		eValues.push_back(std::make_pair(vEdges[(edgeInVert + 3) % count], 0.125 - alpha));
	}
	else
	{
		// Fig6 top [Wang et al. 2006]
		for (int i = 0; i < count; ++i)
		{
			Scalar val = 0.;
			if (i == 0) val = 0.375 - alpha - 0.25 * beta;
			else if (i == 1 || i == count - 1) val = 0.125 - alpha;
			else if (i == 2 || i == count - 2) val = 0.125 * beta - alpha;
			else val = -alpha;
			eValues.push_back(std::make_pair(vEdges[(edgeInVert + i) % count], val));
		}
	}

	if (count == 3)
	{
		// Fig6 bot [Wang et al. 2006]
		fValues.push_back(std::make_pair(vFaces[(edgeInVert + 0) % count], -0.125 * beta));
		fValues.push_back(std::make_pair(vFaces[(edgeInVert + 2) % count], 0.125 * beta));
	}
	else
	{
		// Fig6 top [Wang et al. 2006]
		fValues.push_back(std::make_pair(vFaces[(edgeInVert + 0) % count], -0.125 * beta));
		fValues.push_back(std::make_pair(vFaces[(edgeInVert + 1) % count], -0.125 * beta));
		fValues.push_back(std::make_pair(vFaces[(edgeInVert + count - 1) % count], 0.125 * beta));
		fValues.push_back(std::make_pair(vFaces[(edgeInVert + count - 2) % count], 0.125 * beta));
	}

	for (size_t i = 0; i < eValues.size(); ++i)
	{
		_InsertEdgeEdgeValue(row, eValues[i].first, vert, rSign, eValues[i].second, out);
	}

	for (size_t i = 0; i < fValues.size(); ++i)
	{
		_InsertEdgeFaceValue(row, fValues[i].first, vert, rSign, fValues[i].second, out);
	}
}

void
ComplexLoopNew::_AssembleEdgeOdd(int face, int edgeInFace, TripletInserter out) const
{
	int row = _GetFaceEdgeIndex(face, edgeInFace);

	int vertInFace = (edgeInFace + 1) % 3;
	int vert = _meshPtr->GetFaceVerts(face)[vertInFace];

	int nEdge = _meshPtr->GetFaceEdges(face)[vertInFace];
	int nSign = _meshPtr->GetEdgeSignInFace(face, vertInFace);

	int oEdge = _meshPtr->GetFaceEdges(face)[(vertInFace + 1) % 3];
	int oSign = _meshPtr->GetEdgeSignInFace(face, (vertInFace + 1) % 3);

	int pEdge = _meshPtr->GetFaceEdges(face)[(vertInFace + 2) % 3];
	int pSign = _meshPtr->GetEdgeSignInFace(face, (vertInFace + 2) % 3);

	int rSign = (_GetEdgeVertIndex(nEdge) < _GetEdgeVertIndex(pEdge)) ? 1 : -1;

	bool nBdry = _meshPtr->IsEdgeBoundary(nEdge);
	bool pBdry = _meshPtr->IsEdgeBoundary(pEdge);

	if (nBdry && pBdry)
	{
		// Ear case not covered in [Wang et al. 2006]
		*out++ = TripletX(row, oEdge, (oSign == rSign) ? 0.25 : -0.25);
		*out++ = TripletX(row, pEdge, (pSign == rSign) ? -0.25 : 0.25);
		*out++ = TripletX(row, nEdge, (nSign == rSign) ? -0.25 : 0.25);
	}
	else if (nBdry)
	{
		// Symmetric case of Fig10 top-left in [Wang et al. 2006]
		*out++ = TripletX(row, oEdge, (oSign == rSign) ? 0.21875 : -0.21875);
		*out++ = TripletX(row, pEdge, (pSign == rSign) ? -0.1875 : 0.1875);
		*out++ = TripletX(row, nEdge, (nSign == rSign) ? -0.15625 : 0.15625);
	}
	else if (pBdry)
	{
		// Fig10 top-left in [Wang et al. 2006]
		*out++ = TripletX(row, oEdge, (oSign == rSign) ? 0.21875 : -0.21875);
		*out++ = TripletX(row, pEdge, (pSign == rSign) ? -0.15625 : 0.15625);
		*out++ = TripletX(row, nEdge, (nSign == rSign) ? -0.1875 : 0.1875);
	}
	else
	{
		// Fig4 mid-bot [Wang et al. 2006]
		*out++ = TripletX(row, oEdge, (oSign == rSign) ? 0.1875 : -0.1875);
		*out++ = TripletX(row, pEdge, (pSign == rSign) ? -0.09375 : 0.09375);
		*out++ = TripletX(row, nEdge, (nSign == rSign) ? -0.09375 : 0.09375);
	}

	std::vector<int> flaps;
	flaps.push_back(pEdge);
	flaps.push_back(nEdge);
	for (int i = 0; i < flaps.size(); ++i)
	{
		int edge = flaps[i];
		if (_meshPtr->IsEdgeBoundary(edge)) continue;

		int faceInEdge = _meshPtr->GetFaceIndexInEdge(edge, face);
		int flap = _meshPtr->GetEdgeFaces(edge)[(faceInEdge + 1) % 2];
		int edgeInFlap = _meshPtr->GetEdgeIndexInFace(flap, edge);

		for (int j = 1; j <= 2; ++j)
		{
			// Fig4 mid-bot [Wang et al. 2006]
			Scalar val = 0.09375;
			int cEdge = _meshPtr->GetFaceEdges(flap)[(edgeInFlap + j) % 3];
			int cSign = _meshPtr->GetEdgeSignInFace(flap, (edgeInFlap + j) % 3);
			if (_meshPtr->GetVertIndexInEdge(cEdge, vert) == -1) val = -0.03125;
			*out++ = TripletX(row, cEdge, (cSign == rSign) ? -val : val);
		}
	}
}

std::complex<double> interpZ(const std::vector<std::complex<double>>& zList, const std::vector<Eigen::Vector3d>& gradThetaList, std::vector<double>& coords, const std::vector<Eigen::Vector3d>& pList, Eigen::Vector3cd *gradZ)
{
	Eigen::Vector3d P = Eigen::Vector3d::Zero();
	
	int n = pList.size();
	for (int i = 0; i < n; i++)
	{
		P += coords[i] * pList[i];
	}
	std::complex<double> z = 0;
    Eigen::MatrixXd pseduInv;
    if(gradZ)
    {
        gradZ->setZero();
        Eigen::MatrixXd dalpha(n, 3);
        for(int i = 0; i < n; i++)
            dalpha.row(i) = pList[i];

        pseduInv = (dalpha.transpose() * dalpha).inverse() * dalpha.transpose();
    }


    Eigen::VectorXcd dzdalpha;
    dzdalpha.setZero(n);

    std::complex<double> I = std::complex<double>(0, 1);

	for (int i = 0; i < n; i++)
	{
        double deltaTheta = (P - pList[i]).dot(gradThetaList[i]);
		z += coords[i] * zList[i] * std::complex<double>(std::cos(deltaTheta), std::sin(deltaTheta)) ;

        if(gradZ)
        {
            for(int j = 0; j < n; j++)
            {
                double deltaThetaj = (P - pList[j]).dot(gradThetaList[j]);
                dzdalpha(i) += coords[j] * zList[j] * std::complex<double>(std::cos(deltaThetaj), std::sin(deltaThetaj)) * I * gradThetaList[j].dot(pList[j]);
            }

            dzdalpha(i) += zList[i] * std::complex<double>(std::cos(deltaTheta), std::sin(deltaTheta));
        }

	}
	return z;
}

std::complex<double> computeZandGradZ(const Mesh& mesh, const Eigen::VectorXd& omega, const std::vector<std::complex<double>>& zvals, int fid, const Eigen::Vector3d& bary, Eigen::Vector3cd* gradz)
{
	Eigen::Matrix3d gradBary;
	Eigen::Vector3d P0 = mesh.GetVertPos(mesh.GetFaceVerts(fid)[0]);
	Eigen::Vector3d P1 = mesh.GetVertPos(mesh.GetFaceVerts(fid)[1]);
	Eigen::Vector3d P2 = mesh.GetVertPos(mesh.GetFaceVerts(fid)[2]);
	
	computeBaryGradient(P0, P1, P2, bary, gradBary);

	std::complex<double> z = 0;

	if (gradz)
	{
		gradz->setZero();
	}
	
	
	std::complex<double> tau = std::complex<double>(0, 1);

	for (int i = 0; i < 3; i++)
	{
		double alphai = bary(i);
		int vid = mesh.GetFaceVerts(fid)[i];
		int eid0 = mesh.GetFaceEdges(fid)[i];
		int eid1 = mesh.GetFaceEdges(fid)[(i + 2) % 3];

		double w1 = omega(eid0);
		double w2 = omega(eid1);

		if (vid == mesh.GetEdgeVerts(eid0)[1])
			w1 *= -1;
		if (vid == mesh.GetEdgeVerts(eid1)[1])
			w2 *= -1;

		double deltaTheta = w1 * bary((i + 1) % 3) + w2 * bary((i + 2) % 3);
		std::complex<double> expip = std::complex<double>(std::cos(deltaTheta), std::sin(deltaTheta));

		z += alphai * zvals[vid] * expip;

		if(gradz)
			(*gradz) += zvals[vid] * expip * (gradBary.row(i) + tau * alphai * w1 * gradBary.row((i + 1) % 3) + tau * alphai * w2 * gradBary.row((i + 2) % 3));
	}
	return z;
}

void updateLoopedZvals(const Mesh& mesh, const Eigen::VectorXd& omega, const std::vector<std::complex<double>>& zvals, std::vector<std::complex<double>>& upZvals, std::vector<Eigen::Vector3d> *upOmega)
{
	int V = mesh.GetVertCount();
	int E = mesh.GetEdgeCount();
	
	upZvals.resize(V + E);

	// Even verts
	for (int vi = 0; vi < V; ++vi)
	{
		if (mesh.IsVertBoundary(vi))
		{
			std::vector<int> boundary(2);
			boundary[0] = mesh.GetVertEdges(vi).front();
			boundary[1] = mesh.GetVertEdges(vi).back();

			std::vector<std::complex<double>> zp(2);
			std::vector<Eigen::Vector3d> gradthetap(2);
			std::vector<double> coords = { 1. / 2, 1. / 2 };
			std::vector<Eigen::Vector3d> pList(2);

			for (int j = 0; j < boundary.size(); ++j)
			{
				int edge = boundary[j];
				assert(mesh.IsEdgeBoundary(edge));
				int face = mesh.GetEdgeFaces(edge)[0];
				int viInface = mesh.GetVertIndexInFace(face, vi);

				int viInEdge = mesh.GetVertIndexInEdge(edge, vi);
				int vj = mesh.GetEdgeVerts(edge)[(viInEdge + 1) % 2];

				int vjInface = mesh.GetVertIndexInFace(face, vj);

				Eigen::Vector3d bary = Eigen::Vector3d::Zero();
				bary(viInface) = 3. / 4;
				bary(vjInface) = 1. / 4;

                Eigen::Vector3cd gradZ;
				zp[j] = computeZandGradZ(mesh, omega, zvals, face, bary, &(gradZ));

				pList[j] = 3. / 4 * mesh.GetVertPos(vi) + 1. / 4 * mesh.GetVertPos(vj);
                gradthetap[j] = (std::conj(zp[j]) * gradZ).imag();

                if(std::abs(zp[j]))
                    gradthetap[j] = gradthetap[j] / (std::abs(zp[j]) * std::abs(zp[j]));

			}
			upZvals[vi] = interpZ(zp, gradthetap, coords, pList);
		}
		else
		{
			const std::vector<int>& vFaces = mesh.GetVertFaces(vi);
			int nNeiFaces = vFaces.size();

			// Fig5 left [Wang et al. 2006]
			Scalar alpha = 0.375;
			if (nNeiFaces == 3) alpha /= 2;
			else                    alpha /= nNeiFaces;

			double beta = nNeiFaces / 2. * alpha;

			std::vector<std::complex<double>> zp(nNeiFaces);
			std::vector<Eigen::Vector3d> gradthetap(nNeiFaces);
			std::vector<double> coords;
			coords.resize(nNeiFaces, 1. / nNeiFaces);
			std::vector<Eigen::Vector3d> pList(nNeiFaces);

			for (int k = 0; k < nNeiFaces; ++k)
			{
				int face = vFaces[k];
				int viInface = mesh.GetVertIndexInFace(face, vi);
				Eigen::Vector3d bary;
				bary.setConstant(beta);
				bary(viInface) = 1 - 2 * beta;

				pList[k] = Eigen::Vector3d::Zero();
				for (int i = 0; i < 3; i++)
				{
					pList[k] += bary(i) * mesh.GetVertPos(mesh.GetFaceVerts(face)[i]);
				}
                Eigen::Vector3cd gradZ;
				zp[k] = computeZandGradZ(mesh, omega, zvals, face, bary, &(gradZ));

                gradthetap[k] = (std::conj(zp[k]) * gradZ).imag();

                if(std::abs(zp[k]))
                    gradthetap[k] = gradthetap[k] / (std::abs(zp[k]) * std::abs(zp[k]));
			}
			upZvals[vi] = interpZ(zp, gradthetap, coords, pList);
		}
	}

	// Odd verts
	for (int edge = 0; edge < E; ++edge)
	{
		int row = edge + V;
		if (mesh.IsEdgeBoundary(edge))
		{
			int face = mesh.GetEdgeFaces(edge)[0];
			int eindexFace = mesh.GetEdgeIndexInFace(face, edge);
			Eigen::Vector3d bary;
			bary.setConstant(0.5);
			bary((eindexFace + 2) % 3) = 0;

			upZvals[row] = computeZandGradZ(mesh, omega, zvals, face, bary, NULL);
		}
		else
		{
			
			std::vector<std::complex<double>> zp(2);
			std::vector<Eigen::Vector3d> gradthetap(2);
			std::vector<double> coords = { 1. / 2, 1. / 2 };
			std::vector<Eigen::Vector3d> pList(2);

			for (int j = 0; j < 2; ++j)
			{
				int face = mesh.GetEdgeFaces(edge)[j];
				int offset = mesh.GetEdgeIndexInFace(face, edge);

				Eigen::Vector3d bary;
				bary.setConstant(3. / 8.);
				bary((offset + 2) % 3) = 0.25;

				pList[j] = Eigen::Vector3d::Zero();
				for (int i = 0; i < 3; i++)
				{
					pList[j] += bary(i) * mesh.GetVertPos(mesh.GetFaceVerts(face)[i]);
				}
                Eigen::Vector3cd gradZ;
                zp[j] = computeZandGradZ(mesh, omega, zvals, face, bary, &(gradZ));

                gradthetap[j] = (std::conj(zp[j]) * gradZ).imag();

                if(std::abs(zp[j]))
                    gradthetap[j] = gradthetap[j] / (std::abs(zp[j]) * std::abs(zp[j]));
			}

			upZvals[row] = interpZ(zp, gradthetap, coords, pList);
		}
	}
}

void updateLoopedZvalsNew(const Mesh& mesh, const Eigen::VectorXd& omega, const std::vector<std::complex<double>>& zvals, std::vector<std::complex<double>>& upZvals)
{
    int V = mesh.GetVertCount();
    int E = mesh.GetEdgeCount();

    upZvals.resize(V + E);

    // Even verts
    for (int vi = 0; vi < V; ++vi)
    {
        if (mesh.IsVertBoundary(vi))
        {
            std::vector<int> boundary(2);
            boundary[0] = mesh.GetVertEdges(vi).front();
            boundary[1] = mesh.GetVertEdges(vi).back();

            std::vector<std::complex<double>> zp(3);
            std::vector<Eigen::Vector3d> gradthetap(3);
            std::vector<double> coords = { 1. / 4, 1. / 4, 3. / 4 };
            std::vector<Eigen::Vector3d> pList(3);

            pList[2] = mesh.GetVertPos(vi);
            zp[2] = 0;
            gradthetap[2].setZero();


            for (int j = 0; j < boundary.size(); ++j)
            {
                int edge = boundary[j];
                assert(mesh.IsEdgeBoundary(edge));
                int face = mesh.GetEdgeFaces(edge)[0];
                int viInface = mesh.GetVertIndexInFace(face, vi);

                int viInEdge = mesh.GetVertIndexInEdge(edge, vi);
                int vj = mesh.GetEdgeVerts(edge)[(viInEdge + 1) % 2];

                int vjInface = mesh.GetVertIndexInFace(face, vj);

                Eigen::Vector3d bary = Eigen::Vector3d::Zero();
                bary(vjInface)  = 1;

                Eigen::Vector3cd gradZ;
                zp[j] = computeZandGradZ(mesh, omega, zvals, face, bary, &(gradZ));

                pList[j] = mesh.GetVertPos(vj);
                gradthetap[j] = (std::conj(zp[j]) * gradZ).imag();

                if(std::abs(zp[j]))
                    gradthetap[j] = gradthetap[j] / (std::abs(zp[j]) * std::abs(zp[j]));

                bary.setZero();
                bary(viInface) = 1;

                std::complex<double> zi = computeZandGradZ(mesh, omega, zvals, face, bary, &(gradZ));
                zp[2] += zi / 2.0;

                Eigen::Vector3d gradthetai = (std::conj(zi) * gradZ).imag();
                if(std::abs(zi))
                    gradthetai = gradthetai / (std::abs(zi) * std::abs(zi));
                gradthetap[2] += gradthetai / 2;

            }
            upZvals[vi] = interpZ(zp, gradthetap, coords, pList);
        }
        else
        {
            const std::vector<int>& vFaces = mesh.GetVertFaces(vi);
            int nNeiFaces = vFaces.size();

            // Fig5 left [Wang et al. 2006]
            Scalar alpha = 0.375;
            if (nNeiFaces == 3) alpha /= 2;
            else                    alpha /= nNeiFaces;

            double beta = nNeiFaces / 2. * alpha;

            std::vector<std::complex<double>> zp(nNeiFaces + 1);
            std::vector<Eigen::Vector3d> gradthetap(nNeiFaces + 1);
            std::vector<double> coords;
            coords.resize(nNeiFaces + 1, alpha);
            std::vector<Eigen::Vector3d> pList(nNeiFaces + 1);

            zp[nNeiFaces] = 0;
            gradthetap[nNeiFaces].setZero();
            coords[nNeiFaces] = 1 - nNeiFaces * alpha;
            pList[nNeiFaces] = mesh.GetVertPos(vi);

        }
    }

    // Odd verts
    for (int edge = 0; edge < E; ++edge)
    {
        int row = edge + V;
        if (mesh.IsEdgeBoundary(edge))
        {
            int face = mesh.GetEdgeFaces(edge)[0];
            int eindexFace = mesh.GetEdgeIndexInFace(face, edge);
            Eigen::Vector3d bary;
            bary.setConstant(0.5);
            bary((eindexFace + 2) % 3) = 0;

            upZvals[row] = computeZandGradZ(mesh, omega, zvals, face, bary, NULL);
        }
        else
        {

            std::vector<std::complex<double>> zp(2);
            std::vector<Eigen::Vector3d> gradthetap(2);
            std::vector<double> coords = { 1. / 2, 1. / 2 };
            std::vector<Eigen::Vector3d> pList(2);

            for (int j = 0; j < 2; ++j)
            {
                int face = mesh.GetEdgeFaces(edge)[j];
                int offset = mesh.GetEdgeIndexInFace(face, edge);

                Eigen::Vector3d bary;
                bary.setConstant(3. / 8.);
                bary((offset + 2) % 3) = 0.25;

                pList[j] = Eigen::Vector3d::Zero();
                for (int i = 0; i < 3; i++)
                {
                    pList[j] += bary(i) * mesh.GetVertPos(mesh.GetFaceVerts(face)[i]);
                }
                Eigen::Vector3cd gradZ;
                zp[j] = computeZandGradZ(mesh, omega, zvals, face, bary, &(gradZ));

                gradthetap[j] = (std::conj(zp[j]) * gradZ).imag();

                if(std::abs(zp[j]))
                    gradthetap[j] = gradthetap[j] / (std::abs(zp[j]) * std::abs(zp[j]));
            }

            upZvals[row] = interpZ(zp, gradthetap, coords, pList);
        }
    }
}

void SubdivideNew(const Mesh& mesh, const Eigen::VectorXd& omega, const std::vector<std::complex<double>>& zvals, Eigen::VectorXd& omegaNew, std::vector<std::complex<double>>& upZvals, int level, Mesh& meshNew)
{
	std::unique_ptr<ComplexLoopNew> subd = std::make_unique<ComplexLoopNew>();
	subd->SetMesh(mesh);

	int nverts = mesh.GetVertCount();
	omegaNew = omega;
	upZvals = zvals;


	MatrixX X;
	mesh.GetPos(X);

	meshNew = mesh;
	omegaNew = omega;

	Eigen::VectorXd amp(nverts);
	
	for (int i = 0; i < nverts; i++)
	{
		amp(i) = std::abs(zvals[i]);
	}

	for (int l = 0; l < level; ++l) 
	{
		subd->SetMesh(meshNew);
		SparseMatrixX tmpS0, tmpS1;
		subd->BuildS0(tmpS0);
		subd->BuildS1(tmpS1);

		X = tmpS0 * X;
		amp = tmpS0 * amp;

        std::vector<std::complex<double>> upZvalsNew;

		updateLoopedZvals(meshNew, omegaNew, upZvals, upZvalsNew);
		omegaNew = tmpS1 * omegaNew;

		std::vector<Vector3> points;
		ConvertToVector3(X, points);

		std::vector< std::vector<int> > edgeToVert;
		subd->GetSubdividedEdges(edgeToVert);

		std::vector< std::vector<int> > faceToVert;
		subd->GetSubdividedFaces(faceToVert);

		meshNew.Populate(points, faceToVert, edgeToVert);

        upZvals.swap(upZvalsNew);
	}
}
