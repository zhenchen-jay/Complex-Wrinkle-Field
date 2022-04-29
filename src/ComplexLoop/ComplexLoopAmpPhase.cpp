#include "../../include/ComplexLoop/ComplexLoopAmpPhase.h"
#include <iostream>
#include <cassert>
#include <memory>

void ComplexLoopAmpPhase::Subdivide(const Eigen::VectorXd& omega, const std::vector<std::complex<double>>& zvals, Eigen::VectorXd& omegaNew, std::vector<std::complex<double>>& upZvals, int level)
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

		X = tmpS0 * X;
		amp = tmpS0 * amp;

		theta = tmpS0 * theta;

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