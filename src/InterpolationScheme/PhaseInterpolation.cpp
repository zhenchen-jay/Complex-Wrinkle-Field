#include <iostream>

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

#ifndef GRAIN_SIZE
#define GRAIN_SIZE 10
#endif

#include "../../include/InterpolationScheme/PhaseInterpolation.h"

PhaseInterpolation::PhaseInterpolation(const Eigen::MatrixXd& restV, const MeshConnectivity& restMesh, const Eigen::MatrixXd& upsampledRestV, const MeshConnectivity& upsampledRestMesh, const Eigen::MatrixXd& refV, const MeshConnectivity& refMesh, const Eigen::MatrixXd& upsampledRefV, const MeshConnectivity& upsampledRefMesh)
{
	_restV = restV;
	_restMesh = restMesh;

	_upsampledRestV = upsampledRestV;
	_upsampledRestMesh = upsampledRestMesh;

	_refV = refV;
	_refMesh = refMesh;

	_upsampledRefV = upsampledRefV;
	_upsampledRefMesh = upsampledRefMesh;

	initialization();
}

void PhaseInterpolation::initialization()
{
	locateCuts();
	computeBaryCoordinates();
}

void PhaseInterpolation::getAngleMagnitude(const std::vector<std::complex<double>>& Phi, Eigen::VectorXd& angle, Eigen::VectorXd& mag)
{
	int num = Phi.size();
	
	angle.setZero(num);
	mag.setZero(num);

	for (int i = 0; i < num; i++)
	{
		mag(i) = std::abs(Phi[i]);
		angle(i) = std::arg(Phi[i]);
	}
}

void PhaseInterpolation::locateCuts()
{
	_cuts.clear();
	int nfaces = _restMesh.nFaces();

	assert(nfaces == _refMesh.nFaces());

	std::vector<bool> isVisited(_refMesh.nEdges(), false);


	for (int i = 0; i < nfaces; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			int restEid = _restMesh.faceEdge(i, j);
			int refEid = _refMesh.faceEdge(i, j);

			if (_restMesh.edgeFace(restEid, 0) == -1 || _restMesh.edgeFace(restEid, 1) == -1) // this edge is on the boundary of the rest shape
			{
				if (_refMesh.edgeFace(refEid, 0) != -1 && _refMesh.edgeFace(refEid, 1) != -1) // but it is not on the boundary of the reference shape, then it is a cut edge
				{
					if (!isVisited[refEid]) // if this edge is not visited, we add it to the cut edge vector
					{
						CutEdge cut;
						cut.refEid = refEid;
						cut.restEid = restEid;

						cut.restAdjFid = i;
						cut.refAdjFid(0) = _refMesh.edgeFace(refEid, 0);
						cut.refAdjFid(1) = _refMesh.edgeFace(refEid, 1);

						_cuts.push_back(cut);
						isVisited[refEid] = true;
					}
				}
			}
		}
	}
	std::cout << "Find " << _cuts.size() << " cut edges. " << std::endl;
}

void PhaseInterpolation::cur2restMap(const Eigen::MatrixXd& V2D, const MeshConnectivity& mesh2D, const Eigen::MatrixXd& V3D, const MeshConnectivity& mesh3D, std::vector<Eigen::Vector2i>& map)
{
	int nverts = V3D.rows();
	int nfaces = mesh3D.nFaces();

	assert(nfaces == mesh3D.nFaces());

	map.resize(nverts, Eigen::Vector2i::Constant(-1));


	for (int i = 0; i < nfaces; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			int vid3d = mesh3D.faceVertex(i, j);
			int vid2d = mesh2D.faceVertex(i, j);

			if (map[vid3d](0) == -1)
				map[vid3d](0) = vid2d;
			else
				map[vid3d](1) = vid2d;
		}
	}
}

void PhaseInterpolation::rest2curMap(const Eigen::MatrixXd& V2D, const MeshConnectivity& mesh2D, const Eigen::MatrixXd& V3D, const MeshConnectivity& mesh3D, std::vector<int>& map)
{
	int nverts = V2D.rows();
	int nfaces = mesh2D.nFaces();

	assert(nfaces == mesh3D.nFaces());

	map.resize(nverts, -1);

	for (int i = 0; i < nfaces; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			int vid2d = mesh2D.faceVertex(i, j);
			int vid3d = mesh3D.faceVertex(i, j);

			map[vid2d] = vid3d;
		}
	}
}

double PhaseInterpolation::pointFaceCoord(const Eigen::Vector3d& p, const Eigen::MatrixXd& V, const MeshConnectivity& mesh, int faceId, Eigen::Vector3d& coords)
{
	Eigen::Matrix<double, 3, 2> H;
	Eigen::MatrixXi F = mesh.faces();
	H.col(0) = (V.row(F(faceId, 0)) - V.row(F(faceId, 2))).transpose();
	H.col(1) = (V.row(F(faceId, 1)) - V.row(F(faceId, 2))).transpose();

	Eigen::Vector3d b = p - V.row(F(faceId, 2)).transpose();

	Eigen::ColPivHouseholderQR<Eigen::Matrix<double, 3, 2>> solver(H);
	Eigen::Vector2d sol = solver.solve(b);

	double residual = (H * sol - b).norm();
	coords << sol(0), sol(1), 1 - sol(0) - sol(1);

	return residual;
}

void PhaseInterpolation::computeBaryCoordinates()
{
	std::vector<Eigen::Vector2i> map;
	cur2restMap(_upsampledRestV, _upsampledRestMesh, _upsampledRefV, _upsampledRefMesh, map);

	int nUpsampledverts = _upsampledRefV.rows();
	int nfaces = _restMesh.nFaces();

	// compute the barycentric coordinate information
	_baryCoords.resize(nUpsampledverts);
	double eps = 1e-8;

	auto getBaryPervertex = [&](const tbb::blocked_range<uint32_t>& range)
	{
		for (uint32_t i = range.begin(); i < range.end(); ++i)
		{
			Eigen::Vector3d p = _upsampledRestV.row(map[i](0)).transpose();
			Eigen::Vector3d coord;
			int faceId = -1;
			coord.setZero();
			for (int j = 0; j < nfaces; j++)
			{
				double res = pointFaceCoord(p, _restV, _restMesh.faces(), j, coord);
				if (res < 1e-6 && coord(0) >= -eps && coord(1) >= -eps && coord(2) >= -eps)
				{
					faceId = j;
					break;
				}
			}
			_baryCoords[i].first = faceId;
			_baryCoords[i].second = coord;
		}
	};

	tbb::blocked_range<uint32_t> rangex(0u, (uint32_t)nUpsampledverts, GRAIN_SIZE);
	tbb::parallel_for(rangex, getBaryPervertex);
}

std::complex<double> PhaseInterpolation::waterPoolBasis(Eigen::VectorXd p, Eigen::VectorXd v, Eigen::VectorXd omega)
{
	if (omega.norm() == 0)
		return 1;
	Eigen::Vector2d p0 = v.segment<2>(0) + Eigen::Vector2d(-omega(1), omega(0)) / omega.squaredNorm();
	
	return std::complex<double>(p(0) - p0(0), p(1) - p0(0)) / std::complex<double>(v(0) - p0(0), v(1) - p0(0));

}

std::complex<double> PhaseInterpolation::planWaveBasis(Eigen::VectorXd p, Eigen::VectorXd v, Eigen::VectorXd omega)
{
	if (omega.norm() == 0)
		return 1;
	double deltatheta = omega.dot(p - v);

	return std::complex<double>(std::cos(deltatheta), std::sin(deltatheta));
}

void PhaseInterpolation::estimatePhasePerface(const Eigen::MatrixXd& vertexOmega, const Eigen::VectorXd& globalOmega, const std::vector<std::complex<double>>& vertexPhi, int faceId, Eigen::Vector3d baryCoord, std::complex<double>& Phi, int interpolationType)
{
	Phi = 0;
	Eigen::VectorXd P = baryCoord(0) * _restV.row(_restMesh.faceVertex(faceId, 0)) + baryCoord(1) * _restV.row(_restMesh.faceVertex(faceId, 1)) + baryCoord(2) * _restV.row(_restMesh.faceVertex(faceId, 2));

	double prod = baryCoord(0) * baryCoord(1) * baryCoord(2);

	for (int i = 0; i < 3; i++)
	{
		int vid = _refMesh.faceVertex(faceId, i);
		Eigen::VectorXd Pi = _restV.row(_restMesh.faceVertex(faceId, i));

		std::complex<double> planewaveValue = 1;
		std::complex<double> waterpoolValue = 1;

		if (interpolationType == 0)
		{
			planewaveValue = planWaveBasis(P, Pi, globalOmega);
			waterpoolValue = waterPoolBasis(P, Pi, vertexOmega.row(i).transpose() - globalOmega);
		}
		else if (interpolationType == 1)
		{
			planewaveValue = planWaveBasis(P, Pi, vertexOmega.row(i).transpose());
			waterpoolValue = 1;
		}
		else
		{
			planewaveValue = 1;
			waterpoolValue = waterPoolBasis(P, Pi, vertexOmega.row(i).transpose());
		}
		

		double weight = 3 * baryCoord(i) * baryCoord(i) - 2 * std::pow(baryCoord(i), 3) + 2 * prod;

		Phi += weight * vertexPhi[vid] * planewaveValue * waterpoolValue;
	}
}

void PhaseInterpolation::estimatePhase(const Eigen::MatrixXd& vertexOmega, const std::vector<std::complex<double>>& vertexPhi, std::vector<std::complex<double>>& upsampledPhi, int interpolationType)
{
	if (vertexOmega.size() == 0)
	{
		std::cerr << "empty vertex omega!" << std::endl;
		exit(1);
	}
	int nUpsampledVerts = _upsampledRefV.rows();
	upsampledPhi.resize(nUpsampledVerts);

	Eigen::VectorXd globalOmega = vertexOmega.row(0);
	for (int i = 1; i < vertexOmega.rows(); i++)
		globalOmega += vertexOmega.row(i);
	globalOmega /= vertexOmega.rows();

	for (int i = 0; i < nUpsampledVerts; i++)
	{
		std::complex<double> interiorPhi;
		estimatePhasePerface(vertexOmega, globalOmega, vertexPhi, _baryCoords[i].first, _baryCoords[i].second, interiorPhi, interpolationType);
		upsampledPhi[i] = interiorPhi;
	}
}