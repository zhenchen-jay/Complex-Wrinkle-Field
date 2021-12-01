#pragma once

#include "../MeshLib/MeshConnectivity.h"
#include <Eigen/Dense>
#include <complex>
#include <vector>

class PhaseInterpolation
{
public:
	struct CutEdge
	{
		int refEid;		// edge id in the reference shape
		int restEid;		// edge id in the rest shape

		Eigen::Vector2i refAdjFid;		// face id which contains the edge in the reference shape (each edge should have two adjacent faces)
		int restAdjFid;		// face id which contains the edge in the rest shape (each edge should have exact one adjacent face)
	};

public:
	PhaseInterpolation() {}
	PhaseInterpolation(const Eigen::MatrixXd& restV, const MeshConnectivity& restMesh, const Eigen::MatrixXd& upsampledRestV, const MeshConnectivity& upsampledRestMesh, const Eigen::MatrixXd& refV, const MeshConnectivity& refMesh, const Eigen::MatrixXd& upsampledRefV, const MeshConnectivity& upsampledRefMesh);
	/*
	* Input:
	* 1. rest shape (restV, restMesh).
	* 2. upsampled rest mesh (upsampledRestV, upsampledRestMesh).
	* 3. reference 3D shape (refV, refMesh), this is used to locate the cuts and its connectivity remains the same during the simulation
	* 4. upsampled reference 3D shape
	* Required: For the upsampling, we should use the same scheme for rest shape and reference 3D shape
	*/

	void estimatePhase(const Eigen::MatrixXd& vertexOmega, const std::vector<std::complex<double>>& vertexPhi, std::vector<std::complex<double>>& upsampledPhi, int interpolationType = 0);
	/*
	* Main function: take per-vertex omega and per-vertex phi (complex value a+bi), returns per-vertex upsampled phi (complex value a+bi).
	* All the (upsampled) values are for the (upsampled) reference 3D shape. In this way, we can handle the cutting issue.
	* interpolation_type: 0: composite, 1: planewave, 2: waterpool
	*/

	void estimatePhasePerface(const Eigen::MatrixXd& vertexOmega, const Eigen::VectorXd& globalOmega, const std::vector<std::complex<double>>& vertexPhi, int faceId, Eigen::Vector3d baryCoord, std::complex<double>& Phi, int interpolationType = 0);
	/*
	* Note: baryCoord: barycentric coordinate (w, u, v), w + u + v = 1
	*/

	void estimatePhase(const Eigen::MatrixXd& planeOmega, const Eigen::MatrixXd& waterpoolOmega, const std::vector<std::complex<double>>& vertexPhi, std::vector<std::complex<double>>& upsampledPhi);
	void estimatePhasePerface(const Eigen::MatrixXd& planeOmega, const Eigen::MatrixXd& waterpoolOmega, const std::vector<std::complex<double>>& vertexPhi, int faceId, Eigen::Vector3d baryCoord, std::complex<double>& Phi);
	/*
	* Note: baryCoord: barycentric coordinate (w, u, v), w + u + v = 1
	*/
	

	void getAngleMagnitude(const std::vector<std::complex<double>>& Phi, Eigen::VectorXd& angle, Eigen::VectorXd& mag);

private:
	void initialization();	// initialization: locate the cuts and compute barycentric coordinates. These are only needed to compute once
	void locateCuts();
	void computeBaryCoordinates();

	void cur2restMap(const Eigen::MatrixXd& V2D, const MeshConnectivity& mesh2D, const Eigen::MatrixXd& V3D, const MeshConnectivity& mesh3D, std::vector<Eigen::Vector2i>& map);
	void rest2curMap(const Eigen::MatrixXd& V2D, const MeshConnectivity& mesh2D, const Eigen::MatrixXd& V3D, const MeshConnectivity& mesh3D, std::vector<int>& map);
	double pointFaceCoord(const Eigen::Vector3d& p, const Eigen::MatrixXd& V, const MeshConnectivity& mesh, int faceId, Eigen::Vector3d& coords);
	// compute the barycentric coordinates, where p = (x, y, 0) and V is flat (z = 0).

	// basis function
	std::complex<double> waterPoolBasis(Eigen::VectorXd p, Eigen::VectorXd v, Eigen::VectorXd omega); // we now only consider the 2d case
	std::complex<double> planWaveBasis(Eigen::VectorXd p, Eigen::VectorXd v, Eigen::VectorXd omega);



public:
	Eigen::MatrixXd _restV, _upsampledRestV, _refV, _upsampledRefV;
	MeshConnectivity _restMesh, _upsampledRestMesh, _refMesh, _upsampledRefMesh;
	std::vector<CutEdge> _cuts;
	std::vector<std::pair<int, Eigen::Vector3d>> _baryCoords;
};