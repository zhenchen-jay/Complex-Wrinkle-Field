#pragma once
#include "../CommonTools.h"
#include "../MeshLib/MeshConnectivity.h"
#include <Eigen/Dense>

namespace EuclideanFormula
{
	std::complex<double> planeWaveBasis(Eigen::Vector3d p, Eigen::Vector3d pi, Eigen::Vector3d omega, Eigen::Vector3cd* deriv, Eigen::Matrix3cd* hess);

	std::complex<double> getZvalsFromEuclideanOmega(const Eigen::MatrixXd& basePos, const MeshConnectivity& baseMesh, const int& faceId, const Eigen::Vector3d& bary, const std::vector<std::complex<double>> vertZvals, const Eigen::MatrixXd& w, Eigen::Matrix<std::complex<double>, 15, 1>* deriv, Eigen::Matrix<std::complex<double>, 15, 15>* hess);
	/*
	* compute z values, given the barycentric coordinates
	bary = (alphi0, alpha1, alpha2).
	w = V by 3, each row (wx, wy, wz), wx, y, z is the x,y,z-coordinates 
	vertZvals = (z0, z1, z2)
	*/

	void testZvalsFromEuclideanOmega(const Eigen::MatrixXd& basePos, const MeshConnectivity& baseMesh, const int& faceId, const Eigen::Vector3d& bary, const std::vector<std::complex<double>> vertZvals, const Eigen::MatrixXd& edgeOmega);

	std::vector<std::complex<double>> upsamplingZvals(const Eigen::MatrixXd& triV, const MeshConnectivity& mesh, const std::vector<std::complex<double>>& zvals, const Eigen::MatrixXd& w, const std::vector<std::pair<int, Eigen::Vector3d>>& bary);

}