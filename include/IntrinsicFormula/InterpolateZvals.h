#pragma once
#include "../CommonTools.h"
#include "../MeshLib/MeshConnectivity.h"
#include <Eigen/Dense>

namespace IntrinsicFormula
{
	std::complex<double> getZvalsFromEdgeOmega(const Eigen::Vector3d& bary, const std::vector<std::complex<double>> vertZvals, const Eigen::Vector3d& edgeOmega, Eigen::Matrix<std::complex<double>, 9, 1>* deriv = NULL, Eigen::Matrix<std::complex<double>, 9, 9>* hess = NULL);
	/*
	* compute z values, given the barycentric coordinates 
	bary = (alphi0, alpha1, alpha2). 
	edgeomega = (w12, w20, w01), wij is the one form of the angle change wij = thetaj - thetai
	vertZvals = (z0, z1, z2)
	*/

	std::complex<double> getZvalsFromHalfEdgeOmega(const Eigen::Vector3d& bary, const std::vector<std::complex<double>> vertZvals, const Eigen::Matrix<double, 3, 2>& edgeOmega, Eigen::Matrix<std::complex<double>, 12, 1>* deriv = NULL, Eigen::Matrix<std::complex<double>, 12, 12>* hess = NULL);
	/*
	* compute z values, given the barycentric coordinates
	bary = (alphi0, alpha1, alpha2).
	edgeomega = (w12, w21, 
				 w20, w02, 
				 w01, w10), 
	wij is the one form of the angle change wij = thetaj - thetai
	vertZvals = (z0, z1, z2)
	*/

	void testZvalsFromEdgeOmega(const Eigen::Vector3d& bary, const std::vector<std::complex<double>> vertZvals, const Eigen::Vector3d& edgeOmega);
	void testZvalsFromHalfEdgeOmega(const Eigen::Vector3d& bary, const std::vector<std::complex<double>> vertZvals, const Eigen::Matrix<double, 3, 2>& edgeOmega);

	std::vector<std::complex<double>> upsamplingZvals(const MeshConnectivity& mesh, const std::vector<std::complex<double>>& zvals, const Eigen::VectorXd& w, const std::vector<std::pair<int, Eigen::Vector3d>> &bary);
	std::vector<std::complex<double>> upsamplingZvals(const MeshConnectivity& mesh, const std::vector<std::complex<double>>& zvals, const Eigen::MatrixXd& w, const std::vector<std::pair<int, Eigen::Vector3d>>& bary);

}