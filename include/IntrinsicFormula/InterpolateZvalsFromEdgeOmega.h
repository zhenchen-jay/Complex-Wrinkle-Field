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

	void testZvalsFromEdgeOmega(const Eigen::Vector3d& bary, const std::vector<std::complex<double>> vertZvals, const Eigen::Vector3d& edgeOmega);

	std::vector<std::complex<double>> upsamplingZvals(const MeshConnectivity& mesh, const std::vector<std::complex<double>>& zvals, const Eigen::VectorXd& w, const std::vector<std::pair<int, Eigen::Vector3d>> &bary);

}