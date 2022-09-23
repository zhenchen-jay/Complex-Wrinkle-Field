#pragma once
#include <Eigen/Dense>
#include "../MeshLib/MeshConnectivity.h"

/*
 * In this file, I implemented the algorithm mentioned by [Zuenko et al] "Wrinkles, Folds, Creases, Buckles: Small-Scale Surface Deformations as Periodic Functions on 3D Meshes"
 */

namespace ZuenkoAlg
{
	void spherigonSmoothing(const Eigen::MatrixXd& pos, const MeshConnectivity& mesh, const Eigen::MatrixXd& vertN, const std::vector<std::pair<int, Eigen::Vector3d>>& bary, Eigen::MatrixXd& upPos, Eigen::MatrixXd& upN);
	// The smoothing mentioed in the paper: The SPHERIGON: A Simple Polygon Patch for Smoothing Quickly your Polygonal Meshes.
	
	void getZuenkoSurfacePerframe(
		const Eigen::MatrixXd& baseV, const MeshConnectivity& baseMesh,
		const std::vector<std::complex<double>>& unitZvals, const Eigen::VectorXd& vertAmp, const Eigen::VectorXd& edgeOmega,
		const Eigen::MatrixXd& upsampledV, const Eigen::MatrixXi& upsampledF, const Eigen::MatrixXd& upsampledN,
		const std::vector<std::pair<int, Eigen::Vector3d>>& bary,
		Eigen::MatrixXd& wrinkledV, Eigen::MatrixXi& wrinkledF,
		Eigen::VectorXd& upsampledAmp, Eigen::VectorXd& upsampledPhi, double ampScaling = 1.0);
	// upsampling phase w.r.t. sec. 5.2 in Zuenko's formula, and linear interpolating amplitude
	
	void getZuenkoSurfaceSequence(
		const Eigen::MatrixXd& baseV, const MeshConnectivity& baseMesh,
		const std::vector<std::complex<double>>& initZvals,
		const std::vector<Eigen::VectorXd>& ampList, const std::vector<Eigen::VectorXd>& omegaList,
		Eigen::MatrixXd& upsampledV, Eigen::MatrixXi& upsampledF,
		std::vector<Eigen::MatrixXd>& wrinkledVList, std::vector<Eigen::MatrixXi>& wrinkledFList,
		std::vector<Eigen::VectorXd>& upsampledAmpList, std::vector<Eigen::VectorXd>& upsampledPhiList,
		int numSubdivs = 0, double ampScaling = 1.0,
		int innerIter = 5, double blurCoeff = 0.1);
	// get the "simulated" wrinkled mesh, according to the algorithm mentioned in Sec. 5.1 

}