#pragma once
#include <Eigen/Dense>
#include "../MeshLib/MeshConnectivity.h"

/*
 * In this file, I implemented the algorithm mentioned by [Zuenko et al] "Wrinkles, Folds, Creases, Buckles: Small-Scale Surface Deformations as Periodic Functions on 3D Meshes"
 */

namespace ZuenkoAlg
{
    void spherigonSmoothing(const Eigen::MatrixXd& pos, const MeshConnectivity& mesh, const Eigen::MatrixXd& vertN, const std::vector<std::pair<int, Eigen::Vector3d>>& bary, Eigen::MatrixXd& upPos, Eigen::MatrixXd& upN, bool isG1);
    // The smoothing mentioed in the paper: The SPHERIGON: A Simple Polygon Patch for Smoothing Quickly your Polygonal Meshes.
    // for more clear G1 formula, please refer page 31 in the paper: The Construction of Optimized High-Order Surface Meshes by Energy-Minimizatio

    void spherigonSmoothingSequentially(const Eigen::MatrixXd& pos, const MeshConnectivity& mesh, const Eigen::MatrixXd& vertN, Eigen::MatrixXd& upPos, MeshConnectivity& upMesh, Eigen::MatrixXd& upN, int numSubdiv, bool isG1);
    // This is the sequential version of the spherigon approach, where we upsample once, then update vertex normal. This will produce kinda better results. But it is NOT the one implemented in the original paper.

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
		int numSubdivs = 0, bool isG1 = false, double ampScaling = 1.0,
		int innerIter = 5, double blurCoeff = 0.1);
	// get the "simulated" wrinkled mesh, according to the algorithm mentioned in Sec. 5.1 

}