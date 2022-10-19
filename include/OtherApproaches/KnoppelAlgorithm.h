#pragma once
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "../MeshLib/MeshConnectivity.h"

namespace KnoppelAlg
{
	void getKnoppelPhasePerframe(
		const Eigen::MatrixXd& baseV, const MeshConnectivity& baseMesh,
		const Eigen::VectorXd& omega, const Eigen::VectorXd& edgeWeight,
		const Eigen::VectorXd& vertArea, 
		const std::vector<std::pair<int, Eigen::Vector3d>>& bary, 
		Eigen::VectorXd& upsampledPhi);

	void getKnoppelPhaseSequence(
		const Eigen::MatrixXd& baseV, const MeshConnectivity& baseMesh,
		const std::vector<Eigen::VectorXd>& omegaList,
		Eigen::MatrixXd& upsampledV, Eigen::MatrixXi& upsampledF,
		std::vector<Eigen::VectorXd>& upsampledPhiList,
		int numSubdivs = 0);

    void getKnoppelWrinkledMeshPerframe(
            const Eigen::MatrixXd& baseV, const MeshConnectivity& baseMesh,
            const Eigen::VectorXd& omega, const Eigen::VectorXd& edgeWeight,
            const Eigen::VectorXd& amp, // this is tricky, we just linear upsample amp
            const Eigen::VectorXd& vertArea,
            const std::vector<std::pair<int, Eigen::Vector3d>>& bary,
            Eigen::VectorXd& upsampledAmp,
            Eigen::VectorXd& upsampledPhi);

    void getKnoppelWrinkledMeshSequence(
            const Eigen::MatrixXd& baseV, const MeshConnectivity& baseMesh,
            const std::vector<Eigen::VectorXd>& omegaList,
            const std::vector<Eigen::VectorXd>& ampList,
            Eigen::MatrixXd& upsampledV, Eigen::MatrixXi& upsampledF,
            std::vector<Eigen::VectorXd>& upsampledAmpList,
            std::vector<Eigen::VectorXd>& upsampledPhiList,
            std::vector<Eigen::MatrixXd>& wrinkledVList,
            std::vector<Eigen::MatrixXi>& wrinkledFList,
            double wrinkleAmpRatio = 1,
            int numSubdivs = 0);
}