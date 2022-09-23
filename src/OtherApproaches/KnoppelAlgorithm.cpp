#include "../../include/OtherApproaches/KnoppelAlgorithm.h"
#include "../../include/IntrinsicFormula/KnoppelStripePatternEdgeOmega.h"
#include "../../include/MeshLib/MeshUpsampling.h"
#include <tbb/tbb.h>

namespace KnoppelAlg
{
	void getKnoppelPhasePerframe(
		const Eigen::MatrixXd& baseV, const MeshConnectivity& baseMesh,
		const Eigen::VectorXd& omega, const Eigen::VectorXd& edgeWeight,
		const Eigen::VectorXd& vertArea,
		const std::vector<std::pair<int, Eigen::Vector3d>>& bary,
		Eigen::VectorXd& upsampledPhi)
	{
		std::vector<std::complex<double>> zvals;
		IntrinsicFormula::roundZvalsFromEdgeOmega(baseMesh, omega, edgeWeight, vertArea, baseV.rows(), zvals);
		IntrinsicFormula::getUpsamplingThetaFromEdgeOmega(baseMesh, omega, zvals, bary, upsampledPhi);
	}

	void getKnoppelPhaseSequence(
		const Eigen::MatrixXd& baseV, const MeshConnectivity& baseMesh,
		const std::vector<Eigen::VectorXd>& omegaList,
		Eigen::MatrixXd& upsampledV, Eigen::MatrixXi& upsampledF,
		std::vector<Eigen::VectorXd>& upsampledPhiList,
		int numSubdivs)
	{
		Eigen::VectorXd vertArea = getVertArea(baseV, baseMesh);
		Eigen::VectorXd faceArea = getFaceArea(baseV, baseMesh);


		Eigen::VectorXd edgeWeight(baseMesh.nEdges());
		for (int i = 0; i < baseMesh.nEdges(); i++)
		{
			edgeWeight[i] = 0;
			for (int j = 0; j < 2; j++)
			{
				int fid = baseMesh.edgeFace(i, j);
				if (fid != -1)
				{
					edgeWeight[i] += faceArea[fid] / 3.0;
				}
			}
		}
		
		std::vector<std::pair<int, Eigen::Vector3d>> bary;
		meshUpSampling(baseV, baseMesh.faces(), upsampledV, upsampledF, numSubdivs, NULL, NULL, &bary);

		int nframes = omegaList.size();

		upsampledPhiList.resize(nframes);

		auto frameUpsampling = [&](const tbb::blocked_range<uint32_t>& range)
		{
			for (uint32_t i = range.begin(); i < range.end(); ++i)
			{
				getKnoppelPhasePerframe(baseV, baseMesh, omegaList[i], edgeWeight, vertArea, bary, upsampledPhiList[i]);
			}
		};

		tbb::blocked_range<uint32_t> rangex(0u, (uint32_t)nframes);
		tbb::parallel_for(rangex, frameUpsampling);

	}
}