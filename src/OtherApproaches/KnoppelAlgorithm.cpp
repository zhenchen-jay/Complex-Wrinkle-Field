#include "../../include/OtherApproaches/KnoppelAlgorithm.h"
#include "../../include/IntrinsicFormula/KnoppelStripePatternEdgeOmega.h"
#include "../../include/MeshLib/MeshUpsampling.h"
#include <tbb/tbb.h>
#include <igl/per_vertex_normals.h>

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

    void getKnoppelWrinkledMeshPerframe(
            const Eigen::MatrixXd& baseV, const MeshConnectivity& baseMesh,
            const Eigen::VectorXd& omega, const Eigen::VectorXd& edgeWeight,
            const Eigen::VectorXd& amp, // this is tricky, we just linear upsample amp
            const Eigen::VectorXd& vertArea,
            const std::vector<std::pair<int, Eigen::Vector3d>>& bary,
            Eigen::VectorXd& upsampledAmp,
            Eigen::VectorXd& upsampledPhi)
    {
        std::vector<std::complex<double>> zvals;
        IntrinsicFormula::roundZvalsFromEdgeOmega(baseMesh, omega, edgeWeight, vertArea, baseV.rows(), zvals);
        IntrinsicFormula::getUpsamplingThetaFromEdgeOmega(baseMesh, omega, zvals, bary, upsampledPhi);

        int nupverts = bary.size();
        upsampledAmp.resize(nupverts);
        for(int i = 0; i < nupverts; i++)
        {
            int fid = bary[i].first;
            upsampledAmp[i] = 0;
            for(int j = 0; j < 3; j++)
            {
                int vid = baseMesh.faceVertex(fid, j);
                upsampledAmp[i] += bary[i].second[j] * amp[vid];
            }
        }
    }

    void getKnoppelWrinkledMeshSequence(
            const Eigen::MatrixXd& baseV, const MeshConnectivity& baseMesh,
            const std::vector<Eigen::VectorXd>& omegaList,
            const std::vector<Eigen::VectorXd>& ampList,
            Eigen::MatrixXd& upsampledV, Eigen::MatrixXi& upsampledF,
            std::vector<Eigen::VectorXd>& upsampledAmpList,
            std::vector<Eigen::VectorXd>& upsampledPhiList,
            std::vector<Eigen::MatrixXd>& wrinkledVList,
            std::vector<Eigen::MatrixXi>& wrinkledFList,
            double wrinkleAmpRatio,
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
        Eigen::MatrixXd upsampledN;
        igl::per_vertex_normals(upsampledV, upsampledF, upsampledN);

        int nframes = omegaList.size();

        upsampledPhiList.resize(nframes);
        upsampledAmpList.resize(nframes);
        wrinkledFList.resize(nframes);
        wrinkledVList.resize(nframes);

       /* for (uint32_t i = 0; i < nframes; ++i)
        {
            getKnoppelWrinkledMeshPerframe(baseV, baseMesh, omegaList[i], edgeWeight, ampList[i], vertArea, bary, upsampledAmpList[i], upsampledPhiList[i]);
            wrinkledVList[i] = upsampledV;
            wrinkledFList[i] = upsampledF;
            for (int j = 0; j < upsampledV.rows(); j++)
            {
                wrinkledVList[i].row(j) += wrinkleAmpRatio * upsampledAmpList[i][j] * std::cos(upsampledPhiList[i][j]) * upsampledN.row(j);
            }
        }*/

        auto frameUpsampling = [&](const tbb::blocked_range<uint32_t>& range)
        {
            for (uint32_t i = range.begin(); i < range.end(); ++i)
            {
                getKnoppelWrinkledMeshPerframe(baseV, baseMesh, omegaList[i], edgeWeight, ampList[i], vertArea, bary, upsampledAmpList[i], upsampledPhiList[i]);
                wrinkledVList[i] = upsampledV;
                wrinkledFList[i] = upsampledF;
                for(int j = 0; j < upsampledV.rows(); j++)
                {
                    wrinkledVList[i].row(j) += wrinkleAmpRatio * upsampledAmpList[i][j] * std::cos(upsampledPhiList[i][j]) * upsampledN.row(j);
                }
            }
        };

        tbb::blocked_range<uint32_t> rangex(0u, (uint32_t)nframes);
        tbb::parallel_for(rangex, frameUpsampling);
    }
}