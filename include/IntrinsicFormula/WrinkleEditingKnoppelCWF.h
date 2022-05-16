#pragma once

#include "WrinkleEditingModel.h"
#include <igl/heat_geodesics.h>
#include <igl/avg_edge_length.h>
#include <iostream>

namespace IntrinsicFormula
{
	class WrinkleEditingKnoppelCWF : public WrinkleEditingModel
	{
	public:
        WrinkleEditingKnoppelCWF(const Eigen::MatrixXd& pos, const MeshConnectivity& mesh, const std::vector<VertexOpInfo>& vertexOpts, const Eigen::VectorXi& faceFlag, int quadOrd, double spatialAmpRatio, double spatialEdgeRatio, double spatialKnoppelRatio, int effectivedistFactor) :
			WrinkleEditingModel(pos, mesh, vertexOpts, faceFlag, quadOrd, spatialAmpRatio, spatialEdgeRatio, spatialKnoppelRatio, effectivedistFactor)
		{}
        virtual void solveIntermeditateFrames(Eigen::VectorXd& x, int numIter, double gradTol = 1e-6, double xTol = 0, double fTol = 0, bool isdisplayInfo = false, std::string workingFolder = "") override;
		virtual void convertVariable2List(const Eigen::VectorXd& x) override;
		virtual void convertList2Variable(Eigen::VectorXd& x) override;


        virtual void save(const Eigen::VectorXd& x0, std::string* workingFolder) override
        {
            convertVariable2List(x0);
            std::string tmpFolder;
            if(workingFolder)
                tmpFolder = (*workingFolder) + "/tmpRes/";
            else
                tmpFolder = _savingFolder + "tmpRes/";
            mkdir(tmpFolder);

            std::string outputFolder = tmpFolder + "/optZvals/";
            mkdir(outputFolder);

            std::string omegaOutputFolder = tmpFolder + "/optOmega/";
            mkdir(omegaOutputFolder);

            std::string refOmegaOutputFolder = tmpFolder + "/refOmega/";
            mkdir(refOmegaOutputFolder);

            // save reference
            std::string refAmpOutputFolder = tmpFolder + "/refAmp/";
            mkdir(refAmpOutputFolder);

            int nframes = _zvalsList.size();
            auto savePerFrame = [&](const tbb::blocked_range<uint32_t>& range)
            {
                for (uint32_t i = range.begin(); i < range.end(); ++i)
                {

                    saveVertexZvals(outputFolder + "zvals_" + std::to_string(i) + ".txt", _zvalsList[i]);
                    saveEdgeOmega(omegaOutputFolder + "omega_" + std::to_string(i) + ".txt", _edgeOmegaList[i]);
                    saveVertexAmp(refAmpOutputFolder + "amp_" + std::to_string(i) + ".txt", _combinedRefAmpList[i]);
                    saveEdgeOmega(refOmegaOutputFolder + "omega_" + std::to_string(i) + ".txt", _combinedRefOmegaList[i]);
                }
            };

            tbb::blocked_range<uint32_t> rangex(0u, (uint32_t)nframes, GRAIN_SIZE);
            tbb::parallel_for(rangex, savePerFrame);
        }

		virtual void getComponentNorm(const Eigen::VectorXd& x, double& znorm, double& wnorm) override
		{
			int nverts = _pos.rows();
			int nedges = _mesh.nEdges();

			int numFrames = _zvalsList.size() - 2;
			int nDOFs = 2 * nverts + nedges;

			znorm = 0;
			wnorm = 0;

			for (int i = 0; i < numFrames; i++)
			{
				for (int j = 0; j < nverts; j++)
				{
					znorm = std::max(znorm, std::abs(x(i * nDOFs + 2 * j)));
					znorm = std::max(znorm, std::abs(x(i * nDOFs + 2 * j + 1)));
				}
				for (int j = 0; j < nedges; j++)
				{
					wnorm = std::max(wnorm, std::abs(x(i * nDOFs + 2 * nverts + j)));
				}
			}
		}

        virtual double computeEnergy(const Eigen::VectorXd& x, Eigen::VectorXd* deriv = NULL, Eigen::SparseMatrix<double>* hess = NULL, bool isProj = false) override
        {
            return 0;
        }

		// spatial-temporal energies
        virtual double temporalAmpDifference(int frameId, Eigen::VectorXd* deriv = NULL, std::vector<Eigen::Triplet<double>>* hessT = NULL, bool isProj = false) override
        {
            return 0;
        }
        virtual double temporalOmegaDifference(int frameId, Eigen::VectorXd* deriv = NULL, std::vector<Eigen::Triplet<double>>* hessT = NULL, bool isProj = false) override
        {
            return 0;
        }
        virtual double spatialKnoppelEnergy(int frameId, Eigen::VectorXd* deriv = NULL, std::vector<Eigen::Triplet<double>>* hessT = NULL, bool isProj = false) override
        {
            return 0;
        }
        virtual double kineticEnergy(int frameId, Eigen::VectorXd* deriv = NULL, std::vector<Eigen::Triplet<double>>* hessT = NULL, bool isProj = false) override
        {
            return 0;
        }

	};
}