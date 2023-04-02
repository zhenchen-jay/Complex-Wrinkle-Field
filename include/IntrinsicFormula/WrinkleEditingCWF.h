#pragma once

#include "WrinkleEditingModel.h"
#include <igl/heat_geodesics.h>
#include <igl/avg_edge_length.h>
#include <iostream>

namespace IntrinsicFormula
{
    // this is actually the one we used to generate the paper results, where we manaully blend to get omega (w_blend + dw_blend), and ask the w-z consistency for (z, w_blend), and our variables are the vertex zvals
	class WrinkleEditingCWF : public WrinkleEditingModel
	{
	public:
        WrinkleEditingCWF(const Eigen::MatrixXd& pos, const MeshConnectivity& mesh, const std::vector<VertexOpInfo>& vertexOpts, const Eigen::VectorXi& faceFlag, int quadOrd, double spatialAmpRatio, double spatialEdgeRatio, double spatialKnoppelRatio, int effectivedistFactor) :
			WrinkleEditingModel(pos, mesh, vertexOpts, faceFlag, quadOrd, spatialAmpRatio, spatialEdgeRatio, spatialKnoppelRatio, effectivedistFactor)
		{
			int nverts = _pos.rows();
			int nfaces = _mesh.nFaces();

			_faceWeight.setZero(nfaces);
			_vertWeight.setZero(nverts);

            if(_selectedVids.size() && effectivedistFactor > 0)
            {

                // selected edges and verts
                Eigen::VectorXi selectedEdgeFlags, selectedVertFlags;
                getVertIdinGivenDomain(_selectedFids, selectedVertFlags);
                getEdgeIdinGivenDomain(_selectedFids, selectedEdgeFlags);

                // unselected edges and verts
                Eigen::VectorXi unselectedEdgeFlags, unselectedVertFlags;
                getVertIdinGivenDomain(_unselectedFids, unselectedVertFlags);
                getEdgeIdinGivenDomain(_unselectedFids, unselectedEdgeFlags);

                // interface edges and verts
                Eigen::VectorXi interfaceEdgeFlags, interfaceVertFlags;
                getVertIdinGivenDomain(_interfaceFids, interfaceVertFlags);
                getEdgeIdinGivenDomain(_interfaceFids, interfaceEdgeFlags);

                // build the list
                int nverts = _pos.rows();
                int nedges = _mesh.nEdges();

                for (int i = 0; i < nverts; i++)
                {
                    if (selectedVertFlags(i))
                        _selectedVids.push_back(i);
                    else if (unselectedVertFlags(i))
                        _unselectedVids.push_back(i);
                    else
                        _interfaceVids.push_back(i);
                }

                for (int i = 0; i < nedges; i++)
                {
                    if (selectedEdgeFlags(i))
                        _selectedEids.push_back(i);
                    else if (unselectedEdgeFlags(i))
                        _unselectedEids.push_back(i);
                    else
                        _interfaceEids.push_back(i);
                }

                std::vector<int> sourceVerts;
                for (int i = 0; i < nverts; i++)
                {
                    if (selectedVertFlags(i))
                        sourceVerts.push_back(i);
                    else if (interfaceVertFlags(i))
                        sourceVerts.push_back(i);
                }
                if(sourceVerts.size() == nverts)
                {
                    _faceWeight.setConstant(1.0);
                    _vertWeight.setConstant(1.0);
                }
                else
                {
                    // build geodesics
                    // Precomputation
                    igl::HeatGeodesicsData<double> data;
                    double t = std::pow(igl::avg_edge_length(_pos, _mesh.faces()), 2);
                    const auto precompute = [&]()
                    {
                        if (!igl::heat_geodesics_precompute(_pos, _mesh.faces(), t, data))
                        {
                            std::cerr << "Error: heat_geodesics_precompute failed." << std::endl;
                            exit(EXIT_FAILURE);
                        };
                    };
                    precompute();
                    
                    Eigen::VectorXi gamma(sourceVerts.size());
                    for (int i = 0; i < gamma.rows(); i++)
                        gamma(i) = sourceVerts[i];

                    Eigen::VectorXd dis;
                    igl::heat_geodesics_solve(data, gamma, dis);

                    for (int i = 0; i < nverts; i++)
                    {
                        if (selectedVertFlags(i) || interfaceVertFlags(i))
                            dis(i) = 0;
                    }

                    double min = dis.minCoeff();
                    double max = dis.maxCoeff();

                    std::cout << "min geo: " << min << ", max geo: " << max << std::endl;

                    double mu = 0;
                    double sigma = (max - min) / (2 * effectivedistFactor);

                    for (int i = 0; i < nfaces; i++)
                    {
                        for (int j = 0; j < 3; j++)
                        {
                            int vid = _mesh.faceVertex(i, j);
                            double weight = std::min(expGrowth(dis(vid), mu, sigma), 1e10);
                            _vertWeight(vid) = weight;
                            _faceWeight(i) += weight / 3;
                        }
                    }
                    std::cout << "face weight min: " << _faceWeight.minCoeff() << ", face weight max: " << _faceWeight.maxCoeff() << std::endl;
                }

            }
            else
            {
                _faceWeight.setConstant(1.0);
                _vertWeight.setConstant(1.0);
            }


		}
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
                    saveVertexZvals(outputFolder + "unitZvals_" + std::to_string(i) + ".txt", _unitZvalsList[i]);
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
			
			int numFrames = _unitZvalsList.size() - 2;
			int nDOFs = 2 * nverts;

			znorm = 0;
			wnorm = 0;

			for (int i = 0; i < numFrames; i++)
			{
				for (int j = 0; j < nverts; j++)
                {
                    znorm = std::max(znorm, std::abs(x(i * nDOFs + 2 * j)));
                    znorm = std::max(znorm, std::abs(x(i * nDOFs + 2 * j + 1)));
                }
			}
		}

        virtual std::vector<std::vector<std::complex<double>>> getVertValsList() override
        {
            for(int i = 1; i < _zvalsList.size() - 1; i++)
            {
                for(int j = 0; j < _zvalsList[i].size(); j++)
                {
                    _zvalsList[i][j] = _unitZvalsList[i][j] * _combinedRefAmpList[i][j];
                }
            }
            return _zvalsList;
        }

		virtual double computeEnergy(const Eigen::VectorXd& x, Eigen::VectorXd* deriv = NULL, Eigen::SparseMatrix<double>* hess = NULL, bool isProj = false) override;


		// spatial-temporal energies
        virtual double temporalAmpDifference(int frameId, Eigen::VectorXd* deriv = NULL, std::vector<Eigen::Triplet<double>>* hessT = NULL, bool isProj = false) override;
        virtual double temporalOmegaDifference(int frameId, Eigen::VectorXd* deriv = NULL, std::vector<Eigen::Triplet<double>>* hessT = NULL, bool isProj = false) override { return 0; }
		virtual double spatialKnoppelEnergy(int frameId, Eigen::VectorXd* deriv = NULL, std::vector<Eigen::Triplet<double>>* hessT = NULL, bool isProj = false) override;
		virtual double kineticEnergy(int frameId, Eigen::VectorXd* deriv = NULL, std::vector<Eigen::Triplet<double>>* hessT = NULL, bool isProj = false) override;

        double kineticEnergyWithoutFSq(int frameId, Eigen::VectorXd* deriv = NULL, std::vector<Eigen::Triplet<double>>* hessT = NULL, bool isProj = false);

        void computeAmpSqOmegaQuaticAverage();

	private:
		double expGrowth(double x, double mu, double sigma)		// f = exp((x-mu)^2 / sigma^2)
		{
			return std::exp((x - mu) * (x - mu) / sigma / sigma);
		}
        
        double _ampSqOmegaQuaticAverage;
        std::vector<double> _ampSqOmegaQauticAverageList;

	};
}