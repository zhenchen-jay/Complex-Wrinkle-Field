#pragma once

#include "WrinkleEditingModel.h"
#include <igl/heat_geodesics.h>
#include <igl/avg_edge_length.h>
#include <iostream>

namespace IntrinsicFormula
{
	class WrinkleEditingGlobalModel : public WrinkleEditingModel
	{
	public:
		WrinkleEditingGlobalModel(const Eigen::MatrixXd& pos, const MeshConnectivity& mesh, const std::vector<VertexOpInfo>& vertexOpts, const Eigen::VectorXi& faceFlag, int quadOrd, double spatialAmpRatio, double spatialEdgeRatio, double spatialKnoppelRatio) :
			WrinkleEditingModel(pos, mesh, vertexOpts, faceFlag, quadOrd, spatialAmpRatio, spatialEdgeRatio, spatialKnoppelRatio)
		{
			int nverts = _pos.rows();
			int nfaces = _mesh.nFaces();

			_faceWeight.setZero(nfaces);
			_vertWeight.setZero(nverts);

            if(_selectedVids.size())
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

                Eigen::VectorXi selectedEdgeFlags, selectedVertFlags;
                getVertIdinGivenDomain(_selectedFids, selectedVertFlags);
                getEdgeIdinGivenDomain(_selectedFids, selectedEdgeFlags);

                // interface edges and verts
                Eigen::VectorXi interfaceEdgeFlags, interfaceVertFlags;
                getVertIdinGivenDomain(_interfaceFids, interfaceVertFlags);
                getEdgeIdinGivenDomain(_interfaceFids, interfaceEdgeFlags);

                std::vector<int> sourceVerts;
                for (int i = 0; i < nverts; i++)
                {
                    if (selectedVertFlags(i))
                        sourceVerts.push_back(i);
                    else if (interfaceVertFlags(i))
                        sourceVerts.push_back(i);
                }
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
                double sigma = (max - min) / 4;

                for (int i = 0; i < nfaces; i++)
                {
                    for (int j = 0; j < 3; j++)
                    {
                        int vid = _mesh.faceVertex(i, j);
                        double weight = expGrowth(dis(vid), mu, sigma);
                        _vertWeight(vid) = weight;
                        _faceWeight(i) += weight / 3;
                    }
                }
                std::cout << "face weight min: " << _faceWeight.minCoeff() << ", face weight max: " << _faceWeight.maxCoeff() << std::endl;
            }
            else
                _faceWeight.setConstant(1.0);

		}
		void warmstart();

		virtual void convertVariable2List(const Eigen::VectorXd& x) override;
		virtual void convertList2Variable(Eigen::VectorXd& x) override;


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

		virtual double computeEnergy(const Eigen::VectorXd& x, Eigen::VectorXd* deriv = NULL, Eigen::SparseMatrix<double>* hess = NULL, bool isProj = false) override;

	protected:
		// spatial-temporal energies
		virtual double temporalAmpDifference(int frameId, Eigen::VectorXd* deriv = NULL, std::vector<Eigen::Triplet<double>>* hessT = NULL, bool isProj = false) override;
		virtual double temporalOmegaDifference(int frameId, Eigen::VectorXd* deriv = NULL, std::vector<Eigen::Triplet<double>>* hessT = NULL, bool isProj = false) override;
		virtual double spatialKnoppelEnergy(int frameId, Eigen::VectorXd* deriv = NULL, std::vector<Eigen::Triplet<double>>* hessT = NULL, bool isProj = false) override;
		virtual double kineticEnergy(int frameId, Eigen::VectorXd* deriv = NULL, std::vector<Eigen::Triplet<double>>* hessT = NULL, bool isProj = false) override;
		virtual double naiveKineticEnergy(int frameId, Eigen::VectorXd* deriv = NULL, std::vector<Eigen::Triplet<double>>* hessT = NULL, bool isProj = false) override;

	private:
		double expGrowth(double x, double mu, double sigma)		// f = exp((x-mu)^2 / sigma^2)
		{
			return std::exp((x - mu) * (x - mu) / sigma / sigma);
		}


	private:
		Eigen::VectorXd _faceWeight;
		Eigen::VectorXd _vertWeight;

	};
}