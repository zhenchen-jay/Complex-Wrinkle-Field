#pragma once

#include "WrinkleEditingModel.h"

namespace IntrinsicFormula
{
	class WrinkleEditingLocalModel : public WrinkleEditingModel
	{
	public:
		WrinkleEditingLocalModel(const Eigen::MatrixXd& pos, const MeshConnectivity& mesh, const std::vector<VertexOpInfo>& vertexOpts, const Eigen::VectorXi& faceFlag, int quadOrd, double spatialAmpRatio, double spatialEdgeRatio, double spatialKnoppelRatio) :
			WrinkleEditingModel(pos, mesh, vertexOpts, faceFlag, quadOrd, spatialAmpRatio, spatialEdgeRatio, spatialKnoppelRatio)
		{
			int nfaces = _mesh.nFaces();
			for (int i = 0; i < nfaces; i++)
			{
				if (faceFlag(i) == 1)
				{
					_selectedFids.push_back(i);
					_freeFids.push_back(i);
				}
				else if (faceFlag(i) == -1)
				{
					_interfaceFids.push_back(i);
					_freeFids.push_back(i);
				}
				else
					_unselectedFids.push_back(i);
			}
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

			// free dofs
			Eigen::VectorXi freeVertFlags, freeEdgeFlags;
			getVertIdinGivenDomain(_freeFids, freeVertFlags);
			getEdgeIdinGivenDomain(_freeFids, freeEdgeFlags);

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

				if (freeVertFlags(i))
					_freeVids.push_back(i);
			}

			for (int i = 0; i < nedges; i++)
			{
				if (selectedEdgeFlags(i))
					_selectedEids.push_back(i);
				else if (unselectedEdgeFlags(i))
					_unselectedEids.push_back(i);
				else
					_interfaceEids.push_back(i);

				if (freeEdgeFlags(i))
					_freeEids.push_back(i);
			}



			// building the map
			_actualVid2Free.resize(nverts);
			_actualVid2Free.setConstant(-1);

			for (int i = 0; i < _freeVids.size(); i++)
				_actualVid2Free[_freeVids[i]] = i;

			_actualEid2Free.resize(nedges);
			_actualEid2Free.setConstant(-1);

			for (int j = 0; j < _freeEids.size(); j++)
				_actualEid2Free[_freeEids[j]] = j;
		}
		void warmstart();

		virtual void convertVariable2List(const Eigen::VectorXd& x) override;
		virtual void convertList2Variable(Eigen::VectorXd& x) override;


		virtual void getComponentNorm(const Eigen::VectorXd& x, double& znorm, double& wnorm) override
		{
			int nFreeVerts = _freeVids.size();
			int nFreeEdges = _freeEids.size();

			int numFrames = _zvalsList.size() - 2;
			int nDOFs = 2 * nFreeVerts + nFreeEdges;

			znorm = 0;
			wnorm = 0;

			for (int i = 0; i < numFrames; i++)
			{
				for (int j = 0; j < nFreeVerts; j++)
				{
					znorm = std::max(znorm, std::abs(x(i * nDOFs + 2 * j)));
					znorm = std::max(znorm, std::abs(x(i * nDOFs + 2 * j + 1)));
				}
				for (int j = 0; j < nFreeEdges; j++)
				{
					wnorm = std::max(wnorm, std::abs(x(i * nDOFs + 2 * nFreeVerts + j)));
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
		std::vector<int> _unselectedFids;
		std::vector<int> _unselectedVids;
		std::vector<int> _unselectedEids;
		
		std::vector<int> _freeVids;		// the vertices inside the selected and interface faces
		std::vector<int> _freeEids;		// the edges inside the selected and interface faces 
		std::vector<int> _freeFids;		// selected faces + interface faces

		Eigen::VectorXi _actualVid2Free;	// a vector indicates the map from the i-th vertex to its corresponding index in the _freeVids, -1 indicate the fixed vids
		Eigen::VectorXi _actualEid2Free;	// a vector indicates the map from the i-th edge to its corresponding index in the _freeVids, -1 indicate the fixed vids


	};
}