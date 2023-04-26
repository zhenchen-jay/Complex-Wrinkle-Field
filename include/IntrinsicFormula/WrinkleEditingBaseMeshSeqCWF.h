#pragma once

#include "../CommonTools.h"
#include "../LoadSaveIO.h"
#include "ComputeZdotFromEdgeOmega.h"
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <set>

namespace IntrinsicFormula
{
	class WrinkleEditingBaseMeshSeqCWF
	{
	public:
		WrinkleEditingBaseMeshSeqCWF()
		{

		}

		~WrinkleEditingBaseMeshSeqCWF()
		{

		}

		WrinkleEditingBaseMeshSeqCWF(const std::vector<Eigen::MatrixXd>& pos, const MeshConnectivity& mesh, int quadOrd, double spatialAmpRatio, double spatialKnoppelRatio);
		// faceFlag: 0 for unselected faces, 1 for selected faces, -1 for the interface faces

		void adjustOmegaForConsistency(const std::vector<std::complex<double>>& zvals, const Eigen::VectorXd& omega, Eigen::VectorXd& newOmega, Eigen::VectorXd& deltaOmega, Eigen::VectorXi* edgeFlags = NULL);
		void vecFieldLERP(const Eigen::VectorXd& initVec, const Eigen::VectorXd& tarVec, std::vector<Eigen::VectorXd>& vecList, int numFrames, Eigen::VectorXi* edgeFlag = NULL);
		void vecFieldSLERP(const Eigen::VectorXd& initVec, const Eigen::VectorXd& tarVec, std::vector<Eigen::VectorXd>& vecList, int numFrames, Eigen::VectorXi* edgeFlag = NULL);
		void ampFieldLERP(const Eigen::VectorXd& initAmp, const Eigen::VectorXd& tarAmp, std::vector<Eigen::VectorXd>& ampList, int numFrames, Eigen::VectorXi* vertFlag = NULL);
		void initialization(const std::vector<std::complex<double>>& initZvals, const Eigen::VectorXd& initOmega, const std::vector<std::complex<double>>& tarZvals, const Eigen::VectorXd& tarOmega, bool applyAdj = true);
		
		void initialization(const std::vector<std::vector<std::complex<double>>>& zList, const std::vector<Eigen::VectorXd>& omegaList, const std::vector<Eigen::VectorXd>& refAmpList, const std::vector<Eigen::VectorXd>& refOmegaList, bool applyAdj = true);

		Eigen::VectorXd ampTimeOmegaSqInitialization(const Eigen::VectorXd& initAmpOmegaSq, const Eigen::VectorXd& tarAmpOmegaSq, double t);
		Eigen::VectorXd ampInitialization(const Eigen::VectorXd& initAmp, const Eigen::VectorXd& tarAmp, double t);
		Eigen::VectorXd omegaInitialization(const Eigen::VectorXd& initOmega, const Eigen::VectorXd& tarOmega, const Eigen::VectorXd& initAmp, const Eigen::VectorXd& tarAmp, int frameId);
	
		void solveIntermeditateFrames(Eigen::VectorXd& x, int numIter, double gradTol = 1e-6, double xTol = 0, double fTol = 0, bool isdisplayInfo = false, std::string workingFolder = "");


		std::vector<Eigen::VectorXd> getWList()
		{
			std::vector<Eigen::VectorXd> finalWList = _edgeOmegaList;		// _edgeOmega is the one after adjust
			for(int i = 0;  i < _edgeOmegaList.size(); i++)
				finalWList[i] += _deltaOmegaList[i];
			return finalWList;
		}
		std::vector<std::vector<std::complex<double>>> getVertValsList()
		{
			for (int i = 1; i < _zvalsList.size() - 1; i++)
			{
				for (int j = 0; j < _zvalsList[i].size(); j++)
				{
					_zvalsList[i][j] = _unitZvalsList[i][j] * _combinedRefAmpList[i][j];
				}
			}
			return _zvalsList;
		}

		std::vector<Eigen::VectorXd> getRefWList() { return _combinedRefOmegaList; }
		std::vector<Eigen::VectorXd> getRefAmpList() { return _combinedRefAmpList; }

		std::vector<Eigen::VectorXd> getDeltaWList() {return _deltaOmegaList;}
		std::vector<Eigen::VectorXd> getActualOptWList() { return _edgeOmegaList; }


		std::vector<Eigen::VectorXd> getF2List() {return _ampTimesOmegaSq;}


		void setSaveFolder(const std::string& savingFolder)
		{
			_savingFolder = savingFolder;
		}

		void setwzLists(std::vector<std::vector<std::complex<double>>>& zList, std::vector<Eigen::VectorXd>& wList)
		{
			_zvalsList = zList;
			_edgeOmegaList = wList;

			_deltaOmegaList.resize(_edgeOmegaList.size(), Eigen::VectorXd::Zero(_mesh.nEdges()));
		}

		void save(const Eigen::VectorXd& x0, std::string* workingFold = NULL);
		void convertVariable2List(const Eigen::VectorXd& x);
		void convertList2Variable(Eigen::VectorXd& x);
		void getComponentNorm(const Eigen::VectorXd& x, double& znorm, double& wnorm);
		double computeEnergy(const Eigen::VectorXd& x, Eigen::VectorXd* deriv = NULL, Eigen::SparseMatrix<double>* hess = NULL, bool isProj = false);

		// spatial-temporal energies
		double temporalAmpDifference(int frameId, Eigen::VectorXd* deriv = NULL, std::vector<Eigen::Triplet<double>>* hessT = NULL, bool isProj = false);
		double spatialKnoppelEnergy(int frameId, Eigen::VectorXd* deriv = NULL, std::vector<Eigen::Triplet<double>>* hessT = NULL, bool isProj = false);
		double kineticEnergy(int frameId, Eigen::VectorXd* deriv = NULL, std::vector<Eigen::Triplet<double>>* hessT = NULL, bool isProj = false);

		void computeAmpOmegaSq(const Eigen::MatrixXd& pos, const MeshConnectivity& mesh, const Eigen::VectorXd& amp, const Eigen::VectorXd& omega, Eigen::VectorXd& ampOmegaSq);
		void computeAmpSqOmegaQuaticAverage();

	private:
		std::vector<Eigen::MatrixXd> _posList;
		MeshConnectivity _mesh;

		std::vector<Eigen::VectorXd> _combinedRefAmpList;
		std::vector<Eigen::VectorXd> _combinedRefOmegaList;

		std::vector<Eigen::VectorXd> _edgeOmegaList;
		std::vector<Eigen::VectorXd> _deltaOmegaList;
		std::vector<std::vector<std::complex<double>>> _zvalsList;
		std::vector<std::vector<std::complex<double>>> _unitZvalsList;  // the optimization variables for new formula

		int _quadOrd;
		std::vector<std::vector<int>> _vertNeiFaces;
		std::vector<std::vector<int>> _vertNeiEdges;

		std::vector<Eigen::VectorXd> _faceAreaList;
		std::vector<Eigen::VectorXd> _vertAreaList;
		std::vector<Eigen::VectorXd> _edgeAreaList;
		std::vector<Eigen::VectorXd> _edgeCotCoeffsList;


		std::vector<std::vector<std::vector<Eigen::Matrix2d>>> _faceVertMetricsList;
		double _spatialAmpRatio;
		double _spatialKnoppelRatio;
		std::string _savingFolder;


		std::vector<Eigen::VectorXd> _ampTimesOmegaSq;
		std::vector<Eigen::VectorXd> _ampTimesDeltaOmegaSq;
		Eigen::VectorXd _ampSqOmegaQauticAverageList;
		double _ampSqOmegaQuaticAverage;
	};
}