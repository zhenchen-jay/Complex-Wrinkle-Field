#pragma once

#include "../CommonTools.h"
#include "../LoadSaveIO.h"
#include "ComputeZdotFromEdgeOmega.h"
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <set>

namespace IntrinsicFormula
{
	class WrinkleEditingModel
	{
	public:
		WrinkleEditingModel()
		{}
		virtual ~WrinkleEditingModel() = default;

		WrinkleEditingModel(const Eigen::MatrixXd& pos, const MeshConnectivity& mesh, const std::vector<VertexOpInfo>& vertexOpts, const Eigen::VectorXi& faceFlag, int quadOrd, double spatialAmpRatio, double spatialEdgeRatio, double spatialKnoppelRatio, int effectivedistFactor);
		// faceFlag: 0 for unselected faces, 1 for selected faces, -1 for the interface faces

		void faceFlagsSetup(const Eigen::VectorXi& faceFlags);

        void adjustOmegaForConsistency(const std::vector<std::complex<double>>& zvals, const Eigen::VectorXd& omega, Eigen::VectorXd& newOmega, Eigen::VectorXd& deltaOmega, Eigen::VectorXi* edgeFlags = NULL);
		void vecFieldSLERP(const Eigen::VectorXd& initVec, const Eigen::VectorXd& tarVec, std::vector<Eigen::VectorXd>& vecList, int numFrames, Eigen::VectorXi* edgeFlag = NULL);
        void vecFieldLERP(const Eigen::VectorXd& initVec, const Eigen::VectorXd& tarVec, std::vector<Eigen::VectorXd>& vecList, int numFrames, Eigen::VectorXi* edgeFlag = NULL);
		void ampFieldLERP(const Eigen::VectorXd& initAmp, const Eigen::VectorXd& tarAmp, std::vector<Eigen::VectorXd>& ampList, int numFrames, Eigen::VectorXi* vertFlag = NULL);

        void editCWFBasedOnVertOp(const std::vector<std::complex<double>>& initZvals, const Eigen::VectorXd& initOmega, std::vector<std::complex<double>>& editZvals, Eigen::VectorXd& editOmga);

        void initialization(const std::vector<std::complex<double>>& initZvals, const Eigen::VectorXd& initOmega, const std::vector<std::complex<double>>& tarZvals, const Eigen::VectorXd& tarOmega, int numFrames, bool applyAdj = true);
		void initialization(const std::vector<std::vector<std::complex<double>>>& zList, const std::vector<Eigen::VectorXd>& omegaList, const std::vector<Eigen::VectorXd>& refAmpList, const std::vector<Eigen::VectorXd>& refOmegaList, bool applyAdj = true);

        Eigen::VectorXd ampTimeOmegaSqInitialization(const Eigen::VectorXd& initAmpOmegaSq, const Eigen::VectorXd& tarAmpOmegaSq, double t);
        Eigen::VectorXd ampInitialization(const Eigen::VectorXd& initAmp, const Eigen::VectorXd& tarAmp, const Eigen::VectorXd& initAmpOmegaSq, const Eigen::VectorXd& tarAmpOmegaSq, const Eigen::VectorXd curAmpOmegaSq, double t);
        Eigen::VectorXd omegaInitialization(const Eigen::VectorXd& initOmega, const Eigen::VectorXd& tarOmega, const Eigen::VectorXd& initAmp, const Eigen::VectorXd& tarAmp, double t);

		virtual void solveIntermeditateFrames(Eigen::VectorXd& x, int numIter, double gradTol = 1e-6, double xTol = 0, double fTol = 0, bool isdisplayInfo = false, std::string workingFolder = "") = 0;

	
		std::vector<Eigen::VectorXd> getWList()
        {
            std::vector<Eigen::VectorXd> finalWList = _edgeOmegaList;		// _edgeOmega is the one after adjust
            for(int i = 0;  i < _edgeOmegaList.size(); i++)
                finalWList[i] += _deltaOmegaList[i];
            return finalWList;
        }
		virtual std::vector<std::vector<std::complex<double>>> getVertValsList() = 0;

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

		void getVertIdinGivenDomain(const std::vector<int> faceList, Eigen::VectorXi& vertFlags);
		void getEdgeIdinGivenDomain(const std::vector<int> faceList, Eigen::VectorXi& edgeFlags);

		virtual void save(const Eigen::VectorXd& x0, std::string* workingFold = NULL) = 0;
		virtual void convertVariable2List(const Eigen::VectorXd& x) = 0;
		virtual void convertList2Variable(Eigen::VectorXd& x) = 0;
		virtual void getComponentNorm(const Eigen::VectorXd& x, double& znorm, double& wnorm) = 0;
		virtual double computeEnergy(const Eigen::VectorXd& x, Eigen::VectorXd* deriv = NULL, Eigen::SparseMatrix<double>* hess = NULL, bool isProj = false) = 0;

		// spatial-temporal energies
		virtual double temporalAmpDifference(int frameId, Eigen::VectorXd* deriv = NULL, std::vector<Eigen::Triplet<double>>* hessT = NULL, bool isProj = false) = 0;
		virtual double temporalOmegaDifference(int frameId, Eigen::VectorXd* deriv = NULL, std::vector<Eigen::Triplet<double>>* hessT = NULL, bool isProj = false) = 0;
		virtual double spatialKnoppelEnergy(int frameId, Eigen::VectorXd* deriv = NULL, std::vector<Eigen::Triplet<double>>* hessT = NULL, bool isProj = false) = 0;
		virtual double kineticEnergy(int frameId, Eigen::VectorXd* deriv = NULL, std::vector<Eigen::Triplet<double>>* hessT = NULL, bool isProj = false) = 0;

	protected:
		void computeCombinedRefAmpList(const std::vector<Eigen::VectorXd>& refAmpList, std::vector<Eigen::VectorXd>* combinedOmegaList = NULL);

		void computeCombinedRefOmegaList(const std::vector<Eigen::VectorXd>& refOmegaList);


		Eigen::VectorXd computeCombinedRefAmp(const Eigen::VectorXd& curAmp, const Eigen::VectorXd& refAmp, Eigen::VectorXd* combinedOmega = NULL);

		Eigen::VectorXd computeCombinedRefOmega(const Eigen::VectorXd& curOmega, const Eigen::VectorXd& refOmega);


		double amplitudeEnergyWithGivenOmega(const Eigen::VectorXd& amp, const Eigen::VectorXd& w, Eigen::VectorXd* deriv = NULL, std::vector<Eigen::Triplet<double>>* hessT = NULL);
		double amplitudeEnergyWithGivenOmegaPerface(const Eigen::VectorXd& amp, const Eigen::VectorXd& w, int fid, Eigen::Vector3d* deriv = NULL, Eigen::Matrix3d* hess = NULL);

		double curlFreeEnergy(const Eigen::MatrixXd& w, Eigen::VectorXd* deriv = NULL,
			std::vector<Eigen::Triplet<double>>* hessT = NULL);

		double curlFreeEnergyPerface(const Eigen::MatrixXd& w, int faceId, Eigen::Matrix<double, 3, 1>* deriv = NULL,
			Eigen::Matrix<double, 3, 3>* hess = NULL);

		double divFreeEnergy(const Eigen::MatrixXd& w, Eigen::VectorXd* deriv = NULL, std::vector<Eigen::Triplet<double>>* hessT = NULL);
		double divFreeEnergyPervertex(const Eigen::MatrixXd& w, int vertId, Eigen::VectorXd* deriv = NULL, Eigen::MatrixXd* hess = NULL);

		void buildWeights();

        Eigen::Vector3d rot3dVec(const Eigen::Vector3d& v, const Eigen::Vector3d& axis, double angle);
		void computeAmpOmegaSq(const Eigen::VectorXd& amp, const Eigen::VectorXd& omega, Eigen::VectorXd& ampOmegaSq);


	public:
		// testing functions
		void testEnergy(Eigen::VectorXd x);
		void testCurlFreeEnergy(const Eigen::VectorXd& w);
		void testCurlFreeEnergyPerface(const Eigen::VectorXd& w, int faceId);

		void testDivFreeEnergy(const Eigen::VectorXd& w);
		void testDivFreeEnergyPervertex(const Eigen::VectorXd& w, int vertId);

		void testAmpEnergyWithGivenOmega(const Eigen::VectorXd& amp, const Eigen::VectorXd& w);
		void testAmpEnergyWithGivenOmegaPerface(const Eigen::VectorXd& amp, const Eigen::VectorXd& w, int faceId);

	protected:
		Eigen::MatrixXd _pos;
		MeshConnectivity _mesh;

		// interface indicator
		std::vector<VertexOpInfo> _vertexOpts;

		std::vector<int> _selectedFids;
		std::vector<int> _unselectedFids;
		std::vector<int> _interfaceFids;

		std::vector<int> _selectedVids;
		std::vector<int> _unselectedVids;
		std::vector<int> _interfaceVids;

		std::vector<int> _selectedEids;
		std::vector<int> _unselectedEids;
		std::vector<int> _interfaceEids;

		std::vector<double> _refAmpAveList;

		std::vector<Eigen::VectorXd> _combinedRefAmpList;
		std::vector<Eigen::VectorXd> _combinedRefOmegaList;

		std::vector<Eigen::VectorXd> _edgeOmegaList;
        std::vector<Eigen::VectorXd> _deltaOmegaList;
		std::vector<std::vector<std::complex<double>>> _zvalsList;
        std::vector<std::vector<std::complex<double>>> _unitZvalsList;  // the optimization variables for new formula

		int _quadOrd;
		std::vector<std::vector<int>> _vertNeiFaces;
		std::vector<std::vector<int>> _vertNeiEdges;

		Eigen::VectorXd _faceArea;
		Eigen::VectorXd _vertArea;
		Eigen::VectorXd _edgeArea;
		Eigen::VectorXd _edgeCotCoeffs;


		std::vector<std::vector<Eigen::Matrix2d>> _faceVertMetrics;
		double _spatialAmpRatio;
		double _spatialEdgeRatio;
		double _spatialKnoppelRatio;

		int _effectiveFactor;

		int _nInterfaces;
		std::string _savingFolder;

	public:	// should be private, when publishing
		ComputeZdotFromEdgeOmega _zdotModel;

	protected:
		double expGrowth(double x, double mu, double sigma)		// f = exp((x-mu)^2 / sigma^2)
		{
			return std::exp((x - mu) * (x - mu) / sigma / sigma);
		}


	protected:
		Eigen::VectorXd _faceWeight;
		Eigen::VectorXd _vertWeight;
        std::vector<Eigen::VectorXd> _ampTimesOmegaSq;
        std::vector<Eigen::VectorXd> _ampTimesDeltaOmegaSq;

	};
}