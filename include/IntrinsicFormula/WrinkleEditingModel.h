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

		WrinkleEditingModel(const Eigen::MatrixXd& pos, const MeshConnectivity& mesh, const std::vector<VertexOpInfo>& vertexOpts, const Eigen::VectorXi& faceFlag, int quadOrd, double spatialAmpRatio, double spatialEdgeRatio, double spatialKnoppelRatio);
		// faceFlag: 0 for unselected faces, 1 for selected faces, -1 for the interface faces

		void faceFlagsSetup(const Eigen::VectorXi& faceFlags);

		void initialization(const std::vector<std::complex<double>>& initZvals, const Eigen::VectorXd& initOmega, double numFrames, InitializationType initType, double zuenkoTau = 0.1, int zuenkoIter = 5);
        void initialization(const std::vector<std::complex<double>>& initZvals, const Eigen::VectorXd& initOmega, const std::vector<std::complex<double>>& tarZvals, const Eigen::VectorXd& tarOmega, const std::vector<Eigen::VectorXd>& refAmpList, const std::vector<Eigen::VectorXd>& refOmegaList, InitializationType initType);
		void initialization(const std::vector<std::vector<std::complex<double>>>& zList, const std::vector<Eigen::VectorXd>& omegaList, const std::vector<Eigen::VectorXd>& refAmpList, const std::vector<Eigen::VectorXd>& refOmegaList);

		void ZuenkoAlgorithm(const std::vector<std::complex<double>>& initZvals, const std::vector<Eigen::VectorXd>& refOmegaList, std::vector<std::vector<std::complex<double>>>& zList, double zuenkoTau = 0.1, int zuenkoIter = 5);

		std::vector<Eigen::VectorXd> getWList() { return _edgeOmegaList; }
		std::vector<std::vector<std::complex<double>>> getVertValsList() { return _zvalsList; }

		std::vector<Eigen::VectorXd> getRefWList() { return _combinedRefOmegaList; }
		std::vector<Eigen::VectorXd> getRefAmpList() { return _combinedRefAmpList; }


		void setSaveFolder(const std::string& savingFolder)
		{
			_savingFolder = savingFolder;
		}

		void setwzLists(std::vector<std::vector<std::complex<double>>>& zList, std::vector<Eigen::VectorXd>& wList)
		{
			_zvalsList = zList;
			_edgeOmegaList = wList;
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


		double amplitudeEnergyWithGivenOmega(const Eigen::VectorXd& amp, const Eigen::VectorXd& w, Eigen::VectorXd* deriv = NULL, std::vector<Eigen::Triplet<double>>* hessT = NULL);
		double amplitudeEnergyWithGivenOmegaPerface(const Eigen::VectorXd& amp, const Eigen::VectorXd& w, int fid, Eigen::Vector3d* deriv = NULL, Eigen::Matrix3d* hess = NULL);

		double curlFreeEnergy(const Eigen::MatrixXd& w, Eigen::VectorXd* deriv = NULL,
			std::vector<Eigen::Triplet<double>>* hessT = NULL);

		double curlFreeEnergyPerface(const Eigen::MatrixXd& w, int faceId, Eigen::Matrix<double, 3, 1>* deriv = NULL,
			Eigen::Matrix<double, 3, 3>* hess = NULL);

		double divFreeEnergy(const Eigen::MatrixXd& w, Eigen::VectorXd* deriv = NULL, std::vector<Eigen::Triplet<double>>* hessT = NULL);
		double divFreeEnergyPervertex(const Eigen::MatrixXd& w, int vertId, Eigen::VectorXd* deriv = NULL, Eigen::MatrixXd* hess = NULL);

		


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
		std::vector<std::vector<std::complex<double>>> _zvalsList;

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

		int _nInterfaces;
		std::string _savingFolder;

	public:	// should be private, when publishing
		ComputeZdotFromEdgeOmega _zdotModel;


	};
}