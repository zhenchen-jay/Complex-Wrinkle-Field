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
        {}

        WrinkleEditingBaseMeshSeqCWF(const std::vector<Eigen::MatrixXd>& pos, const MeshConnectivity& mesh, int quadOrd, double spatialAmpRatio, double spatialKnoppelRatio);
        // faceFlag: 0 for unselected faces, 1 for selected faces, -1 for the interface faces

        void adjustOmegaForConsistency(const std::vector<std::complex<double>>& zvals, const Eigen::VectorXd& omega, Eigen::VectorXd& newOmega, Eigen::VectorXd& deltaOmega, Eigen::VectorXi* edgeFlags = NULL);
        void vecFieldSLERP(const Eigen::VectorXd& initVec, const Eigen::VectorXd& tarVec, std::vector<Eigen::VectorXd>& vecList, int numFrames, Eigen::VectorXi* edgeFlag = NULL);
        void vecFieldLERP(const Eigen::VectorXd& initVec, const Eigen::VectorXd& tarVec, std::vector<Eigen::VectorXd>& vecList, int numFrames, Eigen::VectorXi* edgeFlag = NULL);
        void ampFieldLERP(const Eigen::VectorXd& initAmp, const Eigen::VectorXd& tarAmp, std::vector<Eigen::VectorXd>& ampList, int numFrames, Eigen::VectorXi* vertFlag = NULL);

        void initialization(const std::vector<std::complex<double>>& initZvals, const Eigen::VectorXd& initOmega, const std::vector<std::complex<double>>& tarZvals, const Eigen::VectorXd& tarOmega, bool applyAdj = true);

        Eigen::VectorXd ampTimeOmegaSqInitialization(const Eigen::VectorXd& initAmpOmegaSq, const Eigen::VectorXd& tarAmpOmegaSq, double t);
        Eigen::VectorXd ampInitialization(const Eigen::VectorXd& initAmp, const Eigen::VectorXd& tarAmp, const Eigen::VectorXd& initAmpOmegaSq, const Eigen::VectorXd& tarAmpOmegaSq, const Eigen::VectorXd curAmpOmegaSq, double t);
        Eigen::VectorXd omegaInitialization(const Eigen::VectorXd& initOmega, const Eigen::VectorXd& tarOmega, const Eigen::VectorXd& initAmp, const Eigen::VectorXd& tarAmp, double t);

        void solveIntermeditateFrames(Eigen::VectorXd& x, int numIter, double gradTol = 1e-6, double xTol = 0, double fTol = 0, bool isdisplayInfo = false, std::string workingFolder = "");


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

        void save(const Eigen::VectorXd& x0, std::string* workingFold = NULL);
        void convertVariable2List(const Eigen::VectorXd& x);
        void convertList2Variable(Eigen::VectorXd& x);
        void getComponentNorm(const Eigen::VectorXd& x, double& znorm, double& wnorm);
        double computeEnergy(const Eigen::VectorXd& x, Eigen::VectorXd* deriv = NULL, Eigen::SparseMatrix<double>* hess = NULL, bool isProj = false);

        // spatial-temporal energies
        double temporalAmpDifference(int frameId, Eigen::VectorXd* deriv = NULL, std::vector<Eigen::Triplet<double>>* hessT = NULL, bool isProj = false);
        double spatialKnoppelEnergy(int frameId, Eigen::VectorXd* deriv = NULL, std::vector<Eigen::Triplet<double>>* hessT = NULL, bool isProj = false);
        double kineticEnergy(int frameId, Eigen::VectorXd* deriv = NULL, std::vector<Eigen::Triplet<double>>* hessT = NULL, bool isProj = false);

        Eigen::Vector3d rot3dVec(const Eigen::Vector3d& v, const Eigen::Vector3d& axis, double angle);
        void computeAmpOmegaSq(const Eigen::VectorXd& amp, const Eigen::VectorXd& omega, Eigen::VectorXd& ampOmegaSq);


    public:
        // testing functions
        void testEnergy(Eigen::VectorXd x);

    private:
        std::vector<Eigen::MatrixXd> _posList;
        MeshConnectivity _mesh;
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

        std::vector<Eigen::VectorXd> _faceArea;
        std::vector<Eigen::VectorXd> _vertArea;
        std::vector<Eigen::VectorXd> _edgeArea;
        std::vector<Eigen::VectorXd> _edgeCotCoeffs;


        std::vector<std::vector<Eigen::Matrix2d>> _faceVertMetrics;
        double _spatialAmpRatio;
        double _spatialKnoppelRatio;
        std::string _savingFolder;

        Eigen::VectorXd _faceWeight;
        Eigen::VectorXd _vertWeight;
        std::vector<Eigen::VectorXd> _ampTimesOmegaSq;
        std::vector<Eigen::VectorXd> _ampTimesDeltaOmegaSq;

    public:	// should be private, when publishing
        ComputeZdotFromEdgeOmega _zdotModel;

    };
}