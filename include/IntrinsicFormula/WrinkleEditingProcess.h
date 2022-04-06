#pragma once

#include "../CommonTools.h"
#include "ComputeZdotFromHalfEdgeOmega.h"
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <set>

namespace IntrinsicFormula
{
    class WrinkleEditingProcess
    {
    public:
        WrinkleEditingProcess()
        {}

        WrinkleEditingProcess(const Eigen::MatrixXd& pos, const MeshConnectivity& mesh, const std::vector<int>& selectedVids,  const Eigen::VectorXi& faceFlag, int quadOrd, double spatialRatio = 1);

        void initialization(const std::vector<Eigen::VectorXd>& intRefAmpList, const std::vector<Eigen::MatrixXd>& initTefOmegaList, const std::vector<Eigen::VectorXd>& refAmpList, const std::vector<Eigen::MatrixXd>& refOmegaList);

        void convertVariable2List(const Eigen::VectorXd& x);
        void convertList2Variable(Eigen::VectorXd& x);

        std::vector<Eigen::MatrixXd> getWList() { return _edgeOmegaList; }
        std::vector<std::vector<std::complex<double>>> getVertValsList() { return _zvalsList; }

        void setwzLists(std::vector<std::vector<std::complex<double>>>& zList, std::vector<Eigen::MatrixXd>& wList)
        {
            _zvalsList = zList;
            _edgeOmegaList = wList;
        }

        void getComponentNorm(const Eigen::VectorXd& x, double& znorm, double& wnorm)
        {
            int nverts = _zvalsList[0].size();
            int nedges = _edgeOmegaList[0].rows();

            int numFrames = _zvalsList.size() - 2;

            znorm = 0;
            wnorm = 0;

            for (int i = 0; i < numFrames; i++)
            {
                for (int j = 0; j < nverts; j++)
                {
                    znorm = std::max(znorm, std::abs(x(i * (2 * nverts + 2 * nedges) + 2 * j)));
                    znorm = std::max(znorm, std::abs(x(i * (2 * nverts + 2 * nedges) + 2 * j + 1)));
                }
                for (int j = 0; j < nedges; j++)
                {
                    wnorm = std::max(wnorm, std::abs(x(i * (2 * nverts + 2 * nedges) + 2 * nverts + 2 * j)));
                    wnorm = std::max(wnorm, std::abs(x(i * (2 * nverts + 2 * nedges) + 2 * nverts + 2 * j + 1)));
                }
            }
        }

        double computeEnergy(const Eigen::VectorXd& x, Eigen::VectorXd* deriv = NULL, Eigen::SparseMatrix<double>* hess = NULL, bool isProj = false);
        void testEnergy(Eigen::VectorXd x);

        std::vector<Eigen::MatrixXd> getRefWList() { return _combinedRefOmegaList; }
        std::vector<Eigen::VectorXd> getRefAmpList() { return _combinedRefAmpList; }
        Eigen::VectorXi getVertFlag() { return _vertFlag; }
        Eigen::VectorXi getEdgeFlag() { return _edgeFlag; }
        Eigen::VectorXi getFaceFlag() { return _faceFlag; }


    private:
        void computeCombinedRefAmpList(const std::vector<Eigen::VectorXd>& refAmpList, std::vector<Eigen::MatrixXd>* combinedOmegaList = NULL);

        void computeCombinedRefOmegaList(const std::vector<Eigen::MatrixXd>& refOmegaList);


        double amplitudeEnergyWithGivenOmega(const Eigen::VectorXd& amp, const Eigen::MatrixXd& w, Eigen::VectorXd* deriv = NULL, std::vector<Eigen::Triplet<double>>* hessT = NULL);
        double amplitudeEnergyWithGivenOmegaPerface(const Eigen::VectorXd& amp, const Eigen::MatrixXd& w, int fid, Eigen::Vector3d* deriv = NULL, Eigen::Matrix3d* hess = NULL);

        double curlFreeEnergy(const Eigen::MatrixXd& w, Eigen::VectorXd* deriv = NULL,
            std::vector<Eigen::Triplet<double>>* hessT = NULL);

        double curlFreeEnergyPerface(const Eigen::MatrixXd& w, int faceId, Eigen::Matrix<double, 6, 1>* deriv = NULL,
            Eigen::Matrix<double, 6, 6>* hess = NULL);

        double divFreeEnergy(const Eigen::MatrixXd& w, Eigen::VectorXd* deriv = NULL, std::vector<Eigen::Triplet<double>>* hessT = NULL);
        double divFreeEnergyPervertex(const Eigen::MatrixXd& w, int vertId, Eigen::VectorXd* deriv = NULL, Eigen::MatrixXd* hess = NULL);


    public:
        // testing functions
        void testCurlFreeEnergy(const Eigen::MatrixXd& w);
        void testCurlFreeEnergyPerface(const Eigen::MatrixXd& w, int faceId);

        void testDivFreeEnergy(const Eigen::MatrixXd& w);
        void testDivFreeEnergyPervertex(const Eigen::MatrixXd& w, int vertId);

        void testAmpEnergyWithGivenOmega(const Eigen::VectorXd& amp, const Eigen::MatrixXd& w);
        void testAmpEnergyWithGivenOmegaPerface(const Eigen::VectorXd& amp, const Eigen::MatrixXd& w, int faceId);

    private:
        Eigen::MatrixXd _pos;
        MeshConnectivity _mesh;

        // interface indicator
        Eigen::VectorXi _faceFlag;  
        Eigen::VectorXi _vertFlag;
        Eigen::VectorXi _edgeFlag;

        std::vector<int> _selectedVids;

        std::vector<int> _effectiveFids;
        std::vector<int> _effectiveEids;
        std::vector<int> _effectiveVids;

        Eigen::MatrixXd _cotMatrixEntries;
        Eigen::VectorXd _faceArea;
        std::vector<Eigen::VectorXd> _combinedRefAmpList;
        std::vector<Eigen::MatrixXd> _combinedRefOmegaList;

        std::vector<Eigen::MatrixXd> _edgeOmegaList;
        std::vector<std::vector<std::complex<double>>> _zvalsList;

        int _quadOrd;
        std::vector<std::vector<int>> _vertNeiFaces;
        std::vector<std::vector<int>> _vertNeiEdges;
        Eigen::VectorXd _vertArea;
        Eigen::VectorXd _edgeCotCoeffs;

        std::vector<std::vector<Eigen::Matrix2d>> _faceVertMetrics;
        double _spatialRatio;

        int _nInterfaces;

    public:	// should be private, when publishing
        ComputeZdotFromHalfEdgeOmega _zdotModel;


    };
}