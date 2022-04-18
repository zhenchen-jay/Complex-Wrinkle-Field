#pragma once

#include "../CommonTools.h"
#include "ComputeZdotFromEdgeOmega.h"
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <set>

namespace IntrinsicFormula
{
    class WrinkleEditingStaticEdgeModel
    {
    public:
        WrinkleEditingStaticEdgeModel()
        {}

        WrinkleEditingStaticEdgeModel(const Eigen::MatrixXd& pos, const MeshConnectivity& mesh, const std::vector<VertexOpInfo>& vertexOpts, const Eigen::VectorXi& faceFlag, int quadOrd, double spatialAmpRatio, double spatialEdgeRatio, double spatialKnoppelRatio);

        void initialization(const Eigen::VectorXd& initAmp, const Eigen::VectorXd& initOmega, double numFrames);
        void warmstart();

        void convertVariable2List(const Eigen::VectorXd& x);
        void convertList2Variable(Eigen::VectorXd& x);

        std::vector<Eigen::VectorXd> getWList() { return _edgeOmegaList; }
        std::vector<std::vector<std::complex<double>>> getVertValsList() { return _zvalsList; }

        void setwzLists(std::vector<std::vector<std::complex<double>>>& zList, std::vector<Eigen::VectorXd>& wList)
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
                    znorm = std::max(znorm, std::abs(x(i * (2 * nverts + nedges) + 2 * j)));
                    znorm = std::max(znorm, std::abs(x(i * (2 * nverts + nedges) + 2 * j + 1)));
                }
                for (int j = 0; j < nedges; j++)
                {
                    wnorm = std::max(wnorm, std::abs(x(i * (2 * nverts + nedges) + 2 * nverts + j)));
                }
            }
        }

        double computeEnergy(const Eigen::VectorXd& x, Eigen::VectorXd* deriv = NULL, Eigen::SparseMatrix<double>* hess = NULL, bool isProj = false);
        void testEnergy(Eigen::VectorXd x);

        std::vector<Eigen::VectorXd> getRefWList() { return _combinedRefOmegaList; }
        std::vector<Eigen::VectorXd> getRefAmpList() { return _combinedRefAmpList; }
        Eigen::VectorXi getVertFlag() { return _vertFlag; }
        Eigen::VectorXi getEdgeFlag() { return _edgeFlag; }
        Eigen::VectorXi getFaceFlag() { return _faceFlag; }


    private:
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
        void testCurlFreeEnergy(const Eigen::VectorXd& w);
        void testCurlFreeEnergyPerface(const Eigen::VectorXd& w, int faceId);

        void testDivFreeEnergy(const Eigen::VectorXd& w);
        void testDivFreeEnergyPervertex(const Eigen::VectorXd& w, int vertId);

        void testAmpEnergyWithGivenOmega(const Eigen::VectorXd& amp, const Eigen::VectorXd& w);
        void testAmpEnergyWithGivenOmegaPerface(const Eigen::VectorXd& amp, const Eigen::VectorXd& w, int faceId);

    private:
        Eigen::MatrixXd _pos;
        MeshConnectivity _mesh;

        // interface indicator
        std::vector<VertexOpInfo> _vertexOpts;
        Eigen::VectorXi _faceFlag;
        Eigen::VectorXi _vertFlag;
        Eigen::VectorXi _edgeFlag;

        std::vector<int> _effectiveFids;
        std::vector<int> _effectiveEids;
        std::vector<int> _effectiveVids;

        Eigen::MatrixXd _cotMatrixEntries;
        Eigen::VectorXd _faceArea;
        std::vector<Eigen::VectorXd> _combinedRefAmpList;
        std::vector<Eigen::VectorXd> _combinedRefOmegaList;

        std::vector<Eigen::VectorXd> _edgeOmegaList;
        std::vector<std::vector<std::complex<double>>> _zvalsList;

        int _quadOrd;
        std::vector<std::vector<int>> _vertNeiFaces;
        std::vector<std::vector<int>> _vertNeiEdges;
        Eigen::VectorXd _vertArea;
        Eigen::VectorXd _edgeArea;
        Eigen::VectorXd _edgeCotCoeffs;

        std::vector<std::vector<Eigen::Matrix2d>> _faceVertMetrics;
        double _spatialAmpRatio;
        double _spatialEdgeRatio;
        double _spatialKnoppelRatio;

        int _nInterfaces;

    public:	// should be private, when publishing
        ComputeZdotFromEdgeOmega _zdotModel;


    };
}