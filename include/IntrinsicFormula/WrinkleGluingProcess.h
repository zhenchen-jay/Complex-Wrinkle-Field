#pragma once

#include "../CommonTools.h"
#include "IntrinsicKnoppelDrivenFormula.h"
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <set>

namespace IntrinsicFormula
{
    class WrinkleGluingProcess
    {
    public:
        WrinkleGluingProcess()
        {}

        WrinkleGluingProcess(const Eigen::MatrixXd &pos, const MeshConnectivity &mesh, const Eigen::VectorXi &faceFlag, int quadOrd);

        void initialization(const std::vector<std::vector<Eigen::VectorXd>>& refAmpList, std::vector<std::vector<Eigen::MatrixXd>>& refOmegaList);

        std::vector<Eigen::MatrixXd> getWList() { return _model.getWList(); }
        std::vector<std::vector<std::complex<double>>> getVertValsList() { return _model.getVertValsList(); }

        std::vector<Eigen::MatrixXd> getRefWList() { return _combinedRefOmegaList; }
        std::vector<Eigen::VectorXd> getRefAmpList() { return _combinedRefAmpList; }
        Eigen::VectorXi getVertFlag() { return _vertFlag; }
        Eigen::VectorXi getEdgeFlag() { return _edgeFlag; }
        Eigen::VectorXi getFaceFlag() { return _faceFlag; }


    private:
        void computeCombinedRefAmpList(const std::vector<std::vector<Eigen::VectorXd>> &refAmpList, std::vector<Eigen::MatrixXd>* combinedOmegaList = NULL);

        void computeCombinedRefOmegaList(const std::vector<std::vector<Eigen::MatrixXd>> &refOmegaList);


        double amplitudeEnergyWithGivenOmega(const Eigen::VectorXd& amp, const Eigen::MatrixXd& w, Eigen::VectorXd *deriv = NULL, std::vector<Eigen::Triplet<double>> *hessT = NULL);
        double amplitudeEnergyWithGivenOmegaPerface(const Eigen::VectorXd& amp, const Eigen::MatrixXd& w, int fid, Eigen::Vector3d* deriv = NULL, Eigen::Matrix3d* hess = NULL);

        double curlFreeEnergy(const Eigen::MatrixXd &w, Eigen::VectorXd *deriv = NULL,
                              std::vector<Eigen::Triplet<double>> *hessT = NULL);

        double curlFreeEnergyPerface(const Eigen::MatrixXd &w, int faceId, Eigen::Matrix<double, 6, 1> *deriv = NULL,
                                     Eigen::Matrix<double, 6, 6> *hess = NULL);

        double divFreeEnergy(const Eigen::MatrixXd &w, Eigen::VectorXd *deriv = NULL, std::vector<Eigen::Triplet<double>> *hessT = NULL);
        double divFreeEnergyPervertex(const Eigen::MatrixXd& w, int vertId, Eigen::VectorXd *deriv = NULL, Eigen::MatrixXd *hess = NULL);

    public:
        // testing functions
        void testCurlFreeEnergy(const Eigen::MatrixXd &w);
        void testCurlFreeEnergyPerface(const Eigen::MatrixXd &w, int faceId);

        void testDivFreeEnergy(const Eigen::MatrixXd &w);
        void testDivFreeEnergyPervertex(const Eigen::MatrixXd &w, int vertId);

        void testAmpEnergyWithGivenOmega(const Eigen::VectorXd& amp, const Eigen::MatrixXd& w);
        void testAmpEnergyWithGivenOmegaPerface(const Eigen::VectorXd& amp, const Eigen::MatrixXd& w, int faceId);

    private:
        Eigen::MatrixXd _pos;
        MeshConnectivity _mesh;
        Eigen::VectorXi _faceFlag;
        Eigen::VectorXi _vertFlag;
        Eigen::VectorXi _edgeFlag;

        std::vector<int> _effectiveFids;
        std::vector<int> _effectiveEids;
        std::vector<int> _effectiveVids;

        Eigen::MatrixXd _cotMatrixEntries;
        Eigen::VectorXd _faceArea;
        std::vector<Eigen::VectorXd> _combinedRefAmpList;
        std::vector<Eigen::MatrixXd> _combinedRefOmegaList;

        int _quadOrd;
        std::vector<std::vector<int>> _vertNeiFaces;
        std::vector<std::vector<int>> _vertNeiEdges;
        Eigen::VectorXd _vertArea;
        Eigen::VectorXd _edgeCotCoeffs;

        std::vector<std::vector<Eigen::Matrix2d>> _faceVertMetrics;

    public:
        IntrinsicKnoppelDrivenFormula _model;

    };
}