#pragma once

#include "../CommonTools.h"
#include "IntrinsicKnoppelDrivenFormula.h"
#include <Eigen/Sparse>
#include <Eigen/Dense>

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

    private:
        void computeCombinedRefAmpList(const std::vector<std::vector<Eigen::VectorXd>> &refAmpList);

        void computeCombinedRefOmegaList(const std::vector<std::vector<Eigen::MatrixXd>> &refOmegaList);

        double curlFreeEnergy(const Eigen::MatrixXd &w, Eigen::VectorXd *deriv = NULL,
                              std::vector<Eigen::Triplet<double>> *hessT = NULL);

        double curlFreeEnergyPerface(const Eigen::MatrixXd &w, int faceId, Eigen::Matrix<double, 6, 1> *deriv = NULL,
                                     Eigen::Matrix<double, 6, 6> *hess = NULL);
    public:
        // testing functions
        void testCurlFreeEnergy(const Eigen::MatrixXd &w);

        void testCurlFreeEnergyPerface(const Eigen::MatrixXd &w, int faceId);

    private:
        Eigen::MatrixXd _pos;
        MeshConnectivity _mesh;
        Eigen::VectorXi _faceFlag;
        Eigen::VectorXi _vertFlag;
        Eigen::VectorXi _edgeFlag;

        Eigen::MatrixXd _cotMatrixEntries;
        Eigen::VectorXd _faceArea;
        std::vector<Eigen::VectorXd> _combinedRefAmpList;
        std::vector<Eigen::MatrixXd> _combinedRefOmegaList;

        int _quadOrd;

    public:
        IntrinsicKnoppelDrivenFormula _model;

    };
}