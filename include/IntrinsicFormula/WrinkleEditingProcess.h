#pragma once

#include "../CommonTools.h"
#include "IntrinsicKnoppelDrivenFormula.h"
#include <Eigen/Sparse>
#include <Eigen/Dense>

namespace IntrinsicFormula
{
    class WrinkleEditingProcess
    {
    public:
        WrinkleEditingProcess()
        {}

        WrinkleEditingProcess(const Eigen::MatrixXd &pos, const MeshConnectivity &mesh, int quadOrd, const std::vector<Eigen::VectorXd>& refAmpList, std::vector<Eigen::MatrixXd>& refOmegaList);

        void initialization(const std::vector<std::vector<VertexOpInfo>>& vertOptInfoList);

        std::vector<Eigen::MatrixXd> getWList() { return _model.getWList(); }
        std::vector<std::vector<std::complex<double>>> getVertValsList() { return _model.getVertValsList(); }

        std::vector<Eigen::MatrixXd> getRefWList() { return _refOmegaList; }
        std::vector<Eigen::VectorXd> getRefAmpList() { return _refAmpList; }

    private:
        Eigen::MatrixXd _pos;
        MeshConnectivity _mesh;
        Eigen::MatrixXd _cotMatrixEntries;
        Eigen::VectorXd _faceArea;
        std::vector<Eigen::VectorXd> _refAmpList;
        std::vector<Eigen::MatrixXd> _refOmegaList;

        int _quadOrd;

    public:
        IntrinsicKnoppelDrivenFormula _model;

    };
}