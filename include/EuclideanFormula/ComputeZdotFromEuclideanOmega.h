#pragma once
#include "InterpolateZvalsFromEuclideanOmega.h"

namespace EuclideanFormula
{
    class ComputeZdotFromEuclideanOmega
    {
    public:
        ComputeZdotFromEuclideanOmega() {}
        ComputeZdotFromEuclideanOmega(const Eigen::MatrixXd& pos, const MeshConnectivity& mesh, const Eigen::VectorXd& faceArea, int quadOrder, double dt = 1)
        {
            _pos = pos;
            _mesh = mesh;
            _faceArea = faceArea;
            _quadpts = buildQuadraturePoints(quadOrder);
            _dt = dt;
        }

        void resetQuadpts(int quadOrder)
        {
            _quadpts = buildQuadraturePoints(quadOrder);
        }

        double computeZdotIntegration(const std::vector<std::complex<double>>& curZvals, const Eigen::MatrixXd& curw, const std::vector<std::complex<double>>& nextZvals, const Eigen::MatrixXd& nextw, Eigen::VectorXd* deriv, std::vector<Eigen::Triplet<double>>* hessT, bool isProj = false);

        // test functions (for debug usage)
        void testZdotIntegration(const std::vector<std::complex<double>>& curZvals, const Eigen::MatrixXd& curw, const std::vector<std::complex<double>>& nextZvals, const Eigen::MatrixXd& nextw);
        void testZdotIntegrationFromQuad(const std::vector<std::complex<double>>& curZvals, const Eigen::MatrixXd& curw, const std::vector<std::complex<double>>& nextZvals, const Eigen::MatrixXd& nextw, int fid, int qid);
        void testZdotIntegrationPerface(const std::vector<std::complex<double>>& curZvals, const Eigen::MatrixXd& curw, const std::vector<std::complex<double>>& nextZvals, const Eigen::MatrixXd& nextw, int fid);

    private:
        double computeZdotIntegrationFromQuad(const std::vector<std::complex<double>>& curZvals, const Eigen::MatrixXd& curw, const std::vector<std::complex<double>>& nextZvals, const Eigen::MatrixXd& nextw, int fid, int qid, Eigen::Matrix<double, 30, 1>* deriv, Eigen::Matrix<double, 30, 30>* hess);

        double computeZdotIntegrationPerface(const std::vector<std::complex<double>>& curZvals, const Eigen::MatrixXd& curw, const std::vector<std::complex<double>>& nextZvals, const Eigen::MatrixXd& nextw, int fid, Eigen::Matrix<double, 30, 1>* deriv, Eigen::Matrix<double, 30, 30>* hess, bool isProj = false);

    private:
        Eigen::MatrixXd _pos;
        MeshConnectivity _mesh;
        Eigen::VectorXd _faceArea;
        double _dt;
        std::vector<QuadraturePoints> _quadpts;
    };
}
