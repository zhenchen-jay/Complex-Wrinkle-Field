#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "../MeshLib/MeshConnectivity.h"

class VecFieldsSplit{
public:
    VecFieldsSplit() {}
    VecFieldsSplit(const Eigen::MatrixXd& pos, const MeshConnectivity& mesh, const Eigen::MatrixXd& inputFields)
    : _pos(pos), _mesh(mesh), _inputFields(inputFields)
    {}

    double planeWaveSmoothness(const Eigen::MatrixXd vecFields, Eigen::VectorXd *deriv = NULL, Eigen::SparseMatrix<double> *hess = NULL);
    double planeWaveSmoothnessPerface(const Eigen::MatrixXd vecFields, int faceId, Eigen::VectorXd *deriv = NULL, Eigen::MatrixXd *hess = NULL);

    double whirlpoolSmoothness(const Eigen::MatrixXd vecFields, Eigen::VectorXd *deriv = NULL, Eigen::SparseMatrix<double> *hess = NULL, bool isProj = false);
    double whirlpoolSmoothnessPerface(const Eigen::MatrixXd vecFields, int faceId, Eigen::VectorXd *deriv = NULL, Eigen::MatrixXd *hess = NULL, bool isPorj = false);

    double optEnergy(const Eigen::VectorXd& x, Eigen::VectorXd *deriv = NULL, Eigen::SparseMatrix<double> *hess = NULL, bool isProj = false);

public: // test function
    void testPlaneWaveSmoothness(Eigen::MatrixXd vecFields);
    void testPlaneWaveSmoothnessPerface(Eigen::MatrixXd vecFields, int faceId);

    void testWhirlpoolSmoothness(Eigen::MatrixXd vecFields);
    void testWhirlpoolSmoothnessPerface(Eigen::MatrixXd vecFields, int faceId);

    void testOptEnergy(Eigen::VectorXd x);
private:

    Eigen::MatrixXd SPDProjection(const Eigen::MatrixXd &A)      // A is symmetric
    {
        Eigen::MatrixXd posHess = A;
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es;
        es.compute(posHess);
        Eigen::VectorXd evals = es.eigenvalues();

        for (int i = 0; i < evals.size(); i++)
        {
            if (evals(i) < 0)
                evals(i) = 0;
        }
        Eigen::MatrixXd D = evals.asDiagonal();
        Eigen::MatrixXd V = es.eigenvectors();
        posHess = V * D * V.inverse();

        return posHess;
    }


private:
    Eigen::MatrixXd _pos;
    MeshConnectivity _mesh;
    Eigen::MatrixXd _inputFields;
};