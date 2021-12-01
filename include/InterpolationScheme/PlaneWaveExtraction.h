#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "../MeshLib/MeshConnectivity.h"

class PlaneWaveExtraction{
public:
    PlaneWaveExtraction() {}
    PlaneWaveExtraction(const Eigen::MatrixXd& pos, const MeshConnectivity& mesh, const Eigen::MatrixXd& inputFields)
    : _pos(pos), _mesh(mesh), _inputFields(inputFields)
    {}

    bool extractPlaneWave(Eigen::MatrixXd& planeFields);
    /*
     * Input: vertex based vector field [_inputFields]
     * Output: a face based vector field [planeFields], which is curve-free (wf1.e = wf2.e), by minimizing the direchelet energy
     */
    double faceFieldDifference(const Eigen::MatrixXd faceFields, Eigen::VectorXd *deriv = NULL, Eigen::SparseMatrix<double> *hess = NULL);
    double faceFieldDifferencePerEdge(const Eigen::MatrixXd faceFields, int eid, Eigen::Vector4d *deriv = NULL, Eigen::Matrix4d *hess = NULL);

    double optEnergy(const Eigen::VectorXd& x, Eigen::VectorXd *deriv = NULL, Eigen::SparseMatrix<double> *hess = NULL);
    double optEnergyPerEdge(const Eigen::VectorXd& x, int eid, Eigen::Matrix<double, 6, 1> *deriv = NULL, Eigen::Matrix<double, 6, 6> *hess = NULL);

public: // test function
    void testFaceFieldDifference(Eigen::MatrixXd faceFields);
    void testFaceFieldDifferencePerEdge(Eigen::MatrixXd faceField, int eid);

    void testOptEnergy(Eigen::VectorXd x);
    void testOptEnergyPerEdge(Eigen::VectorXd x, int eid);

private:
    void getNumIter(double acc_thresh, int& inner_iter_ref, int& outer_iter_ref)
    {
        if (acc_thresh >= 1e-3)
        {
            inner_iter_ref = outer_iter_ref = 0;
        }
        else if (acc_thresh < 1e-3 && acc_thresh >= 1e-6)
        {
            inner_iter_ref = outer_iter_ref = 1;
        }
        // if (acc_thresh >= 1e-3)
        // {
        //     inner_iter_ref = outer_iter_ref = 1;
        // }
        else if (acc_thresh < 1e-6 && acc_thresh >= 1e-10)
        {
            inner_iter_ref = outer_iter_ref = 2;
        }
        else if (acc_thresh < 1e-10 && acc_thresh >= 1e-13)
        {
            inner_iter_ref = outer_iter_ref = 3;
        }
        else
        {
            inner_iter_ref = outer_iter_ref = 9;
        }
    }


private:
    Eigen::MatrixXd _pos;
    MeshConnectivity _mesh;
    Eigen::MatrixXd _inputFields;
};