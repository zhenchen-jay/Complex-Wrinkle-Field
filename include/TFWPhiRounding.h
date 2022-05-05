#pragma once
#include "CommonTools.h"
#include "MeshLib/IntrinsicGeometry.h"
#include <set>

namespace TFWPhiRounding
{
    void buildFirstFoundForms(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, std::vector<Eigen::Matrix2d>& abars);
    void combField(const Eigen::MatrixXi &F, const std::vector<Eigen::Matrix2d> &abars, const Eigen::VectorXd* weight, const Eigen::MatrixXd &w, Eigen::MatrixXd &combedW);
    void reindex(Eigen::MatrixXi& F);
    void punctureMesh(const Eigen::MatrixXi& F, const std::vector<int>& singularities, Eigen::MatrixXi& puncturedF, Eigen::VectorXi& newFacesToOld);
    void findCuts(const Eigen::MatrixXi& F, std::vector<std::vector<int> >& cuts);
    void cutMesh(const Eigen::MatrixXi& F,
            // list of cuts, each of which is a list (in order) of vertex indices of one cut.
            // Cuts can be closed loops (in which case the last vertex index should equal the
            // first) or open (in which case the two endpoint vertices should be distinct).
            // Multiple cuts can cross but there may be strange behavior if cuts share endpoint
            // vertices, or are non-edge-disjoint.
                 const std::vector<std::vector<int> >& cuts,
            // new vertices and faces
            // **DO NOT ALIAS V OR F!**
                 Eigen::MatrixXi& newF
    );

    void faceDPhi2EdgeDPhi(const Eigen::MatrixXd& faceDphi, const std::set<int>& tensionFaces, const std::vector<Eigen::Matrix2d>& abars, Eigen::MatrixXi F, Eigen::VectorXd& dphi);
// convert face dphi (one-form) to edge dphi which satisfies the local integrability constraints except for the pure tension faces.

    void vectorFieldSingularities(const Eigen::MatrixXi& F, const std::vector<Eigen::Matrix2d>& abars, const Eigen::MatrixXd& w, std::vector<int>& singularities);
    void roundPhiFromDphiCutbyTension(
            const Eigen::MatrixXd& V,
            const Eigen::MatrixXi& F,
            Eigen::MatrixXd& seamedV,
            Eigen::MatrixXi& seamedF,
            const std::vector<Eigen::Matrix2d>& abars,
            const Eigen::VectorXd& amp,
            const Eigen::VectorXd& dPhi, // |E| vector of phi jumps
            Eigen::VectorXd& phi,
            Eigen::VectorXd& seamedPhi,
            Eigen::VectorXd& seamedAmp);
}
