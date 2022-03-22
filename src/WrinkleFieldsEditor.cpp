#include "../include/WrinkleFieldsEditor.h"
#include <igl/per_vertex_normals.h>

void WrinkleFieldsEditor::editWrinkles(const Eigen::MatrixXd &pos, const MeshConnectivity &mesh,
                                       const Eigen::VectorXd &amp, const Eigen::MatrixXd &omega,
                                       const std::vector<VertexOpInfo> &vertInfo, Eigen::VectorXd &ampNew,
                                       Eigen::MatrixXd &omegaNew)
{
    ampNew = amp;
    omegaNew = omega;

    int nverts = pos.rows();
    Eigen::MatrixXd normals;
    igl::per_vertex_normals(pos, mesh.faces(), normals);

//    auto vertEditor = [&](const tbb::blocked_range<uint32_t>& range) {
//        for (uint32_t i = range.begin(); i < range.end(); ++i)
    for (uint32_t i = 0; i < nverts; ++i)
        {
            Eigen::Vector3d omegaVert;
            editWrinklesPerVertex(pos, mesh, normals, amp, omega, vertInfo, i, ampNew(i), omegaVert);
            omegaNew.row(i) = omegaVert;
        }
//    };
//
//    tbb::blocked_range<uint32_t> rangex(0u, (uint32_t)(nverts), GRAIN_SIZE);
//    tbb::parallel_for(rangex, vertEditor);
}

void WrinkleFieldsEditor::editWrinklesPerVertex(const Eigen::MatrixXd &pos, const MeshConnectivity &mesh,
                                                const Eigen::MatrixXd &vertNormals, const Eigen::VectorXd &amp,
                                                const Eigen::MatrixXd &omega, const std::vector<VertexOpInfo> &vertInfo,
                                                int vid, double &ampNew, Eigen::Vector3d &omegaNew)
{
    ampNew = amp(vid);
    omegaNew = omega.row(vid);
    Eigen::Vector3d axis = vertNormals.row(vid);
    if(vertInfo[vid].optType == Rotate)
    {
        // first normalize axis
        double ux = axis(0) / axis.norm(), uy = axis(1) / axis.norm(), uz = axis(2) / axis.norm();
        Eigen::Matrix3d rotMat;

        double c = std::cos(vertInfo[vid].optValue);
        double s = std::sin(vertInfo[vid].optValue);
        rotMat << c + ux * ux * (1 - c), ux* uy* (1 - c) - uz * s, ux* uz* (1 - c) + uy * s,
                uy* ux* (1 - c) + uz * s, c + uy * uy * (1 - c), uy* uz* (1 - c) - ux * s,
                uz* ux* (1 - c) - uy * s, uz* uy* (1 - c) + ux * s, c + uz * uz * (1 - c);

        omegaNew = rotMat * omegaNew;
    }
    else if (vertInfo[vid].optType == Enlarge)
    {
        omegaNew = omega.row(vid) * vertInfo[vid].optValue;
        ampNew = amp(vid) / vertInfo[vid].optValue;
    }
    else if (vertInfo[vid].optType == Tilt)
    {
        double ux = axis(0) / axis.norm(), uy = axis(1) / axis.norm(), uz = axis(2) / axis.norm();
        Eigen::Matrix3d rotMat;

        double c = std::cos(vertInfo[vid].optValue);
        double s = std::sin(vertInfo[vid].optValue);
        rotMat << c + ux * ux * (1 - c), ux* uy* (1 - c) - uz * s, ux* uz* (1 - c) + uy * s,
                uy* ux* (1 - c) + uz * s, c + uy * uy * (1 - c), uy* uz* (1 - c) - ux * s,
                uz* ux* (1 - c) - uy * s, uz* uy* (1 - c) + ux * s, c + uz * uz * (1 - c);

        omegaNew = rotMat * omegaNew;
        omegaNew *= omegaNew.norm() / c;
        ampNew *= c;
    }


}