#include "../include/getSideVertexPhi.h"
#include "../include/MeshLib/MeshUpsampling.h"
#include "../include/InterpolationScheme/SideVertexInterpolation.h"
#include "../include/InterpolationScheme/IntrinsicSideVertexSchemes.h"

void getSideVertexPhi(const Eigen::MatrixXd& V, const MeshConnectivity& mesh, const Eigen::VectorXd& edgeOmega, const Eigen::VectorXd& vertPhi, Eigen::MatrixXd& upV, const MeshConnectivity& upMesh, Eigen::VectorXd& upPhi, int upLevel, int interpType)
{
    // use the intrinsic formula
    Eigen::SparseMatrix<double> mat;
    std::vector<int> facemap;
    std::vector<std::pair<int, Eigen::Vector3d>> bary;

    Eigen::MatrixXi upF;
    meshUpSampling(V, mesh.faces(), upV, upF, upLevel, &mat, &facemap, &bary);

    int nUpVerts = upV.rows();
    upPhi.resize(nUpVerts);

    for(int i = 0; i < nUpVerts; i++)
    {
        int fid = bary[i].first;
        Eigen::Vector3d bcoord = bary[i].second;
        std::vector<Eigen::Vector3d> tri;
        tri.push_back(V.row(mesh.faceVertex(fid, 0)));
        tri.push_back(V.row(mesh.faceVertex(fid, 1)));
        tri.push_back(V.row(mesh.faceVertex(fid, 2)));


        if(interpType == 0)
        {

        }
    }

}

void getSideVertexPhi(const Eigen::MatrixXd& V, const MeshConnectivity& mesh, const Eigen::MatrixXd& vertOmega, const Eigen::VectorXd& vertPhi, Eigen::MatrixXd& upV, const MeshConnectivity& upMesh, Eigen::VectorXd& upPhi, int upLevel, int interpType )
{
    // use the extrinsic formula
}