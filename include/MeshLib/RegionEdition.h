#pragma once
#include <Eigen/Dense>
#include <igl/adjacency_list.h>
#include <set>
#include "MeshConnectivity.h"

void faceFlags2VertFlags(const MeshConnectivity& mesh, int nverts, const Eigen::VectorXi& faceFlags, Eigen::VectorXi& vertFlags);
void vertFlags2faceFlags(const MeshConnectivity& mesh, const Eigen::VectorXi& vertFlags, Eigen::VectorXi& faceFlags);

class RegionEdition
{
public:
    RegionEdition() {}
    RegionEdition(const MeshConnectivity& mesh, const int nverts)
    {
        _mesh = mesh;
        _nverts = nverts;
        int nfaces = _mesh.nFaces();
        _faceNeighboring.resize(nfaces, {});
        for(int i = 0; i < nfaces; i++)
        {
            for(int j = 0; j < 3; j++)
            {
                int eid = _mesh.faceEdge(i, j);
                int fid0 = _mesh.edgeFace(eid, 0);
                int fid1 = _mesh.edgeFace(eid, 1);
                if(fid0 == i)
                    _faceNeighboring[i].push_back(fid1);
                else
                    _faceNeighboring[i].push_back(fid0);
            }
        }
        igl::adjacency_list(_mesh.faces(), _vertNeighboring);
    }
    void faceErosion(const Eigen::VectorXi& faceFlag, Eigen::VectorXi& faceFlagNew);    // one-ring erosion of the selected region (marked as true)
    void faceDilation(const Eigen::VectorXi& faceFlag, Eigen::VectorXi& faceFlagNew);  // one-ring dilation of the selected region

    void faceErosion(const Eigen::VectorXi& faceFlag, Eigen::VectorXi& faceFlagNew, int times = 1);    // one-ring erosion of the selected region (marked as true)
    void faceDilation(const Eigen::VectorXi& faceFlag, Eigen::VectorXi& faceFlagNew, int times);  // one-ring dilation of the selected region

    void vertexErosion(const Eigen::VectorXi& vertFlag, Eigen::VectorXi& vertFlagNew);
    void vertexDilation(const Eigen::VectorXi& vertFlag, Eigen::VectorXi& vertFlagNew);

private:
    MeshConnectivity _mesh;
    int _nverts;
    std::vector<std::vector<int>> _faceNeighboring;
    std::vector<std::vector<int>> _vertNeighboring;
};

