#include "../../include/MeshLib/RegionEdition.h"
#include <iostream>

void RegionEdition::faceErosion(const Eigen::VectorXi &faceFlag,
                            Eigen::VectorXi &faceFlagNew)
{
    faceFlagNew = faceFlag;
    int nfaces = _mesh.nFaces();
    for(int i = 0; i < nfaces; i++)
    {
        if(faceFlag[i])
        {
            int nNeis = _faceNeighboring[i].size();
            bool isOnSelectedBnds = false;
            std::cout << "\nfid: " << i;
            for(int j = 0; j < nNeis; j++)
            {
                int fid = _faceNeighboring[i][j];
                std::cout << ", neighboring faceId: (" << fid << " , " << faceFlag[fid] << "), ";
                if(!faceFlag[fid])
                    isOnSelectedBnds = true;
            }

            if(isOnSelectedBnds)
                faceFlagNew[i] = 0;
//            faceFlagNew[i] = isOnSelectedBnds ? 0 : 1;   // only keep the interior selected faces
        }

    }
}

void RegionEdition::vertexErosion(const Eigen::VectorXi &vertFlag, Eigen::VectorXi &vertFlagNew)
{
    vertFlagNew = vertFlag;
    int nverts = vertFlag.size();
    if(_vertNeighboring.size() != nverts)
    {
        std::cerr << "num of verts doesn't match!" << std::endl;
        exit(1);
    }
    for(int i = 0; i < nverts; i++)
    {
        if(vertFlag[i])
        {
            int nNeis = _vertNeighboring[i].size();
            bool isOnSelectedBnds = false;
            for(int j = 0; j < nNeis; j++)
            {
                int vid = _vertNeighboring[i][j];
                if(!vertFlag[vid])
                    isOnSelectedBnds = true;
            }
            vertFlagNew[i] = isOnSelectedBnds ? 0 : 1;   // only keep the interior selected vertices
        }

    }
}

void RegionEdition::faceDilation( const Eigen::VectorXi &faceFlag,
                              Eigen::VectorXi &faceFlagNew)
{
    int nfaces = _mesh.nFaces();
    Eigen::VectorXi oneVec = Eigen::VectorXi::Ones(nfaces);
    Eigen::VectorXi oppFaceFlags = oneVec - faceFlag;
    faceErosion(oppFaceFlags, oppFaceFlags);
    faceFlagNew = oneVec - oppFaceFlags;
}

void RegionEdition::vertexDilation(const Eigen::VectorXi &vertFlag, Eigen::VectorXi &vertFlagNew)
{
    int nverts = vertFlag.size();
    if(_vertNeighboring.size() != nverts)
    {
        std::cerr << "num of verts doesn't match!" << std::endl;
        exit(1);
    }
    Eigen::VectorXi oneVec = Eigen::VectorXi::Ones(nverts);
    Eigen::VectorXi oppVertFlags = oneVec - vertFlag;
    vertexErosion(oppVertFlags, oppVertFlags);
    vertFlagNew = oneVec - oppVertFlags;
}