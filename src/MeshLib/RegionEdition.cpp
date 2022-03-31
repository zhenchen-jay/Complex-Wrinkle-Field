#include "../../include/MeshLib/RegionEdition.h"
#include <iostream>

void faceFlags2VertFlags(const MeshConnectivity& mesh, int nverts, const Eigen::VectorXi& faceFlags, Eigen::VectorXi& vertFlags)
{
	vertFlags.setZero(nverts);
	int nfaces = mesh.nFaces();

	for (int i = 0; i < nfaces; i++)
	{
		if (faceFlags(i))
		{
			for (int j = 0; j < 3; j++)
			{
				int vid = mesh.faceVertex(i, j);
				vertFlags(vid) = 1;
			}
		}
	}
}

void vertFlags2faceFlags(const MeshConnectivity& mesh, const Eigen::VectorXi& vertFlags, Eigen::VectorXi& faceFlags)
{
	int nfaces = mesh.nFaces();
	faceFlags.setZero(nfaces);

	for (int i = 0; i < nfaces; i++)
	{
		int flag = 1;
		for (int j = 0; j < 3; j++)
		{
			int vid = mesh.faceVertex(i, j);
			flag *= vertFlags(vid);
		}
		faceFlags(i) = flag;
	}
}

void RegionEdition::faceErosion(const Eigen::VectorXi &faceFlag,
							Eigen::VectorXi &faceFlagNew)
{
	int nfaces = _mesh.nFaces();
	Eigen::VectorXi oneVec = Eigen::VectorXi::Ones(nfaces);
	Eigen::VectorXi oppFaceFlags = oneVec - faceFlag;
	faceDilation(oppFaceFlags, oppFaceFlags);
	faceFlagNew = oneVec - oppFaceFlags;
}

void RegionEdition::vertexErosion(const Eigen::VectorXi &vertFlag, Eigen::VectorXi &vertFlagNew)
{
	int nverts = vertFlag.size();
	if (_vertNeighboring.size() != nverts)
	{
		std::cerr << "num of verts doesn't match!" << std::endl;
		exit(1);
	}
	Eigen::VectorXi oneVec = Eigen::VectorXi::Ones(nverts);
	Eigen::VectorXi oppVertFlags = oneVec - vertFlag;
	vertexDilation(oppVertFlags, oppVertFlags);
	vertFlagNew = oneVec - oppVertFlags;
}

void RegionEdition::faceDilation( const Eigen::VectorXi &faceFlag,
							  Eigen::VectorXi &faceFlagNew)
{

	faceFlagNew = faceFlag;
	int nfaces = _mesh.nFaces();
	for (int i = 0; i < nfaces; i++)
	{
		if (faceFlag[i])
		{
			int nNeis = _faceNeighboring[i].size();
			for (int j = 0; j < nNeis; j++)
			{
				int fid = _faceNeighboring[i][j];
				if(fid != -1)
					faceFlagNew[fid] = 1;
			}
		}

	}
}

void RegionEdition::vertexDilation(const Eigen::VectorXi &vertFlag, Eigen::VectorXi &vertFlagNew)
{
	vertFlagNew = vertFlag;
	int nverts = vertFlag.size();
	if (_vertNeighboring.size() != nverts)
	{
		std::cerr << "num of verts doesn't match!" << std::endl;
		exit(1);
	}
	for (int i = 0; i < nverts; i++)
	{
		if (vertFlag[i])
		{
			int nNeis = _vertNeighboring[i].size();
			bool isOnSelectedBnds = false;
			for (int j = 0; j < nNeis; j++)
			{
				int vid = _vertNeighboring[i][j];
				vertFlagNew[vid] = 1;
			}
		}

	}
}