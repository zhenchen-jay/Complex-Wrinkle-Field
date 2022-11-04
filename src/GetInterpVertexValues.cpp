#include "../include/GetInterpVertexValues.h"
#include "../include/MeshLib/MeshUpsampling.h"
#include "../include/InterpolationScheme/SideVertexSchemes.h"
#include "../include/InterpolationScheme/IntrinsicSideVertexSchemes.h"
#include "../include/InterpolationScheme/IntrinsicClouhTocherScheme.h"

void getSideVertexPhi(const Eigen::MatrixXd& V, const MeshConnectivity& mesh, const Eigen::VectorXd& edgeOmega, const Eigen::VectorXd& vertPhi, const std::vector<std::pair<int, Eigen::Vector3d>>& bary, Eigen::VectorXd& upPhi, int interpType)
{
	int nUpVerts = bary.size();
	upPhi.resize(nUpVerts);

	for (int i = 0; i < nUpVerts; i++)
	{
		int fid = bary[i].first;
		Eigen::Vector3d bcoord = bary[i].second;
		std::vector<Eigen::Vector3d> tri;
		tri.push_back(V.row(mesh.faceVertex(fid, 0)));
		tri.push_back(V.row(mesh.faceVertex(fid, 1)));
		tri.push_back(V.row(mesh.faceVertex(fid, 2)));

		std::vector<double> triEdgeOmega(3);
		std::vector<double> vertVals(3);
		for (int j = 0; j < 3; j++)
		{
			int eid = mesh.faceEdge(fid, j);
			if (mesh.edgeVertex(eid, 0) == mesh.faceVertex(fid, (j + 2) % 3))
				triEdgeOmega[j] = -edgeOmega[eid];
			else
				triEdgeOmega[j] = edgeOmega[eid];
			vertVals[j] = vertPhi[mesh.faceVertex(fid, j)];
		}


		if (interpType == 0)
		{
			upPhi[i] = intrinsicLinearSideVertexInterpolation<double>(vertVals, bcoord);
		}
		else if (interpType == 1)
		{
			upPhi[i] = intrinsicCubicSideVertexInterpolation<double>(vertVals, triEdgeOmega, tri, bcoord);
		}
		else
		{
			upPhi[i] = intrinsicWojtanSideVertexInterpolation<double>(vertVals, triEdgeOmega, tri, bcoord);
		}
	}
}

void getSideVertexPhi(const Eigen::MatrixXd& V, const MeshConnectivity& mesh, const Eigen::VectorXd& edgeOmega, const std::vector<std::complex<double>>& vertZvals, const std::vector<std::pair<int, Eigen::Vector3d>>& bary, Eigen::VectorXd& upPhi, int interpType)
{
    int nUpVerts = bary.size();
    upPhi.resize(nUpVerts);

    for (int i = 0; i < nUpVerts; i++)
    {
        int fid = bary[i].first;
        Eigen::Vector3d bcoord = bary[i].second;
        std::vector<Eigen::Vector3d> tri;
        tri.push_back(V.row(mesh.faceVertex(fid, 0)));
        tri.push_back(V.row(mesh.faceVertex(fid, 1)));
        tri.push_back(V.row(mesh.faceVertex(fid, 2)));

        std::vector<double> triEdgeOmega(3);
        std::vector<double> vertVals(3);
        for (int j = 0; j < 3; j++)
        {
            int eid = mesh.faceEdge(fid, j);
            if (mesh.edgeVertex(eid, 0) == mesh.faceVertex(fid, (j + 2) % 3))
                triEdgeOmega[j] = -edgeOmega[eid];
            else
                triEdgeOmega[j] = edgeOmega[eid];
            vertVals[j] = std::arg(vertZvals[mesh.faceVertex(fid, j)]);
        }

        int k2 = std::round((vertVals[0] - vertVals[2] - triEdgeOmega[1]) / 2 / M_PI);
        int k1 = std::round((vertVals[0] - vertVals[1] + triEdgeOmega[2]) / 2 / M_PI);
        vertVals[1] += 2 * k1 * M_PI;
        vertVals[2] += 2 * k2 * M_PI;
        
        if (interpType == 0)
        {
            upPhi[i] = intrinsicLinearSideVertexInterpolation<double>(vertVals, bcoord);
        }
        else if (interpType == 1)
        {
            upPhi[i] = intrinsicCubicSideVertexInterpolation<double>(vertVals, triEdgeOmega, tri, bcoord);
        }
        else
        {
            upPhi[i] = intrinsicWojtanSideVertexInterpolation<double>(vertVals, triEdgeOmega, tri, bcoord);
        }
    }
}

void getClouhTocherPhi(const Eigen::MatrixXd& V, const MeshConnectivity& mesh, const Eigen::VectorXd& edgeOmega, const std::vector<std::complex<double>>& vertZvals, const std::vector<std::pair<int, Eigen::Vector3d>>& bary, Eigen::VectorXd& upPhi)
{
	int nUpVerts = bary.size();
	upPhi.resize(nUpVerts);

	for (int i = 0; i < nUpVerts; i++)
	{
		int fid = bary[i].first;
		Eigen::Vector3d bcoord = bary[i].second;
		std::vector<Eigen::Vector3d> tri;
		tri.push_back(V.row(mesh.faceVertex(fid, 0)));
		tri.push_back(V.row(mesh.faceVertex(fid, 1)));
		tri.push_back(V.row(mesh.faceVertex(fid, 2)));

		std::vector<double> triEdgeOmega(3);
		std::vector<double> vertVals(3);
		for (int j = 0; j < 3; j++)
		{
			int eid = mesh.faceEdge(fid, j);
			if (mesh.edgeVertex(eid, 0) == mesh.faceVertex(fid, (j + 2) % 3))
				triEdgeOmega[j] = -edgeOmega[eid];
			else
				triEdgeOmega[j] = edgeOmega[eid];
			vertVals[j] = std::arg(vertZvals[mesh.faceVertex(fid, j)]);
		}

		int k2 = std::round((vertVals[0] - vertVals[2] - triEdgeOmega[1]) / 2 / M_PI);
		int k1 = std::round((vertVals[0] - vertVals[1] + triEdgeOmega[2]) / 2 / M_PI);

		vertVals[1] += 2 * k1 * M_PI;
		vertVals[2] += 2 * k2 * M_PI;


		upPhi[i] = intrinsicClouthTocherInterpolation<double>(vertVals, triEdgeOmega, tri, bcoord);
	}
}

void getSideVertexPhi(const Eigen::MatrixXd& V, const MeshConnectivity& mesh, const Eigen::MatrixXd& vertOmega, const Eigen::VectorXd& vertPhi, const std::vector<std::pair<int, Eigen::Vector3d>>& bary, Eigen::VectorXd& upPhi, int interpType)
{
	int nUpVerts = bary.size();
	upPhi.resize(nUpVerts);

	for (int i = 0; i < nUpVerts; i++)
	{
		int fid = bary[i].first;
		Eigen::Vector3d bcoord = bary[i].second;
		std::vector<Eigen::Vector3d> tri;
		tri.push_back(V.row(mesh.faceVertex(fid, 0)));
		tri.push_back(V.row(mesh.faceVertex(fid, 1)));
		tri.push_back(V.row(mesh.faceVertex(fid, 2)));

		std::vector<Eigen::Matrix<double, 3, 1>> triVertGrad(3);
		std::vector<double> vertVals(3);
		for (int j = 0; j < 3; j++)
		{
			vertVals[j] = vertPhi[mesh.faceVertex(fid, j)];
			triVertGrad[j] = vertOmega.row(mesh.faceVertex(fid, j)).transpose();
		}


		if (interpType == 0)
		{
			upPhi[i] = linearSideVertexInterpolation<double>(vertVals, bcoord);
		}
		else if (interpType == 1)
		{
			upPhi[i] = cubicSideVertexInterpolation<double>(vertVals, triVertGrad, tri, bcoord);
		}
		else
		{
			upPhi[i] = WojtanSideVertexInterpolation<double>(vertVals, triVertGrad, tri, bcoord);
		}
	}
}

void getSideVertexPhi(const Eigen::MatrixXd& V, const MeshConnectivity& mesh, const Eigen::VectorXd& edgeOmega, const Eigen::VectorXd& vertPhi, Eigen::MatrixXd& upV, const MeshConnectivity& upMesh, Eigen::VectorXd& upPhi, int upLevel, int interpType)
{
	// use the intrinsic formula
	Eigen::SparseMatrix<double> mat;
	std::vector<int> facemap;
	std::vector<std::pair<int, Eigen::Vector3d>> bary;

	Eigen::MatrixXi upF;
	meshUpSampling(V, mesh.faces(), upV, upF, upLevel, &mat, &facemap, &bary);

	getSideVertexPhi(V, mesh, edgeOmega, vertPhi, bary, upPhi, interpType);

}

void getSideVertexPhi(const Eigen::MatrixXd& V, const MeshConnectivity& mesh, const Eigen::MatrixXd& vertOmega, const Eigen::VectorXd& vertPhi, Eigen::MatrixXd& upV, const MeshConnectivity& upMesh, Eigen::VectorXd& upPhi, int upLevel, int interpType )
{
	// use the extrinsic formula
	Eigen::SparseMatrix<double> mat;
	std::vector<int> facemap;
	std::vector<std::pair<int, Eigen::Vector3d>> bary;

	Eigen::MatrixXi upF;
	meshUpSampling(V, mesh.faces(), upV, upF, upLevel, &mat, &facemap, &bary);

	getSideVertexPhi(V, mesh, vertOmega, vertPhi, bary, upPhi, interpType);
}

void getSideVertexPhi(const Eigen::MatrixXd& V, const MeshConnectivity& mesh, const Eigen::VectorXd& edgeOmega, const std::vector<std::complex<double>>& vertZvals, Eigen::MatrixXd& upV, const MeshConnectivity& upMesh, Eigen::VectorXd& upPhi, int upLevel, int interpType)
{
    // use the intrinsic formula
    Eigen::SparseMatrix<double> mat;
    std::vector<int> facemap;
    std::vector<std::pair<int, Eigen::Vector3d>> bary;

    Eigen::MatrixXi upF;
    meshUpSampling(V, mesh.faces(), upV, upF, upLevel, &mat, &facemap, &bary);

    getSideVertexPhi(V, mesh, edgeOmega, vertZvals, bary, upPhi, interpType);
}

void getClouhTocherPhi(const Eigen::MatrixXd& V, const MeshConnectivity& mesh, const Eigen::VectorXd& edgeOmega, const std::vector<std::complex<double>>& vertZvals, Eigen::MatrixXd& upV, const MeshConnectivity& upMesh, Eigen::VectorXd& upPhi, int upLevel)
{
	// use the intrinsic formula
	Eigen::SparseMatrix<double> mat;
	std::vector<int> facemap;
	std::vector<std::pair<int, Eigen::Vector3d>> bary;

	Eigen::MatrixXi upF;
	meshUpSampling(V, mesh.faces(), upV, upF, upLevel, &mat, &facemap, &bary);

	getClouhTocherPhi(V, mesh, edgeOmega, vertZvals, bary, upPhi);
}


void getSideVertexAmp(const Eigen::MatrixXd& V, const MeshConnectivity& mesh, const std::vector<std::complex<double>>& vertZvals, const std::vector<std::pair<int, Eigen::Vector3d>>& bary, Eigen::VectorXd& upAmp, int interpType)
{
	int nUpVerts = bary.size();
	upAmp.resize(nUpVerts);

	for (int i = 0; i < nUpVerts; i++)
	{
		int fid = bary[i].first;
		Eigen::Vector3d bcoord = bary[i].second;
		std::vector<Eigen::Vector3d> tri;
		tri.push_back(V.row(mesh.faceVertex(fid, 0)));
		tri.push_back(V.row(mesh.faceVertex(fid, 1)));
		tri.push_back(V.row(mesh.faceVertex(fid, 2)));

		std::vector<double> triEdge1Form(3) = {0, 0, 0};
		std::vector<double> vertVals(3);
		for (int j = 0; j < 3; j++)
		{
			int eid = mesh.faceEdge(fid, j);
			vertVals[j] = std::abs(vertZvals[mesh.faceVertex(fid, j)]);
		}


		if (interpType == 0)
		{
			upAmp[i] = intrinsicLinearSideVertexInterpolation<double>(vertVals, bcoord);
		}
		else if (interpType == 1)
		{
			upAmp[i] = intrinsicCubicSideVertexInterpolation<double>(vertVals, triEdge1Form, tri, bcoord);
		}
		else
		{
			upAmp[i] = intrinsicWojtanSideVertexInterpolation<double>(vertVals, triEdge1Form, tri, bcoord);
		}
	}
}

void getClouhTocherAmp(const Eigen::MatrixXd& V, const MeshConnectivity& mesh, const std::vector<std::complex<double>>& vertZvals, const std::vector<std::pair<int, Eigen::Vector3d>>& bary, Eigen::VectorXd& upAmp)
{
	int nUpVerts = bary.size();
	upPhi.resize(nUpVerts);

	for (int i = 0; i < nUpVerts; i++)
	{
		int fid = bary[i].first;
		Eigen::Vector3d bcoord = bary[i].second;
		std::vector<Eigen::Vector3d> tri;
		tri.push_back(V.row(mesh.faceVertex(fid, 0)));
		tri.push_back(V.row(mesh.faceVertex(fid, 1)));
		tri.push_back(V.row(mesh.faceVertex(fid, 2)));

		std::vector<double> triEdge1Form = {0, 0, 0};
		std::vector<double> vertVals(3);
		for (int j = 0; j < 3; j++)
		{
			vertVals[j] = std::abs(vertZvals[mesh.faceVertex(fid, j)]);
		}


		upAmp[i] = intrinsicClouthTocherInterpolation<double>(vertVals, triEdge1Form, tri, bcoord);
	}
}