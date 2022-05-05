#pragma once
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include "MeshLib/MeshConnectivity.h"
#include "../dep/SecStencils/Mesh.h"

#ifndef GRAIN_SIZE
#define GRAIN_SIZE 10
#endif

enum KnoppelModelType {
	W_WTar = 0,
	Z_WTar = 1,
	WZ_Tar = 2,
	Z_W = 3
};

struct QuadraturePoints
{
    double u;
    double v;
    double weight;
};

struct RotateVertexInfo
{
	int vid;
	double angle;
};

enum VecMotionType
{
    Rotate = 0,
    Tilt = 1,
    Enlarge = 2,
    None = 3
};

enum InitializationType
{
    Linear = 0,
    Zuenko = 1, 
    Knoppel = 2
};

struct VertexOpInfo
{
    VecMotionType vecOptType = None;
    bool isMagOptCoupled = false;
    double vecOptValue = 0;
    double vecMagValue = 1;
};


std::vector<QuadraturePoints> buildQuadraturePoints(int order); // this is based one the paper: http://lsec.cc.ac.cn/~tcui/myinfo/paper/quad.pdf and the corresponding source codes: http://lsec.cc.ac.cn/phg/download.htm (quad.c)

Eigen::Vector3d computeHatWeight(double u, double v);

Eigen::MatrixXd SPDProjection(Eigen::MatrixXd A);

Eigen::VectorXd vertexVec2IntrinsicVec(const Eigen::MatrixXd& v, const Eigen::MatrixXd& pos, const MeshConnectivity& mesh);
Eigen::MatrixXd vertexVec2IntrinsicHalfEdgeVec(const Eigen::MatrixXd& v, const Eigen::MatrixXd& pos, const MeshConnectivity& mesh);
Eigen::MatrixXd intrinsicHalfEdgeVec2VertexVec(const Eigen::MatrixXd& v, const Eigen::MatrixXd& pos, const MeshConnectivity& mesh);

Eigen::MatrixXd intrinsicEdgeVec2FaceVec(const Eigen::VectorXd& v, const Eigen::MatrixXd& pos, const MeshConnectivity& mesh);
Eigen::MatrixXd intrinsicHalfEdgeVec2FaceVec(const Eigen::MatrixXd& v, const Eigen::MatrixXd& pos, const MeshConnectivity& mesh);

void combField(const Eigen::MatrixXi& F, const Eigen::MatrixXd& w, Eigen::MatrixXd& combedW);
double unitMagEnergy(const std::vector<std::complex<double>>& zvals, Eigen::VectorXd* deriv, std::vector<Eigen::Triplet<double>>* hess, bool isProj = false);

Eigen::Vector3d rotateSingleVector(const Eigen::Vector3d& vec, const Eigen::Vector3d& axis, double angle);
void rotateIntrinsicVector(const Eigen::MatrixXd& V, const MeshConnectivity& mesh, const Eigen::MatrixXd& halfEdgeW, const std::vector<RotateVertexInfo>& rotVerts, Eigen::MatrixXd& rotHalfEdgeW);

void buildVertexNeighboringInfo(const MeshConnectivity& mesh, int nverts, std::vector<std::vector<int>>& vertNeiEdges, std::vector<std::vector<int>>& vertNeiFaces);

Eigen::SparseMatrix<double> buildD0(const MeshConnectivity& mesh, int nverts);

Eigen::VectorXd getFaceArea(const Eigen::MatrixXd& V, const MeshConnectivity& mesh);
Eigen::VectorXd getEdgeArea(const Eigen::MatrixXd& V, const MeshConnectivity& mesh);
Eigen::VectorXd getVertArea(const Eigen::MatrixXd& V, const MeshConnectivity& mesh);

void laplacianSmoothing(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, Eigen::MatrixXd& newV, double smoothingRatio = 0.95, int opTimes = 3);

void laplacianSmoothing(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, const Eigen::VectorXd& oldData, Eigen::VectorXd& newData, double smoothingRatio = 0.95, int opTimes = 3);


void curvedPNTriangleUpsampling(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, const Eigen::MatrixXd& VN, const std::vector<std::pair<int, Eigen::Vector3d>>& baryList, Eigen::MatrixXd& NV, Eigen::MatrixXd& newVN);

void getWrinkledMesh(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, const std::vector<std::complex<double>>& zvals, std::vector<std::vector<int>> *vertNeiFaces, Eigen::MatrixXd& wrinkledV, double scaleRatio, bool isTangentCorrection);

void computeBaryGradient(const Eigen::Vector3d& P0, const Eigen::Vector3d& P1, const Eigen::Vector3d& P2, const Eigen::Vector3d& bary, Eigen::Matrix3d& baryGrad);

void mkdir(const std::string& foldername);
