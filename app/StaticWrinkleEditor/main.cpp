#include "polyscope/polyscope.h"
#include "polyscope/pick.h"

#include <igl/invert_diag.h>
#include <igl/per_vertex_normals.h>
#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>
#include <igl/boundary_loop.h>
#include <igl/doublearea.h>
#include <igl/loop.h>
#include <igl/file_dialog_open.h>
#include <igl/file_dialog_save.h>
#include <igl/cotmatrix_entries.h>
#include "polyscope/messages.h"
#include "polyscope/point_cloud.h"
#include "polyscope/surface_mesh.h"
#include "polyscope/view.h"

#include <iostream>
#include <filesystem>
#include <utility>

#include "../../include/CommonTools.h"
#include "../../include/MeshLib/MeshUpsampling.h"
#include "../../include/Visualization/PaintGeometry.h"
#include "../../include/Optimization/NewtonDescent.h"
#include "../../include/IntrinsicFormula/InterpolateZvals.h"
#include "../../include/IntrinsicFormula/WrinkleEditingStaticEdgeModel.h"
#include "../../include/WrinkleFieldsEditor.h"
#include "../../include/SpherigonSmoothing.h"
#include "../../dep/SecStencils/types.h"
#include "../../dep/SecStencils/Subd.h"
#include "../../dep/SecStencils/utils.h"

#include "../../include/json.hpp"
#include "../../include/ComplexLoop.h"
#include "../../include/MeshLib/RegionEdition.h"

enum RegionOpType
{
	Dilation = 0,
	Erosion = 1
};

std::vector<VertexOpInfo> vertOpts;

Eigen::MatrixXd triV, loopTriV, upsampledTriV;
Eigen::MatrixXi triF, loopTriF, upsampledTriF;
MeshConnectivity triMesh;
Mesh secMesh, subSecMesh;
std::vector<std::pair<int, Eigen::Vector3d>> bary;

Eigen::VectorXd initAmp;
Eigen::VectorXd initOmega;

std::vector<Eigen::VectorXd> omegaList;
std::vector<Eigen::MatrixXd> faceOmegaList;
std::vector<std::vector<std::complex<double>>> zList;

std::vector<Eigen::VectorXd> subOmegaList;
std::vector<Eigen::MatrixXd> subFaceOmegaList;

std::vector<Eigen::VectorXd> phaseFieldsList;
std::vector<Eigen::VectorXd> ampFieldsList;

// reference amp and omega
std::vector<Eigen::VectorXd> refOmegaList;
std::vector<Eigen::VectorXd> refAmpList;

Eigen::MatrixXd dataV;
Eigen::MatrixXi dataF;
Eigen::MatrixXd dataVec;
Eigen::MatrixXd curColor;

int loopLevel = 1;
int upsampleTimes = 2;

bool isForceOptimize = false;
bool isShowVectorFields = true;
bool isShowWrinkels = true;

PaintGeometry mPaint;

int numFrames = 20;
int curFrame = 0;

double globalAmpMax = 1;
double globalAmpMin = 0;

double dragSpeed = 0.5;

float vecratio = 0.001;

double gradTol = 1e-6;
double xTol = 0;
double fTol = 0;
int numIter = 1000;
int quadOrder = 4;
float wrinkleAmpScalingRatio = 1;

double spatialAmpRatio = 1;
double spatialEdgeRatio = 1;
double spatialKnoppelRatio = 1;

std::string workingFolder;

IntrinsicFormula::WrinkleEditingStaticEdgeModel editModel;
VecMotionType selectedMotion = Enlarge;
bool isSelectAll = false;
double selectedMotionValue = 2;
double selectedMagValue = 1;
bool isCoupled = false;

Eigen::VectorXi initSelectedFids;
Eigen::VectorXi selectedFids;
Eigen::VectorXi faceFlags;

RegionOpType regOpType = Dilation;
int optTimes = 5;

bool isLoadOpt;

int clickedFid = -1;
int dilationTimes = 10;

bool isShowWrinkleColorField = false;
bool isWarmStart = false;

// smoothing
int smoothingTimes = 3;
double smoothingRatio = 0.95;


std::map<std::pair<int, int>, int> he2Edge(const Eigen::MatrixXi& faces)
{
	std::map< std::pair<int, int>, int > heToEdge;
	std::vector< std::vector<int> > edgeToVert;
	for (int face = 0; face < faces.rows(); ++face)
	{
		for (int i = 0; i < 3; ++i)
		{
			int vi = faces(face, i);
			int vj = faces(face, (i + 1) % 3);
			assert(vi != vj);

			std::pair<int, int> he = std::make_pair(vi, vj);
			if (he.first > he.second) std::swap(he.first, he.second);
			if (heToEdge.find(he) != heToEdge.end()) continue;

			heToEdge[he] = edgeToVert.size();
			edgeToVert.push_back(std::vector<int>(2));
			edgeToVert.back()[0] = he.first;
			edgeToVert.back()[1] = he.second;
		}
	}
	return heToEdge;
}

std::map<std::pair<int, int>, int> he2Edge(const std::vector< std::vector<int>>& edgeToVert)
{
	std::map< std::pair<int, int>, int > heToEdge;
	for (int i = 0; i < edgeToVert.size(); i++)
	{
		std::pair<int, int> he = std::make_pair(edgeToVert[i][0], edgeToVert[i][1]);
		heToEdge[he] = i;
	}
	return heToEdge;
}

Eigen::VectorXd swapEdgeVec(const std::vector< std::vector<int>>& edgeToVert, const Eigen::VectorXd& edgeVec, int flag)
{
	Eigen::VectorXd  edgeVecSwap = edgeVec;
	std::map< std::pair<int, int>, int > heToEdge = he2Edge(edgeToVert);

	int idx = 0;
	for (auto it : heToEdge)
	{
		if (flag == 0)   // ours to secstencils
			edgeVecSwap(it.second) = edgeVec(idx);
		else
			edgeVecSwap(idx) = edgeVec(it.second);
		idx++;
	}
	return edgeVecSwap;
}

Eigen::VectorXd swapEdgeVec(const Eigen::MatrixXi& faces, const Eigen::VectorXd& edgeVec, int flag)
{
	Eigen::VectorXd  edgeVecSwap = edgeVec;
	std::map< std::pair<int, int>, int > heToEdge = he2Edge(faces);

	int idx = 0;
	for (auto it : heToEdge)
	{
		if (flag == 0)   // ours to secstencils
			edgeVecSwap(it.second) = edgeVec(idx);
		else
			edgeVecSwap(idx) = edgeVec(it.second);
		idx++;
	}
	return edgeVecSwap;
}

std::vector<std::vector<int>> swapEdgeIndices(const Eigen::MatrixXi& faces, const std::vector<std::vector<int>>& edgeIndices, int flag)
{
	std::vector<std::vector<int>> edgeIndicesSwap = edgeIndices;
	std::map< std::pair<int, int>, int > heToEdge = he2Edge(faces);

	int idx = 0;
	for (auto it : heToEdge)
	{
		if (flag == 0)   // ours to secstencils
		{
			edgeIndicesSwap[it.second] = edgeIndices[idx];
		}
		else
		{
			edgeIndicesSwap[idx] = edgeIndices[it.second];
		}
		idx++;
	}

	return edgeIndicesSwap;
}

Eigen::MatrixXd edgeVec2FaceVec(const Mesh& mesh, Eigen::VectorXd& edgeVec)
{
	int nfaces = mesh.GetFaceCount();
	int nedges = mesh.GetEdgeCount();
	Eigen::MatrixXd fVec(nfaces, 3);
	fVec.setZero();

	for (int f = 0; f < nfaces; f++)
	{
		std::vector<int> faceEdges = mesh.GetFaceEdges(f);
		std::vector<int> faceVerts = mesh.GetFaceVerts(f);
		for (int j = 0; j < 3; j++)
		{
			int vid = faceVerts[j];
			int eid0 = faceEdges[j];
			int eid1 = faceEdges[(j + 2) % 3];

			Eigen::Vector3d e0 = mesh.GetVertPos(faceVerts[(j + 1) % 3]) - mesh.GetVertPos(vid);
			Eigen::Vector3d e1 = mesh.GetVertPos(faceVerts[(j + 2) % 3]) - mesh.GetVertPos(vid);

			int flag0 = 1, flag1 = 1;
			Eigen::Vector2d rhs;

			if (mesh.GetEdgeVerts(eid0)[0] == vid)
			{
				flag0 = 1;
			}
			else
			{
				flag0 = -1;
			}


			if (mesh.GetEdgeVerts(eid1)[0] == vid)
			{
				flag1 = 1;
			}
			else
			{
				flag1 = -1;
			}
			rhs(0) = flag0 * edgeVec(eid0);
			rhs(1) = flag1 * edgeVec(eid1);

			Eigen::Matrix2d I;
			I << e0.dot(e0), e0.dot(e1), e1.dot(e0), e1.dot(e1);
			Eigen::Vector2d sol = I.inverse() * rhs;

			fVec.row(f) += (sol(0) * e0 + sol(1) * e1) / 3;
		}
	}
	return fVec;
}

bool loadEdgeOmega(const std::string& filename, const int& nlines, Eigen::VectorXd& edgeOmega)
{
	std::ifstream infile(filename);
	if (!infile)
	{
		std::cerr << "invalid edge omega file name" << std::endl;
		return false;
	}
	else
	{
		Eigen::MatrixXd halfEdgeOmega(nlines, 2);
		edgeOmega.setZero(nlines);
		for (int i = 0; i < nlines; i++)
		{
			std::string line;
			std::getline(infile, line);
			std::stringstream ss(line);

			std::string x, y;
			ss >> x;
			if (!ss)
				return false;
			ss >> y;
			if (!ss)
			{
				halfEdgeOmega.row(i) << std::stod(x), -std::stod(x);
			}
			else
				halfEdgeOmega.row(i) << std::stod(x), std::stod(y);
		}
		edgeOmega = (halfEdgeOmega.col(0) - halfEdgeOmega.col(1)) / 2;
	}
	return true;
}

bool loadVertexZvals(const std::string& filePath, const int& nlines, std::vector<std::complex<double>>& zvals)
{
	std::ifstream zfs(filePath);
	if (!zfs)
	{
		std::cerr << "invalid zvals file name" << std::endl;
		return false;
	}

	zvals.resize(nlines);

	for (int j = 0; j < nlines; j++) {
		std::string line;
		std::getline(zfs, line);
		std::stringstream ss(line);
		std::string x, y;
		ss >> x;
		ss >> y;
		zvals[j] = std::complex<double>(std::stod(x), std::stod(y));
	}
	return true;
}

bool loadVertexAmp(const std::string& filePath, const int& nlines, Eigen::VectorXd& amp)
{
	std::ifstream afs(filePath);

	if (!afs)
	{
		std::cerr << "invalid ref amp file name" << std::endl;
		return false;
	}

	amp.setZero(nlines);

	for (int j = 0; j < nlines; j++)
	{
		std::string line;
		std::getline(afs, line);
		std::stringstream ss(line);
		std::string x;
		ss >> x;
		if (!ss)
			return false;
		amp(j) = std::stod(x);
	}
	return true;
}

void getSelecteFids()
{
	if (isSelectAll)
	{
		selectedFids.setOnes(triMesh.nFaces());
		initSelectedFids = selectedFids;
		return;
	}

	selectedFids.setZero();

	if (clickedFid == -1)
		return;
	else
	{
		selectedFids(clickedFid) = 1;
		initSelectedFids = selectedFids;

		RegionEdition regEdt = RegionEdition(triMesh);

		for (int i = 0; i < dilationTimes; i++)
		{
			regEdt.faceDilation(initSelectedFids, selectedFids);
			initSelectedFids = selectedFids;
		}
	}
	initSelectedFids = selectedFids;
}

double sampling(double t, double offset, double A, double mu, double sigma)
{
	return offset + A * std::exp(-0.5 * (t - mu) * (t - mu) / sigma / sigma);
}

void buildWrinkleMotions()
{
	int nverts = triV.rows();
	Eigen::VectorXi initSelectedVids;

	faceFlags2VertFlags(triMesh, nverts, initSelectedFids, initSelectedVids);

	int nselectedV = 0;
	for (int i = 0; i < nverts; i++)
		if (initSelectedVids(i))
			nselectedV++;
	std::cout << "num of selected vertices: " << nselectedV << std::endl;

	vertOpts.clear();
	vertOpts.resize(nverts, { None, isCoupled, 0, 1 });

	for (int i = 0; i < nverts; i++)
	{
		if (initSelectedVids(i))
			vertOpts[i] = { selectedMotion, isCoupled, selectedMotionValue, selectedMagValue };
	}

}

void updateMagnitudePhase(const std::vector<Eigen::VectorXd>& wFrames, const std::vector<std::vector<std::complex<double>>>& zFrames, std::vector<Eigen::VectorXd>& magList, std::vector<Eigen::VectorXd>& phaseList)
{
	std::vector<std::vector<std::complex<double>>> interpZList(wFrames.size());
	magList.resize(wFrames.size());
	phaseList.resize(wFrames.size());

	subOmegaList.resize(wFrames.size());
	subFaceOmegaList.resize(wFrames.size());

	MeshConnectivity mesh(triF);

	auto computeMagPhase = [&](const tbb::blocked_range<uint32_t>& range)
	{
		for (uint32_t i = range.begin(); i < range.end(); ++i)
		{
			interpZList[i] = IntrinsicFormula::upsamplingZvals(mesh, zFrames[i], wFrames[i], bary);
			magList[i].setZero(interpZList[i].size());
			phaseList[i].setZero(interpZList[i].size());

			for (int j = 0; j < magList[i].size(); j++)
			{
				magList[i](j) = std::abs(interpZList[i][j]);
				phaseList[i](j) = std::arg(interpZList[i][j]);
			}
		}
	};

	tbb::blocked_range<uint32_t> rangex(0u, (uint32_t)interpZList.size(), GRAIN_SIZE);
	tbb::parallel_for(rangex, computeMagPhase);

	/*
	auto computeMagPhase = [&](const tbb::blocked_range<uint32_t> &range)
	{
		for (uint32_t i = range.begin(); i < range.end(); ++i)
		{
			Eigen::VectorXd edgeVec = wFrames[i];
			edgeVec = swapEdgeVec(triF, edgeVec, 0);

			Mesh tmpMesh;

			Subdivide(secMesh, edgeVec, zFrames[i], subOmegaList[i], interpZList[i], loopLevel, tmpMesh);

			std::vector<std::vector<int>> edge2Verts;

			for (int e = 0; e < tmpMesh.GetEdgeCount(); e++)
			{
				edge2Verts.push_back({ tmpMesh.GetEdgeVerts(e)[0], tmpMesh.GetEdgeVerts(e)[1] });
			}
			edgeVec = swapEdgeVec(edge2Verts, subOmegaList[i], 1);

			MeshConnectivity loopMesh = MeshConnectivity(loopTriF);

			subFaceOmegaList[i] = intrinsicEdgeVec2FaceVec(edgeVec, loopTriV, loopMesh);

			interpZList[i] = IntrinsicFormula::upsamplingZvals(loopMesh, interpZList[i], edgeVec, bary);



			magList[i].setZero(interpZList[i].size());
			phaseList[i].setZero(interpZList[i].size());

			Eigen::MatrixXd faceVec = edgeVec2FaceVec(tmpMesh, subOmegaList[i]);

			subFaceOmegaList[i] = faceVec;

			for (int j = 0; j < magList[i].size(); j++)
			{
				magList[i](j) = std::abs(interpZList[i][j]);
				phaseList[i](j) = std::arg(interpZList[i][j]);
			}
		}
	};

	tbb::blocked_range<uint32_t> rangex(0u, (uint32_t) interpZList.size(), GRAIN_SIZE);
	tbb::parallel_for(rangex, computeMagPhase);
	*/

	/*for (uint32_t i = 0; i < interpZList.size(); ++i)
	{
		Eigen::VectorXd edgeVec = wFrames[i];
		edgeVec = swapEdgeVec(triF, edgeVec, 0);

		Mesh tmpMesh;

		if (loopLevel <= 1)
			Subdivide(secMesh, edgeVec, zFrames[i], subOmegaList[i], interpZList[i], loopLevel, tmpMesh);

		else
		{
			Subdivide(secMesh, edgeVec, zFrames[i], subOmegaList[i], interpZList[i], 1, tmpMesh);

			std::vector<std::vector<int>> edge2Verts;

			for (int e = 0; e < tmpMesh.GetEdgeCount(); e++)
			{
				edge2Verts.push_back({ tmpMesh.GetEdgeVerts(e)[0], tmpMesh.GetEdgeVerts(e)[1] });
			}
			edgeVec = swapEdgeVec(edge2Verts, subOmegaList[i], 1);

			MeshConnectivity loopMesh = MeshConnectivity(loopTriF);

			subFaceOmegaList[i] = intrinsicEdgeVec2FaceVec(edgeVec, loopTriV, loopMesh);

			interpZList[i] = IntrinsicFormula::upsamplingZvals(loopMesh, interpZList[i], edgeVec, bary);

		}



		magList[i].setZero(interpZList[i].size());
		phaseList[i].setZero(interpZList[i].size());

		Eigen::MatrixXd faceVec = edgeVec2FaceVec(tmpMesh, subOmegaList[i]);

		subFaceOmegaList[i] = faceVec;

		for (int j = 0; j < magList[i].size(); j++)
		{
			magList[i](j) = std::abs(interpZList[i][j]);
			phaseList[i](j) = std::arg(interpZList[i][j]);
		}
	}*/


}

void updateSubOmega(const std::vector<Eigen::VectorXd>& wFrames, std::vector<Eigen::VectorXd>& subOmegaList, std::vector<Eigen::MatrixXd>& subFaceOmegaList)
{
	/* std::vector<std::vector<std::complex<double>>> interpZList(wFrames.size());
	 subOmegaList.resize(wFrames.size());
	 subFaceOmegaList.resize(wFrames.size());

	 MeshConnectivity mesh(triF);

	 auto computeMagPhase = [&](const tbb::blocked_range<uint32_t> &range)
	 {
		 for (uint32_t i = range.begin(); i < range.end(); ++i)
		 {
			 Eigen::VectorXd edgeVec = wFrames[i];

			 edgeVec = swapEdgeVec(triF, edgeVec, 0);
			 edgeVec = loopMatOneForm * edgeVec;

			 subOmegaList[i] = edgeVec;
			 Eigen::MatrixXd faceVec = edgeVec2FaceVec(subSecMesh, edgeVec);

			 subFaceOmegaList[i] = faceVec;
		 }
	 };

	 tbb::blocked_range<uint32_t> rangex(0u, (uint32_t) interpZList.size(), GRAIN_SIZE);
	 tbb::parallel_for(rangex, computeMagPhase);*/
}


void initialization(const Eigen::MatrixXd& triV, const Eigen::MatrixXi& triF, Eigen::MatrixXd& upsampledTriV, Eigen::MatrixXi& upsampledTriF)
{
	Eigen::SparseMatrix<double> S;
	std::vector<int> facemap;

	triMesh = MeshConnectivity(triF);

	loopTriV = triV;
	loopTriF = triF;

	std::vector<Eigen::Vector3d> pos;
	std::vector<std::vector<int>> faces;

	pos.resize(triV.rows());
	for (int i = 0; i < triV.rows(); i++)
	{
		pos[i] = triV.row(i);
	}

	faces.resize(triF.rows());
	for (int i = 0; i < triF.rows(); i++)
	{
		faces[i] = { triF(i, 0), triF(i, 1), triF(i, 2) };
	}

	secMesh.Populate(pos, faces);
	subSecMesh = secMesh;

	if (loopLevel > 0)
	{
		Subd* subd = ChooseSubdivisionScheme(subSecMesh, false);

		Subdivide(subSecMesh, loopLevel, subd, 0);

		subSecMesh.GetPos(loopTriV);
		loopTriF.resize(subSecMesh.GetFaceCount(), 3);
		for (int i = 0; i < loopTriF.rows(); i++)
		{
			for (int j = 0; j < 3; j++)
				loopTriF(i, j) = subSecMesh.GetFaceVerts(i)[j];
		}
		meshUpSampling(loopTriV, loopTriF, upsampledTriV, upsampledTriF, upsampleTimes - loopLevel, NULL, NULL, &bary);
	}
	else
		meshUpSampling(loopTriV, loopTriF, upsampledTriV, upsampledTriF, upsampleTimes, NULL, NULL, &bary);


	selectedFids.setZero(triMesh.nFaces());
	initSelectedFids = selectedFids;

}


void updateEditionDomain()
{
	getSelecteFids();
	RegionEdition regOpt(triMesh);
	selectedFids = initSelectedFids;

	int nselected0 = 0;
	for (int i = 0; i < initSelectedFids.rows(); i++)
	{
		if (initSelectedFids(i) == 1)
		{
			nselected0++;
		}
	}

	for (int i = 0; i < optTimes; i++)
	{
		std::cout << "dilation option to get interface, step: " << i << std::endl;
		Eigen::VectorXi selectedFidNew;
		if (regOpType == Dilation)
			regOpt.faceDilation(selectedFids, selectedFidNew);

		else
			regOpt.faceErosion(selectedFids, selectedFidNew);


		selectedFids = selectedFidNew;
	}
	faceFlags = initSelectedFids - selectedFids;

	int nselected = 0;
	for (int i = 0; i < selectedFids.rows(); i++)
	{
		if (selectedFids(i) == 1)
		{
			nselected++;
		}
	}

	int ninterfaces = 0;
	for (int i = 0; i < faceFlags.rows(); i++)
	{
		if (faceFlags(i) == -1)
			ninterfaces++;
	}
	std::cout << "initial selected faces: " << nselected0 << ", selected faces: " << nselected << ", num of interfaces: " << nselected - nselected0 << " " << ninterfaces << std::endl;

	std::cout << "build wrinkle motions. " << std::endl;
	buildWrinkleMotions();

}

void updatePaintingItems()
{
	// get interploated amp and phase frames
	std::cout << "compute upsampled phase: " << std::endl;
	updateMagnitudePhase(omegaList, zList, ampFieldsList, phaseFieldsList);
	std::cout << "compute upsampled omega: " << std::endl;
	updateSubOmega(omegaList, subOmegaList, subFaceOmegaList);


	std::cout << "compute face vector fields:" << std::endl;
	faceOmegaList.resize(omegaList.size());
	for (int i = 0; i < omegaList.size(); i++)
	{
		faceOmegaList[i] = intrinsicEdgeVec2FaceVec(omegaList[i], triV, triMesh);
	}


	// update global maximum amplitude
	std::cout << "update max and min amp. " << std::endl;

	globalAmpMax = std::max(ampFieldsList[0].maxCoeff(), editModel.getRefAmpList()[0].maxCoeff());
	globalAmpMin = std::min(ampFieldsList[0].minCoeff(), editModel.getRefAmpList()[0].minCoeff());
	for (int i = 1; i < ampFieldsList.size(); i++)
	{
		globalAmpMax = std::max(globalAmpMax, std::max(ampFieldsList[i].maxCoeff(), editModel.getRefAmpList()[i].maxCoeff()));
		globalAmpMin = std::min(globalAmpMin, std::min(ampFieldsList[i].minCoeff(), editModel.getRefAmpList()[i].minCoeff()));
	}

	std::cout << "start to update viewer." << std::endl;
}





void solveKeyFrames(const Eigen::VectorXd& initAmp, const Eigen::VectorXd& initOmega, const Eigen::VectorXi& faceFlags, std::vector<Eigen::VectorXd>& wFrames, std::vector<std::vector<std::complex<double>>>& zFrames)
{
	editModel = IntrinsicFormula::WrinkleEditingStaticEdgeModel(triV, triMesh, vertOpts, faceFlags, quadOrder, spatialAmpRatio, spatialEdgeRatio, spatialKnoppelRatio);

	editModel.initialization(initAmp, initOmega, numFrames - 2);

	std::cout << "initilization finished!" << std::endl;
	Eigen::VectorXd x;
	std::cout << "convert list to variable." << std::endl;
	editModel.convertList2Variable(x);

	refOmegaList = editModel.getRefWList();
	refAmpList = editModel.getRefAmpList();


	if (isForceOptimize)
	{
		if (isWarmStart)
			editModel.warmstart();

		else
		{

			auto funVal = [&](const Eigen::VectorXd& x, Eigen::VectorXd* grad, Eigen::SparseMatrix<double>* hess, bool isProj) {
				Eigen::VectorXd deriv;
				Eigen::SparseMatrix<double> H;
				double E = editModel.computeEnergy(x, grad ? &deriv : NULL, hess ? &H : NULL, isProj);

				if (grad)
				{
					(*grad) = deriv;
				}

				if (hess)
				{
					(*hess) = H;
				}

				return E;
			};
			auto maxStep = [&](const Eigen::VectorXd& x, const Eigen::VectorXd& dir) {
				return 1.0;
			};

			auto getVecNorm = [&](const Eigen::VectorXd& x, double& znorm, double& wnorm) {
				editModel.getComponentNorm(x, znorm, wnorm);
			};



			OptSolver::testFuncGradHessian(funVal, x);

			auto x0 = x;
			OptSolver::newtonSolver(funVal, maxStep, x, numIter, gradTol, xTol, fTol, true, getVecNorm, &workingFolder);
			std::cout << "before optimization: " << x0.norm() << ", after optimization: " << x.norm() << std::endl;
		}
	}
	std::cout << "convert variable to list." << std::endl;
	editModel.convertVariable2List(x);
	std::cout << "get w list" << std::endl;
	wFrames = editModel.getWList();
	std::cout << "get z list" << std::endl;
	zFrames = editModel.getVertValsList();
}



void registerMeshByPart(const Eigen::MatrixXd& basePos, const Eigen::MatrixXi& baseF,
	const Eigen::MatrixXd& upPos, const Eigen::MatrixXi& upF, const double& shiftz, const double& ampMin,
	const double& ampMax,
	const Eigen::VectorXd ampVec, const Eigen::VectorXd& phaseVec, const Eigen::MatrixXd& omegaVec,
	const Eigen::VectorXd& refAmp, const Eigen::MatrixXd& refOmega, const Eigen::VectorXi& vertFlag,
	Eigen::MatrixXd& renderV, Eigen::MatrixXi& renderF, Eigen::MatrixXd& renderVec, Eigen::MatrixXd& renderColor)
{
	int nverts = basePos.rows();
	int nfaces = baseF.rows();

	int nupverts = upPos.rows();
	int nupfaces = upF.rows();

	int ndataVerts = 2 * nverts + 2 * nupverts;
	int ndataFaces = 2 * nfaces + 2 * nupfaces;

	Eigen::MatrixXd smoothPos;
	Eigen::MatrixXi smoothF;
	Eigen::SparseMatrix<double> S;

	if (!isShowVectorFields)
	{
		ndataVerts = 2 * nupverts + nverts;
		ndataFaces = 2 * nupfaces + nverts;
	}
	if (isShowWrinkels)
	{
		ndataVerts += nupverts;
		ndataFaces += nupfaces;
		//        loopUpsampling(upPos, upF, smoothPos, smoothF, 2, &S);
		//        igl::loop(upPos, upF, smoothPos, smoothF, 2);
		//
		//        ndataVerts += smoothPos.rows();
		//        ndataFaces += smoothF.rows();
	}
	std::cout << "num of vertices: " << ndataVerts << ", num of faces: " << ndataFaces << std::endl;

	double shiftx = 1.5 * (basePos.col(0).maxCoeff() - basePos.col(0).minCoeff());

	renderV.resize(ndataVerts, 3);
	renderVec.setZero(ndataFaces, 3);
	renderF.resize(ndataFaces, 3);
	renderColor.setZero(ndataVerts, 3);

	renderColor.col(0).setConstant(1.0);
	renderColor.col(1).setConstant(1.0);
	renderColor.col(2).setConstant(1.0);

	int curVerts = 0;
	int curFaces = 0;

	Eigen::MatrixXd shiftV = basePos;
	shiftV.col(0).setConstant(-shiftx);
	shiftV.col(1).setConstant(0);
	shiftV.col(2).setConstant(shiftz);

	for (int i = 0; i < nverts; i++)
	{
		if (vertFlag(i) == 1)
			renderColor.row(i) << 1.0, 0, 0;
		else if (vertFlag(i) == -1)
			renderColor.row(i) << 0, 1.0, 0;
	}
	renderV.block(curVerts, 0, nverts, 3) = basePos - shiftV;
	renderF.block(curFaces, 0, nfaces, 3) = baseF;
	if (isShowVectorFields)
	{
		for (int i = 0; i < nfaces; i++)
			renderVec.row(i + curFaces) = omegaVec.row(i);
	}

	curVerts += nverts;
	curFaces += nfaces;


	Eigen::MatrixXi shiftF = baseF;
	shiftF.setConstant(curVerts);
	shiftV.col(0).setConstant(0);
	shiftV.col(1).setConstant(0);
	shiftV.col(2).setConstant(shiftz);
	mPaint.setNormalization(false);
	Eigen::VectorXd normoalizedRefAmpVec = refAmp;
	for (int i = 0; i < normoalizedRefAmpVec.rows(); i++)
	{
		normoalizedRefAmpVec(i) = (refAmp(i) - ampMin) / (ampMax - ampMin);
	}
	std::cout << "ref amp (min, max): " << refAmp.minCoeff() << " " << refAmp.maxCoeff() << std::endl;
	Eigen::MatrixXd refColor = mPaint.paintAmplitude(normoalizedRefAmpVec);
	renderColor.block(curVerts, 0, nverts, 3) = refColor;
	renderV.block(curVerts, 0, nverts, 3) = basePos - shiftV;
	renderF.block(curFaces, 0, nfaces, 3) = baseF + shiftF;

	if (isShowVectorFields)
	{
		for (int i = 0; i < nfaces; i++)
			renderVec.row(i + curFaces) = refOmega.row(i);
	}
	curVerts += nverts;
	curFaces += nfaces;



	// interpolated amp
	shiftV = upPos;
	shiftV.col(0).setConstant(shiftx);
	shiftV.col(1).setConstant(0);
	shiftV.col(2).setConstant(shiftz);


	shiftF = upF;
	shiftF.setConstant(curVerts);

	// interpolated phase
	renderV.block(curVerts, 0, nupverts, 3) = upPos - shiftV;
	renderF.block(curFaces, 0, nupfaces, 3) = upF + shiftF;

	mPaint.setNormalization(false);
	Eigen::MatrixXd phiColor = mPaint.paintPhi(phaseVec);
	renderColor.block(curVerts, 0, nupverts, 3) = phiColor;

	curVerts += nupverts;
	curFaces += nupfaces;

	// interpolated amp
	shiftF.setConstant(curVerts);
	shiftV.col(0).setConstant(2 * shiftx);
	renderV.block(curVerts, 0, nupverts, 3) = upPos - shiftV;
	renderF.block(curFaces, 0, nupfaces, 3) = upF + shiftF;

	mPaint.setNormalization(false);
	Eigen::VectorXd normoalizedAmpVec = ampVec;
	for (int i = 0; i < normoalizedAmpVec.rows(); i++)
	{
		normoalizedAmpVec(i) = (ampVec(i) - ampMin) / (ampMax - ampMin);
	}
	std::cout << "amp (min, max): " << ampVec.minCoeff() << " " << ampVec.maxCoeff() << std::endl;
	Eigen::MatrixXd ampColor = mPaint.paintAmplitude(normoalizedAmpVec);
	renderColor.block(curVerts, 0, nupverts, 3) = ampColor;

	curVerts += nupverts;
	curFaces += nupfaces;

	// interpolated amp
	if (isShowWrinkels)
	{
		std::cout << "show wrinkles, nupverts: " << upPos.rows() << std::endl;
		shiftF.setConstant(curVerts);
		shiftV.col(0).setConstant(3 * shiftx);
		Eigen::MatrixXd tmpV = upPos - shiftV;
		Eigen::MatrixXd tmpN;
		igl::per_vertex_normals(tmpV, upF, tmpN);

		Eigen::MatrixXd wrinkledV = upPos;

		Eigen::VectorXd ampCosVec(nupverts);

		for (int i = 0; i < nupverts; i++)
		{
			wrinkledV.row(i) = upPos.row(i) + wrinkleAmpScalingRatio * ampVec(i) * std::cos(phaseVec(i)) * tmpN.row(i);
			ampCosVec(i) = normoalizedAmpVec(i) * std::cos(phaseVec(i));
		}
        Eigen::MatrixXd wrinkledVNew;
        laplacianSmoothing(wrinkledV, upsampledTriF, wrinkledVNew, smoothingRatio, smoothingTimes);
        std::cout << "smoothing finished" << std::endl;

        for (int i = 0; i < nupverts; i++)
        {
            renderV.row(curVerts + i) = tmpV.row(i) + (wrinkledVNew.row(i) - upsampledTriV.row(i));
        }

        igl::writeOBJ(workingFolder + "wrinkledMesh.obj", wrinkledV, upsampledTriF);
        igl::writeOBJ(workingFolder + "wrinkledMesh_smoothed.obj", wrinkledVNew, upsampledTriF);

		renderF.block(curFaces, 0, nupfaces, 3) = upF + shiftF;

		mPaint.setNormalization(false);
		Eigen::MatrixXd ampCosColor = mPaint.paintAmplitude(ampCosVec);

		if (isShowWrinkleColorField)
			renderColor.block(curVerts, 0, nupverts, 3) = ampCosColor;
		else
		{
			for (int i = 0; i < nupverts; i++)
			{
				renderColor.row(i + curVerts) << 80 / 255.0, 122 / 255.0, 91 / 255.0;
			}
		}


		curVerts += nupverts;
		curFaces += nupfaces;


	}

}

void registerMesh(int frameId)
{
	Eigen::MatrixXd initP, interpP;
	Eigen::MatrixXi initF, interpF;
	Eigen::MatrixXd initVec, interpVec;
	Eigen::MatrixXd initColor, interpColor;

	double shiftz = 1.5 * (triV.col(2).maxCoeff() - triV.col(2).minCoeff());
	int totalfames = ampFieldsList.size();
	Eigen::MatrixXd refFaceOmega = intrinsicEdgeVec2FaceVec(editModel.getRefWList()[frameId], triV, triMesh);

	Eigen::VectorXi selectedVids, initSelectedVids;
	faceFlags2VertFlags(triMesh, triV.rows(), selectedFids, selectedVids);
	faceFlags2VertFlags(triMesh, triV.rows(), initSelectedFids, initSelectedVids);

	for (int i = 0; i < selectedVids.rows(); i++)
	{
		if (selectedVids(i) && !initSelectedVids(i))
			selectedVids(i) = -1;
	}

	registerMeshByPart(triV, triF, upsampledTriV, upsampledTriF, 0, globalAmpMin, globalAmpMax,
		ampFieldsList[frameId], phaseFieldsList[frameId], faceOmegaList[frameId], editModel.getRefAmpList()[frameId], refFaceOmega, selectedVids, interpP, interpF, interpVec, interpColor);

	std::cout << "register mesh finished" << std::endl;

	dataV = interpP;
	curColor = interpColor;
	dataVec = interpVec;
	dataF = interpF;

	polyscope::registerSurfaceMesh("input mesh", dataV, dataF);
}

void updateFieldsInView(int frameId)
{
	std::cout << "update viewer. " << std::endl;
	registerMesh(frameId);
	polyscope::getSurfaceMesh("input mesh")->addVertexColorQuantity("VertexColor", curColor);
	polyscope::getSurfaceMesh("input mesh")->getQuantity("VertexColor")->setEnabled(true);

	if (isShowVectorFields)
	{
		polyscope::getSurfaceMesh("input mesh")->addFaceVectorQuantity("face vector field", dataVec * vecratio, polyscope::VectorType::AMBIENT);
		polyscope::getSurfaceMesh("input mesh")->getQuantity("face vector field")->setEnabled(true);
	}

	/*double shiftx = 1.5 * (triV.col(0).maxCoeff() - triV.col(0).minCoeff());

	polyscope::registerSurfaceMesh("init vector field mesh", triV, triF);

	polyscope::getSurfaceMesh("init vector field mesh")->translate(glm::vec3(shiftx, 0, 0));
	polyscope::getSurfaceMesh("init vector field mesh")->setEnabled(false);

	if (isShowVectorFields)
	{
		polyscope::getSurfaceMesh("init vector field mesh")->addFaceVectorQuantity("vector field", vecratio * faceOmegaList[frameId], polyscope::VectorType::AMBIENT);
	}


	polyscope::registerSurfaceMesh("vector field mesh", loopTriV, loopTriF);
	polyscope::getSurfaceMesh("vector field mesh")->setEnabled(false);

	if (isShowVectorFields)
	{
		polyscope::getSurfaceMesh("vector field mesh")->addFaceVectorQuantity("upsampled vector field", vecratio * subFaceOmegaList[frameId], polyscope::VectorType::AMBIENT);
	}*/

}

int getSelectedFaceId()
{
	if (polyscope::pick::haveSelection())
	{
		unsigned long id = polyscope::pick::getSelection().second;
		int nverts = polyscope::getSurfaceMesh("input mesh")->nVertices();


		int nlocalFaces = triMesh.nFaces();

		if (id >= nverts && id < nlocalFaces + nverts)
		{
			return id - nverts;
		}
		else
			return -1;
	}
	else
		return -1;
}


bool loadProblem()
{
	std::string loadFileName = igl::file_dialog_open();

	std::cout << "load file in: " << loadFileName << std::endl;
	using json = nlohmann::json;
	std::ifstream inputJson(loadFileName);
	if (!inputJson) {
		std::cerr << "missing json file in " << loadFileName << std::endl;
		return false;
	}

	std::string filePath = loadFileName;
	std::replace(filePath.begin(), filePath.end(), '\\', '/'); // handle the backslash issue for windows
	int id = filePath.rfind("/");
	std::string workingFolder = filePath.substr(0, id + 1);
	std::cout << "working folder: " << workingFolder << std::endl;

	json jval;
	inputJson >> jval;

	std::string meshFile = jval["mesh_name"];
	upsampleTimes = jval["upsampled_times"];
	loopLevel = 0;

	quadOrder = jval["quad_order"];
	numFrames = jval["num_frame"];

	isCoupled = jval["operation_details"]["amp_omega_coupling"];
	selectedMagValue = jval["operation_details"]["amp_operation_value"];
	selectedMotionValue = jval["operation_details"]["omega_operation_value"];

	std::string optype = jval["operation_details"]["omega_operation_type"];
	if (optype == "None")
		selectedMotion = None;
	else if (optype == "Enlarge")
		selectedMotion = Enlarge;
	else if (optype == "Rotate")
		selectedMotion = Rotate;
	else
		selectedMotion = None;

	isSelectAll = jval["region_details"]["select_all"];
	clickedFid = jval["region_details"]["selected_fid"];
	dilationTimes = jval["region_details"]["selected_domain_dilation"];
	optTimes = jval["region_details"]["interface_dilation"];

	//spatialAmpRatio = jval["spatial_ratio"]["amp_ratio"];
	//spatialEdgeRatio = jval["spatial_ratio"]["edge_ratio"];
	//spatialKnoppelRatio = jval["spatial_ratio"]["knoppel_ratio"];

	meshFile = workingFolder + meshFile;


	igl::readOBJ(meshFile, triV, triF);
	triMesh = MeshConnectivity(triF);
    
	initialization(triV, triF, upsampledTriV, upsampledTriF);
	updateEditionDomain();

	int nedges = triMesh.nEdges();
	int nverts = triV.rows();

	std::string initAmpPath = jval["init_amp"];
	std::string initOmegaPath = jval["init_omega"];

	if (!loadVertexAmp(workingFolder + initAmpPath, triV.rows(), initAmp))
	{
		std::cout << "missing init amp file: " << std::endl;
		return false;
	}

	if (!loadEdgeOmega(workingFolder + initOmegaPath, nedges, initOmega)) {
		std::cout << "missing init edge omega file." << std::endl;
		return false;
	}


	std::string refAmp = jval["reference"]["ref_amp"];
	std::string refOmega = jval["reference"]["ref_omega"];

	std::string optZvals = jval["solution"]["opt_zvals"];
	std::string optOmega = jval["solution"]["opt_omega"];

	// edge omega List
	int iter = 0;
	bool isLoadRef = true;
	refAmpList.resize(numFrames);
	refOmegaList.resize(numFrames);

	for (uint32_t i = 0; i < numFrames; ++i) {

		if (!loadVertexAmp(workingFolder + refAmp + "/amp_" + std::to_string(i) + ".txt", triV.rows(), refAmpList[i]))
		{
			std::cout << "missing amp file: " << std::endl;
			isLoadRef = false;
			break;
		}

		std::string edgePath = workingFolder + refOmega + "/omega_" + std::to_string(i) + ".txt";
		if (!loadEdgeOmega(edgePath, nedges, refOmegaList[i])) {
			std::cout << "missing edge file." << std::endl;
			isLoadRef = false;
			break;
		}
	}

	if (!isLoadRef)
	{
		editModel = IntrinsicFormula::WrinkleEditingStaticEdgeModel(triV, triMesh, vertOpts, faceFlags, quadOrder, spatialAmpRatio, spatialEdgeRatio, spatialKnoppelRatio);

		editModel.initialization(initAmp, initOmega, numFrames - 2);
		refAmpList = editModel.getRefAmpList();
		refOmegaList = editModel.getRefWList();
	}


	isLoadOpt = true;
	zList.clear();
	omegaList.clear();
	for (int i = 0; i < numFrames; i++)
	{
		std::string zvalFile = workingFolder + optZvals + "/zvals_" + std::to_string(i) + ".txt";
		std::string edgeOmegaFile = workingFolder + optOmega + "/omega_" + std::to_string(i) + ".txt";
		std::vector<std::complex<double>> zvals;
		if (!loadVertexZvals(zvalFile, nverts, zvals))
		{
			isLoadOpt = false;
			break;
		}
		Eigen::VectorXd edgeOmega;
		if (!loadEdgeOmega(edgeOmegaFile, nedges, edgeOmega)) {
			isLoadOpt = false;
			break;
		}

		zList.push_back(zvals);
		omegaList.push_back(edgeOmega);
	}

	if (isLoadOpt)
	{
		std::cout << "load zvals and omegas from file!" << std::endl;
	}
	else
	{
		std::cout << "failed to load zvals and omegas from file, set them to be random values!" << std::endl;
		zList.resize(numFrames);
		omegaList.resize(numFrames);
		omegaList = refOmegaList;

		for (int i = 0; i < numFrames; i++)
		{
			//omegaList[i].setRandom(nedges);
			Eigen::Vector2d rnd = Eigen::Vector2d::Random();
			//zList[i].resize(nverts, std::complex<double>(rnd(0), rnd(1)));
			zList[i].resize(nverts, std::complex<double>(1, 0));
		}
	}

	updatePaintingItems();

	curFrame = 0;

	return true;
}


bool saveProblem()
{
	std::string saveFileName = igl::file_dialog_save();

	std::string curOpt = "None";
	if (selectedMotion == Enlarge)
		curOpt = "Enlarge";
	else if (selectedMotion == Rotate)
		curOpt = "Rotate";

	using json = nlohmann::json;
	json jval =
	{
			{"mesh_name",         "mesh.obj"},
			{"num_frame",         zList.size()},
			{"quad_order",        quadOrder},
			{"spatial_ratio",     {
										   {"amp_ratio", spatialAmpRatio},
										   {"edge_ratio", spatialEdgeRatio},
										   {"knoppel_ratio", spatialKnoppelRatio}

								  }
			},
			{"upsampled_times", loopLevel},
			{"init_omega",        "omega.txt"},
			{"init_amp",          "amp.txt"},
			{
			 "region_details",    {
										  {"select_all", isSelectAll},
										  {"selected_fid", clickedFid},
										  {"selected_domain_dilation", dilationTimes},
										  {"interface_dilation", optTimes}

								  }
			},
			{
			 "operation_details", {
										  {"omega_operation_type", curOpt},
										  {"omega_operation_value", selectedMotionValue},
										  {"amp_omega_coupling", isCoupled},
										  {"amp_operation_value", selectedMagValue}
								  }
			},
			{
			 "reference",         {
										  {"ref_amp", "/refAmp/"},
										  {"ref_omega", "/refOmega/"}
								  }
			},
			{
			 "solution",          {
										  {"opt_zvals", "/optZvals/"},
										  {"opt_omega", "/optOmega/"},
										  {"wrinkle_mesh", "/wrinkledMesh/"},
										  {"upsampled_amp", "/upsampledAmp/"},
										  {"upsampled_phase", "/upsampledPhase/"}
								  }
			}
	};

	std::string filePath = saveFileName;
	std::replace(filePath.begin(), filePath.end(), '\\', '/'); // handle the backslash issue for windows
	int id = filePath.rfind("/");
	std::string workingFolder = filePath.substr(0, id + 1);

	std::ofstream iwfs(workingFolder + "omega.txt");
	iwfs << std::setprecision(std::numeric_limits<long double>::digits10 + 1) << initOmega << std::endl;

	std::ofstream iafs(workingFolder + "amp.txt");
	iafs << std::setprecision(std::numeric_limits<long double>::digits10 + 1) << initAmp << std::endl;


	igl::writeOBJ(workingFolder + "mesh.obj", triV, triF);

	std::string outputFolder = workingFolder + "/optZvals/";
	mkdir(outputFolder);

	std::string omegaOutputFolder = workingFolder + "/optOmega/";
	mkdir(omegaOutputFolder);



	for (int i = 0; i < zList.size(); i++)
	{
		std::ofstream zfs(outputFolder + "zvals_" + std::to_string(i) + ".txt");
		std::ofstream wfs(omegaOutputFolder + "omega_" + std::to_string(i) + ".txt");
		wfs << std::setprecision(std::numeric_limits<long double>::digits10 + 1) << omegaList[i] << std::endl;
		for (int j = 0; j < zList[i].size(); j++)
		{
			zfs << std::setprecision(std::numeric_limits<long double>::digits10 + 1) << zList[i][j].real() << " " << zList[i][j].imag() << std::endl;
		}


	}

	Eigen::MatrixXd N;
	igl::per_vertex_normals(upsampledTriV, upsampledTriF, N);

	outputFolder = workingFolder + "/upsampledAmp/";
	mkdir(outputFolder);

	std::string outputFolderPhase = workingFolder + "/upsampledPhase/";
	mkdir(outputFolderPhase);

	std::string outputFolderWrinkles = workingFolder + "/wrinkledMesh/";
	mkdir(outputFolderWrinkles);

	for (int i = 0; i < ampFieldsList.size(); i++)
	{
		Eigen::MatrixXd wrinkledV = upsampledTriV;
		for (int j = 0; j < upsampledTriV.rows(); j++)
		{
			wrinkledV.row(j) = upsampledTriV.row(j) + ampFieldsList[i](j) * std::cos(phaseFieldsList[i][j]) * N.row(j);
		}

		igl::writeOBJ(outputFolderWrinkles + "wrinkledMesh_" + std::to_string(i) + ".obj", wrinkledV, upsampledTriF);
        laplacianSmoothing(wrinkledV, upsampledTriF, wrinkledV, smoothingRatio, smoothingTimes);
        igl::writeOBJ(outputFolderWrinkles + "wrinkledMeshSmoothed_" + std::to_string(i) + ".obj", wrinkledV, upsampledTriF);

		/*Eigen::MatrixXd upWrinkledV;
		Eigen::MatrixXi upWrinkledF;
		loopUpsampling(wrinkledV, upsampledTriF, upWrinkledV, upWrinkledF, 2);
		igl::writeOBJ(outputFolderWrinkles + "loopedWrinkledMesh_" + std::to_string(i) + ".obj", upWrinkledV, upWrinkledF);*/

		std::ofstream afs(outputFolder + "upAmp_" + std::to_string(i) + ".txt");
		std::ofstream pfs(outputFolderPhase + "upPhase" + std::to_string(i) + ".txt");
		afs << std::setprecision(std::numeric_limits<long double>::digits10 + 1) << ampFieldsList[i] << std::endl;
		pfs << std::setprecision(std::numeric_limits<long double>::digits10 + 1) << phaseFieldsList[i] << std::endl;
	}

	// save reference
	outputFolder = workingFolder + "/refAmp/";
	mkdir(outputFolder);
	for (int i = 0; i < refAmpList.size(); i++)
	{
		std::ofstream afs(outputFolder + "amp_" + std::to_string(i) + ".txt");
		afs << std::setprecision(std::numeric_limits<long double>::digits10 + 1) << refAmpList[i] << std::endl;
	}

	outputFolder = workingFolder + "/refOmega/";
	mkdir(outputFolder);
	for (int i = 0; i < refOmegaList.size(); i++)
	{
		std::ofstream wfs(outputFolder + "omega_" + std::to_string(i) + ".txt");
		wfs << std::setprecision(std::numeric_limits<long double>::digits10 + 1) << refOmegaList[i] << std::endl;
	}

	std::ofstream o(saveFileName);
	o << std::setw(4) << jval << std::endl;
	std::cout << "save file in: " << saveFileName << std::endl;

	return true;
}

void callback() {
	int newId = getSelectedFaceId();
	clickedFid = newId > 0 && newId < triMesh.nFaces() ? newId : clickedFid;
	ImGui::PushItemWidth(100);
	float w = ImGui::GetContentRegionAvailWidth();
	float p = ImGui::GetStyle().FramePadding.x;
	if (ImGui::Button("Load", ImVec2(ImVec2((w - p) / 2.f, 0))))
	{
		loadProblem();
		updateMagnitudePhase(omegaList, zList, ampFieldsList, phaseFieldsList);
		updateFieldsInView(curFrame);
	}
	ImGui::SameLine(0, p);
	if (ImGui::Button("Save", ImVec2((w - p) / 2.f, 0)))
	{
		saveProblem();
	}
	if (ImGui::Button("Reset", ImVec2(-1, 0)))
	{
		curFrame = 0;
		updateFieldsInView(curFrame);
	}

	if (ImGui::InputInt("underline loop level", &loopLevel))
	{
		if (loopLevel < 0)
			loopLevel = 0;
		if (loopLevel > upsampleTimes)
			loopLevel = upsampleTimes;
	}

	if (ImGui::InputInt("upsampled times", &upsampleTimes))
	{
		if (upsampleTimes >= loopLevel)
		{
			initialization(triV, triF, upsampledTriV, upsampledTriF);
			if (isForceOptimize)	//already solve for the interp states
			{
				updateMagnitudePhase(omegaList, zList, ampFieldsList, phaseFieldsList);
				updateFieldsInView(curFrame);
			}
		}
	}
	if (ImGui::CollapsingHeader("Visualization Options", ImGuiTreeNodeFlags_DefaultOpen))
	{
		if (ImGui::Checkbox("is show vector fields", &isShowVectorFields))
		{
			updateFieldsInView(curFrame);
		}
		if (ImGui::Checkbox("is show wrinkled mesh", &isShowWrinkels))
		{
			updateFieldsInView(curFrame);
		}
		if (ImGui::Checkbox("is show wrinkle color fields", &isShowWrinkleColorField))
		{
			updateFieldsInView(curFrame);
		}
		if (ImGui::DragFloat("wrinkle amp scaling ratio", &wrinkleAmpScalingRatio, 0.0005, 0, 1))
        {
            if (wrinkleAmpScalingRatio >= 0)
                updateFieldsInView(curFrame);
        }
	}

    if (ImGui::CollapsingHeader("smoothing Options", ImGuiTreeNodeFlags_DefaultOpen))
    {
        if(ImGui::InputInt("smoothing times", &smoothingTimes))
        {
            smoothingTimes = smoothingTimes > 0 ? smoothingTimes : 0;
            updateFieldsInView(curFrame);
        }
        if(ImGui::InputDouble("smoothing ratio", &smoothingRatio))
        {
            smoothingRatio = smoothingRatio > 0 ? smoothingRatio : 0;
            updateFieldsInView(curFrame);
        }
    }



	if (ImGui::CollapsingHeader("Selected Region", ImGuiTreeNodeFlags_DefaultOpen))
	{
		ImGui::Checkbox("Select all", &isSelectAll);
		ImGui::InputInt("clicked face id", &clickedFid);

		if (ImGui::InputInt("dilation times", &dilationTimes))
		{
			if (dilationTimes < 0)
				dilationTimes = 3;
		}
		ImGui::Combo("reg opt func", (int*)&regOpType, "Dilation\0Erosion\0");
		if (ImGui::InputInt("opt times", &optTimes))
		{
			if (optTimes < 0 || optTimes > 20)
				optTimes = 0;
		}
	}
	if (ImGui::CollapsingHeader("Wrinkle Edition Options", ImGuiTreeNodeFlags_DefaultOpen))
	{
		ImGui::Combo("edition motion", (int*)&selectedMotion, "Ratate\0Tilt\0Enlarge\0None\0");
		if (ImGui::InputDouble("motion value", &selectedMotionValue))
		{
			if (selectedMotionValue < 0 && selectedMotion == Enlarge)
				selectedMotionValue = 0;
		}
		ImGui::Checkbox("vec mag coupled", &isCoupled);
		if (ImGui::InputDouble("mag motion value", &selectedMagValue))
		{
			if (selectedMagValue < 0)
				selectedMagValue = 1;
		}
	}

	if (ImGui::CollapsingHeader("Frame Visualization Options", ImGuiTreeNodeFlags_DefaultOpen))
	{
		if (ImGui::SliderInt("current frame", &curFrame, 0, numFrames - 1))
		{
			if (curFrame >= 0 && curFrame <= numFrames - 1)
				updateFieldsInView(curFrame);
		}
		if (ImGui::DragFloat("vec ratio", &(vecratio), 0.00005, 0, 1))
		{
			updateFieldsInView(curFrame);
		}
	}

	if (ImGui::CollapsingHeader("optimzation parameters", ImGuiTreeNodeFlags_DefaultOpen))
	{
		if (ImGui::InputInt("num of frames", &numFrames))
		{
			if (numFrames <= 0)
				numFrames = 10;
		}

		if (ImGui::InputInt("num iterations", &numIter))
		{
			if (numIter < 0)
				numIter = 1000;
		}
		if (ImGui::InputDouble("grad tol", &gradTol))
		{
			if (gradTol < 0)
				gradTol = 1e-6;
		}
		if (ImGui::InputDouble("x tol", &xTol))
		{
			if (xTol < 0)
				xTol = 0;
		}
		if (ImGui::InputDouble("f tol", &fTol))
		{
			if (fTol < 0)
				fTol = 0;
		}
		if (ImGui::InputInt("quad order", &quadOrder))
		{
			if (quadOrder <= 0 || quadOrder > 20)
				quadOrder = 4;
		}

		if (ImGui::InputDouble("spatial amp ratio", &spatialAmpRatio))
		{
			if (spatialAmpRatio < 0)
				spatialAmpRatio = 1;
		}

		if (ImGui::InputDouble("spatial edge ratio", &spatialEdgeRatio))
		{
			if (spatialEdgeRatio < 0)
				spatialEdgeRatio = 1;
		}

		if (ImGui::InputDouble("spatial knoppel ratio", &spatialKnoppelRatio))
		{
			if (spatialKnoppelRatio < 0)
				spatialKnoppelRatio = 1;
		}

		ImGui::Checkbox("warm start", &isWarmStart);

	}


	ImGui::Checkbox("Try Optimization", &isForceOptimize);

	if (ImGui::Button("comb fields & updates", ImVec2(-1, 0)))
	{

		updateEditionDomain();
		// solve for the path from source to target
		solveKeyFrames(initAmp, initOmega, faceFlags, omegaList, zList);
		updatePaintingItems();
		updateFieldsInView(curFrame);
	}

	if (ImGui::Button("output images", ImVec2(-1, 0)))
	{
		std::string curFolder = std::filesystem::current_path().string();
		std::cout << "save folder: " << curFolder << std::endl;
		for (int i = 0; i < ampFieldsList.size(); i++)
		{
			updateFieldsInView(i);
			//polyscope::options::screenshotExtension = ".jpg";
			std::string name = curFolder + "/output_" + std::to_string(i) + ".jpg";
			polyscope::screenshot(name);
		}
	}


	ImGui::PopItemWidth();
}


int main(int argc, char** argv)
{
	if (!loadProblem())
	{
		std::cout << "failed to load file." << std::endl;
		return 1;
	}


	// Options
	polyscope::options::autocenterStructures = true;
	polyscope::view::windowWidth = 1024;
	polyscope::view::windowHeight = 1024;

	// Initialize polyscope
	polyscope::init();

	// Register the mesh with Polyscope
//    polyscope::registerSurfaceMesh("input mesh", triV, triF);


	polyscope::view::upDir = polyscope::view::UpDir::ZUp;

	// Add the callback
	polyscope::state::userCallback = callback;

	polyscope::options::groundPlaneHeightFactor = 0.25; // adjust the plane height

	updateFieldsInView(curFrame);
	// Show the gui
	polyscope::show();


	return 0;
}