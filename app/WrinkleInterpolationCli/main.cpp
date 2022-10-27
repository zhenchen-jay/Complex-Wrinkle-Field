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

#include <iostream>
#include <filesystem>
#include <utility>
#include <CLI/CLI.hpp>

#include "../../include/CommonTools.h"
#include "../../include/MeshLib/MeshUpsampling.h"
#include "../../include/Optimization/NewtonDescent.h"
#include "../../include/IntrinsicFormula/InterpolateZvals.h"
#include "../../include/IntrinsicFormula/WrinkleEditingModel.h"
#include "../../include/IntrinsicFormula/WrinkleEditingCWF.h"

#include "../../include/IntrinsicFormula/KnoppelStripePatternEdgeOmega.h"
#include "../../include/WrinkleFieldsEditor.h"
#include "../../include/SpherigonSmoothing.h"
#include "../../dep/SecStencils/types.h"
#include "../../dep/SecStencils/Subd.h"
#include "../../dep/SecStencils/utils.h"

#include "../../include/json.hpp"
#include "../../include/ComplexLoop/ComplexLoop.h"
#include "../../include/ComplexLoop/ComplexLoopAmpPhase.h"
#include "../../include/ComplexLoop/ComplexLoopAmpPhaseEdgeJump.h"
#include "../../include/ComplexLoop/ComplexLoopReIm.h"
#include "../../include/ComplexLoop/ComplexLoopZuenko.h"
#include "../../include/LoadSaveIO.h"
#include "../../include/SecMeshParsing.h"
#include "../../include/MeshLib/RegionEdition.h"

std::vector<VertexOpInfo> vertOpts;

Eigen::MatrixXd triV, upsampledTriV;
Eigen::MatrixXi triF, upsampledTriF;
MeshConnectivity triMesh;
Mesh secMesh, subSecMesh;

// initial information
Eigen::VectorXd initAmp;
Eigen::VectorXd initOmega;
std::vector<std::complex<double>> initZvals;

// target information
Eigen::VectorXd tarAmp;
Eigen::VectorXd tarOmega;
std::vector<std::complex<double>> tarZvals;

// base mesh information list
std::vector<Eigen::VectorXd> omegaList;
std::vector<Eigen::MatrixXd> faceOmegaList;
std::vector<std::vector<std::complex<double>>> zList;


// upsampled informations
std::vector<Eigen::VectorXd> subOmegaList;
std::vector<Eigen::MatrixXd> subFaceOmegaList;

std::vector<Eigen::VectorXd> phaseFieldsList;
std::vector<Eigen::VectorXd> ampFieldsList;
std::vector<std::vector<std::complex<double>>> upZList;
std::vector<Eigen::MatrixXd> wrinkledVList;

std::vector<Eigen::VectorXd> consistencyVec;
std::vector<Eigen::VectorXd> upConsistencyVec;

// reference amp and omega
std::vector<Eigen::VectorXd> refOmegaList;
std::vector<Eigen::VectorXd> refAmpList;

// region edition
RegionEdition regEdt;

int numFrames = 51;

double globalAmpMax = 1;
double globalAmpMin = 0;

double globalInconMax = 1;
double globalInconMin = 0;

double globalCoarseInconMax = 1;
double globalCoarseInconMin = 0;

int quadOrder = 4;

double spatialAmpRatio = 1000;
double spatialEdgeRatio = 1000;
double spatialKnoppelRatio = 1000;

std::string workingFolder;

std::shared_ptr<IntrinsicFormula::WrinkleEditingModel> editModel;

// smoothing
int smoothingTimes = 2;
double smoothingRatio = 0.95;

bool isFixedBnd = false;
int effectivedistFactor = 4;

bool isSelectAll = false;
VecMotionType selectedMotion = Enlarge;

double selectedMotionValue = 2;
double selectedMagValue = 1;
bool isCoupled = false;

Eigen::VectorXi selectedFids;
Eigen::VectorXi interfaceFids;
Eigen::VectorXi faceFlags;	// -1 for interfaces, 0 otherwise
Eigen::VectorXi selectedVertices;

int optTimes = 5;

bool isLoadOpt = false;
bool isLoadTar = false;

int clickedFid = -1;
int dilationTimes = 10;

bool isUseV2 = false;
int upsampleTimes = 3;


// Default arguments
struct {
	std::string input;
	std::string method = "CWF";
	double gradTol = 1e-6;
	double xTol = 0;
	double fTol = 0;
	int numIter = 1000;
	double ampScale = 1;
	bool reOptimize = false;
} args;


struct PickedFace
{
	int fid = -1;
	double ampChangeRatio = 1.;
	int effectiveRadius = 5;
	int interfaceDilation = 5;
	VecMotionType freqVecMotion = Enlarge;
	double freqVecChangeValue = 1.;
	bool isFreqAmpCoupled = false;

	std::vector<int> effectiveFaces = {};
	std::vector<int> interFaces = {};
	std::vector<int> effectiveVerts = {};
	std::vector<int> interVerts = {};

	void buildEffectiveFaces(int nfaces)
	{
		effectiveFaces.clear();
		effectiveVerts.clear();
		interFaces.clear();
		interVerts.clear();

		if (fid == -1 || fid >= nfaces)
			return;
		else
		{
			Eigen::VectorXi curFaceFlags = Eigen::VectorXi::Zero(triF.rows());
			curFaceFlags(fid) = 1;
			Eigen::VectorXi curFaceFlagsNew = curFaceFlags;
			regEdt.faceDilation(curFaceFlagsNew, curFaceFlags, effectiveRadius);
			regEdt.faceDilation(curFaceFlags, curFaceFlagsNew, interfaceDilation);

			Eigen::VectorXi vertFlags, vertFlagsNew;

			faceFlags2VertFlags(triMesh, triV.rows(), curFaceFlags, vertFlags);
			faceFlags2VertFlags(triMesh, triV.rows(), curFaceFlagsNew, vertFlagsNew);

			for (int i = 0; i < curFaceFlags.rows(); i++)
			{
				if (curFaceFlags(i))
					effectiveFaces.push_back(i);
				else if (curFaceFlagsNew(i))
					interFaces.push_back(i);
			}


			for (int i = 0; i < vertFlags.rows(); i++)
			{
				if (vertFlags(i))
					effectiveVerts.push_back(i);
				else if (vertFlagsNew(i))
					interVerts.push_back(i);
			}
		}
	}
};

std::vector<PickedFace> pickFaces;


void buildWrinkleMotions(const std::vector<PickedFace>& faceList, std::vector<VertexOpInfo>& vertOpInfo)
{
	int nverts = triV.rows();
	vertOpInfo.clear();
	vertOpInfo.resize(nverts, { None, isCoupled, 0, 1 });

	if (isSelectAll)
	{
		for (int i = 0; i < nverts; i++)
		{
			vertOpts[i] = { selectedMotion, isCoupled, selectedMotionValue, selectedMagValue };
		}
	}
	else
	{
		Eigen::VectorXi tmpFlags;
		tmpFlags.setZero(nverts);
		int nselectedV = 0;

		// make sure there is no overlap
		for (auto& pf : faceList)
		{
			for (auto& v : pf.effectiveVerts)
			{
				if (tmpFlags(v))
				{
					std::cerr << "overlap happens on the effective vertices. " << std::endl;
					exit(EXIT_FAILURE);
				}
				tmpFlags(v) = 1;
				nselectedV++;
			}
		}

		std::cout << "num of selected vertices: " << nselectedV << std::endl;

		vertOpInfo.clear();
		vertOpInfo.resize(nverts, { None, isCoupled, 0, 1 });

		for (auto& pf : faceList)
		{
			for (auto& v : pf.effectiveVerts)
			{
				vertOpInfo[v] = { pf.freqVecMotion, pf.isFreqAmpCoupled, pf.freqVecChangeValue, pf.ampChangeRatio };
			}
		}
	}

}

bool addSelectedFaces(const PickedFace face, Eigen::VectorXi& curFaceFlags, Eigen::VectorXi& curVertFlags)
{
	for (auto& f : face.effectiveFaces)
		if (curFaceFlags(f))
			return false;

	for (auto& v : face.effectiveVerts)
		if (curVertFlags(v))
			return false;

	for (auto& f : face.effectiveFaces)
		curFaceFlags(f) = 1;
	for (auto& v : face.effectiveVerts)
		curVertFlags(v) = 1;

	return true;
}


InitializationType initType = SeperateLinear;
double zuenkoTau = 0.1;
int zuenkoIter = 5;

static void buildEditModel(const Eigen::MatrixXd& pos, const MeshConnectivity& mesh, const std::vector<VertexOpInfo>& vertexOpts, const Eigen::VectorXi& faceFlag, int quadOrd, double spatialAmpRatio, double spatialEdgeRatio, double spatialKnoppelRatio, int effectivedistFactor, std::shared_ptr<IntrinsicFormula::WrinkleEditingModel>& editModel)
{
	editModel = std::make_shared<IntrinsicFormula::WrinkleEditingCWF>(pos, mesh, vertexOpts, faceFlag, quadOrd, spatialAmpRatio, spatialEdgeRatio, spatialKnoppelRatio, effectivedistFactor);
}

void updateMagnitudePhase(const std::vector<Eigen::VectorXd>& wFrames, const std::vector<std::vector<std::complex<double>>>& zFrames, 
	std::vector<Eigen::VectorXd>& magList, 
	std::vector<Eigen::VectorXd>& phaseList,
	std::vector<std::vector<std::complex<double>>>& upZFrames)
{
	magList.resize(wFrames.size());
	phaseList.resize(wFrames.size());
	upZFrames.resize(wFrames.size());

	consistencyVec.resize(wFrames.size());
	upConsistencyVec.resize(wFrames.size());

	subOmegaList.resize(wFrames.size());
	subFaceOmegaList.resize(wFrames.size());
	
	MeshConnectivity mesh(triF);


	auto computeMagPhase = [&](const tbb::blocked_range<uint32_t>& range)
	{
		for (uint32_t i = range.begin(); i < range.end(); ++i)
		{
			Eigen::VectorXd edgeVec = swapEdgeVec(triF, wFrames[i], 0);

			consistencyVec[i] = inconsistencyComputation(secMesh, edgeVec, zFrames[i]);

			std::shared_ptr<ComplexLoop> complexLoopOpt = std::make_shared<ComplexLoopZuenko>();
			complexLoopOpt->setBndFixFlag(isFixedBnd);
			complexLoopOpt->SetMesh(secMesh);
			complexLoopOpt->Subdivide(edgeVec, zFrames[i], subOmegaList[i], upZFrames[i], upsampleTimes);
			Mesh tmpMesh = complexLoopOpt->GetMesh();

			upConsistencyVec[i] = inconsistencyComputation(tmpMesh, subOmegaList[i], upZFrames[i]);

			subFaceOmegaList[i] = edgeVec2FaceVec(tmpMesh, subOmegaList[i]);
			
			magList[i].setZero(upZFrames[i].size());
			phaseList[i].setZero(upZFrames[i].size());

			for (int j = 0; j < magList[i].size(); j++)
			{
				magList[i](j) = std::abs(upZFrames[i][j]);
				phaseList[i](j) = std::arg(upZFrames[i][j]);
			}
		}
	};

	tbb::blocked_range<uint32_t> rangex(0u, (uint32_t)upZFrames.size());
	tbb::parallel_for(rangex, computeMagPhase);

}


void updateWrinkles(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, const std::vector<std::vector<std::complex<double>>>& zFrames, std::vector<Eigen::MatrixXd>& wrinkledVFrames, double scaleRatio, bool isUseV2)
{
	std::vector<std::vector<std::complex<double>>> interpZList(zFrames.size());
	wrinkledVFrames.resize(zFrames.size());

	std::vector<std::vector<int>> vertNeiEdges;
	std::vector<std::vector<int>> vertNeiFaces;

	buildVertexNeighboringInfo(MeshConnectivity(F), V.rows(), vertNeiEdges, vertNeiFaces);

	auto computeWrinkles = [&](const tbb::blocked_range<uint32_t>& range)
	{
		for (uint32_t i = range.begin(); i < range.end(); ++i)
		{
			getWrinkledMesh(V, F, zFrames[i], &vertNeiFaces, wrinkledVFrames[i], scaleRatio, isUseV2);
		}
	};

	tbb::blocked_range<uint32_t> rangex(0u, (uint32_t)interpZList.size());
	tbb::parallel_for(rangex, computeWrinkles);

}

void updateInterfaces(const std::vector<PickedFace>& faces, Eigen::VectorXi& interFaceFlags)
{
	int nfaces = triMesh.nFaces();
	interFaceFlags.setZero(nfaces);

	for (auto& f : faces)
	{
		for (auto& interf : f.interFaces)
			interFaceFlags(interf) = 1;
	}
}

void updateEditionDomain()
{
	int nselected = 0;
	for (int i = 0; i < selectedFids.rows(); i++)
	{
		if (selectedFids(i) == 1)
		{
			nselected++;
		}
	}

	Eigen::VectorXi interfaces;
	updateInterfaces(pickFaces, interfaces);

	faceFlags = selectedFids;
	int ninterfaces = 0;
	for (int i = 0; i < selectedFids.rows(); i++)
	{
		if (selectedFids(i) == 0 && interfaces(i) == 1)
		{
			ninterfaces++;
			faceFlags(i) = -1;
		}

	}

	std::cout << "selected effective faces: " << nselected << ", num of interfaces: " << ninterfaces << std::endl;

	std::cout << "build wrinkle motions. " << std::endl;
	buildWrinkleMotions(pickFaces, vertOpts);

}

void updateEverythingForSaving()
{
	// get interploated amp and phase frames
	std::cout << "compute upsampled phase: " << std::endl;
	updateMagnitudePhase(omegaList, zList, ampFieldsList, phaseFieldsList, upZList);

	std::cout << "compute wrinkle meshes: " << std::endl;
	updateWrinkles(upsampledTriV, upsampledTriF, upZList, wrinkledVList, args.ampScale, isUseV2);


	std::cout << "compute face vector fields:" << std::endl;
	faceOmegaList.resize(omegaList.size());
	for (int i = 0; i < omegaList.size(); i++)
	{
		faceOmegaList[i] = intrinsicEdgeVec2FaceVec(omegaList[i], triV, triMesh);
	}


	// update global maximum amplitude
	std::cout << "update max and min amp. " << std::endl;

	globalAmpMax = std::max(ampFieldsList[0].maxCoeff(), refAmpList[0].maxCoeff());
	globalAmpMin = std::min(ampFieldsList[0].minCoeff(), refAmpList[0].minCoeff());
	for (int i = 1; i < ampFieldsList.size(); i++)
	{
		globalAmpMax = std::max(globalAmpMax, std::max(ampFieldsList[i].maxCoeff(), refAmpList[i].maxCoeff()));
		globalAmpMin = std::min(globalAmpMin, std::min(ampFieldsList[i].minCoeff(), refAmpList[i].minCoeff()));
	}

	// update global maximum amplitude
	std::cout << "update max and min consistency. " << std::endl;

	globalInconMax = upConsistencyVec[0].maxCoeff();
	globalInconMin = upConsistencyVec[0].minCoeff();
	for (int i = 1; i < upConsistencyVec.size(); i++)
	{
		globalInconMax = std::max(globalInconMax, upConsistencyVec[i].maxCoeff());
		globalInconMin = std::min(globalInconMin, upConsistencyVec[i].minCoeff());
	}

	// update global maximum amplitude
	std::cout << "update max and min consistency. " << std::endl;

	globalCoarseInconMax = consistencyVec[0].maxCoeff();
	globalCoarseInconMin = consistencyVec[0].minCoeff();
	for (int i = 1; i < upConsistencyVec.size(); i++)
	{
		globalCoarseInconMax = std::max(globalCoarseInconMax, consistencyVec[i].maxCoeff());
		globalCoarseInconMin = std::min(globalCoarseInconMin, consistencyVec[i].minCoeff());
	}

	std::cout << "start to update viewer." << std::endl;
}

void getUpsampledMesh(const Eigen::MatrixXd& triV, const Eigen::MatrixXi& triF, Eigen::MatrixXd& upsampledTriV, Eigen::MatrixXi& upsampledTriF)
{
	secMesh = convert2SecMesh(triV, triF);
	subSecMesh = secMesh;

	std::shared_ptr<ComplexLoop> complexLoopOpt = std::make_shared<ComplexLoopZuenko>();
	complexLoopOpt->setBndFixFlag(isFixedBnd);
	complexLoopOpt->SetMesh(secMesh);
	complexLoopOpt->meshSubdivide(upsampleTimes);
	subSecMesh = complexLoopOpt->GetMesh();
	parseSecMesh(subSecMesh, upsampledTriV, upsampledTriF);
}

void initialization(const Eigen::MatrixXd& triV, const Eigen::MatrixXi& triF, Eigen::MatrixXd& upsampledTriV, Eigen::MatrixXi& upsampledTriF)
{
	getUpsampledMesh(triV, triF, upsampledTriV, upsampledTriF);
	selectedFids.setZero(triMesh.nFaces());
	interfaceFids = selectedFids;
	regEdt = RegionEdition(triMesh, triV.rows());
	selectedVertices.setZero(triV.rows());
}

void solveKeyFrames(const std::vector<std::complex<double>>& initzvals, const Eigen::VectorXd& initOmega, const Eigen::VectorXi& faceFlags, std::vector<Eigen::VectorXd>& wFrames, std::vector<std::vector<std::complex<double>>>& zFrames)
{
	Eigen::VectorXd x;
	editModel->setSaveFolder(workingFolder);

	buildEditModel(triV, triMesh, vertOpts, faceFlags, quadOrder, spatialAmpRatio, spatialEdgeRatio, spatialKnoppelRatio, effectivedistFactor, editModel);
    if(tarZvals.empty())
	    editModel->initialization(initZvals, initOmega, numFrames - 2, initType, zuenkoTau, zuenkoIter);
    else
        editModel->initialization(initZvals, initOmega, tarZvals, tarOmega, numFrames - 2, true);
	editModel->convertList2Variable(x);

	editModel->solveIntermeditateFrames(x, args.numIter, args.gradTol, args.xTol, args.fTol, true, workingFolder);
	editModel->convertVariable2List(x);
	refOmegaList = editModel->getRefWList();
	refAmpList = editModel->getRefAmpList();

	std::cout << "get w list" << std::endl;
	wFrames = editModel->getWList();
	std::cout << "get z list" << std::endl;
	zFrames = editModel->getVertValsList();
}

bool loadProblem()
{
	std::string loadFileName = args.input;
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
	workingFolder = filePath.substr(0, id + 1);
	std::cout << "working folder: " << workingFolder << std::endl;

	json jval;
	inputJson >> jval;

	std::string meshFile = jval["mesh_name"];
	upsampleTimes = jval["upsampled_times"];


	meshFile = workingFolder + meshFile;
	igl::readOBJ(meshFile, triV, triF);
	triMesh = MeshConnectivity(triF);
	initialization(triV, triF, upsampledTriV, upsampledTriF);
	

	quadOrder = jval["quad_order"];
	numFrames = jval["num_frame"];
    if (jval.contains(std::string_view{ "wrinkle_amp_scale" }))
    {
        if(args.ampScale == 1)
            args.ampScale = jval["wrinkle_amp_scale"];
    }

	isSelectAll = jval["region_global_details"]["select_all"];
	isCoupled = jval["region_global_details"]["amp_omega_coupling"];
	selectedMagValue = jval["region_global_details"]["amp_operation_value"];
	selectedMotionValue = jval["region_global_details"]["omega_operation_value"];
	std::string optype = jval["region_global_details"]["omega_operation_motion"];

	if (optype == "None")
		selectedMotion = None;
	else if (optype == "Enlarge")
		selectedMotion = Enlarge;
	else if (optype == "Rotate")
		selectedMotion = Rotate;
	else
		selectedMotion = None;

	pickFaces.clear();

	if (jval.contains(std::string_view{ "region_local_details" }))
	{
		int npicked = jval["region_local_details"].size();
		for (int i = 0; i < npicked; i++)
		{
			PickedFace pf;
			pf.fid = jval["region_local_details"][i]["face_id"];
			pf.effectiveRadius = jval["region_local_details"][i]["effective_radius"];
			pf.interfaceDilation = jval["region_local_details"][i]["interface_dilation"];

			optype = jval["region_local_details"][i]["omega_operation_motion"];
			if (optype == "None")
				pf.freqVecMotion = None;
			else if (optype == "Enlarge")
				pf.freqVecMotion = Enlarge;
			else if (optype == "Rotate")
				pf.freqVecMotion = Rotate;
			else
				pf.freqVecMotion = None;

			pf.isFreqAmpCoupled = jval["region_local_details"][i]["amp_omega_coupling"];
			pf.freqVecChangeValue = jval["region_local_details"][i]["omega_opereation_value"];
			pf.ampChangeRatio = jval["region_local_details"][i]["amp_operation_value"];
			

			pf.buildEffectiveFaces(triF.rows());
			if (!addSelectedFaces(pf, selectedFids, selectedVertices))
			{
				std::cerr << "something wrong happened in the store setup file!" << std::endl;
				exit(EXIT_FAILURE);
			}
			pickFaces.push_back(pf);
		}
	}
	std::cout << "num of picked faces: " << pickFaces.size() << std::endl;
	
	updateEditionDomain();
	

	if (jval.contains(std::string_view{ "spatial_ratio" }))
	{
		if (jval["spatial_ratio"].contains(std::string_view{ "amp_ratio" }))
			spatialAmpRatio = jval["spatial_ratio"]["amp_ratio"];
		else
			spatialAmpRatio = 1000;

		if (jval["spatial_ratio"].contains(std::string_view{ "edge_ratio" }))
			spatialEdgeRatio = jval["spatial_ratio"]["edge_ratio"];
		else
			spatialEdgeRatio = 1000;

		if (jval["spatial_ratio"].contains(std::string_view{ "knoppel_ratio" }))
			spatialKnoppelRatio = jval["spatial_ratio"]["knoppel_ratio"];
		else
			spatialKnoppelRatio = 1000;
	}
	else
	{
		spatialAmpRatio = 1000;
		spatialEdgeRatio = 1000;
		spatialKnoppelRatio = 1000;
	}


	int nedges = triMesh.nEdges();
	int nverts = triV.rows();

	std::string initAmpPath = jval["init_amp"];
	std::string initOmegaPath = jval["init_omega"];
	std::string initZValsPath = "zvals.txt";
	if (jval.contains(std::string_view{ "init_zvals" }))
	{
		initZValsPath = jval["init_zvals"];
	}

	if (!loadEdgeOmega(workingFolder + initOmegaPath, nedges, initOmega)) {
		std::cout << "missing init edge omega file." << std::endl;
		return false;
	}

	if (!loadVertexZvals(workingFolder + initZValsPath, triV.rows(), initZvals))
	{
		std::cout << "missing init zval file, try to load amp file, and round zvals from amp and omega" << std::endl;
		if (!loadVertexAmp(workingFolder + initAmpPath, triV.rows(), initAmp))
		{
			std::cout << "missing init amp file: " << std::endl;
			return false;
		}

		else
		{
			Eigen::VectorXd edgeArea, vertArea;
			edgeArea = getEdgeArea(triV, triMesh);
			vertArea = getVertArea(triV, triMesh);
			IntrinsicFormula::roundZvalsFromEdgeOmegaVertexMag(triMesh, initOmega, initAmp, edgeArea, vertArea, triV.rows(), initZvals);
		}
	}
	else
	{
		initAmp.setZero(triV.rows());
		for (int i = 0; i < initZvals.size(); i++)
			initAmp(i) = std::abs(initZvals[i]);
	}

    std::string tarAmpPath = "amp_tar.txt";
    if (jval.contains(std::string_view{ "tar_amp" }))
    {
        tarAmpPath = jval["tar_amp"];
    }
    std::string tarOmegaPath = "omega_tar.txt";
    if (jval.contains(std::string_view{ "tar_omega" }))
    {
        tarOmegaPath = jval["tar_omega"];
    }
    std::string tarZValsPath = "zvals_tar.txt";
    if (jval.contains(std::string_view{ "tar_zvals" }))
    {
        tarZValsPath = jval["tar_zvals"];
    }
    
    tarOmega.resize(0);
    tarZvals = {};

	isLoadTar = true;

    if (!loadEdgeOmega(workingFolder + tarOmegaPath, nedges, tarOmega)) {
        std::cout << "missing tar edge omega file." << std::endl;
        isLoadTar = false;
    }

    if (!loadVertexZvals(workingFolder + tarZValsPath, triV.rows(), tarZvals))
    {
        std::cout << "missing tar zval file, try to load amp file, and round zvals from amp and omega" << std::endl;
        if (!loadVertexAmp(workingFolder + tarAmpPath, triV.rows(), tarAmp))
        {
            std::cout << "missing tar amp file: " << std::endl;
			isLoadTar = false;
        }

        else
        {
            Eigen::VectorXd edgeArea, vertArea;
            edgeArea = getEdgeArea(triV, triMesh);
            vertArea = getVertArea(triV, triMesh);
            IntrinsicFormula::roundZvalsFromEdgeOmegaVertexMag(triMesh, tarOmega, tarAmp, edgeArea, vertArea, triV.rows(), tarZvals);
        }
    }
    else
    {
        tarAmp.setZero(triV.rows());
        for (int i = 0; i < tarZvals.size(); i++)
            tarAmp(i) = std::abs(tarZvals[i]);
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
	if (!isLoadOpt || !isLoadRef)
	{
		buildEditModel(triV, triMesh, vertOpts, faceFlags, quadOrder, spatialAmpRatio, spatialEdgeRatio, spatialKnoppelRatio, effectivedistFactor, editModel);

        if(!isLoadTar)
        {
            editModel->initialization(initZvals, initOmega, numFrames - 2, initType, 0.1);
        }
        else
        {
            editModel->initialization(initZvals, initOmega, tarZvals, tarOmega, numFrames - 2, true);
        }

		refAmpList = editModel->getRefAmpList();
		refOmegaList = editModel->getRefWList();

		if (!isLoadOpt)
		{
			zList = editModel->getVertValsList();
			omegaList = editModel->getWList();
		}

	}
	else
	{
		buildEditModel(triV, triMesh, vertOpts, faceFlags, quadOrder, spatialAmpRatio, spatialEdgeRatio, spatialKnoppelRatio, effectivedistFactor, editModel);
		editModel->initialization(zList, omegaList, refAmpList, refOmegaList);
	}
	

	return true;
}


bool saveProblem()
{
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
            {"wrinkle_amp_ratio", args.ampScale},
			{"quad_order",        quadOrder},
			{"spatial_ratio",     {
										   {"amp_ratio", spatialAmpRatio},
										   {"edge_ratio", spatialEdgeRatio},
										   {"knoppel_ratio", spatialKnoppelRatio}

								  }
			},
			{"upsampled_times",  upsampleTimes},
			{"init_omega",        "omega.txt"},
			{"init_amp",          "amp.txt"},
			{"init_zvals",        "zvals.txt"},
			{"tar_omega",         "omega_tar.txt"},
			{"tar_amp",           "amp_tar.txt"},
			{"tar_zvals",         "zvals_tar.txt"},
			{"region_global_details",	  {
										{"select_all", isSelectAll},
										{"omega_operation_motion", curOpt},
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

	for (int i = 0; i < pickFaces.size(); i++)
	{
		curOpt = "None";
		if (pickFaces[i].freqVecMotion == Enlarge)
			curOpt = "Enlarge";
		else if (pickFaces[i].freqVecMotion == Rotate)
			curOpt = "Rotate";
		json pfJval =
		{
			{"face_id", pickFaces[i].fid},
			{"effective_radius", pickFaces[i].effectiveRadius},
			{"interface_dilation", pickFaces[i].interfaceDilation},
			{"omega_operation_motion", curOpt},
			{"omega_opereation_value", pickFaces[i].freqVecChangeValue},
			{"amp_operation_value", pickFaces[i].ampChangeRatio},
			{"amp_omega_coupling", pickFaces[i].isFreqAmpCoupled}
		};
		jval["region_local_details"].push_back(pfJval);
	}

	saveEdgeOmega(workingFolder + "omega.txt", initOmega);
	saveVertexAmp(workingFolder + "amp.txt", initAmp);
	saveVertexZvals(workingFolder + "zvals.txt", initZvals);

	if (isLoadTar)
	{
		std::cout << "save target" << std::endl;
		saveEdgeOmega(workingFolder + "omega_tar.txt", tarOmega);
		saveVertexAmp(workingFolder + "amp_tar.txt", tarAmp);
		saveVertexZvals(workingFolder + "zvals_tar.txt", tarZvals);
	}

	igl::writeOBJ(workingFolder + "mesh.obj", triV, triF);

	std::string outputFolder = workingFolder + "/optZvals/";
	mkdir(outputFolder);

	std::string omegaOutputFolder = workingFolder + "/optOmega/";
	mkdir(omegaOutputFolder);

	std::string refOmegaOutputFolder = workingFolder + "/refOmega/";
	mkdir(refOmegaOutputFolder);

	// save reference
	std::string refAmpOutputFolder = workingFolder + "/refAmp/";
	mkdir(refAmpOutputFolder);

	int nframes = zList.size();
	auto savePerFrame = [&](const tbb::blocked_range<uint32_t>& range)
	{
		for (uint32_t i = range.begin(); i < range.end(); ++i)
		{

			saveVertexZvals(outputFolder + "zvals_" + std::to_string(i) + ".txt", zList[i]);
			saveEdgeOmega(omegaOutputFolder + "omega_" + std::to_string(i) + ".txt", omegaList[i]);
			saveVertexAmp(refAmpOutputFolder + "amp_" + std::to_string(i) + ".txt", refAmpList[i]);
			saveEdgeOmega(refOmegaOutputFolder + "omega_" + std::to_string(i) + ".txt", refOmegaList[i]);
		}
	};

	tbb::blocked_range<uint32_t> rangex(0u, (uint32_t)nframes, GRAIN_SIZE);
	tbb::parallel_for(rangex, savePerFrame);


	std::ofstream o(args.input);
	o << std::setw(4) << jval << std::endl;
	std::cout << "save file in: " << args.input << std::endl;

	return true;
}

bool saveForRender()
{
	// render information
	std::string renderFolder = workingFolder + "/render/";
	mkdir(renderFolder);
	igl::writeOBJ(renderFolder + "basemesh.obj", triV, triF);
	igl::writeOBJ(renderFolder + "upmesh.obj", upsampledTriV, upsampledTriF);


	saveFlag4Render(faceFlags, renderFolder + "faceFlags.cvs");

	std::string outputFolderAmp = renderFolder + "/upsampledAmp/";
	mkdir(outputFolderAmp);

	std::string outputFolderPhase = renderFolder + "/upsampledPhase/";
	mkdir(outputFolderPhase);

	std::string outputFolderWrinkles = renderFolder + "/wrinkledMesh/";
	mkdir(outputFolderWrinkles);

	std::string refAmpFolder = renderFolder + "/refAmp/";
	mkdir(refAmpFolder);
	std::string refOmegaFolder = renderFolder + "/refOmega/";
	mkdir(refOmegaFolder);

	std::string optAmpFolder = renderFolder + "/optAmp/";
	mkdir(optAmpFolder);
	std::string optOmegaFolder = renderFolder + "/optOmega/";
	mkdir(optOmegaFolder);

	int nframes = ampFieldsList.size();

	auto savePerFrame = [&](const tbb::blocked_range<uint32_t>& range)
	{
		for (uint32_t i = range.begin(); i < range.end(); ++i)
		{

			// upsampled information
			igl::writeOBJ(outputFolderWrinkles + "wrinkledMesh_" + std::to_string(i) + ".obj", wrinkledVList[i], upsampledTriF);
			Eigen::MatrixXd lapWrinkledV;
			laplacianSmoothing(wrinkledVList[i], upsampledTriF, lapWrinkledV, smoothingRatio, smoothingTimes, isFixedBnd);
			igl::writeOBJ(outputFolderWrinkles + "wrinkledMeshSmoothed_" + std::to_string(i) + ".obj", lapWrinkledV, upsampledTriF);

			saveAmp4Render(ampFieldsList[i], outputFolderAmp + "upAmp_" + std::to_string(i) + ".cvs", globalAmpMin, globalAmpMax);
			savePhi4Render(phaseFieldsList[i], outputFolderPhase + "upPhase" + std::to_string(i) + ".cvs");
			saveDphi4Render(subFaceOmegaList[i], subSecMesh, outputFolderPhase + "upOmega" + std::to_string(i) + ".cvs");

			// reference information
			Eigen::MatrixXd refFaceOmega = intrinsicEdgeVec2FaceVec(refOmegaList[i], triV, triMesh);
			saveAmp4Render(refAmpList[i], refAmpFolder + "refAmp_" + std::to_string(i) + ".cvs", globalAmpMin, globalAmpMax);
			saveDphi4Render(refFaceOmega, triMesh, triV, refOmegaFolder + "refOmega_" + std::to_string(i) + ".cvs");

			// optimal information
			saveDphi4Render(faceOmegaList[i], triMesh, triV, optOmegaFolder + "optOmega_" + std::to_string(i) + ".cvs");
			Eigen::VectorXd baseAmplitude = refAmpList[i];
			for(int j = 0 ; j < refAmpList[i].size(); j++)
			{
				baseAmplitude(j) = std::abs(zList[i][j]);
			}

			saveAmp4Render(baseAmplitude, optAmpFolder + "optAmp_" + std::to_string(i) + ".cvs", globalAmpMin, globalAmpMax);
		}
	};

	tbb::blocked_range<uint32_t> rangex(0u, (uint32_t)nframes);
	tbb::parallel_for(rangex, savePerFrame);

	return true;
}


int main(int argc, char** argv)
{
	CLI::App app("Wrinkle Interpolation");
	app.add_option("input,-i,--input", args.input, "Input model")->required()->check(CLI::ExistingFile);
	app.add_option("-g,--gradTol", args.gradTol, "The gradient tolerance for optimization.");
	app.add_option("-x,--xTol", args.xTol, "The variable update tolerance for optimization.");
	app.add_option("-f,--fTol", args.fTol, "The functio value update tolerance for optimization.");
	app.add_option("-n,--numIter", args.numIter, "The number of iteration for optimization.");
	app.add_option("-a,--ampScaling", args.ampScale, "The amplitude scaling for wrinkled surface upsampling.");
	app.add_option("-r,--reoptimizae", args.reOptimize, "The amplitude scaling for wrinkled surface upsampling.");

	try {
		app.parse(argc, argv);
	}
	catch (const CLI::ParseError& e) {
		return app.exit(e);
	}

	if (!loadProblem())
	{
		std::cout << "failed to load file." << std::endl;
		return 1;
	}

	updateEditionDomain();
	if (args.reOptimize)
	{
		// solve for the path from source to target
		solveKeyFrames(initZvals, initOmega, faceFlags, omegaList, zList);
	}
	updateEverythingForSaving();
	
	saveProblem();
	saveForRender();

	return 0;
}