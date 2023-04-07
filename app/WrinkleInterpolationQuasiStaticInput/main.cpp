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
#include <igl/cylinder.h>
#include "polyscope/messages.h"
#include "polyscope/point_cloud.h"
#include "polyscope/surface_mesh.h"
#include "polyscope/view.h"

#include <iostream>
#include <filesystem>
#include <utility>

#include "../../include/testMeshGeneration.h"

#include "../../include/CommonTools.h"
#include "../../include/MeshLib/MeshUpsampling.h"
#include "../../include/Visualization/PaintGeometry.h"
#include "../../include/Optimization/NewtonDescent.h"
#include "../../include/IntrinsicFormula/InterpolateZvals.h"
#include "../../include/IntrinsicFormula/WrinkleEditingBaseMeshSeqCWF.h"

#include "../../include/IntrinsicFormula/KnoppelStripePatternEdgeOmega.h"
#include "../../include/WrinkleFieldsEditor.h"
#include "../../dep/SecStencils/types.h"
#include "../../dep/SecStencils/Subd.h"
#include "../../dep/SecStencils/utils.h"

#include "../../include/json.hpp"
#include "../../include/ComplexLoop/ComplexLoop.h"
#include "../../include/ComplexLoop/ComplexLoopZuenko.h"
#include "../../include/LoadSaveIO.h"
#include "../../include/SecMeshParsing.h"
#include "../../include/MeshLib/RegionEdition.h"

#include <CLI/CLI.hpp>

std::vector<Eigen::MatrixXd> triVList, upsampledTriVList;
std::vector<Mesh> secMeshList, subSecMeshList;

Eigen::MatrixXi triF, upsampledTriF;
MeshConnectivity triMesh;

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
std::vector<Eigen::VectorXd> refF2List;

// delta omega
std::vector<Eigen::VectorXd> deltaOmegaList;
std::vector<Eigen::VectorXd> actualOmegaList;

int upsampleTimes = 0;

bool isShowVectorFields = true;
bool isShowWrinkels = true;

PaintGeometry mPaint;

int numFrames = 20;
int curFrame = 0;

double globalAmpMax = 1;
double globalAmpMin = 0;

double globalFMax = 1;
double globalFmin = 0;

double globalInconMax = 1;
double globalInconMin = 0;

double globalCoarseInconMax = 1;
double globalCoarseInconMin = 0;

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

std::shared_ptr<IntrinsicFormula::WrinkleEditingBaseMeshSeqCWF> editModel;

// smoothing
int smoothingTimes = 0;
double smoothingRatio = 0.95;

bool isFixedBnd = false;
int effectivedistFactor = 4;

bool isShowActualKnoppel = false;
std::vector<Eigen::VectorXd> knoppelPhiList;
Eigen::MatrixXd knoppelUpV;
Eigen::MatrixXi knoppelUpF;

int optTimes = 5;

bool isLoadOpt;

int clickedFid = -1;
int dilationTimes = 10;

bool isUseV2 = false;
bool isForceReinitilaize = true;

static void buildEditModel(const std::vector<Eigen::MatrixXd>& posList, const MeshConnectivity& mesh, int quadOrd, double spatialAmpRatio, double spatialKnoppelRatio, std::shared_ptr<IntrinsicFormula::WrinkleEditingBaseMeshSeqCWF>& editModel)
{
	editModel = std::make_shared<IntrinsicFormula::WrinkleEditingBaseMeshSeqCWF>(posList, mesh, quadOrd, spatialAmpRatio, spatialKnoppelRatio);
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

			consistencyVec[i] = inconsistencyComputation(secMeshList[i], edgeVec, zFrames[i]);

			std::shared_ptr<ComplexLoop> complexLoopOpt = std::make_shared<ComplexLoopZuenko>();
			complexLoopOpt->setBndFixFlag(isFixedBnd);
			complexLoopOpt->SetMesh(secMeshList[i]);
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


void updateWrinkles(const std::vector<Eigen::MatrixXd>& VList, const Eigen::MatrixXi& F, const std::vector<std::vector<std::complex<double>>>& zFrames, std::vector<Eigen::MatrixXd>& wrinkledVFrames, double scaleRatio, bool isUseV2)
{
	std::vector<std::vector<std::complex<double>>> interpZList(zFrames.size());
	wrinkledVFrames.resize(zFrames.size());

	std::vector<std::vector<int>> vertNeiEdges;
	std::vector<std::vector<int>> vertNeiFaces;

	buildVertexNeighboringInfo(MeshConnectivity(F), VList[0].rows(), vertNeiEdges, vertNeiFaces);

	auto computeWrinkles = [&](const tbb::blocked_range<uint32_t>& range)
	{
		for (uint32_t i = range.begin(); i < range.end(); ++i)
		{
			getWrinkledMesh(VList[i], F, zFrames[i], &vertNeiFaces, wrinkledVFrames[i], scaleRatio, isUseV2);
		}
	};

	tbb::blocked_range<uint32_t> rangex(0u, (uint32_t)interpZList.size());
	tbb::parallel_for(rangex, computeWrinkles);


}

void getUpsampledMesh(const std::vector<Eigen::MatrixXd>& triVList, const Eigen::MatrixXi& triF, std::vector<Eigen::MatrixXd>& upsampledTriVList, Eigen::MatrixXi& upsampledTriF)
{
	secMeshList.resize(triVList.size());
	subSecMeshList.resize(triVList.size());
	upsampledTriVList.resize(triVList.size());

	for (int i = 0; i < triVList.size(); i++)
	{
		secMeshList[i] = convert2SecMesh(triVList[i], triF);
		subSecMeshList[i] = secMeshList[i];

		std::shared_ptr<ComplexLoop> complexLoopOpt = std::make_shared<ComplexLoopZuenko>();
		complexLoopOpt->setBndFixFlag(isFixedBnd);
		complexLoopOpt->SetMesh(secMeshList[i]);
		complexLoopOpt->meshSubdivide(upsampleTimes);
		subSecMeshList[i] = complexLoopOpt->GetMesh();
		parseSecMesh(subSecMeshList[i], upsampledTriVList[i], upsampledTriF);
	}
	
}

void initialization(const std::vector<Eigen::MatrixXd>& triVList, const Eigen::MatrixXi& triF, std::vector<Eigen::MatrixXd>& upsampledTriVList, Eigen::MatrixXi& upsampledTriF)
{
	getUpsampledMesh(triVList, triF, upsampledTriVList, upsampledTriF);
}

void updatePaintingItems()
{
	// get interploated amp and phase frames
	std::cout << "compute upsampled phase: " << std::endl;
	updateMagnitudePhase(omegaList, zList, ampFieldsList, phaseFieldsList, upZList);

	std::cout << "compute wrinkle meshes: " << std::endl;
	updateWrinkles(upsampledTriVList, upsampledTriF, upZList, wrinkledVList, wrinkleAmpScalingRatio, isUseV2);


	std::cout << "compute face vector fields:" << std::endl;
	faceOmegaList.resize(omegaList.size());
	for (int i = 0; i < omegaList.size(); i++)
	{
		faceOmegaList[i] = intrinsicEdgeVec2FaceVec(omegaList[i], triVList[i], triMesh);
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


    // update global maximum f
    std::cout << "update max and min f. " << std::endl;

    globalFMax = std::max(refF2List[0].maxCoeff(), refF2List[0].maxCoeff());
    globalFmin = std::min(refF2List[0].minCoeff(), refF2List[0].minCoeff());
    for (int i = 1; i < refF2List.size(); i++)
    {
        globalFMax = std::max(globalAmpMax, std::max(refF2List[i].maxCoeff(), refF2List[i].maxCoeff()));
        globalFmin = std::min(globalAmpMin, std::min(refF2List[i].minCoeff(), refF2List[i].minCoeff()));
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

void reinitializeKeyFrames(const std::vector<std::complex<double>>& initZvals, const Eigen::VectorXd& initOmega, std::vector<Eigen::VectorXd>& wFrames, std::vector<std::vector<std::complex<double>>>& zFrames)
{
	editModel->initialization(initZvals, initOmega, tarZvals, tarOmega, true);
		
	std::cout << "initilization finished!" << std::endl;
	refOmegaList = editModel->getRefWList();
	refAmpList = editModel->getRefAmpList();
    refF2List = editModel->getF2List();

    deltaOmegaList = editModel->getDeltaWList();
	actualOmegaList = editModel->getActualOptWList();

	std::cout << "get w list" << std::endl;
	wFrames = editModel->getWList();
	std::cout << "get z list" << std::endl;
	zFrames = editModel->getVertValsList();

	std::cout << "check boundary matches: " << std::endl;

	std::cout << "omega match: " << std::endl;
	std::cout << (wFrames[0] - initOmega).norm() << std::endl;
	std::cout << (wFrames[wFrames.size() - 1] - tarOmega).norm() << std::endl;

	std::cout << "z match: " << std::endl;
	std::cout << getZListNorm(zFrames[0]) - getZListNorm(initZvals) << std::endl;
	std::cout << getZListNorm(zFrames[zFrames.size() - 1]) - getZListNorm(tarZvals) << std::endl;
}

void solveKeyFrames(const std::vector<std::complex<double>>& initzvals, const Eigen::VectorXd& initOmega, std::vector<Eigen::VectorXd>& wFrames, std::vector<std::vector<std::complex<double>>>& zFrames)
{
	Eigen::VectorXd x;
	editModel->setSaveFolder(workingFolder);

	if (isForceReinitilaize)
	{
		editModel->initialization(initZvals, initOmega, tarZvals, tarOmega, true);
	}
	editModel->convertList2Variable(x);
//	editModel->testEnergy(x);

	editModel->solveIntermeditateFrames(x, numIter, gradTol, xTol, fTol, true, workingFolder);
	editModel->convertVariable2List(x);
	refOmegaList = editModel->getRefWList();
	refAmpList = editModel->getRefAmpList();
    refF2List = editModel->getF2List();

    deltaOmegaList = editModel->getDeltaWList();
	actualOmegaList = editModel->getActualOptWList();

	std::cout << "get w list" << std::endl;
	wFrames = editModel->getWList();
	std::cout << "get z list" << std::endl;
	zFrames = editModel->getVertValsList();

	std::cout << (wFrames[0] - initOmega).norm() << std::endl;
}

void registerMesh(int frameId, bool isFirstTime = true)
{
	int curShift = 0;
	double shiftx = 1.5 * (triVList[frameId].col(0).maxCoeff() - triVList[frameId].col(0).minCoeff());
	if (isFirstTime)
	{
		polyscope::registerSurfaceMesh("base mesh", triVList[frameId], triF);
	}
	else
		polyscope::getSurfaceMesh("base mesh")->updateVertexPositions(triVList[frameId]);
	
	polyscope::getSurfaceMesh("base mesh")->addFaceVectorQuantity("frequency field", vecratio * faceOmegaList[frameId], polyscope::VectorType::AMBIENT);

	Eigen::VectorXd baseAmplitude = refAmpList[frameId];
	for(int i = 0 ; i < refAmpList[frameId].size(); i++)
	{
		baseAmplitude(i) = std::abs(zList[frameId][i]);
	}
	auto baseAmp = polyscope::getSurfaceMesh("base mesh")->addVertexScalarQuantity("opt amplitude", baseAmplitude);
	baseAmp->setMapRange(std::pair<double, double>(globalAmpMin, globalAmpMax));
	polyscope::getSurfaceMesh("base mesh")->addFaceVectorQuantity("opt frequency field", vecratio * faceOmegaList[frameId], polyscope::VectorType::AMBIENT);

	auto consPatterns = polyscope::getSurfaceMesh("base mesh")->addVertexScalarQuantity("inconsistency", consistencyVec[frameId]);
	consPatterns->setMapRange(std::pair<double, double>(globalCoarseInconMin, globalCoarseInconMax));
	curShift++;

	Eigen::MatrixXd refFaceOmega = intrinsicEdgeVec2FaceVec(refOmegaList[frameId], triVList[frameId], triMesh);
    Eigen::MatrixXd deltaFaceOmega = intrinsicEdgeVec2FaceVec(deltaOmegaList[frameId], triVList[frameId], triMesh);
	Eigen::MatrixXd actualFaceOmega = intrinsicEdgeVec2FaceVec(actualOmegaList[frameId], triVList[frameId], triMesh);
	if (isFirstTime)
	{
		polyscope::registerSurfaceMesh("reference mesh", triVList[frameId], triF);
		polyscope::getSurfaceMesh("reference mesh")->translate({ curShift * shiftx, 0, 0 });
	}
	else
		polyscope::getSurfaceMesh("reference mesh")->updateVertexPositions(triVList[frameId]);
		
	auto refAmp = polyscope::getSurfaceMesh("reference mesh")->addVertexScalarQuantity("reference amplitude", refAmpList[frameId]);
	refAmp->setMapRange(std::pair<double, double>(globalAmpMin, globalAmpMax));
    auto reff = polyscope::getSurfaceMesh("reference mesh")->addVertexScalarQuantity("f", refF2List[frameId]);
    reff->setMapRange(std::pair<double, double>(globalFmin, globalFMax));
	polyscope::getSurfaceMesh("reference mesh")->addFaceVectorQuantity("reference frequency field", vecratio * refFaceOmega, polyscope::VectorType::AMBIENT);
    polyscope::getSurfaceMesh("reference mesh")->addFaceVectorQuantity("delta frequency field", vecratio * deltaFaceOmega, polyscope::VectorType::AMBIENT);
	polyscope::getSurfaceMesh("reference mesh")->addFaceVectorQuantity("actual opt frequency field", vecratio * actualFaceOmega, polyscope::VectorType::AMBIENT);
	curShift++;

	// phase pattern
	if (isFirstTime)
	{
		polyscope::registerSurfaceMesh("phase mesh", upsampledTriVList[frameId], upsampledTriF);
		polyscope::getSurfaceMesh("phase mesh")->translate({ curShift * shiftx, 0, 0 });
	}
	else
		polyscope::getSurfaceMesh("phase mesh")->updateVertexPositions(upsampledTriVList[frameId]);
			
	mPaint.setNormalization(false);
	Eigen::MatrixXd phaseColor = mPaint.paintPhi(phaseFieldsList[frameId]);
	polyscope::getSurfaceMesh("phase mesh")->addVertexColorQuantity("vertex phi", phaseColor);
	curShift++;

	// amp pattern
	if (isFirstTime)
	{
		polyscope::registerSurfaceMesh("upsampled ampliude and frequency mesh", upsampledTriVList[frameId], upsampledTriF);
		polyscope::getSurfaceMesh("upsampled ampliude and frequency mesh")->translate({ curShift * shiftx, 0, 0 });
	}
	else
		polyscope::getSurfaceMesh("upsampled ampliude and frequency mesh")->updateVertexPositions(upsampledTriVList[frameId]);
		
	auto ampPatterns = polyscope::getSurfaceMesh("upsampled ampliude and frequency mesh")->addVertexScalarQuantity("vertex amplitude", ampFieldsList[frameId]);
	ampPatterns->setMapRange(std::pair<double, double>(globalAmpMin, globalAmpMax));
	polyscope::getSurfaceMesh("upsampled ampliude and frequency mesh")->addFaceVectorQuantity("subdivided frequency field", vecratio * subFaceOmegaList[frameId], polyscope::VectorType::AMBIENT);
	auto upconsPatterns = polyscope::getSurfaceMesh("upsampled ampliude and frequency mesh")->addVertexScalarQuantity("inconsistency", upConsistencyVec[frameId]);
	upconsPatterns->setMapRange(std::pair<double, double>(globalInconMin, globalInconMax));
	curShift++;

	// wrinkle mesh
	Eigen::MatrixXd lapWrinkledV = wrinkledVList[frameId];
	laplacianSmoothing(wrinkledVList[frameId], upsampledTriF, lapWrinkledV, smoothingRatio, smoothingTimes, isFixedBnd);
	if(isFirstTime)
	{
		polyscope::registerSurfaceMesh("wrinkled mesh", lapWrinkledV, upsampledTriF);
		polyscope::getSurfaceMesh("wrinkled mesh")->setSurfaceColor({ 80 / 255.0, 122 / 255.0, 91 / 255.0 });
		polyscope::getSurfaceMesh("wrinkled mesh")->translate({ curShift * shiftx, 0, 0 });
	}
		
	else
		polyscope::getSurfaceMesh("wrinkled mesh")->updateVertexPositions(lapWrinkledV);
}

void updateFieldsInView(int frameId, bool isFirstTime)
{
	//std::cout << "update viewer. " << std::endl;
	registerMesh(frameId, isFirstTime);
}


bool loadProblem(std::string loadFileName = "")
{
	if(loadFileName == "")
		loadFileName = igl::file_dialog_open();

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

	std::string meshFile = jval["mesh_name_prefix"];
	upsampleTimes = jval["upsampled_times"];
	if (upsampleTimes > 2)
		upsampleTimes = 2;
	numFrames = jval["num_frame"];
	triVList.resize(numFrames);

	meshFile = workingFolder + "/basemesh/" + meshFile;

	for (int i = 0; i < numFrames; i++)
	{
		igl::readOBJ(meshFile + "_" + std::to_string(i) + ".obj", triVList[i], triF);
	}
	
	triMesh = MeshConnectivity(triF);
	initialization(triVList, triF, upsampledTriVList, upsampledTriF);
	quadOrder = jval["quad_order"];

	if (jval.contains(std::string_view{ "wrinkle_amp_ratio" }))
	{
		wrinkleAmpScalingRatio = jval["wrinkle_amp_ratio"];
	}
	std::cout << "wrinkle amplitude scaling ratio: " << wrinkleAmpScalingRatio << std::endl;

	if (jval.contains(std::string_view{ "spatial_ratio" }))
	{
		if (jval["spatial_ratio"].contains(std::string_view{ "amp_ratio" }))
			spatialAmpRatio = jval["spatial_ratio"]["amp_ratio"];
		else
			spatialAmpRatio = 100;

		if (jval["spatial_ratio"].contains(std::string_view{ "edge_ratio" }))
			spatialEdgeRatio = jval["spatial_ratio"]["edge_ratio"];
		else
			spatialEdgeRatio = 100;

		if (jval["spatial_ratio"].contains(std::string_view{ "knoppel_ratio" }))
			spatialKnoppelRatio = jval["spatial_ratio"]["knoppel_ratio"];
		else
			spatialKnoppelRatio = 100;
	}
	else
	{
		spatialAmpRatio = 100;
		spatialEdgeRatio = 100;
		spatialKnoppelRatio = 100;
	}

	buildEditModel(triVList, triMesh, quadOrder, spatialAmpRatio, spatialKnoppelRatio, editModel);


	int nedges = triMesh.nEdges();
	int nverts = triVList[0].rows();

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

	if (!loadVertexZvals(workingFolder + initZValsPath, nverts, initZvals))
	{
		std::cout << "missing init zval file, try to load amp file, and round zvals from amp and omega" << std::endl;
		if (!loadVertexAmp(workingFolder + initAmpPath, nverts, initAmp))
		{
			std::cout << "missing init amp file: " << std::endl;
			return false;
		}

		else
		{
			Eigen::VectorXd edgeArea, vertArea;
			edgeArea = getEdgeArea(triVList[0], triMesh);
			vertArea = getVertArea(triVList[0], triMesh);

            double rescaling = 18.0 / 16.0;

            initAmp *= rescaling;
            initOmega /= rescaling;

            IntrinsicFormula::roundZvalsFromEdgeOmega(triMesh, initOmega, edgeArea, vertArea, nverts, initZvals);
            for(int i = 0; i < nverts; i++)
            {
                auto z = initZvals[i];
                z = initAmp(i) * std::complex<double>(std::cos(std::arg(z)), std::sin(std::arg(z)));
                initZvals[i] = z;
            }
		}
	}
	else
	{
		initAmp.setZero(nverts);
		for (int i = 0; i < initZvals.size(); i++)
		{
			initAmp(i) = std::abs(initZvals[i]);
		}

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
	bool loadTar = true;
	tarOmega.resize(0);
	tarZvals = {};

	if (!loadEdgeOmega(workingFolder + tarOmegaPath, nedges, tarOmega)) {
		std::cout << "missing tar edge omega file." << std::endl;
		loadTar = false;
	}

	if (!loadVertexZvals(workingFolder + tarZValsPath, nverts, tarZvals))
	{
		std::cout << "missing tar zval file, try to load amp file, and round zvals from amp and omega" << std::endl;
		if (!loadVertexAmp(workingFolder + tarAmpPath, nverts, tarAmp))
		{
			std::cout << "missing tar amp file: " << std::endl;
			loadTar = false;
		}

		else
        {
            Eigen::VectorXd edgeArea, vertArea;
            edgeArea = getEdgeArea(triVList[numFrames - 1], triMesh);
            vertArea = getVertArea(triVList[numFrames - 1], triMesh);

            double rescaling = 26.0 / 13.0;
            tarOmega /= -rescaling;
            tarAmp *= rescaling;

            int nedges = triMesh.nEdges();
            double changeZ = 0.2;

//            for (int i = 0; i < nedges; i++)
//            {
//                double centerZ = triVList[numFrames - 1](triMesh.edgeVertex(i, 0), 2) +
//                                 triVList[numFrames - 1](triMesh.edgeVertex(i, 1), 2);
//                centerZ /= 2.0;
//                centerZ = std::abs(centerZ);
//                if(centerZ > changeZ)
//                {
////                    double scalingRatio = 1 / (0.5 - changeZ) / (0.5 - changeZ) * (centerZ - changeZ) * (centerZ - changeZ) + 1;
//                    double scalingRatio = (centerZ - changeZ) / (0.5 - changeZ) * (rescaling - 1) + 1;
//                    tarOmega[i] *= scalingRatio;
//
////                    tarOmega[i] *= 3;
//                }
//
//            }
//
//            int nverts = triVList[numFrames - 1].rows();
//            for (int i = 0; i < nverts; i++)
//            {
//                double centerZ = triVList[numFrames - 1](i, 2);
//                centerZ = std::abs(centerZ);
//                if(centerZ > changeZ)
//                {
////                    double scalingRatio = 1 / (0.5 - changeZ) / (0.5 - changeZ) * (centerZ - changeZ) * (centerZ - changeZ) + 1;
//                    double scalingRatio = (centerZ - changeZ) / (0.5 - changeZ) * (rescaling - 1) + 1;
//                    tarAmp[i] /= scalingRatio;
//
////                    tarAmp[i] /= 2;
//                }
//            }

			IntrinsicFormula::roundZvalsFromEdgeOmega(triMesh, tarOmega, edgeArea, vertArea, nverts, tarZvals);
            for(int i = 0; i < nverts; i++)
            {
                auto z = tarZvals[i];
                z = tarAmp(i) * std::complex<double>(std::cos(std::arg(z)), std::sin(std::arg(z)));
                tarZvals[i] = z;
            }
		}
	}
	else
	{
		tarAmp.setZero(nverts);
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

		if (!loadVertexAmp(workingFolder + refAmp + "/amp_" + std::to_string(i) + ".txt", nverts, refAmpList[i]))
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
        deltaOmegaList.resize(omegaList.size(), Eigen::VectorXd::Zero(omegaList[0].size()));
		actualOmegaList.resize(omegaList.size(), Eigen::VectorXd::Zero(omegaList[0].size()));
	}
	if (!isLoadOpt || !isLoadRef)
	{
		editModel->initialization(initZvals, initOmega, tarZvals, tarOmega, true);

		refAmpList = editModel->getRefAmpList();
		refOmegaList = editModel->getRefWList();
        deltaOmegaList = editModel->getDeltaWList();
		actualOmegaList = editModel->getActualOptWList();

		if (!isLoadOpt)
		{
			zList = editModel->getVertValsList();
			omegaList = editModel->getWList();
		}
		
	}
	else
	{
		editModel->initialization(zList, omegaList, refAmpList, refOmegaList);
	}
    refF2List = refAmpList;

	updatePaintingItems();

	curFrame = 0;

	return true;
}


bool saveProblem()
{
	std::string saveFileName = igl::file_dialog_save();

	using json = nlohmann::json;
	json jval =
	{
			{"mesh_name_prefix",         "mesh"},
			{"num_frame",         zList.size()},
			{"wrinkle_amp_ratio", wrinkleAmpScalingRatio},
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


	saveEdgeOmega(workingFolder + "omega.txt", initOmega);
	saveVertexAmp(workingFolder + "amp.txt", initAmp);
	saveVertexZvals(workingFolder + "zvals.txt", initZvals);

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


	std::ofstream o(saveFileName);
	o << std::setw(4) << jval << std::endl;
	std::cout << "save file in: " << saveFileName << std::endl;

	return true;
}

bool saveForRender()
{
	std::string saveFileName = igl::file_dialog_save();
	std::string filePath = saveFileName;
	std::replace(filePath.begin(), filePath.end(), '\\', '/'); // handle the backslash issue for windows
	int id = filePath.rfind("/");
	std::string workingFolder = filePath.substr(0, id + 1);

	// render information
	std::string renderFolder = workingFolder + "/render/";
	mkdir(renderFolder);

	std::string basemeshFolder = renderFolder + "/basemesh/";
	mkdir(basemeshFolder);

	for (int i = 0; i < triVList.size(); i++)
	{
		igl::writeOBJ(basemeshFolder + "/basemesh_" + std::to_string(i) + ".obj", triVList[i], triF);
	}

	std::string upmeshFolder = renderFolder + "/upsampledMesh/";
	mkdir(upmeshFolder);

	for (int i = 0; i < triVList.size(); i++)
	{
		igl::writeOBJ(upmeshFolder + "/upmesh_" + std::to_string(i) + ".obj", upsampledTriVList[i], upsampledTriF);
	}

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
			saveDphi4Render(subFaceOmegaList[i], subSecMeshList[i], outputFolderPhase + "upOmega" + std::to_string(i) + ".cvs");

			// reference information
			Eigen::MatrixXd refFaceOmega = intrinsicEdgeVec2FaceVec(refOmegaList[i], triVList[i], triMesh);
			saveAmp4Render(refAmpList[i], refAmpFolder + "refAmp_" + std::to_string(i) + ".cvs", globalAmpMin, globalAmpMax);
			saveDphi4Render(refFaceOmega, triMesh, triVList[i], refOmegaFolder + "refOmega_" + std::to_string(i) + ".cvs");

			// optimal information
			saveDphi4Render(faceOmegaList[i], triMesh, triVList[i], optOmegaFolder + "optOmega_" + std::to_string(i) + ".cvs");
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

void callback() {
	ImGui::PushItemWidth(100);
	float w = ImGui::GetContentRegionAvailWidth();
	float p = ImGui::GetStyle().FramePadding.x;
	if (ImGui::Button("Load", ImVec2((w - p) / 2.f, 0)))
	{
		loadProblem();
		updateMagnitudePhase(omegaList, zList, ampFieldsList, phaseFieldsList, upZList);
		updateFieldsInView(curFrame, true);
	}
	ImGui::SameLine(0, p);
	if (ImGui::Button("Save", ImVec2((w - p) / 2.f, 0)))
	{
		saveProblem();
	}
	if (ImGui::Button("save for render", ImVec2(-1, 0)))
	{
		saveForRender();
	}
	ImGuiTabBarFlags tab_bar_flags = ImGuiTabBarFlags_None;
	if (ImGui::BeginTabBar("Visualization Options", tab_bar_flags))
	{
		if (ImGui::BeginTabItem("Wrinkle Mesh Upsampling"))
		{
			if (ImGui::InputInt("upsampled level", &upsampleTimes))
			{
				if (upsampleTimes >= 0)
				{
					getUpsampledMesh(triVList, triF, upsampledTriVList, upsampledTriF);
					updatePaintingItems();
					updateFieldsInView(curFrame, true);
				}
			}
			if (ImGui::Checkbox("fix bnd", &isFixedBnd))
			{
				getUpsampledMesh(triVList, triF, upsampledTriVList, upsampledTriF);
				updatePaintingItems();
				updateFieldsInView(curFrame, true);
			}
			if (ImGui::Checkbox("show actual knoppel", &isShowActualKnoppel))
			{
				getUpsampledMesh(triVList, triF, upsampledTriVList, upsampledTriF);
				updatePaintingItems();
				updateFieldsInView(curFrame, true);
			}
			ImGui::EndTabItem();
		}
		if (ImGui::BeginTabItem("Wrinkle Mesh Smoothing"))
		{
			if (ImGui::InputInt("smoothing times", &smoothingTimes))
			{
				smoothingTimes = smoothingTimes > 0 ? smoothingTimes : 0;
//				updatePaintingItems();
				updateFieldsInView(curFrame, false);
			}
			if (ImGui::InputDouble("smoothing ratio", &smoothingRatio))
			{
				smoothingRatio = smoothingRatio > 0 ? smoothingRatio : 0;
//				updatePaintingItems();
				updateFieldsInView(curFrame, false);
			}
			ImGui::EndTabItem();
		}
		ImGui::EndTabBar();
	}

	if (ImGui::CollapsingHeader("Visualization Options", ImGuiTreeNodeFlags_DefaultOpen))
	{
		if (ImGui::Checkbox("is show vector fields", &isShowVectorFields))
		{
			updateFieldsInView(curFrame, false);
		}
		if (ImGui::Checkbox("is show wrinkled mesh", &isShowWrinkels))
		{
			updateFieldsInView(curFrame, false);
		}
		if (ImGui::Checkbox("is use v2", &isUseV2))
		{
			updateFieldsInView(curFrame, false);
		}
		if (ImGui::DragFloat("wrinkle amp scaling ratio", &wrinkleAmpScalingRatio, 0.0005, 0, 1))
		{
			if (wrinkleAmpScalingRatio >= 0)
				updateFieldsInView(curFrame, false);
		}
	}

	
	if (ImGui::CollapsingHeader("Frame Visualization Options", ImGuiTreeNodeFlags_DefaultOpen))
	{
//		if (ImGui::SliderInt("current frame", &curFrame, 0, numFrames - 1))
		if (ImGui::DragInt("current frame", &curFrame, 1, 0, numFrames - 1))
		{
			curFrame = curFrame % (numFrames + 1);
			updateFieldsInView(curFrame, false);
		}
		if (ImGui::SliderInt("current frame slider bar", &curFrame, 0, numFrames - 1))
		{
			curFrame = curFrame % (numFrames + 1);
			updateFieldsInView(curFrame, false);
		}
		if (ImGui::DragFloat("vec ratio", &(vecratio), 0.00005, 0, 1))
		{
			updateFieldsInView(curFrame, false);
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

		if (ImGui::InputInt("effective factor", &effectivedistFactor))
		{
			if (effectivedistFactor < 0)
				effectivedistFactor = 4;
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

		ImGui::Checkbox("reinitialize before solve", &isForceReinitilaize);

	}
	if (ImGui::Button("Reinitialization", ImVec2((w - p) / 2.f, 0)))
	{
		// solve for the path from source to target
		reinitializeKeyFrames(initZvals, initOmega, omegaList, zList);
		updatePaintingItems();
		updateFieldsInView(curFrame, false);
	}
	ImGui::SameLine();
	if (ImGui::Button("Solve", ImVec2((w - p) / 2.f, 0)))
	{
		// solve for the path from source to target
		solveKeyFrames(initZvals, initOmega, omegaList, zList);
		updatePaintingItems();
		updateFieldsInView(curFrame, false);
	}

	if (ImGui::Button("update viewer", ImVec2(-1, 0)))
	{
		updatePaintingItems();
		updateFieldsInView(curFrame, false);
	}

	if (ImGui::Button("output images", ImVec2(-1, 0)))
	{
		std::string curFolder = workingFolder + "/screenshots/";
		mkdir(curFolder);
		std::cout << "save folder: " << curFolder << std::endl;
		for (int i = 0; i < ampFieldsList.size(); i++)
		{
			updateFieldsInView(i, false);
			//polyscope::options::screenshotExtension = ".jpg";
			std::string name = curFolder + "/output_" + std::to_string(i) + ".jpg";
			polyscope::screenshot(name);
		}
	}


	ImGui::PopItemWidth();
}


int main(int argc, char** argv)
{
	std::string inputFile = "";
	CLI::App app("Wrinkle Interpolation");
	app.add_option("input,-i,--input", inputFile, "Input model")->check(CLI::ExistingFile);

	try {
		app.parse(argc, argv);
	}
	catch (const CLI::ParseError& e) {
		return app.exit(e);
	}

	if (!loadProblem(inputFile))
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

	updateFieldsInView(curFrame, true);
	// Show the gui
	polyscope::show();


	return 0;
}