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
#include "../../include/IntrinsicFormula/WrinkleEditingLocalModel.h"
#include "../../include/IntrinsicFormula/WrinkleEditingModel.h"
#include "../../include/IntrinsicFormula/WrinkleEditingGlobalModel.h"
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

// reference amp and omega
std::vector<Eigen::VectorXd> refOmegaList;
std::vector<Eigen::VectorXd> refAmpList;

// region edition
RegionEdition regEdt;

int upsampleTimes = 2;

bool isForceOptimize = false;
bool isShowVectorFields = true;
bool isShowWrinkels = true;

PaintGeometry mPaint;

int numFrames = 20;
int curFrame = 0;

double globalAmpMax = 1;
double globalAmpMin = 0;


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

bool isRotate = false;

std::string workingFolder;

std::shared_ptr<IntrinsicFormula::WrinkleEditingModel> editModel;

// smoothing
int smoothingTimes = 3;
double smoothingRatio = 0.95;

bool isFixedBnd = false;
int effectivedistFactor = 6;


bool isLoadOpt;

bool isUseV2 = false;
bool isWarmStart = false;

enum ModelType
{
	GlobalModel = 0,
	LocalModel = 1
};

InitializationType initType = Linear;
double zuenkoTau = 0.1;
int zuenkoIter = 5;

ModelType editModelType = GlobalModel;


void buildRefInfo(const std::vector<std::complex<double>>& initZvals, const Eigen::VectorXd& initOmega, std::vector<Eigen::VectorXd>& didacticRefAmpList, std::vector<Eigen::VectorXd>& didacticRefOmegaList, int numFrames, bool isRotate)
{

    didacticRefAmpList.resize(numFrames + 2);
    didacticRefOmegaList.resize(numFrames + 2);


    didacticRefAmpList[0] = Eigen::VectorXd::Ones(initZvals.size());
    for(int i = 0; i < didacticRefAmpList[0].rows(); i++)
    {
        didacticRefAmpList[0][i] = std::abs(initZvals[i]);
    }

    didacticRefOmegaList[0] = initOmega;

    double dt = 1.0 / (numFrames + 1);
    for (int i = 1; i <= numFrames + 1; i++)
    {
        if(isRotate)
        {
            std::vector<VertexOpInfo> curVertOpts;
            curVertOpts.resize(initAmp.rows(), { Rotate, false, dt * i * 90.0, 1 });
            WrinkleFieldsEditor::edgeBasedWrinkleEdition(triV, triMesh, initAmp, initOmega, curVertOpts, didacticRefAmpList[i], didacticRefOmegaList[i]);
        }

        else
        {
            didacticRefAmpList[i] = initAmp;
            didacticRefOmegaList[i] = initOmega;
            didacticRefOmegaList[i][initOmega.rows() - 1] *= (1 + dt * i);

        }
    }
}

void updateMagnitudePhase(const std::vector<Eigen::VectorXd>& wFrames, const std::vector<std::vector<std::complex<double>>>& zFrames, 
	std::vector<Eigen::VectorXd>& magList, 
	std::vector<Eigen::VectorXd>& phaseList,
	std::vector<std::vector<std::complex<double>>>& upZFrames)
{
	magList.resize(wFrames.size());
	phaseList.resize(wFrames.size());
	upZFrames.resize(wFrames.size());

	subOmegaList.resize(wFrames.size());
	subFaceOmegaList.resize(wFrames.size());

	MeshConnectivity mesh(triF);


	auto computeMagPhase = [&](const tbb::blocked_range<uint32_t>& range)
	{
		for (uint32_t i = range.begin(); i < range.end(); ++i)
		{
			
			Eigen::VectorXd edgeVec = swapEdgeVec(triF, wFrames[i], 0);
			std::shared_ptr<ComplexLoop> complexLoopOpt = std::make_shared<ComplexLoopZuenko>();
			complexLoopOpt->setBndFixFlag(isFixedBnd);
			complexLoopOpt->SetMesh(secMesh);
			complexLoopOpt->Subdivide(edgeVec, zFrames[i], subOmegaList[i], upZFrames[i], upsampleTimes);
			Mesh tmpMesh = complexLoopOpt->GetMesh();
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

	tbb::blocked_range<uint32_t> rangex(0u, (uint32_t)upZFrames.size(), GRAIN_SIZE);
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

	tbb::blocked_range<uint32_t> rangex(0u, (uint32_t)interpZList.size(), GRAIN_SIZE);
	tbb::parallel_for(rangex, computeWrinkles);


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
	regEdt = RegionEdition(triMesh, triV.rows());
}

void updatePaintingItems()
{
	// get interploated amp and phase frames
	std::cout << "compute upsampled phase: " << std::endl;
	updateMagnitudePhase(omegaList, zList, ampFieldsList, phaseFieldsList, upZList);

	std::cout << "compute wrinkle meshes: " << std::endl;
	updateWrinkles(upsampledTriV, upsampledTriF, upZList, wrinkledVList, wrinkleAmpScalingRatio, isUseV2);


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

	std::cout << "start to update viewer." << std::endl;
}

void solveKeyFrames(const Eigen::VectorXd& initAmp, const Eigen::VectorXd& initOmega, const Eigen::VectorXi& faceFlags, std::vector<Eigen::VectorXd>& wFrames, std::vector<std::vector<std::complex<double>>>& zFrames)
{
	if(editModelType == GlobalModel)
		editModel = std::make_shared<IntrinsicFormula::WrinkleEditingGlobalModel>(triV, triMesh, vertOpts, faceFlags, quadOrder, spatialAmpRatio, spatialEdgeRatio, spatialKnoppelRatio, effectivedistFactor);
	else
		editModel = std::make_shared<IntrinsicFormula::WrinkleEditingLocalModel>(triV, triMesh, vertOpts, faceFlags, quadOrder, spatialAmpRatio, spatialEdgeRatio, spatialKnoppelRatio);

    buildRefInfo(initZvals, initOmega, refAmpList, refOmegaList, numFrames - 2, isRotate);
    Eigen::VectorXd vertArea = getVertArea(triV, triMesh);
    Eigen::VectorXd edgeArea = getEdgeArea(triV, triMesh);
    Eigen::VectorXd faceArea = getFaceArea(triV, triMesh);

    std::vector<std::complex<double>> tarZvals;
    if(!isRotate)
        tarZvals = initZvals;
    else
    {
        IntrinsicFormula::roundZvalsFromEdgeOmegaVertexMag(triMesh, refOmegaList[refOmegaList.size() - 1], refAmpList[refAmpList.size()-1], edgeArea, vertArea, vertArea.size(), tarZvals);
    }
    editModel->initialization(initZvals, initOmega, tarZvals, refOmegaList[refOmegaList.size() - 1], refAmpList, refOmegaList, initType);


//	editModel->initialization(initZvals, initOmega, numFrames - 2, initType, zuenkoTau, zuenkoIter);
	
	std::cout << "initilization finished!" << std::endl;
	Eigen::VectorXd x;
	std::cout << "convert list to variable." << std::endl;
	editModel->convertList2Variable(x);

	refOmegaList = editModel->getRefWList();
	refAmpList = editModel->getRefAmpList();


	if (isForceOptimize)
	{
		if (isWarmStart)
			editModel->warmstart();

		else
		{

			auto funVal = [&](const Eigen::VectorXd& x, Eigen::VectorXd* grad, Eigen::SparseMatrix<double>* hess, bool isProj) {
				Eigen::VectorXd deriv;
				Eigen::SparseMatrix<double> H;
				double E = editModel->computeEnergy(x, grad ? &deriv : NULL, hess ? &H : NULL, isProj);

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
				editModel->getComponentNorm(x, znorm, wnorm);
			};
            auto saveTmpRes = [&](const Eigen::VectorXd& x, std::string* folder)
            {
                editModel->save(x, folder);
            };



			OptSolver::testFuncGradHessian(funVal, x);

			auto x0 = x;
			Eigen::VectorXd grad;
			Eigen::SparseMatrix<double> hess;
			double f0 = funVal(x0, &grad, &hess, false);
			std::cout << "initial f: " << f0 << ", grad norm: " << grad.norm() << ", hess norm: " << hess.norm() << ", working folder: " << workingFolder << std::endl;
			OptSolver::newtonSolver(funVal, maxStep, x, numIter, gradTol, xTol, fTol, false, getVecNorm, &workingFolder, saveTmpRes);
			std::cout << "before optimization: " << x0.norm() << ", after optimization: " << x.norm() << std::endl;

			std::cout << "convert variable to list." << std::endl;
			editModel->convertVariable2List(x);
		}
	}
	std::cout << "get w list" << std::endl;
	wFrames = editModel->getWList();
	std::cout << "get z list" << std::endl;
	zFrames = editModel->getVertValsList();

}

void registerMesh(int frameId)
{
	double shiftx = 1.5 * (triV.col(0).maxCoeff() - triV.col(0).minCoeff());
	polyscope::registerSurfaceMesh("base mesh", triV, triF);
	polyscope::getSurfaceMesh("base mesh")->addFaceVectorQuantity("frequency field", vecratio * faceOmegaList[frameId], polyscope::VectorType::AMBIENT);

    Eigen::VectorXd baseAmplitude = refAmpList[frameId];
    for(int i = 0 ; i < refAmpList[frameId].size(); i++)
    {
        baseAmplitude(i) = std::abs(zList[frameId][i]);
    }
    auto baseAmp = polyscope::getSurfaceMesh("base mesh")->addVertexScalarQuantity("opt amplitude", baseAmplitude);
    baseAmp->setMapRange(std::pair<double, double>(globalAmpMin, globalAmpMax));
    polyscope::getSurfaceMesh("base mesh")->addFaceVectorQuantity("opt frequency field", vecratio * faceOmegaList[frameId], polyscope::VectorType::AMBIENT);


	Eigen::MatrixXd refFaceOmega = intrinsicEdgeVec2FaceVec(refOmegaList[frameId], triV, triMesh);
	polyscope::registerSurfaceMesh("reference mesh", triV, triF);
	polyscope::getSurfaceMesh("reference mesh")->translate({ shiftx, 0, 0 });
	auto refAmp = polyscope::getSurfaceMesh("reference mesh")->addVertexScalarQuantity("reference amplitude", refAmpList[frameId]);
	refAmp->setMapRange(std::pair<double, double>(globalAmpMin, globalAmpMax));
	polyscope::getSurfaceMesh("reference mesh")->addFaceVectorQuantity("reference frequency field", vecratio * refFaceOmega, polyscope::VectorType::AMBIENT);

	// phase pattern
	polyscope::registerSurfaceMesh("phase mesh", upsampledTriV, upsampledTriF);
	mPaint.setNormalization(false);
	Eigen::MatrixXd phaseColor = mPaint.paintPhi(phaseFieldsList[frameId]);
	polyscope::getSurfaceMesh("phase mesh")->translate({ 2 * shiftx, 0, 0 });
	polyscope::getSurfaceMesh("phase mesh")->addVertexColorQuantity("vertex phi", phaseColor);

	// amp pattern
	polyscope::registerSurfaceMesh("upsampled ampliude and frequency mesh", upsampledTriV, upsampledTriF);
	polyscope::getSurfaceMesh("upsampled ampliude and frequency mesh")->translate({ 3 * shiftx, 0, 0 });
	auto ampPatterns = polyscope::getSurfaceMesh("upsampled ampliude and frequency mesh")->addVertexScalarQuantity("vertex amplitude", ampFieldsList[frameId]);
	ampPatterns->setMapRange(std::pair<double, double>(globalAmpMin, globalAmpMax));
	polyscope::getSurfaceMesh("upsampled ampliude and frequency mesh")->addFaceVectorQuantity("subdivided frequency field", vecratio * subFaceOmegaList[frameId], polyscope::VectorType::AMBIENT);

	// wrinkle mesh
	Eigen::MatrixXd lapWrinkledV = wrinkledVList[frameId];
	laplacianSmoothing(wrinkledVList[frameId], upsampledTriF, lapWrinkledV, smoothingRatio, smoothingTimes, isFixedBnd);
	polyscope::registerSurfaceMesh("wrinkled mesh", lapWrinkledV, upsampledTriF);
	polyscope::getSurfaceMesh("wrinkled mesh")->setSurfaceColor({ 80 / 255.0, 122 / 255.0, 91 / 255.0 });
	polyscope::getSurfaceMesh("wrinkled mesh")->translate({ 4 * shiftx, 0, 0 });
}

void updateFieldsInView(int frameId)
{
	std::cout << "update viewer. " << std::endl;
	registerMesh(frameId);
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
        Eigen::VectorXi faceFlags;
        faceFlags.setOnes(triMesh.nFaces());

        int nverts = triV.rows();
        vertOpts.clear();
        vertOpts.resize(nverts, { None, false, 0, 1 });

		editModel = std::make_shared<IntrinsicFormula::WrinkleEditingGlobalModel>(triV, triMesh, vertOpts, faceFlags, quadOrder, spatialAmpRatio, spatialEdgeRatio, spatialKnoppelRatio, effectivedistFactor);

        buildRefInfo(initZvals, initOmega, refAmpList, refOmegaList, numFrames - 2, isRotate);

        Eigen::VectorXd vertArea = getVertArea(triV, triMesh);
        Eigen::VectorXd edgeArea = getEdgeArea(triV, triMesh);
        Eigen::VectorXd faceArea = getFaceArea(triV, triMesh);

        std::vector<std::complex<double>> tarZvals;
        if(!isRotate)
            tarZvals = initZvals;
        else
        {
            IntrinsicFormula::roundZvalsFromEdgeOmegaVertexMag(triMesh, refOmegaList[refOmegaList.size() - 1], refAmpList[refAmpList.size()-1], edgeArea, vertArea, vertArea.size(), tarZvals);
        }

        editModel->initialization(initZvals, initOmega, tarZvals, refOmegaList[refOmegaList.size() -1], refAmpList, refOmegaList, Linear);
		refAmpList = editModel->getRefAmpList();
		refOmegaList = editModel->getRefWList();

		if (!isLoadOpt)
		{
			zList = editModel->getVertValsList();
			omegaList = editModel->getWList();
		}
		
	}

	updatePaintingItems();

	curFrame = 0;

	return true;
}


bool saveProblem()
{
	std::string saveFileName = igl::file_dialog_save();

	std::string filePath = saveFileName;
	std::replace(filePath.begin(), filePath.end(), '\\', '/'); // handle the backslash issue for windows
	int id = filePath.rfind("/");
	std::string workingFolder = filePath.substr(0, id + 1);


    saveEdgeOmega(workingFolder + "omega.txt", initOmega);
    saveVertexAmp(workingFolder + "amp.txt", initAmp);
    saveVertexZvals(workingFolder + "zvals.txt", initZvals);

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
    igl::writeOBJ(renderFolder + "basemesh.obj", triV, triF);
    igl::writeOBJ(renderFolder + "upmesh.obj", upsampledTriV, upsampledTriF);

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

    tbb::blocked_range<uint32_t> rangex(0u, (uint32_t)nframes, GRAIN_SIZE);
    tbb::parallel_for(rangex, savePerFrame);

//    for (int i = 0; i < nframes; i++)
//    {
//        // upsampled information
//        igl::writeOBJ(outputFolderWrinkles + "wrinkledMesh_" + std::to_string(i) + ".obj", wrinkledVList[i], upsampledTriF);
//        Eigen::MatrixXd lapWrinkledV;
//        laplacianSmoothing(wrinkledVList[i], upsampledTriF, lapWrinkledV, smoothingRatio, smoothingTimes, isFixedBnd);
//        igl::writeOBJ(outputFolderWrinkles + "wrinkledMeshSmoothed_" + std::to_string(i) + ".obj", lapWrinkledV, upsampledTriF);
//
//        saveAmp4Render(ampFieldsList[i], outputFolderAmp + "upAmp_" + std::to_string(i) + ".cvs");
//        savePhi4Render(phaseFieldsList[i], outputFolderPhase + "upPhase" + std::to_string(i) + ".cvs");
//        saveDphi4Render(subFaceOmegaList[i], subSecMesh, outputFolderPhase + "upOmega" + std::to_string(i) + ".cvs");
//
//        // reference information
//        Eigen::MatrixXd refFaceOmega = intrinsicEdgeVec2FaceVec(refOmegaList[i], triV, triMesh);
//        saveAmp4Render(refAmpList[i], refAmpFolder + "refAmp_" + std::to_string(i) + ".cvs");
//        saveDphi4Render(refFaceOmega, triMesh, triV, refOmegaFolder + "refOmega_" + std::to_string(i) + ".cvs");
//
//        // optimal information
//        saveDphi4Render(faceOmegaList[i], triMesh, triV, optOmegaFolder + "optOmega_" + std::to_string(i) + ".cvs");
//        saveAmp4Render(zList[i], optAmpFolder + "optAmp_" + std::to_string(i) + ".cvs");
//    }

    return true;
}

void callback() {
	ImGui::PushItemWidth(100);
	float w = ImGui::GetContentRegionAvailWidth();
	float p = ImGui::GetStyle().FramePadding.x;
	if (ImGui::Button("Load", ImVec2(ImVec2((w - p) / 2.f, 0))))
	{
		loadProblem();
		updateMagnitudePhase(omegaList, zList, ampFieldsList, phaseFieldsList, upZList);
		updateFieldsInView(curFrame);
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
					getUpsampledMesh(triV, triF, upsampledTriV, upsampledTriF);
					updatePaintingItems();
					updateFieldsInView(curFrame);
				}
			}
			if (ImGui::Checkbox("fix bnd", &isFixedBnd))
			{
				getUpsampledMesh(triV, triF, upsampledTriV, upsampledTriF);
				updatePaintingItems();
				updateFieldsInView(curFrame);
			}
			ImGui::EndTabItem();
		}
		if (ImGui::BeginTabItem("Wrinkle Mesh Smoothing"))
		{
			if (ImGui::InputInt("smoothing times", &smoothingTimes))
			{
				smoothingTimes = smoothingTimes > 0 ? smoothingTimes : 0;
//				updatePaintingItems();
				updateFieldsInView(curFrame);
			}
			if (ImGui::InputDouble("smoothing ratio", &smoothingRatio))
			{
				smoothingRatio = smoothingRatio > 0 ? smoothingRatio : 0;
//				updatePaintingItems();
				updateFieldsInView(curFrame);
			}
			ImGui::EndTabItem();
		}
		ImGui::EndTabBar();
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
		if (ImGui::Checkbox("is use v2", &isUseV2))
		{
			updateFieldsInView(curFrame);
		}
		if (ImGui::DragFloat("wrinkle amp scaling ratio", &wrinkleAmpScalingRatio, 0.0005, 0, 1))
        {
            if (wrinkleAmpScalingRatio >= 0)
                updateFieldsInView(curFrame);
        }
	}
	if (ImGui::CollapsingHeader("Frame Visualization Options", ImGuiTreeNodeFlags_DefaultOpen))
	{
		if (ImGui::SliderInt("current frame", &curFrame, 0, numFrames - 1))
		{
			curFrame = curFrame % numFrames;
			updateFieldsInView(curFrame);
		}
		if (ImGui::DragFloat("vec ratio", &(vecratio), 0.00005, 0, 1))
		{
			updateFieldsInView(curFrame);
		}
	}

	if (ImGui::CollapsingHeader("optimzation parameters", ImGuiTreeNodeFlags_DefaultOpen))
	{
		ImGui::Combo("model type", (int*)&editModelType, "Global\0Local\0");
		ImGui::Combo("Initialization type", (int*)&initType, "Linear\0Zuenko\0Knoppel\0");
        if (ImGui::Checkbox("Rotate", &isRotate))
        {
            
        }
		if (ImGui::InputDouble("Zuenko Tau", &zuenkoTau))
		{
			if (zuenkoTau < 0)
				zuenkoTau = 0;
		}
		if (ImGui::InputInt("Zuenko Inner Iter", &zuenkoIter))
		{
			if (zuenkoIter < 0)
				zuenkoTau = 5;
		}
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

		ImGui::Checkbox("warm start", &isWarmStart);

	}


	ImGui::Checkbox("Try Optimization", &isForceOptimize);

	if (ImGui::Button("comb fields & updates", ImVec2(-1, 0)))
	{
        Eigen::VectorXi faceFlags;
        faceFlags.setOnes(triMesh.nFaces());
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