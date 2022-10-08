#include "polyscope/polyscope.h"
#include "polyscope/pick.h"

#include <igl/invert_diag.h>
#include <igl/per_vertex_normals.h>
#include <igl/readOBJ.h>
#include <igl/readOFF.h>
#include <igl/writeOBJ.h>
#include <igl/boundary_loop.h>
#include <igl/doublearea.h>
#include <igl/loop.h>
#include <igl/decimate.h>
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
#include "../../include/IntrinsicFormula/KnoppelStripePatternEdgeOmega.h"
#include "../../include/testMeshGeneration.h"
#include "../../include/SpherigonSmoothing.h"

#include "../../include/json.hpp"
#include "../../include/ComplexLoop/ComplexLoop.h"
#include "../../include/ComplexLoop/ComplexLoopAmpPhase.h"
#include "../../include/ComplexLoop/ComplexLoopAmpPhaseEdgeJump.h"
#include "../../include/ComplexLoop/ComplexLoopReIm.h"
#include "../../include/ComplexLoop/ComplexLoopZuenko.h"

#include "../../include/LoadSaveIO.h"
#include "../../include/SecMeshParsing.h"
#include "../../include/OtherApproaches/KnoppelAlgorithm.h"
#include "../../include/OtherApproaches/TFWAlgorithm.h"
#include "../../include/OtherApproaches/ZuenkoAlgorithm.h"

Eigen::MatrixXd triV, upsampledKnoppelTriV, upsampledZuenkoTriV, loopTriV, zuenkoFinalV;
Eigen::MatrixXi triF, upsampledKnoppelTriF, upsampledZuenkoTriF, loopTriF, zuenkoFinalF;
MeshConnectivity triMesh;
Mesh secMesh, upSecMesh;

std::vector<Eigen::MatrixXd> wrinkledVList, TFWWrinkledVList, ZuenkoWrinkledVList, TFWPhiVList, TFWProbVList, TFWUpsamplingVList;
std::vector<Eigen::MatrixXi> TFWWrinkledFList, ZuenkoWrinkledFList, TFWPhiFList, TFWProbFList, TFWUpsamplingFList;
std::vector<std::vector<std::complex<double>>> zList, upZList;
std::vector<Eigen::VectorXd> omegaList, ampList, upOmegaList, upPhiList, TFWUpPhiSoupList, TFWUpPhiList, ZuenkoUpPhiList, KnoppelUpPhiList, upAmpList, TFWUpAmpList, ZuenkoUpAmpList;
std::vector<Eigen::MatrixXd> faceOmegaList;
Eigen::VectorXd zuenkoFinalAmp, zuenkoFinalPhi;


int upsamplingLevel = 2;
float wrinkleAmpScalingRatio = 1;
std::string workingFolder = "";
int numFrames = 20;
int curFrame = 0;

double globalAmpMin = 0;
double globalAmpMax = 1;
float vecratio = 0.01;
bool isUseV2 = false;

PaintGeometry mPaint;

static void getUpsampledMesh(const Eigen::MatrixXd& triV, const Eigen::MatrixXi& triF, Eigen::MatrixXd& upsampledTriV, Eigen::MatrixXi& upsampledTriF)
{
	secMesh = convert2SecMesh(triV, triF);
	upSecMesh = secMesh;

	std::shared_ptr<ComplexLoop> complexLoopOpt = std::make_shared<ComplexLoopZuenko>();
	complexLoopOpt->setBndFixFlag(true);
	complexLoopOpt->SetMesh(secMesh);
	complexLoopOpt->meshSubdivide(upsamplingLevel);
	upSecMesh = complexLoopOpt->GetMesh();
	parseSecMesh(upSecMesh, upsampledTriV, upsampledTriF);
}

static void initialization(const Eigen::MatrixXd& triV, const Eigen::MatrixXi& triF, Eigen::MatrixXd& upsampledTriV, Eigen::MatrixXi& upsampledTriF)
{
	getUpsampledMesh(triV, triF, upsampledTriV, upsampledTriF);
}

static void getOursUpsamplingRes(const Mesh& secMesh, const std::vector<Eigen::VectorXd>& edgeOmegaList, const std::vector<std::vector<std::complex<double>>>& zvalsList, Mesh& upMesh, std::vector<Eigen::VectorXd>& upEdgeOmegaList, std::vector<std::vector<std::complex<double>>>& upZvalsList, int upLevel)
{
	int nframes = edgeOmegaList.size();
	upEdgeOmegaList.resize(nframes);
	upZvalsList.resize(nframes);

	auto frameUpsampling = [&](const tbb::blocked_range<uint32_t>& range)
	{
		for (uint32_t i = range.begin(); i < range.end(); ++i)
		{
			Eigen::VectorXd edgeVec = swapEdgeVec(triF, edgeOmegaList[i], 0);

			std::shared_ptr<ComplexLoop> complexLoopOpt = std::make_shared<ComplexLoopZuenko>();
			complexLoopOpt->setBndFixFlag(true);
			complexLoopOpt->SetMesh(secMesh);
			complexLoopOpt->Subdivide(edgeVec, zvalsList[i], upEdgeOmegaList[i], upZvalsList[i], upLevel);
			if(i == 0)
			{
				upMesh = complexLoopOpt->GetMesh();
			}
		}
	};

	tbb::blocked_range<uint32_t> rangex(0u, (uint32_t)nframes);
	tbb::parallel_for(rangex, frameUpsampling);
}


static void updateWrinkles(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, const std::vector<std::vector<std::complex<double>>>& zFrames, std::vector<Eigen::MatrixXd>& wrinkledVFrames, double scaleRatio, bool isUseV2)
// all the things are upsampled
{
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

	tbb::blocked_range<uint32_t> rangex(0u, (uint32_t)zFrames.size());
	tbb::parallel_for(rangex, computeWrinkles);


}

static void upsamplingEveryThingForComparison()
{
	getOursUpsamplingRes(secMesh, omegaList, zList, upSecMesh, upOmegaList, upZList, upsamplingLevel);
	upPhiList.resize(upZList.size());
	upAmpList.resize(upZList.size());
	faceOmegaList.resize(upZList.size());
	ampList.resize(upZList.size());

	parseSecMesh(upSecMesh, loopTriV, loopTriF);

	auto computeOursForVis = [&](const tbb::blocked_range<uint32_t>& range)
	{
		for (uint32_t i = range.begin(); i < range.end(); ++i)
		{
			upPhiList[i].resize(upZList[i].size());
			upAmpList[i].resize(upZList[i].size());

			for (int j = 0; j < upZList[i].size(); j++)
			{
				upPhiList[i][j] = std::arg(upZList[i][j]);
				upAmpList[i][j] = std::abs(upZList[i][j]);
			}
			faceOmegaList[i] = intrinsicEdgeVec2FaceVec(omegaList[i], triV, triMesh);

			ampList[i].resize(zList[i].size());

			for (int j = 0; j < zList[i].size(); j++)
			{
				ampList[i][j] = std::abs(zList[i][j]);
			}
		}
	};
	tbb::blocked_range<uint32_t> rangex(0u, (uint32_t)upZList.size());
	tbb::parallel_for(rangex, computeOursForVis);

	globalAmpMin = ampList[0].minCoeff();
	globalAmpMax = ampList[0].maxCoeff();

	for (uint32_t i = 1; i < upZList.size(); ++i)
	{
		for (int j = 0; j < zList[i].size(); j++)
		{
			globalAmpMin = std::min(ampList[i][j], globalAmpMin);
			globalAmpMax = std::max(ampList[i][j], globalAmpMax);
		}
	}
	updateWrinkles(loopTriV, loopTriF, upZList, wrinkledVList, wrinkleAmpScalingRatio, isUseV2);

	KnoppelAlg::getKnoppelPhaseSequence(triV, triMesh, omegaList, upsampledKnoppelTriV, upsampledKnoppelTriF, KnoppelUpPhiList, upsamplingLevel);

	ZuenkoAlg::getZuenkoSurfaceSequence(triV, triMesh, zList[0], ampList, omegaList, upsampledZuenkoTriV, upsampledZuenkoTriF, ZuenkoWrinkledVList, ZuenkoWrinkledFList, ZuenkoUpAmpList, ZuenkoUpPhiList, upsamplingLevel, true, wrinkleAmpScalingRatio);

	TFWAlg::getTFWSurfaceSequence(triV, triMesh.faces(), ampList, omegaList, TFWWrinkledVList, TFWWrinkledFList, TFWUpsamplingVList, TFWUpsamplingFList, TFWPhiVList, TFWPhiFList, TFWProbVList, TFWProbFList, TFWUpAmpList, TFWUpPhiSoupList, TFWUpPhiList, upsamplingLevel, wrinkleAmpScalingRatio, isUseV2, true);


	std::vector<std::pair<int, Eigen::Vector3d>> bary;
	Eigen::MatrixXd baseN, upsampledN, upsampledV;
	Eigen::MatrixXi upsampledF;

	meshUpSampling(triV, triMesh.faces(), upsampledV, upsampledF, upsamplingLevel, NULL, NULL, &bary);

	igl::per_vertex_normals(triV, triMesh.faces(), baseN);
	ZuenkoAlg::spherigonSmoothing(triV, triMesh, baseN, bary, upsampledV, upsampledN, true);
	igl::per_vertex_normals(upsampledV, upsampledF, upsampledN);

	std::vector<std::complex<double>> curZvals = zList[numFrames - 1];
	for (int i = 0; i < curZvals.size(); i++)
	{
		double phi = std::arg(curZvals[i]);
		curZvals[i] = std::complex<double>(std::cos(phi), std::sin(phi));
	}
	
	ZuenkoAlg::getZuenkoSurfacePerframe(triV, triMesh, curZvals, ampList[numFrames - 1], omegaList[numFrames - 1], upsampledV, upsampledF, upsampledN, bary, zuenkoFinalV, zuenkoFinalF, zuenkoFinalAmp, zuenkoFinalPhi, wrinkleAmpScalingRatio);
}

bool isFirstVis = true;
static void updateView(int frameId)
{
	double shiftx = 1.5 * (triV.col(0).maxCoeff() - triV.col(0).minCoeff());
	int n = 0;

	double shifty = 1.5 * (triV.col(1).maxCoeff() - triV.col(1).minCoeff());
	int m = 0;

	if (isFirstVis)
	{
		auto baseSurf = polyscope::registerSurfaceMesh("base mesh", triV, triF);
	}
   
    auto freqFields = polyscope::getSurfaceMesh("base mesh")->addFaceVectorQuantity("frequency field", vecratio * faceOmegaList[frameId], polyscope::VectorType::AMBIENT);
    auto initAmp = polyscope::getSurfaceMesh("base mesh")->addVertexScalarQuantity("amplitude", ampList[frameId]);
    initAmp->setMapRange(std::pair<double, double>(globalAmpMin, globalAmpMax));
    n++;


	////////////////////////////////////// our stuffs ///////////////////////////////////////////////
	// wrinkled mesh
	if (isFirstVis)
	{
		// wrinkle mesh
		polyscope::registerSurfaceMesh("our wrinkled mesh", wrinkledVList[frameId], loopTriF);
		polyscope::getSurfaceMesh("our wrinkled mesh")->setSurfaceColor({ 80 / 255.0, 122 / 255.0, 91 / 255.0 });
		polyscope::getSurfaceMesh("our wrinkled mesh")->translate({ n * shiftx, 0, 0 });
	}
	else
		polyscope::getSurfaceMesh("our wrinkled mesh")->updateVertexPositions(wrinkledVList[frameId]);
	n++;

	// amp pattern
	if (isFirstVis)
	{
		polyscope::registerSurfaceMesh("our upsampled ampliude mesh", loopTriV, loopTriF);
		polyscope::getSurfaceMesh("our upsampled ampliude mesh")->translate({ n * shiftx, 0, 0 });
	}
	auto ampPatterns = polyscope::getSurfaceMesh("our upsampled ampliude mesh")->addVertexScalarQuantity("vertex amplitude", upAmpList[frameId]);
	ampPatterns->setMapRange(std::pair<double, double>(globalAmpMin, globalAmpMax));
	ampPatterns->setEnabled(true);
	n++;
	

	// phase pattern
	mPaint.setNormalization(false);
	if (isFirstVis)
	{
		polyscope::registerSurfaceMesh("our upsampled phase mesh", loopTriV, loopTriF);
		polyscope::getSurfaceMesh("our upsampled phase mesh")->translate({ n * shiftx, 0, 0 });
	}
		
	Eigen::MatrixXd phaseColor = mPaint.paintPhi(upPhiList[frameId]);
	auto ourPhasePatterns = polyscope::getSurfaceMesh("our upsampled phase mesh")->addVertexColorQuantity("vertex phi", phaseColor);
	ourPhasePatterns->setEnabled(true);
	n++;


	////////////////////////////////////// Zuenko's stuffs ///////////////////////////////////////////////
	n = 0;
	m++;
	if (isFirstVis)
	{
		// wrinkle mesh
		polyscope::registerSurfaceMesh("Zuenko final wrinkled mesh", zuenkoFinalV, zuenkoFinalF);
		polyscope::getSurfaceMesh("Zuenko final wrinkled mesh")->setSurfaceColor({ 80 / 255.0, 122 / 255.0, 91 / 255.0 });
		polyscope::getSurfaceMesh("Zuenko final wrinkled mesh")->translate({ n * shiftx, m * shifty, 0 });
	}
	else
		polyscope::getSurfaceMesh("Zuenko final wrinkled mesh")->updateVertexPositions(zuenkoFinalV);
	n++;
	// wrinkled mesh
	if (isFirstVis)
	{
		// wrinkle mesh
		polyscope::registerSurfaceMesh("Zuenko wrinkled mesh", ZuenkoWrinkledVList[frameId], ZuenkoWrinkledFList[frameId]);
		polyscope::getSurfaceMesh("Zuenko wrinkled mesh")->setSurfaceColor({ 80 / 255.0, 122 / 255.0, 91 / 255.0 });
		polyscope::getSurfaceMesh("Zuenko wrinkled mesh")->translate({ n * shiftx, m * shifty, 0 });
	}
	else
		polyscope::getSurfaceMesh("Zuenko wrinkled mesh")->updateVertexPositions(ZuenkoWrinkledVList[frameId]);
	n++;

	// amp pattern
	if (isFirstVis)
	{
		polyscope::registerSurfaceMesh("Zuenko upsampled ampliude mesh", upsampledZuenkoTriV, upsampledZuenkoTriF);
		polyscope::getSurfaceMesh("Zuenko upsampled ampliude mesh")->translate({ n * shiftx, m * shifty, 0 });
	}
	auto ampZuenkoPatterns = polyscope::getSurfaceMesh("Zuenko upsampled ampliude mesh")->addVertexScalarQuantity("vertex amplitude", ZuenkoUpAmpList[frameId]);
	ampZuenkoPatterns->setMapRange(std::pair<double, double>(globalAmpMin, globalAmpMax));
	ampZuenkoPatterns->setEnabled(true);
	n++;


	// phase pattern
	mPaint.setNormalization(false);
	if (isFirstVis)
	{
		polyscope::registerSurfaceMesh("Zuenko upsampled phase mesh", upsampledZuenkoTriV, upsampledZuenkoTriF);
		polyscope::getSurfaceMesh("Zuenko upsampled phase mesh")->translate({ n * shiftx, m * shifty, 0 });
	}

	Eigen::MatrixXd phaseZuenkoColor = mPaint.paintPhi(ZuenkoUpPhiList[frameId]);
	auto ZuenkoPhasePatterns = polyscope::getSurfaceMesh("Zuenko upsampled phase mesh")->addVertexColorQuantity("vertex phi", phaseZuenkoColor);
	ZuenkoPhasePatterns->setEnabled(true);
	n++;


	////////////////////////////////////// TFW stuffs ///////////////////////////////////////////////
	n = 1;
	m++;
	// wrinkled mesh
	if (isFirstVis)
	{
		// wrinkle mesh
		polyscope::registerSurfaceMesh("TFW wrinkled mesh", TFWWrinkledVList[frameId], TFWWrinkledFList[frameId]);
		polyscope::getSurfaceMesh("TFW wrinkled mesh")->setSurfaceColor({ 80 / 255.0, 122 / 255.0, 91 / 255.0 });
		polyscope::getSurfaceMesh("TFW wrinkled mesh")->translate({ n * shiftx, m * shifty, 0 });
	}
	else
		polyscope::getSurfaceMesh("TFW wrinkled mesh")->updateVertexPositions(TFWWrinkledVList[frameId]);
	n++;

	// amp pattern
	polyscope::registerSurfaceMesh("TFW upsampled ampliude mesh", TFWUpsamplingVList[frameId], TFWUpsamplingFList[frameId]);
	polyscope::getSurfaceMesh("TFW upsampled ampliude mesh")->translate({ n * shiftx, m * shifty, 0 });
	auto ampTFWPatterns = polyscope::getSurfaceMesh("TFW upsampled ampliude mesh")->addVertexScalarQuantity("vertex amplitude", TFWUpAmpList[frameId]);
	ampTFWPatterns->setMapRange(std::pair<double, double>(globalAmpMin, globalAmpMax));
	ampTFWPatterns->setEnabled(true);
	n++;


	// phase pattern
	mPaint.setNormalization(false);
	Eigen::MatrixXd TFWPhaseColor = mPaint.paintPhi(TFWUpPhiList[frameId]);

    polyscope::registerSurfaceMesh("TFW upsampled phase mesh", TFWUpsamplingVList[frameId], TFWUpsamplingFList[frameId]);
    polyscope::getSurfaceMesh("TFW upsampled phase mesh")->translate({ n * shiftx, m * shifty, 0 });
    auto TFWPhasePatterns = polyscope::getSurfaceMesh("TFW upsampled phase mesh")->addVertexColorQuantity("vertex phi", TFWPhaseColor);
    TFWPhasePatterns->setEnabled(true);

	// we compose the problem mesh and phi mesh here. polyscope has some strange bug
//	Eigen::MatrixXd compositePhiV, compositePhiColor;
//	Eigen::MatrixXi compositePhiF;
//	std::cout << TFWPhiVList[frameId].rows() << " " << TFWProbVList[frameId].rows() << std::endl;
//	compositePhiV.resize(TFWPhiVList[frameId].rows() + TFWProbVList[frameId].rows(), 3);
//	compositePhiV.block(0, 0, TFWPhiVList[frameId].rows(), 3) = TFWPhiVList[frameId];
//	compositePhiV.block(TFWPhiVList[frameId].rows(), 0, TFWProbVList[frameId].rows(), 3) = TFWProbVList[frameId];
//
//	compositePhiF.resize(TFWPhiFList[frameId].rows() + TFWProbFList[frameId].rows(), 3);
//	compositePhiF.block(0, 0, TFWPhiFList[frameId].rows(), 3) = TFWPhiFList[frameId];
//
//	Eigen::MatrixXi onesMat = TFWProbFList[frameId];
//	onesMat.setOnes();
//	compositePhiF.block(TFWPhiFList[frameId].rows(), 0, TFWProbFList[frameId].rows(), 3) = TFWProbFList[frameId] + TFWPhiVList[frameId].rows() * onesMat;
//
//	compositePhiColor.setOnes(TFWPhiVList[frameId].rows() + TFWProbVList[frameId].rows(), 3);
//	compositePhiColor.block(0, 0, TFWPhiVList[frameId].rows(), 3) = TFWPhaseColor;
//
//	polyscope::registerSurfaceMesh("TFW upsampled phase mesh", compositePhiV, compositePhiF);
//	polyscope::getSurfaceMesh("TFW upsampled phase mesh")->translate({ n * shiftx, m * shifty, 0 });
//	auto TFWPhasePatterns = polyscope::getSurfaceMesh("TFW upsampled phase mesh")->addVertexColorQuantity("vertex phi", compositePhiColor);
//	TFWPhasePatterns->setEnabled(true);


	/*polyscope::registerSurfaceMesh("TFW upsampled phase mesh", TFWPhiVList[frameId], TFWPhiFList[frameId]);
	polyscope::getSurfaceMesh("TFW upsampled phase mesh")->translate({ n * shiftx, m * shifty, 0 });
	polyscope::registerSurfaceMesh("TFW upsampled problem mesh", TFWProbVList[frameId], TFWProbFList[frameId]);
	polyscope::getSurfaceMesh("TFW upsampled problem mesh")->translate({ n * shiftx, m * shifty, 0 });
	polyscope::getSurfaceMesh("TFW upsampled problem mesh")->setSurfaceColor({ 1, 1, 1 });
	
	auto TFWPhasePatterns = polyscope::getSurfaceMesh("TFW upsampled phase mesh")->addVertexColorQuantity("vertex phi", TFWPhaseColor);
	TFWPhasePatterns->setEnabled(true);*/
	
	n++;

	////////////////////////////////////// Knoppel stuffs ///////////////////////////////////////////////
	n -= 1;
	m++;
	// phase pattern
	mPaint.setNormalization(false);
	if (isFirstVis)
	{
		polyscope::registerSurfaceMesh("Knoppel upsampled phase mesh", upsampledKnoppelTriV, upsampledKnoppelTriF);
		polyscope::getSurfaceMesh("Knoppel upsampled phase mesh")->translate({ n * shiftx, m * shifty, 0 });
	}
	

	Eigen::MatrixXd phaseKnoppelColor = mPaint.paintPhi(KnoppelUpPhiList[frameId]);
	auto KnoppelPhasePatterns = polyscope::getSurfaceMesh("Knoppel upsampled phase mesh")->addVertexColorQuantity("vertex phi", phaseKnoppelColor);
	KnoppelPhasePatterns->setEnabled(true);
	n++;

	isFirstVis = false;
}

static bool loadProblem(std::string *inputpath = NULL)
{
	std::string loadFileName;
	if (!inputpath)
		loadFileName = igl::file_dialog_open();
	else
		loadFileName = *inputpath;

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
	upsamplingLevel = jval["upsampled_times"];
	if (upsamplingLevel > 2)
		upsamplingLevel = 2;


	meshFile = workingFolder + meshFile;
	igl::readOBJ(meshFile, triV, triF);
	triMesh = MeshConnectivity(triF);
	initialization(triV, triF, upsampledKnoppelTriV, upsampledKnoppelTriF);

	numFrames = jval["num_frame"];

	int nedges = triMesh.nEdges();
	int nverts = triV.rows();

	zList.clear();
	omegaList.clear();
	std::string optZvals = jval["solution"]["opt_zvals"];
	std::string optOmega = jval["solution"]["opt_omega"];

	bool isLoadOpt = true;
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
	if (!isLoadOpt)
	{
		std::cout << "missing required zvals and omega files" << std::endl;
		exit(EXIT_FAILURE);
	}

	if (isLoadOpt)
	{
		std::cout << "load zvals and omegas from file!" << std::endl;
	}


	curFrame = 0;
	isFirstVis = true;

	return true;
}

static bool saveProblem()
{
    // save zuenko results
    std::string zuenkoFolder = workingFolder + "/zuenkoRes/";
    mkdir(zuenkoFolder);

    igl::writeOBJ(zuenkoFolder + "zuenkoUpMesh.obj", upsampledZuenkoTriV, upsampledZuenkoTriF);
    // save upsampling things
    tbb::parallel_for(
            tbb::blocked_range<int>(0u, (uint32_t)numFrames),
            [&](const tbb::blocked_range<int> &range)
            {
                for (uint32_t i = range.begin(); i < range.end(); ++i)
                {
                    savePhi4Render(ZuenkoUpPhiList[i], zuenkoFolder + "zuenkoUpPhi_" + std::to_string(i) + ".cvs");
                    saveAmp4Render(ZuenkoUpAmpList[i], zuenkoFolder + "zuenkoUpAmp_" + std::to_string(i) + ".cvs", globalAmpMin, globalAmpMax);
                    igl::writeOBJ(zuenkoFolder + "zuenkoWrinkleMesh_" + std::to_string(i) + ".obj", ZuenkoWrinkledVList[i], ZuenkoWrinkledFList[i]);
                }
            }
    );

//    for(int i = 0; i < numFrames; i++)
//    {
//        savePhi4Render(ZuenkoUpPhiList[i], zuenkoFolder + "zuenkoUpPhi_" + std::to_string(i) + ".cvs");
//        saveAmp4Render(ZuenkoUpAmpList[i], zuenkoFolder + "zuenkoUpAmp_" + std::to_string(i) + ".cvs");
//        igl::writeOBJ(zuenkoFolder + "zuenkoWrinkleMesh_" + std::to_string(i) + ".obj", ZuenkoWrinkledVList[i], ZuenkoWrinkledFList[i]);
//    }
    // save the final thing, for comparison
    savePhi4Render(zuenkoFinalPhi, zuenkoFolder + "zuenkoUpPhi_target.cvs");
    saveAmp4Render(zuenkoFinalAmp, zuenkoFolder + "zuenkoUpAmp_target.cvs", globalAmpMin, globalAmpMax);
    igl::writeOBJ(zuenkoFolder + "zuenkoWrinkleMesh_target.obj", zuenkoFinalV, zuenkoFinalF);


    // save TFW results
    std::string TFWFolder = workingFolder + "/TFWRes/";
    mkdir(TFWFolder);

    tbb::parallel_for(
            tbb::blocked_range<int>(0u, (uint32_t)numFrames),
            [&](const tbb::blocked_range<int> &range)
            {
                for (uint32_t i = range.begin(); i < range.end(); ++i)
                {
                    savePhi4Render(TFWUpPhiList[i], TFWFolder + "TFWUpPhi_" + std::to_string(i) + ".cvs");
                    savePhi4Render(TFWUpPhiSoupList[i], TFWFolder + "TFWUpPhiSoup_" + std::to_string(i) + ".cvs");
                    igl::writeOBJ(TFWFolder + "TFWPhiMesh_" + std::to_string(i) + ".obj", TFWPhiVList[i], TFWPhiFList[i]);
                    igl::writeOBJ(TFWFolder + "TFWProbMesh_" + std::to_string(i) + ".obj", TFWProbVList[i], TFWProbFList[i]);

                    saveAmp4Render(TFWUpAmpList[i], TFWFolder + "TFWUpAmp_" + std::to_string(i) + ".cvs", globalAmpMin, globalAmpMax);
                    igl::writeOBJ(TFWFolder + "TFWUpMesh_" + std::to_string(i) + ".obj", TFWUpsamplingVList[i], TFWUpsamplingFList[i]);

                    igl::writeOBJ(TFWFolder + "TFWWrinkledMesh_" + std::to_string(i) + ".obj", TFWWrinkledVList[i], TFWWrinkledFList[i]);
                }
            }
    );
//    for(int i = 0; i < numFrames; i++)
//    {
//        savePhi4Render(TFWUpPhiList[i], TFWFolder + "TFWUpPhi_" + std::to_string(i) + ".cvs");
//        igl::writeOBJ(TFWFolder + "TFWPhiMesh_" + std::to_string(i) + ".obj", TFWPhiVList[i], TFWPhiFList[i]);
//        igl::writeOBJ(TFWFolder + "TFWProbMesh_" + std::to_string(i) + ".obj", TFWProbVList[i], TFWProbFList[i]);
//
//        savePhi4Render(TFWUpAmpList[i], TFWFolder + "TFWUpAmp_" + std::to_string(i) + ".cvs");
//        igl::writeOBJ(TFWFolder + "TFWUpMesh_" + std::to_string(i) + ".obj", TFWUpsamplingVList[i], TFWUpsamplingFList[i]);
//
//        igl::writeOBJ(TFWFolder + "TFWWrinkledMesh_" + std::to_string(i) + ".obj", TFWWrinkledVList[i], TFWWrinkledFList[i]);
//    }

    // save CWF results
    std::string CWFFolder = workingFolder + "/CWFRes/";
    mkdir(CWFFolder);
    igl::writeOBJ(CWFFolder + "CWFUpMesh.obj", loopTriV, loopTriF);
    tbb::parallel_for(
            tbb::blocked_range<int>(0u, (uint32_t)numFrames),
            [&](const tbb::blocked_range<int> &range)
            {
                for (uint32_t i = range.begin(); i < range.end(); ++i)
                {
                    savePhi4Render(upPhiList[i], CWFFolder + "CWFUpPhi_" + std::to_string(i) + ".cvs");
                    saveAmp4Render(upAmpList[i], CWFFolder + "CWFUpAmp_" + std::to_string(i) + ".cvs", globalAmpMin, globalAmpMax);
                    igl::writeOBJ(CWFFolder + "CWFWrinkleMesh_" + std::to_string(i) + ".obj", wrinkledVList[i], loopTriF);
                }
            }
    );

    // save Knoppel results
    std::string knoppelFolder = workingFolder + "/KnoppelRes/";
    mkdir(knoppelFolder);
    igl::writeOBJ(knoppelFolder + "KnoppelUpMesh.obj", upsampledKnoppelTriV, upsampledKnoppelTriF);
    tbb::parallel_for(
            tbb::blocked_range<int>(0u, (uint32_t)numFrames),
            [&](const tbb::blocked_range<int> &range)
            {
                for (uint32_t i = range.begin(); i < range.end(); ++i)
                {
                    savePhi4Render(KnoppelUpPhiList[i], knoppelFolder + "KnoppelUpPhi_" + std::to_string(i) + ".cvs");
                }
            }
    );
//    for(int i = 0; i < numFrames; i++)
//    {
//        savePhi4Render(KnoppelUpPhiList[i], knoppelFolder + "KnoppelUpPhi_" + std::to_string(i) + ".cvs");
//    }
    

    return true;
}

static void callback() {
	ImGui::PushItemWidth(100);
	float w = ImGui::GetContentRegionAvailWidth();
	float p = ImGui::GetStyle().FramePadding.x;

	if (ImGui::Button("load", ImVec2(-1, 0)))
	{
		if (!loadProblem())
		{
			std::cout << "failed to load file." << std::endl;
			exit(EXIT_FAILURE);
		}
		upsamplingEveryThingForComparison();
		updateView(curFrame);
	}
    if (ImGui::Button("save", ImVec2(-1, 0)))
    {
        if (!saveProblem())
        {
            std::cout << "failed to load file." << std::endl;
            exit(EXIT_FAILURE);
        }
    }

	if (ImGui::InputInt("underline upsampling level", &upsamplingLevel))
	{
		if (upsamplingLevel < 0)
			upsamplingLevel = 2;
	}

	if (ImGui::CollapsingHeader("Visualization Options", ImGuiTreeNodeFlags_DefaultOpen))
	{
		if (ImGui::DragFloat("wrinkle amp scaling ratio", &wrinkleAmpScalingRatio, 0.0005, 0, 100))
		{
			if (wrinkleAmpScalingRatio >= 0)
				updateView(curFrame);
		}

        if (ImGui::DragFloat("freq scaling ratio", &vecratio, 0.0005, 0, 1))
        {
            if (vecratio >= 0)
            {
                polyscope::getSurfaceMesh("base mesh")->addFaceVectorQuantity("frequency field", vecratio * faceOmegaList[curFrame], polyscope::VectorType::AMBIENT);
            }
        }

	}
	if (ImGui::SliderInt("current frame slider bar", &curFrame, 0, numFrames - 1))
	{
		curFrame = curFrame % numFrames;
		updateView(curFrame);
	}
	
	if (ImGui::Button("recompute", ImVec2(-1, 0)))
	{
		upsamplingEveryThingForComparison();
		isFirstVis = true; // reset the mesh
		updateView(curFrame);
	}

	if (ImGui::Button("output images", ImVec2(-1, 0)))
	{
		std::string curFolder = std::filesystem::current_path().string();


        for(int i = 0; i < numFrames; i++)
        {
            std::string name = curFolder + "/output_" + std::to_string(i) + ".jpg";
            updateView(i);
            polyscope::screenshot(name);
        }
        updateView(curFrame);
	}


	ImGui::PopItemWidth();
}


int main(int argc, char** argv)
{
	if(argc < 2)
	{
		if (!loadProblem())
		{
			std::cout << "failed to load file." << std::endl;
			return 1;
		}
	}
	else
	{
		std::string inputPath = argv[2];
		if (!loadProblem(&inputPath))
		{
			std::cout << "failed to load file." << std::endl;
			return 1;
		}
	}
	
	
	// Options
	polyscope::options::autocenterStructures = true;
	polyscope::view::windowWidth = 1024;
	polyscope::view::windowHeight = 1024;

	// Initialize polyscope
	polyscope::init();
	polyscope::view::upDir = polyscope::view::UpDir::ZUp;

	// Add the callback
	polyscope::state::userCallback = callback;

	polyscope::options::groundPlaneHeightFactor = 0.25; // adjust the plane height
	curFrame = 0;

	upsamplingEveryThingForComparison();
	isFirstVis = true;
	updateView(curFrame);
	// Show the gui
	polyscope::show();


	return 0;
}