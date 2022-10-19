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

#include "../../include/IntrinsicFormula/WrinkleEditingModel.h"
#include "../../include/IntrinsicFormula/WrinkleEditingCWF.h"
#include "../../include/MeshLib/RegionEdition.h"


std::shared_ptr<IntrinsicFormula::WrinkleEditingModel> editModel;

Eigen::MatrixXd triV, upsampledKnoppelTriV, upsampledZuenkoTriV, loopTriV, zuenkoFinalV;
Eigen::MatrixXi triF, upsampledKnoppelTriF, upsampledZuenkoTriF, loopTriF, zuenkoFinalF;
MeshConnectivity triMesh;
Mesh secMesh, upSecMesh;

std::vector<Eigen::MatrixXd> wrinkledVList, TFWWrinkledVList, ZuenkoWrinkledVList, TFWPhiVList, TFWProbVList, TFWUpsamplingVList, knoppelWrinkledVList;
std::vector<Eigen::MatrixXi> TFWWrinkledFList, ZuenkoWrinkledFList, TFWPhiFList, TFWProbFList, TFWUpsamplingFList, knoppelWrinkledFList;
std::vector<std::vector<std::complex<double>>> zList, upZList;
std::vector<Eigen::VectorXd> omegaList, ampList, upOmegaList, upPhiList, TFWUpPhiSoupList, TFWUpPhiList, ZuenkoUpPhiList, knoppelUpPhiList, upAmpList, TFWUpAmpList, ZuenkoUpAmpList, knoppelUpAmpList;
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

// region edition
RegionEdition regEdt;

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
std::vector<VertexOpInfo> vertOpts;

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

int quadOrder = 4;

double spatialAmpRatio = 1000;
double spatialEdgeRatio = 1000;
double spatialKnoppelRatio = 1000;

InitializationType initType = SeperateLinear;
double zuenkoTau = 0.1;
int zuenkoIter = 5;

static void buildEditModel(const Eigen::MatrixXd& pos, const MeshConnectivity& mesh, const std::vector<VertexOpInfo>& vertexOpts, const Eigen::VectorXi& faceFlag, int quadOrd, double spatialAmpRatio, double spatialEdgeRatio, double spatialKnoppelRatio, int effectivedistFactor, std::shared_ptr<IntrinsicFormula::WrinkleEditingModel>& editModel)
{
    editModel = std::make_shared<IntrinsicFormula::WrinkleEditingCWF>(pos, mesh, vertexOpts, faceFlag, quadOrd, spatialAmpRatio, spatialEdgeRatio, spatialKnoppelRatio, effectivedistFactor);
}

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

//	KnoppelAlg::getKnoppelPhaseSequence(triV, triMesh, omegaList, upsampledKnoppelTriV, upsampledKnoppelTriF, knoppelUpPhiList, upsamplingLevel);
    KnoppelAlg::getKnoppelWrinkledMeshSequence(triV, triMesh, omegaList, ampList, upsampledKnoppelTriV, upsampledKnoppelTriF, knoppelUpAmpList, knoppelUpPhiList, knoppelWrinkledVList, knoppelWrinkledFList, wrinkleAmpScalingRatio, upsamplingLevel);
    

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
	n++;

	////////////////////////////////////// Knoppel stuffs ///////////////////////////////////////////////
    n = 1;
    m++;
    // wrinkled mesh
    if (isFirstVis)
    {
        // wrinkle mesh
        polyscope::registerSurfaceMesh("knoppel wrinkled mesh", knoppelWrinkledVList[frameId], knoppelWrinkledFList[frameId]);
        polyscope::getSurfaceMesh("knoppel wrinkled mesh")->setSurfaceColor({ 80 / 255.0, 122 / 255.0, 91 / 255.0 });
        polyscope::getSurfaceMesh("knoppel wrinkled mesh")->translate({ n * shiftx, m * shifty, 0 });
    }
    else
        polyscope::getSurfaceMesh("knoppel wrinkled mesh")->updateVertexPositions(TFWWrinkledVList[frameId]);
    n++;

    // amp pattern
    if(isFirstVis)
    {
        polyscope::registerSurfaceMesh("knoppel upsampled ampliude mesh", upsampledKnoppelTriV, upsampledKnoppelTriF);
        polyscope::getSurfaceMesh("knoppel upsampled ampliude mesh")->translate({ n * shiftx, m * shifty, 0 });
    }

    auto ampKnoppelPatterns = polyscope::getSurfaceMesh("knoppel upsampled ampliude mesh")->addVertexScalarQuantity("vertex amplitude", TFWUpAmpList[frameId]);
    ampKnoppelPatterns->setMapRange(std::pair<double, double>(globalAmpMin, globalAmpMax));
    ampKnoppelPatterns->setEnabled(true);
    n++;

	// phase pattern
	mPaint.setNormalization(false);
	if (isFirstVis)
	{
		polyscope::registerSurfaceMesh("Knoppel upsampled phase mesh", upsampledKnoppelTriV, upsampledKnoppelTriF);
		polyscope::getSurfaceMesh("Knoppel upsampled phase mesh")->translate({ n * shiftx, m * shifty, 0 });
	}
	

	Eigen::MatrixXd phaseKnoppelColor = mPaint.paintPhi(knoppelUpPhiList[frameId]);
	auto KnoppelPhasePatterns = polyscope::getSurfaceMesh("Knoppel upsampled phase mesh")->addVertexColorQuantity("vertex phi", phaseKnoppelColor);
	KnoppelPhasePatterns->setEnabled(true);
	n++;

	isFirstVis = false;
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
    if (jval.contains(std::string_view{ "wrinkle_amp_scale" }))
    {
        wrinkleAmpScalingRatio = jval["wrinkle_amp_scale"];
    }


	meshFile = workingFolder + meshFile;
	igl::readOBJ(meshFile, triV, triF);
	triMesh = MeshConnectivity(triF);
	initialization(triV, triF, upsampledKnoppelTriV, upsampledKnoppelTriF);

	numFrames = jval["num_frame"];
    numFrames = jval["num_frame"];
    if (jval.contains(std::string_view{ "wrinkle_amp_scale" }))
    {
        wrinkleAmpScalingRatio = jval["wrinkle_amp_scale"];
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

	int nedges = triMesh.nEdges();
	int nverts = triV.rows();

	zList.resize(numFrames);
	omegaList.resize(numFrames);
    ampList.resize(numFrames);

    // load initial and target zvals and omega
    std::string initAmpPath = jval["init_amp"];
    std::string initOmegaPath = jval["init_omega"];
    std::string initZValsPath = "zvals.txt";
    if (jval.contains(std::string_view{ "init_zvals" }))
    {
        initZValsPath = jval["init_zvals"];
    }

    if (!loadEdgeOmega(workingFolder + initOmegaPath, nedges, omegaList[0])) {
        std::cout << "missing init edge omega file." << std::endl;
        return false;
    }

    if (!loadVertexZvals(workingFolder + initZValsPath, triV.rows(), zList[0]))
    {
        std::cout << "missing init zval file, try to load amp file, and round zvals from amp and omega" << std::endl;
        if (!loadVertexAmp(workingFolder + initAmpPath, triV.rows(), ampList[0]))
        {
            std::cout << "missing init amp file: " << std::endl;
            return false;
        }

        else
        {
            Eigen::VectorXd edgeArea, vertArea;
            edgeArea = getEdgeArea(triV, triMesh);
            vertArea = getVertArea(triV, triMesh);
            IntrinsicFormula::roundZvalsFromEdgeOmegaVertexMag(triMesh, omegaList[0], ampList[0], edgeArea, vertArea, triV.rows(), zList[0]);
        }
    }
    else
    {
        ampList[0].setZero(triV.rows());
        for (int i = 0; i < ampList[0].size(); i++)
            ampList[0](i) = std::abs(zList[0][i]);
    }

    // linear interpolation to get the list
    buildEditModel(triV, triMesh, vertOpts, faceFlags, quadOrder, spatialAmpRatio, spatialEdgeRatio, spatialKnoppelRatio, effectivedistFactor, editModel);

    editModel->initialization(zList[0], omegaList[0], numFrames - 2, initType, 0.1);
    ampList = editModel->getRefAmpList();
    omegaList = editModel->getRefWList();
    zList = editModel->getVertValsList();

    curFrame = 0;
	isFirstVis = true;

	return true;
}

static bool saveProblem()
{
    // save linear results
    std::string linearFolder = workingFolder + "/linearRes/";
    mkdir(linearFolder);

    igl::writeOBJ(linearFolder + "linearUpMesh.obj", loopTriV, loopTriF);
    // save upsampling things
    tbb::parallel_for(
            tbb::blocked_range<int>(0u, (uint32_t)numFrames),
            [&](const tbb::blocked_range<int> &range)
            {
                for (uint32_t i = range.begin(); i < range.end(); ++i)
                {
                    savePhi4Render(upPhiList[i], linearFolder + "linearUpPhi_" + std::to_string(i) + ".cvs");
                    saveAmp4Render(upAmpList[i], linearFolder + "linearUpAmp_" + std::to_string(i) + ".cvs", globalAmpMin, globalAmpMax);
                    igl::writeOBJ(linearFolder + "linearWrinkleMesh_" + std::to_string(i) + ".obj", wrinkledVList[i], loopTriF);
                }
            }
    );

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
                    savePhi4Render(knoppelUpPhiList[i], knoppelFolder + "KnoppelUpPhi_" + std::to_string(i) + ".cvs");
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