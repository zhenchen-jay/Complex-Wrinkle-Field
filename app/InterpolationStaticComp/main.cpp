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
#include "../../include/GetInterpVertexValues.h"

#include "../../include/OtherApproaches/TFWAlgorithm.h"
#include "../../include/OtherApproaches/ZuenkoAlgorithm.h"

Eigen::MatrixXd triV, upsampledTriV, loopTriV, zuenkoUpV, TFWUpV;
Eigen::MatrixXi triF, upsampledTriF, loopTriF, zuenkoUpF, TFWUpF;
MeshConnectivity triMesh;
Mesh secMesh, upSecMesh;

Eigen::MatrixXd wrinkledV, loopWrinkledV, loopReImWrinkledV, zuenkoWrinkledV, TFWWrinkledV;
std::vector<std::complex<double>> zvals, upZvals, upLoopAmpPhi_Zvals, upLoopReIm_Zvals;
Eigen::VectorXd phi, omega, amp, upOmega, upPhi, upAmp, upLoopAmpPhi_Phi, upLoopAmpPhi_Amp, upLoopReIm_Phi, upLoopReIm_Amp, zuenkoPhi, zuenkoAmp, TFWPhi, TFWAmp;
Eigen::MatrixXd faceOmega;

Eigen::VectorXd sideVertexLinearPhi, ClouhTorcherPhi, sideVertexWojtanPhi, knoppelPhi, sideVertexLinearAmp, ClouhTorcherAmp, sideVertexWojtanAmp;
Eigen::MatrixXd sideVertexLinearWrinkledV, sideVertexWojtanWrinkledV, ClouhTorcherWrinkledV;

int upsamplingLevel = 2;
float wrinkleAmpScalingRatio = 1;

double globalAmpMin = 0;
double globalAmpMax = 1;
float vecratio = 0.001;
bool isUseV2 = false;
std::string workingFolder = "";
bool isShowEveryThing = false;

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

static void getClouhTorcherUpsamplingAmpPhi(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, const Eigen::VectorXd& edgeOmega, const std::vector<std::complex<double>>& zvals, Eigen::MatrixXd& upV, Eigen::MatrixXi& upF, Eigen::VectorXd& upAmp, Eigen::VectorXd& upPhi, int upLevel)
{
	Eigen::SparseMatrix<double> mat;
	std::vector<int> facemap;
	std::vector<std::pair<int, Eigen::Vector3d>> bary;


	meshUpSampling(V, F, upV, upF, upLevel, &mat, &facemap, &bary);
	MeshConnectivity mesh(F);
	MeshConnectivity upMesh;

	getClouhTocherPhi(V, mesh, edgeOmega, zvals, bary, upPhi);
    getClouhTocherAmp(V, mesh, zvals, bary, upAmp);
}

static void getSideVertexUpsamplingAmpPhi(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, const Eigen::VectorXd& edgeOmega, const std::vector<std::complex<double>>& zvals, Eigen::MatrixXd& upV, Eigen::MatrixXi& upF, Eigen::VectorXd& upAmpLinear, Eigen::VectorXd& upPhiLinear, Eigen::VectorXd& upAmpWojtan, Eigen::VectorXd& upPhiWojtan, int upLevel)
{
	Eigen::SparseMatrix<double> mat;
	std::vector<int> facemap;
	std::vector<std::pair<int, Eigen::Vector3d>> bary;


	meshUpSampling(V, F, upV, upF, upLevel, &mat, &facemap, &bary);
	MeshConnectivity mesh(F);
	MeshConnectivity upMesh;

	getSideVertexPhi(V, mesh, edgeOmega, zvals, bary, upPhiLinear, 0);
	getSideVertexPhi(V, mesh, edgeOmega, zvals, bary, upPhiWojtan, 2);


    getSideVertexAmp(V, mesh, zvals, bary, upAmpLinear, 0);
    getSideVertexAmp(V, mesh, zvals, bary, upAmpWojtan, 2);
}

static void getKnoppelUpsamplingPhi(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, const Eigen::VectorXd& edgeOmega, const std::vector<std::complex<double>>& zvals, Eigen::MatrixXd& upV, Eigen::MatrixXi& upF, Eigen::VectorXd& upPhi, int upLevel)
{
	Eigen::SparseMatrix<double> mat;
	std::vector<int> facemap;
	std::vector<std::pair<int, Eigen::Vector3d>> bary;


	meshUpSampling(V, F, upV, upF, upLevel, &mat, &facemap, &bary);
	MeshConnectivity mesh(F);

	IntrinsicFormula::getUpsamplingThetaFromEdgeOmega(mesh, edgeOmega, zvals, bary, upPhi); // knoppel's approach
}

static void getOursUpsamplingRes(const Mesh& secMesh, const Eigen::VectorXd& edgeOmega, const std::vector<std::complex<double>>& zvals, Mesh& upMesh, Eigen::VectorXd& upEdgeOmega, std::vector<std::complex<double>>& upZvals, int upLevel)
{
	Eigen::VectorXd edgeVec = swapEdgeVec(triF, edgeOmega, 0);

	std::shared_ptr<ComplexLoop> complexLoopOpt = std::make_shared<ComplexLoopZuenko>();
	complexLoopOpt->setBndFixFlag(true);
	complexLoopOpt->SetMesh(secMesh);
	complexLoopOpt->Subdivide(edgeVec, zvals, upEdgeOmega, upZvals, upLevel);
	upMesh = complexLoopOpt->GetMesh();
}

static void getLoopAmpPhaseUpsamplingRes(const Mesh& secMesh, const Eigen::VectorXd& edgeOmega, const std::vector<std::complex<double>>& zvals, Mesh& upMesh, Eigen::VectorXd& upEdgeOmega, std::vector<std::complex<double>>& upLoopAmpPhi_Zvals, int upLevel)
{
    Eigen::VectorXd edgeVec = swapEdgeVec(triF, edgeOmega, 0);

    std::shared_ptr<ComplexLoop> complexLoopOpt = std::make_shared<ComplexLoopAmpPhase>();
    complexLoopOpt->setBndFixFlag(true);
    complexLoopOpt->SetMesh(secMesh);
    complexLoopOpt->Subdivide(edgeVec, zvals, upEdgeOmega, upLoopAmpPhi_Zvals, upLevel);
    upMesh = complexLoopOpt->GetMesh();
}

static void getLoopReImUpsamplingRes(const Mesh& secMesh, const Eigen::VectorXd& edgeOmega, const std::vector<std::complex<double>>& zvals, Mesh& upMesh, Eigen::VectorXd& upEdgeOmega, std::vector<std::complex<double>>& upLoopReIm_Zvals, int upLevel)
{
    Eigen::VectorXd edgeVec = swapEdgeVec(triF, edgeOmega, 0);

    std::shared_ptr<ComplexLoop> complexLoopOpt = std::make_shared<ComplexLoopReIm>();
    complexLoopOpt->setBndFixFlag(true);
    complexLoopOpt->SetMesh(secMesh);
    complexLoopOpt->Subdivide(edgeVec, zvals, upEdgeOmega, upLoopReIm_Zvals, upLevel);
    upMesh = complexLoopOpt->GetMesh();
}


static void updateWrinkles(const Eigen::MatrixXd& upV, const Eigen::MatrixXi& upF, const std::vector<std::complex<double>>& upZvals, Eigen::MatrixXd& wrinkledV, double scaleRatio, bool isUseV2)
// all the things are upsampled
{
	
	std::vector<std::vector<int>> vertNeiEdges;
	std::vector<std::vector<int>> vertNeiFaces;

	buildVertexNeighboringInfo(MeshConnectivity(upF), upV.rows(), vertNeiEdges, vertNeiFaces);

	getWrinkledMesh(upV, upF, upZvals, &vertNeiFaces, wrinkledV, scaleRatio, isUseV2);
}

static void upsamplingEveryThingForComparison()
{
	getOursUpsamplingRes(secMesh, omega, zvals, upSecMesh, upOmega, upZvals, upsamplingLevel);
    getLoopAmpPhaseUpsamplingRes(secMesh, omega, zvals, upSecMesh, upOmega, upLoopAmpPhi_Zvals, upsamplingLevel);
    getLoopReImUpsamplingRes(secMesh, omega, zvals, upSecMesh, upOmega, upLoopReIm_Zvals, upsamplingLevel);
	getKnoppelUpsamplingPhi(triV, triF, omega, zvals, upsampledTriV, upsampledTriF, knoppelPhi, upsamplingLevel);
	getSideVertexUpsamplingAmpPhi(triV, triF, omega, zvals, upsampledTriV, upsampledTriF, sideVertexLinearAmp, sideVertexLinearPhi, sideVertexWojtanAmp, sideVertexWojtanPhi, upsamplingLevel);
	getClouhTorcherUpsamplingAmpPhi(triV, triF, omega, zvals, upsampledTriV, upsampledTriF, ClouhTorcherAmp, ClouhTorcherPhi, upsamplingLevel);

	parseSecMesh(upSecMesh, loopTriV, loopTriF);


	upPhi.resize(upZvals.size());
	upAmp.resize(upZvals.size());

	for (int j = 0; j < upZvals.size(); j++)
	{
		upPhi[j] = std::arg(upZvals[j]);
		upAmp[j] = std::abs(upZvals[j]);
	}

    upLoopAmpPhi_Phi.resize(upLoopAmpPhi_Zvals.size());
    upLoopAmpPhi_Amp.resize(upLoopAmpPhi_Zvals.size());

    for (int j = 0; j < upZvals.size(); j++)
    {
        upLoopAmpPhi_Phi[j] = std::arg(upLoopAmpPhi_Zvals[j]);
        upLoopAmpPhi_Amp[j] = std::abs(upLoopAmpPhi_Zvals[j]);
    }

    upLoopReIm_Phi.resize(upLoopReIm_Zvals.size());
    upLoopReIm_Amp.resize(upLoopReIm_Zvals.size());

    for (int j = 0; j < upLoopReIm_Zvals.size(); j++)
    {
        upLoopReIm_Phi[j] = std::arg(upLoopReIm_Zvals[j]);
        upLoopReIm_Amp[j] = std::abs(upLoopReIm_Zvals[j]);
    }
    
	faceOmega = intrinsicEdgeVec2FaceVec(omega, triV, triMesh);

	amp.resize(zvals.size());

	for (int j = 0; j < zvals.size(); j++)
    {
		amp[j] = std::abs(zvals[j]);
	}
    globalAmpMin = std::min(0.0, amp.minCoeff());
    globalAmpMax = amp.maxCoeff();

	std::cout << "compute wrinkle meshes: " << std::endl;
	updateWrinkles(loopTriV, loopTriF, upZvals, wrinkledV, wrinkleAmpScalingRatio, isUseV2);

    sideVertexLinearWrinkledV = upsampledTriV;
    sideVertexWojtanWrinkledV = upsampledTriV;
    ClouhTorcherWrinkledV = upsampledTriV;
    loopWrinkledV = loopTriV;
    loopReImWrinkledV = loopTriV;

    Eigen::MatrixXd upsampledN, loopN;
    igl::per_vertex_normals(upsampledTriV, upsampledTriF, upsampledN);
    igl::per_vertex_normals(loopTriV, loopTriF, loopN);

    for(int i = 0; i < upsampledN.rows(); i++)
    {
        sideVertexLinearWrinkledV.row(i) += wrinkleAmpScalingRatio * sideVertexLinearAmp[i] * std::cos(sideVertexLinearPhi[i]) * upsampledN.row(i);
        sideVertexWojtanWrinkledV.row(i) += wrinkleAmpScalingRatio * sideVertexWojtanAmp[i] * std::cos(sideVertexWojtanPhi[i]) * upsampledN.row(i);
        ClouhTorcherWrinkledV.row(i) += wrinkleAmpScalingRatio * ClouhTorcherAmp[i] * std::cos(ClouhTorcherPhi[i]) * upsampledN.row(i);
        loopWrinkledV.row(i) += wrinkleAmpScalingRatio * upLoopAmpPhi_Amp[i] * std::cos(upLoopAmpPhi_Phi[i]) * loopN.row(i);
        loopReImWrinkledV.row(i) += wrinkleAmpScalingRatio * upLoopReIm_Amp[i] * std::cos(upLoopReIm_Phi[i]) * loopN.row(i);
    }

    std::vector<std::pair<int, Eigen::Vector3d>> bary;
    Eigen::MatrixXd baseN, zuenkoN;
    meshUpSampling(triV,  triF, zuenkoUpV, zuenkoUpF, upsamplingLevel, NULL, NULL, &bary);
    igl::per_vertex_normals(triV, triF,  baseN);
    ZuenkoAlg::spherigonSmoothing(triV, triMesh, baseN, bary, zuenkoUpV, zuenkoN, true);
    igl::per_vertex_normals(zuenkoUpV, zuenkoUpF,  zuenkoN);

    std::vector<std::complex<double>> curZvals = zvals;
    for (int i = 0; i < curZvals.size(); i++)
    {
        double phi = std::arg(curZvals[i]);
        curZvals[i] = std::complex<double>(std::cos(phi), std::sin(phi));
    }
    ZuenkoAlg::getZuenkoSurfacePerframe(triV, triMesh, curZvals, amp, omega, zuenkoUpV, zuenkoUpF, zuenkoN, bary, zuenkoWrinkledV, zuenkoUpF, zuenkoAmp, zuenkoPhi, wrinkleAmpScalingRatio);

    TFWAlg::getTFWSurfacePerframe(triV, triMesh.faces(), amp, omega, TFWWrinkledV, TFWUpF, &TFWUpV, &TFWUpF, NULL, NULL, NULL, NULL, TFWAmp, NULL, TFWPhi, upsamplingLevel, wrinkleAmpScalingRatio, false, true);

}


static void updateView()
{
	double shiftx = 1.5 * (triV.col(0).maxCoeff() - triV.col(0).minCoeff());
    double shifty = 1.5 * (triV.col(1).maxCoeff() - triV.col(1).minCoeff());

    if(shifty == 0 && shiftx != 0)
        shifty = shiftx;
    else if(shifty != 0 && shiftx == 0)
        shiftx = shifty;
    if(shifty == 0)
    {
        shiftx = 1.5 * (triV.col(2).maxCoeff() - triV.col(2).minCoeff());
        shifty = shiftx;
    }

	int n = 0;
    int m = 0;

    if (isShowEveryThing)
    {
        polyscope::registerSurfaceMesh("base mesh", triV, triF);
        polyscope::getSurfaceMesh("base mesh")->addFaceVectorQuantity("frequency field", faceOmega);
        auto initAmp = polyscope::getSurfaceMesh("base mesh")->addVertexScalarQuantity("amplitude", amp);
        initAmp->setMapRange(std::pair<double, double>(globalAmpMin, globalAmpMax));
        n++;
    }
    int oldn = n;

    // ours phase pattern
    mPaint.setNormalization(false);
    polyscope::registerSurfaceMesh("CWF phase mesh", loopTriV, loopTriF);
    polyscope::getSurfaceMesh("CWF phase mesh")->translate({ n * shiftx, m * shifty, 0 });
    Eigen::MatrixXd phaseColor = mPaint.paintPhi(upPhi);
    auto ourPhasePatterns = polyscope::getSurfaceMesh("CWF phase mesh")->addVertexColorQuantity("vertex phi", phaseColor);
    ourPhasePatterns->setEnabled(true);
    n++;

    // naive loop amp and pahse pattern
    polyscope::registerSurfaceMesh("loop_amp_phase phase mesh", loopTriV, loopTriF);
    polyscope::getSurfaceMesh("loop_amp_phase phase mesh")->translate({ n * shiftx, m * shifty, 0 });
    phaseColor = mPaint.paintPhi(upLoopAmpPhi_Phi);
    auto loopPhasePatterns = polyscope::getSurfaceMesh("loop_amp_phase phase mesh")->addVertexColorQuantity("vertex phi", phaseColor);
    loopPhasePatterns->setEnabled(true);
    n++;

    // naive loop Re and Im pattern
    polyscope::registerSurfaceMesh("loop_re_im phase mesh", loopTriV, loopTriF);
    polyscope::getSurfaceMesh("loop_re_im phase mesh")->translate({ n * shiftx, m * shifty, 0 });
    phaseColor = mPaint.paintPhi(upLoopReIm_Phi);
    loopPhasePatterns = polyscope::getSurfaceMesh("loop_re_im phase mesh")->addVertexColorQuantity("vertex phi", phaseColor);
    loopPhasePatterns->setEnabled(true);
    n++;


    // zuenko pattern
    polyscope::registerSurfaceMesh("Zuenko phase mesh", zuenkoUpV, zuenkoUpF);
    polyscope::getSurfaceMesh("Zuenko phase mesh")->translate({ n * shiftx, m * shifty, 0 });
    phaseColor = mPaint.paintPhi(zuenkoPhi);
    loopPhasePatterns = polyscope::getSurfaceMesh("Zuenko phase mesh")->addVertexColorQuantity("vertex phi", phaseColor);
    loopPhasePatterns->setEnabled(true);
    n++;


    // TFW pattern
    polyscope::registerSurfaceMesh("TFW phase mesh", TFWUpV, TFWUpF);
    polyscope::getSurfaceMesh("TFW phase mesh")->translate({ n * shiftx, m * shifty, 0 });
    phaseColor = mPaint.paintPhi(TFWPhi);
    loopPhasePatterns = polyscope::getSurfaceMesh("TFW phase mesh")->addVertexColorQuantity("vertex phi", phaseColor);
    loopPhasePatterns->setEnabled(true);
    n++;


    // wojtan side vertex pahse pattern
    polyscope::registerSurfaceMesh("Wojtan-side phase mesh", upsampledTriV, upsampledTriF);
    polyscope::getSurfaceMesh("Wojtan-side phase mesh")->translate({ n * shiftx, m * shifty, 0 });
    phaseColor = mPaint.paintPhi(sideVertexWojtanPhi);
    auto wojtanPhasePatterns = polyscope::getSurfaceMesh("Wojtan-side phase mesh")->addVertexColorQuantity("vertex phi", phaseColor);
    wojtanPhasePatterns->setEnabled(true);
    n++;

    // cubic side Clouh-Torcher pattern
    polyscope::registerSurfaceMesh("Clouh-Torcher phase mesh", upsampledTriV, upsampledTriF);
    polyscope::getSurfaceMesh("Clouh-Torcher phase mesh")->translate({ n * shiftx, m * shifty, 0 });
    phaseColor = mPaint.paintPhi(ClouhTorcherPhi);
    auto cubicPhasePatterns = polyscope::getSurfaceMesh("Clouh-Torcher phase mesh")->addVertexColorQuantity("vertex phi", phaseColor);
    cubicPhasePatterns->setEnabled(true);
    n++;

    m++;
    n = oldn;
    // ours amp pattern
    mPaint.setNormalization(false);
    polyscope::registerSurfaceMesh("CWF amp mesh", loopTriV, loopTriF);
    polyscope::getSurfaceMesh("CWF amp mesh")->translate({ n * shiftx, m * shifty, 0 });
    auto ourAmpPatterns = polyscope::getSurfaceMesh("CWF amp mesh")->addVertexScalarQuantity("vertex amp", upAmp);
    ourAmpPatterns->setColorMap("coolwarm");
    ourAmpPatterns->setMapRange({globalAmpMin, globalAmpMax});
    ourAmpPatterns->setEnabled(true);
    n++;

    // naive loop amp and pahse pattern
    polyscope::registerSurfaceMesh("loop_amp_phase amp mesh", loopTriV, loopTriF);
    polyscope::getSurfaceMesh("loop_amp_phase amp mesh")->translate({ n * shiftx, m * shifty, 0 });
    auto loopAmpPatterns = polyscope::getSurfaceMesh("loop_amp_phase amp mesh")->addVertexScalarQuantity("vertex amp", upLoopAmpPhi_Amp);
    loopAmpPatterns->setColorMap("coolwarm");
    loopAmpPatterns->setMapRange({globalAmpMin, globalAmpMax});
    loopAmpPatterns->setEnabled(true);
    n++;

    // naive loop Re and Im pattern
    polyscope::registerSurfaceMesh("loop_re_im amp mesh", loopTriV, loopTriF);
    polyscope::getSurfaceMesh("loop_re_im amp mesh")->translate({ n * shiftx, m * shifty, 0 });
    loopAmpPatterns = polyscope::getSurfaceMesh("loop_re_im amp mesh")->addVertexScalarQuantity("vertex amp", upLoopReIm_Amp);
    loopAmpPatterns->setColorMap("coolwarm");
    loopAmpPatterns->setMapRange({globalAmpMin, globalAmpMax});
    loopAmpPatterns->setEnabled(true);
    n++;

    // zuenko
    polyscope::registerSurfaceMesh("Zuenko amp mesh", zuenkoUpV, zuenkoUpF);
    polyscope::getSurfaceMesh("Zuenko amp mesh")->translate({ n * shiftx, m * shifty, 0 });
    auto ZuenkoAmpPatterns = polyscope::getSurfaceMesh("Zuenko amp mesh")->addVertexScalarQuantity("vertex amp", zuenkoAmp);
    ZuenkoAmpPatterns->setColorMap("coolwarm");
    ZuenkoAmpPatterns->setMapRange({globalAmpMin, globalAmpMax});
    ZuenkoAmpPatterns->setEnabled(true);
    n++;

    // tfw
    polyscope::registerSurfaceMesh("TFW amp mesh", TFWUpV, TFWUpF);
    polyscope::getSurfaceMesh("TFW amp mesh")->translate({ n * shiftx, m * shifty, 0 });
    auto TFWAmpPatterns = polyscope::getSurfaceMesh("TFW amp mesh")->addVertexScalarQuantity("vertex amp", TFWAmp);
    TFWAmpPatterns->setColorMap("coolwarm");
    TFWAmpPatterns->setMapRange({globalAmpMin, globalAmpMax});
    TFWAmpPatterns->setEnabled(true);
    n++;


    // wojtan side vertex pattern
    polyscope::registerSurfaceMesh("Wojtan-side amp mesh", upsampledTriV, upsampledTriF);
    polyscope::getSurfaceMesh("Wojtan-side amp mesh")->translate({ n * shiftx, m * shifty, 0 });
    auto wojtanAmpPatterns = polyscope::getSurfaceMesh("Wojtan-side amp mesh")->addVertexScalarQuantity("vertex amp", sideVertexWojtanAmp);
    wojtanAmpPatterns->setColorMap("coolwarm");
    wojtanAmpPatterns->setMapRange({globalAmpMin, globalAmpMax});
    wojtanPhasePatterns->setEnabled(true);
    n++;


    // Clouh-Torcher
    polyscope::registerSurfaceMesh("Clouh-Torcher amp mesh", upsampledTriV, upsampledTriF);
    polyscope::getSurfaceMesh("Clouh-Torcher amp mesh")->translate({ n * shiftx, m * shifty, 0 });
    auto CTAmpPatterns = polyscope::getSurfaceMesh("Clouh-Torcher amp mesh")->addVertexScalarQuantity("vertex amp", ClouhTorcherAmp);
    CTAmpPatterns->setColorMap("coolwarm");
    CTAmpPatterns->setMapRange({globalAmpMin, globalAmpMax});
    CTAmpPatterns->setEnabled(true);
    n++;



    m++;
    n = oldn;
    polyscope::registerSurfaceMesh("ours wrinkled mesh", wrinkledV, loopTriF);
    polyscope::getSurfaceMesh("ours wrinkled mesh")->translate({ n * shiftx, m * shifty, 0 });
    n++;

    polyscope::registerSurfaceMesh("loop_amp_phase wrinkled mesh", loopWrinkledV, loopTriF);
    polyscope::getSurfaceMesh("loop_amp_phase wrinkled mesh")->translate({ n * shiftx, m * shifty, 0 });
    n++;

    polyscope::registerSurfaceMesh("loop_re_im wrinkled mesh", loopReImWrinkledV, loopTriF);
    polyscope::getSurfaceMesh("loop_re_im wrinkled mesh")->translate({ n * shiftx, m * shifty, 0 });
    n++;

    polyscope::registerSurfaceMesh("Zuenko wrinkled mesh", zuenkoWrinkledV, zuenkoUpF);
    polyscope::getSurfaceMesh("Zuenko wrinkled mesh")->translate({ n * shiftx, m * shifty, 0 });
    n++;

    polyscope::registerSurfaceMesh("TFW wrinkled mesh", TFWWrinkledV, TFWUpF);
    polyscope::getSurfaceMesh("TFW wrinkled mesh")->translate({ n * shiftx, m * shifty, 0 });
    n++;


    polyscope::registerSurfaceMesh("Wojtan-side wrinkled mesh", sideVertexWojtanWrinkledV, upsampledTriF);
    polyscope::getSurfaceMesh("Wojtan-side wrinkled mesh")->translate({ n * shiftx, m * shifty, 0 });
    n++;


    polyscope::registerSurfaceMesh("Clouh-Torcher wrinkled mesh", ClouhTorcherWrinkledV, upsampledTriF);
    polyscope::getSurfaceMesh("Clouh-Torcher wrinkled mesh")->translate({ n * shiftx, m * shifty, 0 });
    n++;


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
	initialization(triV, triF, upsampledTriV, upsampledTriF);

	int nedges = triMesh.nEdges();
	int nverts = triV.rows();

	bool isLoadOpt = true;

	std::string zvalFile = "zvals.txt";
    if(jval.contains(std::string_view("zvals")))
        zvalFile = jval["zvals"];
	std::string edgeOmegaFile = "omega.txt";
    if(jval.contains(std::string_view("omega")))
        edgeOmegaFile = jval["omega"];
	std::string phiFile = "";
	if (jval.contains(std::string_view{ "phi" }))
	{
		phiFile = jval["phi"];
	}
	zvalFile = workingFolder + "/" + zvalFile;
	edgeOmegaFile = workingFolder + "/" + edgeOmegaFile;
	phiFile = workingFolder + "/" + phiFile;

	if (!loadVertexZvals(zvalFile, nverts, zvals))
	{
		isLoadOpt = false;
	}

	if (!loadEdgeOmega(edgeOmegaFile, nedges, omega)) {
		isLoadOpt = false;
	}

	if (!isLoadOpt)
	{
		std::cout << "missing required zvals and omega files" << std::endl;
		exit(EXIT_FAILURE);
	}

	std::ifstream pfs(phiFile);
    bool loadPhiFile = true;

	if (!pfs)
	{
		std::cerr << "invalid ref phase file name, use the arg of zvals" << std::endl;
        loadPhiFile = false;

	}
	else
	{
		phi.setZero(triV.rows());

		for (int j = 0; j < triV.rows(); j++)
		{
			std::string line;
			std::getline(pfs, line);
			std::stringstream ss(line);
			std::string x;
			ss >> x;
			if (!ss)
            {
                std::cerr << "invalid ref phase file format in: " << phiFile << ", use the arg of zvals" << std::endl;
                loadPhiFile = false;
                break;
            }
			phi(j) = std::stod(x);
		}
	}

    if(!loadPhiFile)
    {
        phi.setZero(triV.rows());
        for (int j = 0; j < triV.rows(); j++)
        {
            phi[j] = std::arg(zvals[j]);
        }
    }

	

	if (isLoadOpt)
	{
		std::cout << "load zvals and omegas from file!" << std::endl;
	}
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
		updateView();
	}
    if (ImGui::Button("save", ImVec2(-1, 0)))
    {
        std::string saveFolder = workingFolder + "/otherUpsampling/";
        mkdir(saveFolder);
        savePhi4Render(sideVertexWojtanPhi, saveFolder + "sideVertexWojtanPhi.cvs");
        savePhi4Render(ClouhTorcherPhi, saveFolder + "ClouhTorcherPhi.cvs");
        savePhi4Render(upPhi, saveFolder + "CWFPhi.cvs");
        savePhi4Render(upLoopAmpPhi_Phi, saveFolder + "LoopAmpPhasePhi.cvs");
        savePhi4Render(upLoopReIm_Phi, saveFolder + "LoopReImPhi.cvs");
        savePhi4Render(zuenkoPhi, saveFolder + "ZuenkoPhi.cvs");
        savePhi4Render(TFWPhi, saveFolder + "TFWPhi.cvs");


        saveAmp4Render(sideVertexWojtanAmp, saveFolder + "sideVertexWojtanAmp.cvs", globalAmpMin, globalAmpMax);
        saveAmp4Render(ClouhTorcherAmp, saveFolder + "ClouhTorcherAmp.cvs", globalAmpMin, globalAmpMax);
        saveAmp4Render(upAmp, saveFolder + "CWFAmp.cvs", globalAmpMin, globalAmpMax);
        saveAmp4Render(upLoopAmpPhi_Amp, saveFolder + "LoopAmpPhaseAmp.cvs", globalAmpMin, globalAmpMax);
        saveAmp4Render(upLoopReIm_Amp, saveFolder + "LoopReImAmp.cvs", globalAmpMin, globalAmpMax);
        saveAmp4Render(zuenkoAmp, saveFolder + "ZuenkoAmp.cvs", globalAmpMin, globalAmpMax);
        saveAmp4Render(TFWAmp, saveFolder + "TFWAmp.cvs", globalAmpMin, globalAmpMax);



        igl::writeOBJ(saveFolder + "ClouhTorcherWrinkledMesh.obj", ClouhTorcherWrinkledV, upsampledTriF);
        igl::writeOBJ(saveFolder + "sideVertexWojtanWrinkledMesh.obj", sideVertexWojtanWrinkledV, upsampledTriF);
        igl::writeOBJ(saveFolder + "CWFWrinkledMesh.obj", wrinkledV, loopTriF);
        igl::writeOBJ(saveFolder + "loopAmpPhaseWrinkledMesh.obj", loopWrinkledV, loopTriF);
        igl::writeOBJ(saveFolder + "loopReImWrinkledMesh.obj", loopReImWrinkledV, loopTriF);
        igl::writeOBJ(saveFolder + "zuenkoWrinkledMesh.obj", zuenkoWrinkledV, zuenkoUpF);
        igl::writeOBJ(saveFolder + "TFWWrinkledMesh.obj", TFWWrinkledV, TFWUpF);


        igl::writeOBJ(saveFolder + "midpointUpMesh.obj", upsampledTriV, upsampledTriF);
        igl::writeOBJ(saveFolder + "loopUpMesh.obj", loopTriV, loopTriF);
        igl::writeOBJ(saveFolder + "TFWUpMesh.obj", TFWUpV, TFWUpF);
        igl::writeOBJ(saveFolder + "ZuenkoUpMesh.obj", zuenkoUpV, zuenkoUpF);
    }

	if (ImGui::InputInt("underline upsampling level", &upsamplingLevel))
	{
		if (upsamplingLevel < 0)
			upsamplingLevel = 2;
	}

	if (ImGui::CollapsingHeader("Visualization Options", ImGuiTreeNodeFlags_DefaultOpen))
	{
		if (ImGui::DragFloat("wrinkle amp scaling ratio", &wrinkleAmpScalingRatio, 0.0005, 0, 1))
		{
			if (wrinkleAmpScalingRatio >= 0)
				updateView();
		}

	}
	if (ImGui::Checkbox("Show wrinkles and Amp", &isShowEveryThing)) {}

	if (ImGui::Button("recompute", ImVec2(-1, 0)))
	{
		upsamplingEveryThingForComparison();
		updateView();
	}

	if (ImGui::Button("output images", ImVec2(-1, 0)))
	{
		std::string curFolder = std::filesystem::current_path().string();
		std::string name = curFolder + "/output.jpg";
		polyscope::screenshot(name);
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

	upsamplingEveryThingForComparison();
	updateView();
	// Show the gui
	polyscope::show();


	return 0;
}