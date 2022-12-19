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

Eigen::MatrixXd triV;
Eigen::MatrixXi triF;
MeshConnectivity triMesh;
Mesh secMesh;

Eigen::MatrixXd TFWWrinkledV, TFWPhiV, TFWProbV, TFWUpsamplingV;
Eigen::MatrixXi TFWWrinkledF, TFWPhiF, TFWProbF, TFWUpsamplingF;
std::vector<std::complex<double>> zvals;
Eigen::VectorXd omega, amp, TFWUpPhiSoup, TFWUpAmp, TFWUpPhi;
Eigen::MatrixXd faceOmega;


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


static void upsamplingEveryThingForComparison()
{
	faceOmega = intrinsicEdgeVec2FaceVec(omega, triV, triMesh);

	amp.resize(zvals.size());

	for (int j = 0; j < zvals.size(); j++)
	{
		amp[j] = std::abs(zvals[j]);
		globalAmpMin = std::min(amp[j], globalAmpMin);
		globalAmpMax = std::max(amp[j], globalAmpMax);
	}

	TFWAlg::getTFWSurfacePerframe(triV, triMesh.faces(), amp, omega, TFWWrinkledV, TFWWrinkledF, &TFWUpsamplingV, &TFWUpsamplingF, &TFWPhiV, &TFWPhiF, &TFWProbV, &TFWProbF, TFWUpAmp, &TFWUpPhiSoup, TFWUpPhi, upsamplingLevel, wrinkleAmpScalingRatio);
}

static void updateView()
{
	double shiftx = 1.5 * (triV.col(0).maxCoeff() - triV.col(0).minCoeff());
	int n = 0;

	auto baseSurf = polyscope::registerSurfaceMesh("base mesh", triV, triF);
   
    auto freqFields = polyscope::getSurfaceMesh("base mesh")->addFaceVectorQuantity("frequency field", vecratio * faceOmega, polyscope::VectorType::AMBIENT);
    auto initAmp = polyscope::getSurfaceMesh("base mesh")->addVertexScalarQuantity("amplitude", amp);
    initAmp->setMapRange(std::pair<double, double>(globalAmpMin, globalAmpMax));
	n++;


	////////////////////////////////////// TFW stuffs ///////////////////////////////////////////////
	// wrinkle mesh
	polyscope::registerSurfaceMesh("TFW wrinkled mesh", TFWWrinkledV, TFWWrinkledF);
	polyscope::getSurfaceMesh("TFW wrinkled mesh")->setSurfaceColor({ 80 / 255.0, 122 / 255.0, 91 / 255.0 });
	polyscope::getSurfaceMesh("TFW wrinkled mesh")->translate({ n * shiftx, 0, 0 });
	n++;

	// amp pattern
	polyscope::registerSurfaceMesh("TFW upsampled ampliude mesh", TFWUpsamplingV, TFWUpsamplingF);
	polyscope::getSurfaceMesh("TFW upsampled ampliude mesh")->translate({ n * shiftx,0, 0 });
	auto ampTFWPatterns = polyscope::getSurfaceMesh("TFW upsampled ampliude mesh")->addVertexScalarQuantity("vertex amplitude", TFWUpAmp);
	ampTFWPatterns->setMapRange(std::pair<double, double>(globalAmpMin, globalAmpMax));
	ampTFWPatterns->setEnabled(true);
	n++;


	// phase pattern
	mPaint.setNormalization(false);
	polyscope::registerSurfaceMesh("TFW upsampled phase mesh", TFWUpsamplingV, TFWUpsamplingF);
	polyscope::getSurfaceMesh("TFW upsampled phase mesh")->translate({ n * shiftx, 0, 0 });

	Eigen::MatrixXd TFWPhiColor = mPaint.paintPhi(TFWUpPhi);
	auto TFWPhasePatterns = polyscope::getSurfaceMesh("TFW upsampled phase mesh")->addVertexColorQuantity("vertex phi", TFWPhiColor);
	TFWPhasePatterns->setEnabled(true);
	n++;
}

static bool loadProblem(std::string* inputpath = NULL)
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

	int nedges = triMesh.nEdges();
	int nverts = triV.rows();

	bool isLoadOpt = true;

	std::string zvalFile = jval["zvals"];
	std::string edgeOmegaFile = jval["omega"];
	zvalFile = workingFolder + "/" + zvalFile;
	edgeOmegaFile = workingFolder + "/" + edgeOmegaFile;
	
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

        if (ImGui::DragFloat("freq scaling ratio", &vecratio, 0.0005, 0, 1))
        {
            if (vecratio >= 0)
            {
                polyscope::getSurfaceMesh("base mesh")->addFaceVectorQuantity("frequency field", vecratio * faceOmega, polyscope::VectorType::AMBIENT);
            }
        }

	}
	
	if (ImGui::Button("recompute", ImVec2(-1, 0)))
	{
		upsamplingEveryThingForComparison();
		updateView();
	}

	if (ImGui::Button("output images", ImVec2(-1, 0)))
	{
		std::string curFolder = std::filesystem::current_path().string();

		std::string name = curFolder + "/output.jpg";
		updateView();
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
	curFrame = 0;

	upsamplingEveryThingForComparison();

	updateView();
	// Show the gui
	polyscope::show();


	return 0;
}