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
#include "../../include/GetInterpVertexPhi.h"

Eigen::MatrixXd triV, upsampledTriV, loopTriV;
Eigen::MatrixXi triF, upsampledTriF, loopTriF;
MeshConnectivity triMesh;
Mesh secMesh, upSecMesh;

Eigen::MatrixXd wrinkledV;
std::vector<std::complex<double>> zvals, upZvals;
Eigen::VectorXd phi, omega, amp, upOmega, upPhi, upAmp;
Eigen::MatrixXd faceOmega;

Eigen::VectorXd sideVertexLinearPhi, ClouhTorcherPhi, sideVertexWojtanPhi, knoppelPhi;

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

static void getClouhTorcherUpsamplingPhi(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, const Eigen::VectorXd& edgeOmega, const std::vector<std::complex<double>>& zvals, Eigen::MatrixXd& upV, Eigen::MatrixXi& upF, Eigen::VectorXd& upPhi, int upLevel)
{
	Eigen::SparseMatrix<double> mat;
	std::vector<int> facemap;
	std::vector<std::pair<int, Eigen::Vector3d>> bary;


	meshUpSampling(V, F, upV, upF, upLevel, &mat, &facemap, &bary);
	MeshConnectivity mesh(F);
	MeshConnectivity upMesh;

	getClouhTocherPhi(V, mesh, edgeOmega, zvals, bary, upPhi);
}

static void getSideVertexUpsamplingPhi(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, const Eigen::VectorXd& edgeOmega, const std::vector<std::complex<double>>& zvals, Eigen::MatrixXd& upV, Eigen::MatrixXi& upF, Eigen::VectorXd& upPhiLinear, Eigen::VectorXd& upPhiWojtan, int upLevel)
{
	Eigen::SparseMatrix<double> mat;
	std::vector<int> facemap;
	std::vector<std::pair<int, Eigen::Vector3d>> bary;


	meshUpSampling(V, F, upV, upF, upLevel, &mat, &facemap, &bary);
	MeshConnectivity mesh(F);
	MeshConnectivity upMesh;

	getSideVertexPhi(V, mesh, edgeOmega, zvals, bary, upPhiLinear, 0);
	getSideVertexPhi(V, mesh, edgeOmega, zvals, bary, upPhiWojtan, 2);
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
	getKnoppelUpsamplingPhi(triV, triF, omega, zvals, upsampledTriV, upsampledTriF, knoppelPhi, upsamplingLevel);
	getSideVertexUpsamplingPhi(triV, triF, omega, zvals, upsampledTriV, upsampledTriF, sideVertexLinearPhi, sideVertexWojtanPhi, upsamplingLevel);
	getClouhTorcherUpsamplingPhi(triV, triF, omega, zvals, upsampledTriV, upsampledTriF, ClouhTorcherPhi, upsamplingLevel);

	parseSecMesh(upSecMesh, loopTriV, loopTriF);


	upPhi.resize(upZvals.size());
	upAmp.resize(upZvals.size());

	for (int j = 0; j < upZvals.size(); j++)
	{
		upPhi[j] = std::arg(upZvals[j]);
		upAmp[j] = std::abs(upZvals[j]);
	}
	faceOmega = intrinsicEdgeVec2FaceVec(omega, triV, triMesh);

	amp.resize(zvals.size());

	for (int j = 0; j < zvals.size(); j++)
	{
		amp[j] = std::abs(zvals[j]);
		globalAmpMin = std::min(amp[j], globalAmpMin);
		globalAmpMax = std::max(amp[j], globalAmpMax);
	}

	std::cout << "compute wrinkle meshes: " << std::endl;
	updateWrinkles(loopTriV, loopTriF, upZvals, wrinkledV, wrinkleAmpScalingRatio, isUseV2);
}


static void updateView()
{
	double shiftx = 1.5 * (triV.col(0).maxCoeff() - triV.col(0).minCoeff());
	int n = 0;

	polyscope::registerSurfaceMesh("base mesh", triV, triF);
	polyscope::getSurfaceMesh("base mesh")->addFaceVectorQuantity("frequency field", faceOmega);
	auto initAmp = polyscope::getSurfaceMesh("base mesh")->addVertexScalarQuantity("amplitude", amp);
	initAmp->setMapRange(std::pair<double, double>(globalAmpMin, globalAmpMax));
	n++;
	
	if (isShowEveryThing)
	{
		// wrinkle mesh
		polyscope::registerSurfaceMesh("wrinkled mesh", wrinkledV, loopTriF);
		polyscope::getSurfaceMesh("wrinkled mesh")->setSurfaceColor({ 80 / 255.0, 122 / 255.0, 91 / 255.0 });
		polyscope::getSurfaceMesh("wrinkled mesh")->translate({ n * shiftx, 0, 0 });
		n++;


		// amp pattern
		polyscope::registerSurfaceMesh("upsampled ampliude mesh", loopTriV, loopTriF);
		polyscope::getSurfaceMesh("upsampled ampliude mesh")->translate({ n * shiftx, 0, 0 });
		auto ampPatterns = polyscope::getSurfaceMesh("upsampled ampliude mesh")->addVertexScalarQuantity("vertex amplitude", upAmp);
		ampPatterns->setMapRange(std::pair<double, double>(globalAmpMin, globalAmpMax));
		ampPatterns->setEnabled(true);
		n++;
	}
	

	// ours phase pattern
	mPaint.setNormalization(false);
	polyscope::registerSurfaceMesh("upsampled phase mesh", loopTriV, loopTriF);
	polyscope::getSurfaceMesh("upsampled phase mesh")->translate({ n * shiftx, 0, 0 });
	Eigen::MatrixXd phaseColor = mPaint.paintPhi(upPhi);
	auto ourPhasePatterns = polyscope::getSurfaceMesh("upsampled phase mesh")->addVertexColorQuantity("vertex phi", phaseColor);
	ourPhasePatterns->setEnabled(true);
	n++;

	// knoppel pahse pattern
	/*polyscope::registerSurfaceMesh("knoppel phase mesh", upsampledTriV, upsampledTriF);
	polyscope::getSurfaceMesh("knoppel phase mesh")->translate({ n * shiftx, 0, 0 });
	phaseColor = mPaint.paintPhi(knoppelPhi);
	auto knoppelPhasePatterns = polyscope::getSurfaceMesh("knoppel phase mesh")->addVertexColorQuantity("vertex phi", phaseColor);
	knoppelPhasePatterns->setEnabled(true);
	n++;*/


	// linear side vertex pahse pattern
	polyscope::registerSurfaceMesh("Linear-side phase mesh", upsampledTriV, upsampledTriF);
	polyscope::getSurfaceMesh("Linear-side phase mesh")->translate({ n * shiftx, 0, 0 });
	phaseColor = mPaint.paintPhi(sideVertexLinearPhi);
	auto linearPhasePatterns = polyscope::getSurfaceMesh("Linear-side phase mesh")->addVertexColorQuantity("vertex phi", phaseColor);
	linearPhasePatterns->setEnabled(true);
	n++;


	// cubic side vertex pahse pattern
	polyscope::registerSurfaceMesh("Clouh-Torcher phase mesh", upsampledTriV, upsampledTriF);
	polyscope::getSurfaceMesh("Clouh-Torcher phase mesh")->translate({ n * shiftx, 0, 0 });
	phaseColor = mPaint.paintPhi(ClouhTorcherPhi);
	auto cubicPhasePatterns = polyscope::getSurfaceMesh("Clouh-Torcher phase mesh")->addVertexColorQuantity("vertex phi", phaseColor);
	cubicPhasePatterns->setEnabled(true);
	n++;


	// wojtan side vertex pahse pattern
	polyscope::registerSurfaceMesh("Wojtan-side phase mesh", upsampledTriV, upsampledTriF);
	polyscope::getSurfaceMesh("Wojtan-side phase mesh")->translate({ n * shiftx, 0, 0 });
	phaseColor = mPaint.paintPhi(sideVertexWojtanPhi);
	auto wojtanPhasePatterns = polyscope::getSurfaceMesh("Wojtan-side phase mesh")->addVertexColorQuantity("vertex phi", phaseColor);
	wojtanPhasePatterns->setEnabled(true);
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

	std::string zvalFile = jval["zvals"];
	std::string edgeOmegaFile = jval["omega"];
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