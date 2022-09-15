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
#include "../../include/InterpolationSchemes.h"

Eigen::MatrixXd triV, upsampledTriV;
Eigen::MatrixXi triF, upsampledTriF;
MeshConnectivity triMesh;
Mesh secMesh, upSecMesh;

std::vector<std::vector<std::complex<double>>> zList, upZList;
std::vector<Eigen::VectorXd> omegaList, upOmegaList;

int upsamplingLevel = 2;
float wrinkleAmpScalingRatio = 0.1;
std::string workingFolder = "";
int numFrames = 20;
int curFrame = 0;

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

static void getKnoppelUpsamplingPhi(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, const std::vector<Eigen::VectorXd>& edgeOmegaList, const std::vector<std::vector<std::complex<double>>>& zvalsList, Eigen::MatrixXd& upV, Eigen::MatrixXi& upF, std::vector<Eigen::VectorXd>& upPhiList, int upLevel)
{
    int nframes = edgeOmegaList.size();
    Eigen::SparseMatrix<double> mat;
    std::vector<int> facemap;
    std::vector<std::pair<int, Eigen::Vector3d>> bary;


    meshUpSampling(V, F, upV, upF, upLevel, &mat, &facemap, &bary);
    upPhiList.resize(nframes);

    MeshConnectivity mesh(F);

    auto frameUpsampling = [&](const tbb::blocked_range<uint32_t>& range)
    {
        for (uint32_t i = range.begin(); i < range.end(); ++i)
        {
            IntrinsicFormula::getUpsamplingThetaFromEdgeOmega(mesh, edgeOmegaList[i], zvalsList[i], bary, upPhiList[i]); // knoppel's approach
        }
    };

    tbb::blocked_range<uint32_t> rangex(0u, (uint32_t)nframes);
    tbb::parallel_for(rangex, frameUpsampling);
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

static bool loadProblem()
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
    upsamplingLevel = jval["upsampled_times"];
    if (upsamplingLevel > 2)
        upsamplingLevel = 2;


    meshFile = workingFolder + meshFile;
    igl::readOBJ(meshFile, triV, triF);
    triMesh = MeshConnectivity(triF);
    initialization(triV, triF, upsampledTriV, upsampledTriF);

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

    return true;
}



static void updateEveryThing()
{
    // get the upsampled phase, amplitude and wrinkles from our approach

}

static void updateView()
{

}

static void callback() {
	ImGui::PushItemWidth(100);
	float w = ImGui::GetContentRegionAvailWidth();
	float p = ImGui::GetStyle().FramePadding.x;

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
    if (ImGui::SliderInt("current frame slider bar", &curFrame, 0, numFrames - 1))
    {
        curFrame = curFrame % numFrames;
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
        std::cout << "Usage: ./InterpolationComparison_bin model_path (should contain all the zvals and omega information)" << std::endl;
        exit(EXIT_FAILURE);
    }
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
	polyscope::view::upDir = polyscope::view::UpDir::ZUp;

	// Add the callback
	polyscope::state::userCallback = callback;

	polyscope::options::groundPlaneHeightFactor = 0.25; // adjust the plane height

    updateView();
	// Show the gui
	polyscope::show();


	return 0;
}