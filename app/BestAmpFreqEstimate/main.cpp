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

static void getFrequencyInfo(const Eigen::MatrixXd& V, const MeshConnectivity& mesh, int axis, double value, double& maxAmp, double& freq)
{
    if (axis < 0 || axis > 2)
    {
        std::cout << "wrong axis direction, should be 0, 1, 2, for x, y, z, set default to x direction." << std::endl;
        axis = 0;
    }
    int nedges = mesh.nEdges();
    std::vector<Eigen::Vector2d> middlepts;
    // double midZ = ( V.col(2).minCoeff() + V.col(2).maxCoeff() ) / 2.0;
    for (int i = 0; i < nedges; i++)
    {
        int v0 = mesh.edgeVertex(i, 0);
        int v1 = mesh.edgeVertex(i, 1);
        Eigen::Vector3d vert0 = V.row(v0).transpose();
        Eigen::Vector3d vert1 = V.row(v1).transpose();
        double z0 = vert0[axis];
        double z1 = vert1[axis];
        if (z0 == z1)
            continue;
        //compute barycentric coordinates
        double t = (value - z0) / (z1 - z0);
        if (t >= 0 && t <= 1.0)
        {
            Eigen::Vector3d mid = (1.0 - t) * vert0 + t * vert1;
            assert(fabs(mid[axis] - midZ) < 1e-8);
            if (axis == 0)
                middlepts.push_back(mid.segment<2>(1));
            else if (axis == 2)
                middlepts.push_back(mid.segment<2>(0));
            else
                middlepts.push_back(Eigen::Vector2d(mid[2], mid[0]));
        }
    }

    int npts = middlepts.size();
    double minr = std::numeric_limits<double>::infinity();
    double maxr = 0;
    double totr = 0;
    for (int i = 0; i < npts; i++)
    {
        double r = middlepts[i].norm();
        minr = std::min(minr, r);
        maxr = std::max(maxr, r);
        totr += r;
    }
    double avgr = totr / double(npts);
    double amplitude = (maxr - minr) / 2.0;
    std::cout << "val: " << value << std::endl;
    std::cout << "amplitude at val: " << amplitude << std::endl;
    std::cout << "Radius of the circle: " << avgr << std::endl;

    int nmodes = npts / 2;
    struct AngPt
    {
        double angle;
        double amplitude;
        bool operator<(const AngPt& other) const
        {
            return angle < other.angle;
        }
    };
    std::vector<AngPt> angpts;
    for (int i = 0; i < npts; i++)
    {
        double theta = std::atan2(middlepts[i][1], middlepts[i][0]);
        double ampt = middlepts[i].norm() - avgr;
        angpts.push_back({ theta,ampt });
    }
    std::sort(angpts.begin(), angpts.end());
    const double PI = 3.1415926535898;
    int bestmode = -1;
    double bestprod = 0;

    for (int mode = 1; mode < nmodes; mode++)
    {
        double sinterm = 0;
        double costerm = 0;
        for (int i = 0; i < npts; i++)
        {
            int next = (i + 1) % npts;
            int prev = (npts + i - 1) % npts;
            double nexttheta = angpts[next].angle;
            double prevtheta = angpts[prev].angle;
            if (i == 0 || i == npts - 1)
                nexttheta += 2.0 * PI;
            double da = 0.5 * (nexttheta - prevtheta);
            double theta = angpts[i].angle;
            costerm += std::cos(mode * theta) * angpts[i].amplitude * da;
            sinterm += std::sin(mode * theta) * angpts[i].amplitude * da;
        }
        double prod = costerm * costerm + sinterm * sinterm;

        if (prod > bestprod)
        {
            bestprod = prod;
            bestmode = mode;
        }
    }
    std::cout << "current val has " << bestmode << " wrinkle periods" << std::endl;

    maxAmp = 0;
    for (int i = 0; i < npts; i++)
    {
        maxAmp += std::abs(angpts[i].amplitude);
    }
    maxAmp /= npts;
    freq = bestmode;
}


Eigen::MatrixXd CIPCV, triV, upsampledV;
Eigen::MatrixXi CIPCF, triF, upsampledF;
Mesh secMesh;
MeshConnectivity triMesh;


std::vector<Eigen::VectorXd> preComputedRotateOmega;
std::vector<std::vector<std::complex<double>>> preComputedZvals, preComputedUpsampledZvals;
std::vector<Eigen::MatrixXd> wrinkledVList;
std::vector<Eigen::MatrixXd> preComputedWrinkledUpdates;

std::vector<Eigen::MatrixXd> faceOmegaList;


Eigen::VectorXd initOmega;
Eigen::VectorXd initAmp;
Eigen::VectorXd tarOmega;
Eigen::VectorXd tarAmp;

float rescalingAmp = 1.0;
double rescalingFreq = 1.0;
int upsampleTimes = 2;

double maxRot = 90;
int numSamples = 180;
float vecRatio = 0.001; // for visualization
std::string workingFolder = "";
int curFrame = 0;

void updateMagnitudePhase(
        const std::vector<Eigen::VectorXd>& wFrames,
        const std::vector<std::vector<std::complex<double>>>& zFrames,
        std::vector<Eigen::VectorXd>* subOmegaList,
        std::vector<std::vector<std::complex<double>>>& upZFrames,
        int upsampleTimes)
{
	upZFrames.resize(wFrames.size());
    if(subOmegaList)
        subOmegaList->resize(wFrames.size());

	MeshConnectivity mesh(triF);

	auto computeMagPhase = [&](const tbb::blocked_range<uint32_t>& range)
	{
		for (uint32_t i = range.begin(); i < range.end(); ++i)
		{
			Eigen::VectorXd edgeVec = swapEdgeVec(triF, wFrames[i], 0);
			std::shared_ptr<ComplexLoop> complexLoopOpt = std::make_shared<ComplexLoopZuenko>();
			complexLoopOpt->setBndFixFlag(true);
			complexLoopOpt->SetMesh(secMesh);
            Eigen::VectorXd subOmega;
			complexLoopOpt->Subdivide(edgeVec, zFrames[i], subOmega, upZFrames[i], upsampleTimes);
            if(subOmegaList)
                subOmegaList->at(i) = subOmega;
		}
	};

	tbb::blocked_range<uint32_t> rangex(0u, (uint32_t)upZFrames.size());
	tbb::parallel_for(rangex, computeMagPhase);

}


void updateWrinkles(
        const Eigen::MatrixXd& V,
        const Eigen::MatrixXi& F,
        const std::vector<std::vector<std::complex<double>>>& zFrames,
        std::vector<Eigen::MatrixXd>& wrinkledVFrames,
        double scaleRatio,
        bool isUseV2)
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

void getUpsampledMesh(const Mesh &secMesh, Eigen::MatrixXd& upsampledTriV, Eigen::MatrixXi& upsampledTriF, int upsampleTimes)
{
    std::shared_ptr<ComplexLoop> complexLoopOpt = std::make_shared<ComplexLoopZuenko>();
    complexLoopOpt->setBndFixFlag(true);
    complexLoopOpt->SetMesh(secMesh);
    complexLoopOpt->meshSubdivide(upsampleTimes);
    parseSecMesh(complexLoopOpt->GetMesh(), upsampledTriV, upsampledTriF);
	
}

void initialization(const Eigen::MatrixXd& triV, const Eigen::MatrixXi& triF, Eigen::MatrixXd& upsampledTriV, Eigen::MatrixXi& upsampledTriF, int upsampleTimes)
{
    secMesh = convert2SecMesh(triV, triF);
    triMesh = MeshConnectivity(triF);
	getUpsampledMesh(secMesh, upsampledTriV, upsampledTriF, upsampleTimes);
}

void updatePaintingItems(bool isReCompute = false)
{
	// get interploated amp and phase frames
    if(isReCompute)
    {
        std::cout << "compute upsampled phase: " << std::endl;
        updateMagnitudePhase(preComputedRotateOmega, preComputedZvals, nullptr, preComputedUpsampledZvals, upsampleTimes);

        std::cout << "compute wrinkle meshes: " << std::endl;
        updateWrinkles(upsampledV, upsampledF, preComputedUpsampledZvals, wrinkledVList, 1, true);
        preComputedWrinkledUpdates.resize(wrinkledVList.size());

        for(int i = 0; i < wrinkledVList.size(); i++)
        {
            preComputedWrinkledUpdates[i] = wrinkledVList[i] - upsampledV;
            wrinkledVList[i] = upsampledV + rescalingAmp * preComputedWrinkledUpdates[i];
        }

    }
    else
    {
        for(int i = 0; i < wrinkledVList.size(); i++)
        {
            wrinkledVList[i] = upsampledV + rescalingAmp * preComputedWrinkledUpdates[i];
        }
    }
	std::cout << "start to update viewer." << std::endl;
}

void registerMesh(int frameId, bool isFirstTime = true)
{
	if (isFirstTime)
	{
		polyscope::registerSurfaceMesh("base mesh", triV, triF);
	}
	else
		polyscope::getSurfaceMesh("base mesh")->updateVertexPositions(triV);
	
	polyscope::getSurfaceMesh("base mesh")->addFaceVectorQuantity("frequency field", vecRatio * faceOmegaList[frameId], polyscope::VectorType::AMBIENT);

	// wrinkle mesh
	if(isFirstTime)
	{
		polyscope::registerSurfaceMesh("wrinkled mesh", wrinkledVList[frameId], upsampledF);
		polyscope::getSurfaceMesh("wrinkled mesh")->setSurfaceColor({ 80 / 255.0, 122 / 255.0, 91 / 255.0 });
	}
		
	else
		polyscope::getSurfaceMesh("wrinkled mesh")->updateVertexPositions(wrinkledVList[frameId]);
}

void updateView(int frameId, bool isFirstTime)
{
	//std::cout << "update viewer. " << std::endl;
	registerMesh(frameId, isFirstTime);
}


void preComputation(int numSamples, double maxAngle)
{
    double deltaTheta = maxAngle / (numSamples - 1);
    preComputedRotateOmega.resize(numSamples);
    preComputedZvals.resize(numSamples);

    preComputedRotateOmega[0] = initOmega * rescalingFreq;
    Eigen::VectorXd edgeArea, vertArea;
    edgeArea = getEdgeArea(triV, triMesh);
    vertArea = getVertArea(triV, triMesh);
    int nverts = triV.rows();

    IntrinsicFormula::roundZvalsFromEdgeOmega(triMesh, preComputedRotateOmega[0], edgeArea, vertArea, nverts, preComputedZvals[0]);

    for(int v = 0; v < nverts; v++)
    {
        auto z = preComputedZvals[0][v];
        preComputedZvals[0][v] = initAmp[v] * std::complex<double>(std::cos(std::arg(z)), std::sin(std::arg(z)));
    }

    Eigen::MatrixXd faceNormals;
    igl::per_face_normals(triV, triF, faceNormals);

    faceOmegaList.resize(preComputedRotateOmega.size());
    faceOmegaList[0] = intrinsicEdgeVec2FaceVec(preComputedRotateOmega[0], triV, triMesh);

    for(int i = 1; i < numSamples; i++)
    {
        double angle = -i * deltaTheta / 180.0 * M_PI;
        faceOmegaList[i] = faceOmegaList[0];
        for(int f = 0; f < triF.rows(); f++)
        {
            Eigen::Vector3d axis = faceNormals.row(f);
            Eigen::Vector3d vec = faceOmegaList[0].row(f).segment<3>(0);
            Eigen::Vector3d rotVec = rotateSingleVector(vec, axis, angle);
            faceOmegaList[i].row(f) = rotVec;
        }
        preComputedRotateOmega[i] = faceVec2IntrinsicEdgeVec(faceOmegaList[i], triV, triMesh);
        IntrinsicFormula::roundZvalsFromEdgeOmega(triMesh, preComputedRotateOmega[i], edgeArea, vertArea, nverts, preComputedZvals[i]);

        for(int v = 0; v < nverts; v++)
        {
            auto z = preComputedZvals[i][v];
            preComputedZvals[i][v] = initAmp[v] * std::complex<double>(std::cos(std::arg(z)), std::sin(std::arg(z)));
        }
    }
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

	std::string meshFile = jval["mesh"];
	upsampleTimes = jval["upsampled_times"];
	if (upsampleTimes > 2)
		upsampleTimes = 2;

	meshFile = workingFolder + meshFile;

    igl::readOBJ(meshFile, triV, triF);

	initialization(triV, triF, upsampledV, upsampledF, upsampleTimes);
	int nedges = triMesh.nEdges();
    int nverts = triV.rows();

	std::string initAmpPath = jval["init_amp"];
	std::string initOmegaPath = jval["init_omega"];

	if (!loadEdgeOmega(workingFolder + initOmegaPath, nedges, initOmega))
    {
		std::cout << "missing init edge omega file." << std::endl;
		return false;
	}

    if (!loadVertexAmp(workingFolder + initAmpPath, nverts, initAmp))
    {
        std::cout << "missing init vert Amp file." << std::endl;
        return false;
    }

    preComputation(numSamples, maxRot);
	updatePaintingItems(true);

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

    tarOmega = preComputedRotateOmega[curFrame];
    tarAmp = initAmp * rescalingAmp;

    Eigen::VectorXd edgeArea, vertArea;
    edgeArea = getEdgeArea(triV, triMesh);
    vertArea = getVertArea(triV, triMesh);
    int nverts = triV.rows();
    std::vector<std::complex<double>> tarZvals;

    IntrinsicFormula::roundZvalsFromEdgeOmega(triMesh, tarOmega, edgeArea, vertArea, triV.rows(), tarZvals);
    for(int i = 0; i < nverts; i++)
    {
        auto z = tarZvals[i];
        z = tarAmp(i) * std::complex<double>(std::cos(std::arg(z)), std::sin(std::arg(z)));
        tarZvals[i] = z;
    }

	saveEdgeOmega(workingFolder + "omega_opt.txt", tarOmega);
	saveVertexAmp(workingFolder + "amp_opt.txt", tarAmp);
	saveVertexZvals(workingFolder + "zvals_opt.txt", tarZvals);

	return true;
}

void callback() {
	ImGui::PushItemWidth(100);
	float w = ImGui::GetContentRegionAvailWidth();
	float p = ImGui::GetStyle().FramePadding.x;
	if (ImGui::Button("Load", ImVec2((w - p) / 2.f, 0)))
	{
		loadProblem();
        updateView(curFrame, true);
	}
	ImGui::SameLine(0, p);
	if (ImGui::Button("Save", ImVec2((w - p) / 2.f, 0)))
	{
		saveProblem();
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
                    initialization(triV, triF, upsampledV, upsampledF, upsampleTimes);
					updatePaintingItems(true);
					updateView(curFrame, true);
				}
			}
			ImGui::EndTabItem();
		}
		ImGui::EndTabBar();
	}

	if (ImGui::CollapsingHeader("Visualization Options", ImGuiTreeNodeFlags_DefaultOpen))
	{
		if (ImGui::DragFloat("wrinkle amp scaling ratio", &(rescalingAmp), 0.0005, 0, 1))
		{
			if (rescalingAmp >= 0)
				updateView(curFrame, false);
		}

        if (ImGui::InputDouble("wrinkle frequency scaling ratio", &(rescalingFreq)))
        {
            if (rescalingFreq > 0)
            {
                preComputation(numSamples, maxRot);
                updatePaintingItems(true);
                updateView(curFrame, true);
            }
        }

		if (ImGui::Button("Compute Optimal Rescaling", ImVec2(-1, 0)))
		{
            double IPCAmp, IPCFreq;
            double guessAmp, guessFreq;

            getFrequencyInfo(CIPCV, MeshConnectivity(CIPCF), 0, (CIPCV.col(0).minCoeff() + CIPCV.maxCoeff()) / 2, IPCAmp, IPCFreq);

            getFrequencyInfo(wrinkledVList[curFrame], MeshConnectivity(upsampledF), 0, (wrinkledVList[curFrame].col(0).minCoeff() + wrinkledVList[curFrame].maxCoeff()) / 2, guessAmp, guessFreq);

            rescalingAmp = IPCAmp / guessAmp;
            rescalingFreq = IPCFreq / guessFreq;

            std::cout << "rescaling Amp: " << rescalingAmp << ", freq: " << rescalingFreq << std::endl;

            preComputation(numSamples, maxRot);
            updatePaintingItems(true);
            updateView(curFrame, true);
		}
	}
	

	
	if (ImGui::CollapsingHeader("Frame Visualization Options", ImGuiTreeNodeFlags_DefaultOpen))
	{
//		if (ImGui::SliderInt("current frame", &curFrame, 0, numFrames - 1))
		if (ImGui::DragInt("current frame", &curFrame, 1, 0, numSamples - 1))
		{
			curFrame = curFrame % (numSamples + 1);
			updateView(curFrame, false);
		}
		if (ImGui::SliderInt("current frame slider bar", &curFrame, 0, numSamples - 1))
		{
			curFrame = curFrame % (numSamples + 1);
			updateView(curFrame, false);
		}
		if (ImGui::DragFloat("vec ratio", &(vecRatio), 0.00005, 0, 1))
		{
			updateView(curFrame, false);
		}
	}

	if (ImGui::Button("output images", ImVec2(-1, 0)))
	{
		std::string curFolder = workingFolder + "/screenshots/";
		mkdir(curFolder);
		std::cout << "save folder: " << curFolder << std::endl;
		for (int i = 0; i < preComputedUpsampledZvals.size(); i++)
		{
			updateView(i, false);
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
    std::string refFile = "";
	CLI::App app("Wrinkle Direction Determination");
	app.add_option("input,-i,--input", inputFile, "Input model")->check(CLI::ExistingFile);
    app.add_option("input,-r,--ref", refFile, "reference model")->check(CLI::ExistingFile);

	try {
		app.parse(argc, argv);
	}
	catch (const CLI::ParseError& e) {
		return app.exit(e);
	}


	if (!loadProblem(inputFile))
	{
		std::cout << "failed to load file." << std::endl;
		return EXIT_FAILURE;
	}

	if (refFile == "")
		refFile = igl::file_dialog_open();
	if (!igl::readOBJ(refFile, CIPCV, CIPCF))
	{
		std::cout << "failed to load C-IPC mesh" << std::endl;
		return EXIT_FAILURE;
	}

	// Options
	polyscope::options::autocenterStructures = false;
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

    polyscope::registerSurfaceMesh("C-IPC mesh", CIPCV, CIPCF);
	updateView(curFrame, true);
	// Show the gui
	polyscope::show();


	return EXIT_SUCCESS;
}