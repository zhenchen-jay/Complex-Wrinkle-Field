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
#include "../../include/MeshLib/RegionEdition.h"

enum TestType
{
    Random = 0,
    EndPts = 1
};

Eigen::MatrixXd triV, loopTriV, NV, newN;
Eigen::MatrixXi triF, loopTriF, NF;

MeshConnectivity triMesh;
Mesh secMesh, subSecMesh;
std::vector<std::pair<int, Eigen::Vector3d>> bary;

Eigen::VectorXd initAmp, tarAmp;
Eigen::VectorXd initOmega, tarOmega;
std::vector<std::complex<double>> initZvals, tarZvals;

Eigen::VectorXd loopedAmp0, loopedOmega0, loopedPhase0;
std::vector<std::complex<double>> loopedZvals0;

Eigen::VectorXd loopedAmp1, loopedOmega1, loopedPhase1;
std::vector<std::complex<double>> loopedZvals1;



int loopLevel = 1;

PaintGeometry mPaint;

float vecratio = 0.001;
float wrinkleAmpScalingRatio = 1;

std::string workingFolder;
double eps = 1e-5;


bool isUseTangentCorrection = false;
bool isFixBnd = true;

std::shared_ptr<ComplexLoop> loopModel0, loopModel1;
TestType testType = Random;
double globalAmpMin = 0;
double globalAmpMax = 1;

void updateMagnitudePhase(const Eigen::VectorXd& omega, const std::vector<std::complex<double>>& zvals, std::shared_ptr<ComplexLoop>& loopModel, Eigen::VectorXd& omegaNew, std::vector<std::complex<double>>& upZvals, Eigen::VectorXd& upsampledAmp, Eigen::VectorXd& upsampledPhase)
{
	NV = triV;
	NF = triF;
	Eigen::MatrixXd VN;
	igl::per_vertex_normals(triV, triF, VN);

	meshUpSampling(triV, triF, NV, NF, loopLevel, NULL, NULL, &bary);
	curvedPNTriangleUpsampling(triV, triF, VN, bary, NV, newN);

	secMesh = convert2SecMesh(triV, triF);
	subSecMesh = secMesh;

	Eigen::VectorXd edgeVec = swapEdgeVec(triF, omega, 0);

    loopModel = std::make_shared<ComplexLoopZuenko>();

    loopModel->SetMesh(secMesh);
    loopModel->setBndFixFlag(isFixBnd);
    loopModel->Subdivide(edgeVec, zvals, omegaNew, upZvals, loopLevel);
	subSecMesh = (loopModel->GetMesh());
	parseSecMesh(subSecMesh, loopTriV, loopTriF);

	upsampledAmp.resize(loopTriV.rows());
	upsampledPhase.resize(loopTriV.rows());
	for(int i = 0; i < upsampledAmp.rows(); i++)
	{
		upsampledAmp(i) = std::abs(upZvals[i]);
		upsampledPhase(i) = std::arg(upZvals[i]);
	}

}


void initialization(const Eigen::MatrixXd& triV, const Eigen::MatrixXi& triF, Eigen::MatrixXd& upsampledTriV, Eigen::MatrixXi& upsampledTriF)
{
	Eigen::SparseMatrix<double> S;
	std::vector<int> facemap;

	triMesh = MeshConnectivity(triF);

	loopTriV = triV;
	loopTriF = triF;

	std::vector<Eigen::Vector3d> pos;
	std::vector<std::vector<int>> faces;

	pos.resize(triV.rows());
	for (int i = 0; i < triV.rows(); i++)
	{
		pos[i] = triV.row(i);
	}

	faces.resize(triF.rows());
	for (int i = 0; i < triF.rows(); i++)
	{
		faces[i] = { triF(i, 0), triF(i, 1), triF(i, 2) };
	}

	secMesh.Populate(pos, faces);
	subSecMesh = secMesh;
}

double getZvalDiffNorm(const std::vector<std::complex<double>>& zvals0, const std::vector<std::complex<double>>& zvals1)
{
    if(zvals0.size() != zvals1.size())
    {
        std::cout << "size doesn't match" << std::endl;
        return 0;
    }
    else
    {
        double diffNorm = 0;
        for(int i = 0; i < zvals1.size(); i++)
        {
            diffNorm += std::norm(zvals1[i] - zvals0[i]);
        }
        if(diffNorm > 0)
            diffNorm = std::sqrt(diffNorm);
        return  diffNorm;
    }
}

double getZvalNorm(const std::vector<std::complex<double>>& zvals)
{
    double diffNorm = 0;
    for(int i = 0; i < zvals.size(); i++)
    {
        diffNorm += std::norm(zvals[i]);
    }
    if(diffNorm > 0)
        diffNorm = std::sqrt(diffNorm);
    return  diffNorm;
}


void updatePaintingItems()
{
	// get interploated amp and phase frames
	std::cout << "compute upsampled phase: " << std::endl;
	updateMagnitudePhase(initOmega, initZvals, loopModel0, loopedOmega0, loopedZvals0, loopedAmp0, loopedPhase0);
    updateMagnitudePhase(tarOmega, tarZvals, loopModel1, loopedOmega1, loopedZvals1, loopedAmp1, loopedPhase1);

    globalAmpMin = std::min(loopedAmp0.minCoeff(), loopedAmp1.minCoeff());
    globalAmpMax = std::min(loopedAmp0.maxCoeff(), loopedAmp1.maxCoeff());

    std::cout << " input omega difference: " << (tarOmega - initOmega).norm() <<  ",  input z difference: " << getZvalDiffNorm(initZvals, tarZvals) << std::endl;
    std::cout << "output omega differnece: " << (loopedOmega1 - loopedOmega0).norm() << ", output z difference: " << getZvalDiffNorm(loopedZvals0, loopedZvals1) << std::endl;
    std::cout << "output phase differnece: " << (loopedPhase1 - loopedPhase0).norm() << ", output amp difference: " << (loopedAmp1 - loopedAmp0).norm() << std::endl;
}

void registerMeshes(const std::string meshName, double shiftx, double shifty, const Eigen::VectorXd& amp, const Eigen::VectorXd& omega, const Eigen::VectorXd& loopedAmp, Eigen::VectorXd& loopedOmega, const Eigen::VectorXd& loopedPhi, const std::vector<std::complex<double>>& loopedZvals)
{
    polyscope::registerSurfaceMesh(meshName, triV, triF);
    polyscope::getSurfaceMesh(meshName)->translate(glm::vec3(shiftx, 3 * shifty, 0));
    polyscope::getSurfaceMesh(meshName)->addVertexScalarQuantity("amp color", amp);

    auto faceOmega = intrinsicEdgeVec2FaceVec(omega, triV, triMesh);
    polyscope::getSurfaceMesh(meshName)->addFaceVectorQuantity("vector field", faceOmega);
    Eigen::MatrixXd wrinkledTriV;

    std::vector<std::vector<int>> vertNeiEdges, vertNeiFaces;
    buildVertexNeighboringInfo(MeshConnectivity(loopTriF), loopTriV.rows(), vertNeiEdges, vertNeiFaces);
    getWrinkledMesh(loopTriV, loopTriF, loopedZvals, &vertNeiFaces, wrinkledTriV, wrinkleAmpScalingRatio, isUseTangentCorrection);

    Eigen::MatrixXd faceColors(loopTriF.rows(), 3);
    for (int i = 0; i < faceColors.rows(); i++)
    {
        faceColors.row(i) << 80 / 255.0, 122 / 255.0, 91 / 255.0;
    }

    polyscope::registerSurfaceMesh(meshName + " looped amp", loopTriV, loopTriF);
    polyscope::getSurfaceMesh(meshName + " looped amp")->translate(glm::vec3(shiftx, 2 * shifty, 0));
    auto ampColor = polyscope::getSurfaceMesh(meshName + " looped amp")->addVertexScalarQuantity("amp color", loopedAmp);
    ampColor->setMapRange(std::pair<double, double>(globalAmpMin, globalAmpMax));
    ampColor->setEnabled(true);

    polyscope::registerSurfaceMesh(meshName + " looped phase", loopTriV, loopTriF);
    polyscope::getSurfaceMesh(meshName + " looped phase")->translate(glm::vec3(shiftx, shifty, 0));

    mPaint.setNormalization(false);
    Eigen::MatrixXd phiColor = mPaint.paintPhi(loopedPhi);
    auto phaseColor = polyscope::getSurfaceMesh(meshName + " looped phase")->addVertexColorQuantity("phase color", phiColor);
    phaseColor->setEnabled(true);

    auto loopedFaceOmega = edgeVec2FaceVec(subSecMesh, loopedOmega);
    polyscope::getSurfaceMesh(meshName + " looped phase")->addFaceVectorQuantity("upsampled vector field", loopedFaceOmega);

    polyscope::registerSurfaceMesh(meshName + " looped wrinkles", wrinkledTriV, loopTriF);
    polyscope::getSurfaceMesh(meshName + " looped wrinkles")->setSurfaceColor({ 80 / 255.0, 122 / 255.0, 91 / 255.0 });
    polyscope::getSurfaceMesh(meshName + " looped wrinkles")->addFaceColorQuantity("wrinkled color", faceColors);
    polyscope::getSurfaceMesh(meshName + " looped wrinkles")->translate(glm::vec3(shiftx, 0, 0));
    polyscope::getSurfaceMesh(meshName + " looped wrinkles")->setEnabled(true);
}

void updateView()
{
	std::cout << "update viewer. " << std::endl;
    // update loop stuffs
    updatePaintingItems();
    double shiftx = 1.5 * (triV.col(0).maxCoeff() - triV.col(0).minCoeff());
    double shifty = 1.5 * (triV.col(1).maxCoeff() - triV.col(1).minCoeff());

    registerMeshes("init mesh", 0, shifty, initAmp, initOmega, loopedAmp0, loopedOmega0, loopedPhase0, loopedZvals0);
    registerMeshes("tar mesh", shiftx, shifty, tarAmp, tarOmega, loopedAmp1, loopedOmega1, loopedPhase1, loopedZvals1);
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
	loopLevel = jval["upsampled_times"];
    if(loopLevel > 2)
        loopLevel = 2;

	meshFile = workingFolder + meshFile;


	igl::readOBJ(meshFile, triV, triF);
	triMesh = MeshConnectivity(triF);
	initialization(triV, triF, loopTriV, loopTriF);

	int nedges = triMesh.nEdges();
	int nverts = triV.rows();

	std::string initAmpPath = "amp.txt";
	std::string initOmegaPath = jval["init_omega"];
	std::string initZValsPath = "zvals.txt";
	if (jval.contains(std::string_view{ "init_zvals" }))
	{
		initZValsPath = jval["init_zvals"];
	}
    if (jval.contains(std::string_view{ "init_amp" }))
    {
        initAmpPath = jval["init_amp"];
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

    bool loadTar = true;
    if (!loadEdgeOmega(workingFolder + tarOmegaPath, nedges, tarOmega)) {
        std::cout << "missing tar edge omega file." << std::endl;
        loadTar = false;
    }

    if (!loadVertexZvals(workingFolder + tarZValsPath, triV.rows(), tarZvals))
    {
        std::cout << "missing tar zval file, try to load amp file, and round zvals from amp and omega" << std::endl;
        if (!loadVertexAmp(workingFolder + tarAmpPath, triV.rows(), tarAmp))
        {
            std::cout << "missing tar amp file: " << std::endl;
            loadTar = false;
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
        for (int i = 0; i < tarAmp.size(); i++)
            tarAmp(i) = std::abs(tarZvals[i]);
    }

    if(loadTar)
        testType = EndPts;
    else
    {
        std::vector<std::complex<double>> updatedZvals(nverts);
        for(int i = 0; i < nverts; i++)
        {
            Eigen::Vector2d rv = Eigen::Vector2d::Random();
            updatedZvals[i] = std::complex<double>(rv[0], rv[1]);
        }

        Eigen::VectorXd updatedOmega(nedges), upOmega;
        updatedOmega.setRandom();

        tarOmega = initOmega + eps * updatedOmega;
        tarZvals.resize(nverts);

        for (int j = 0; j < nverts; j++)
        {
            tarZvals[j] = initZvals[j] + eps * updatedZvals[j];
        }

        tarAmp.setZero(triV.rows());
        for (int i = 0; i < tarAmp.size(); i++)
            tarAmp(i) = std::abs(tarZvals[i]);
    }
    updatePaintingItems();

	return true;
}


void callback() {
	ImGui::PushItemWidth(100);
	if (ImGui::Button("Load", ImVec2(-1, 0)))
	{
		loadProblem();
		updateView();
	}

	if (ImGui::InputInt("underline loop level", &loopLevel))
	{
		if (loopLevel < 0)
			loopLevel = 0;
	}

	if (ImGui::Button("updates", ImVec2(-1, 0)))
	{
		// solve for the path from source to target
		updatePaintingItems();
		updateView();
	}

    if (ImGui::Button("Loop continuous check", ImVec2(-1, 0)))
    {
        std::cout << "test continuity" << std::endl;
        std::shared_ptr<ComplexLoop> opt1, opt2;
        opt1 = std::make_shared<ComplexLoopZuenko>();
        opt2 = std::make_shared<ComplexLoopZuenko>();

        opt1->SetMesh(secMesh);
        opt1->setBndFixFlag(isFixBnd);

        int nverts = secMesh.GetVertCount();
        int nedges = secMesh.GetEdgeCount();

        std::vector<std::complex<double>> updatedZvals(nverts);

        double updatedZnorm = 0;
        for(int i = 0; i < nverts; i++)
        {
            updatedZvals[i] = tarZvals[i] - initZvals[i];
        }

        Eigen::VectorXd updatedOmega = tarOmega - initOmega, upOmega;
        double updateOmegaNorm = updatedOmega.norm();
        std::vector<std::complex<double>> upZvals;

        Eigen::VectorXd edgeVec = swapEdgeVec(triF, initOmega, 0);
        opt1->Subdivide(edgeVec, initZvals, upOmega, upZvals, loopLevel);

        Eigen::VectorXd upAmp, upPhi, upAmp1, upPhi1;
        upAmp.resize(loopTriV.rows());
        upPhi.resize(loopTriV.rows());
        for(int i = 0; i < upAmp.rows(); i++)
        {
            upAmp(i) = std::abs(upZvals[i]);
            upPhi(i) = std::arg(upZvals[i]);
        }

        for(int i = 0; i < 10; i++)
        {
            opt2->SetMesh(secMesh);
            opt2->setBndFixFlag(isFixBnd);
            
            std::vector<std::complex<double>> upZvals1;
            Eigen::VectorXd upOmega1;
            double eps = std::pow(0.1, i);
            Eigen::VectorXd newOmega = initOmega + eps * updatedOmega;
            std::vector<std::complex<double>> newZvals(nverts);

            for(int j = 0; j < nverts; j++)
            {
                newZvals[j] = initZvals[j] + eps * updatedZvals[j];
            }
            Eigen::VectorXd edgeVec1 = swapEdgeVec(triF, newOmega, 0);
            opt2->Subdivide(edgeVec1, newZvals, upOmega1, upZvals1, loopLevel);

            upAmp1.resize(loopTriV.rows());
            upPhi1.resize(loopTriV.rows());
            for(int j = 0; j < upAmp.rows(); j++)
            {
                upAmp1(j) = std::abs(upZvals1[j]);
                upPhi1(j) = std::arg(upZvals1[j]);
            }

            std::cout << "eps: " << eps << ", coarse omega update: " << eps * updateOmegaNorm << ", coarse z update: " << eps * getZvalNorm(updatedZvals) << std::endl;
            std::cout << "updated loop omega norm: " << (upOmega1 - upOmega).norm() << ", updated loop zval norm: " << getZvalDiffNorm(upZvals, upZvals1) << std::endl;
            std::cout << "updated loop amp norm: " << (upAmp1 - upAmp).norm() << ", updated loop phi norm: " << (upPhi1 - upPhi).norm() << std::endl;
        }

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