#include "polyscope/polyscope.h"
#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>
#include "polyscope/messages.h"
#include "polyscope/surface_mesh.h"
#include "polyscope/view.h"

#include <iostream>
#include <filesystem>
#include <utility>

#include "../../include/CommonTools.h"
#include "../../include/Visualization/PaintGeometry.h"


Eigen::MatrixXd upTriV, knoppelUpTriV;
Eigen::MatrixXi upTriF, knoppelUpTriF;

std::vector<Eigen::MatrixXd> wrinkleVList;
std::vector<Eigen::MatrixXd> wrinkleFList;

std::vector<Eigen::VectorXd> CWFPhaseList, KnoppelPhaseList, LinearPhaseList;
std::vector<Eigen::VectorXd> CWFAmpList, KnoppelAmpList, LinearAmpList;

enum DisplayType
{
    comparison = 0,
    CWFRes = 1
};
DisplayType displayType = comparison;
int curFrame = 0;
int nFrames = 10;
std::string loadPath = "";

void loadCVSFile(const std::string& filename, Eigen::VectorXd& vec, int nlines)
{
    vec.setZero(nlines);
    std::ifstream infile(filename);
    if (!infile)
    {
        std::cerr << "invalid edge omega file name" << std::endl;
        exit(EXIT_FAILURE);
    }
    else
    {
        for (int i = 0; i < nlines; i++)
        {
            std::string line;
            std::getline(infile, line);
            std::stringstream ss(line);

//            std::cout << ss.str() << std::endl;

            std::string x, y;
            ss >> x;
            ss >> y;
            vec(i) = std::stod(x);
        }
    }
}

void registerMesh(int frameId)
{
	double shiftx = 1.2 * (upTriV.col(0).maxCoeff() - upTriV.col(0).minCoeff());

    if(displayType == CWFRes)
    {
        PaintGeometry mPaint;
        polyscope::registerSurfaceMesh("phase mesh", upTriV, upTriF);
        mPaint.setNormalization(false);
        Eigen::MatrixXd phaseColor = mPaint.paintPhi(CWFPhaseList[frameId]);
        polyscope::getSurfaceMesh("phase mesh")->translate({ 0, 0, 0 });
        polyscope::getSurfaceMesh("phase mesh")->addVertexColorQuantity("vertex phi", phaseColor);
        polyscope::getSurfaceMesh("phase mesh")->setEnabled(true);

        polyscope::registerSurfaceMesh("amplitude mesh", upTriV, upTriF);
        polyscope::getSurfaceMesh("amplitude mesh")->translate({ shiftx, 0, 0 });
        auto ampPatterns = polyscope::getSurfaceMesh("amplitude mesh")->addVertexScalarQuantity("vertex amplitude", CWFAmpList[frameId]);
        ampPatterns->setMapRange(std::pair<double, double>(0, 1));
        polyscope::getSurfaceMesh("amplitude mesh")->setEnabled(true);

        // wrinkle mesh
        polyscope::registerSurfaceMesh("wrinkled mesh", wrinkleVList[frameId], upTriF);
        polyscope::getSurfaceMesh("wrinkled mesh")->setSurfaceColor({ 80 / 255.0, 122 / 255.0, 91 / 255.0 });
        polyscope::getSurfaceMesh("wrinkled mesh")->translate({ 2 * shiftx, 0, 0 });
        polyscope::getSurfaceMesh("wrinkled mesh")->setEnabled(true);
    }
    else
    {
        PaintGeometry mPaint;
        polyscope::registerSurfaceMesh("CWF phase mesh", upTriV, upTriF);
        mPaint.setNormalization(false);
        Eigen::MatrixXd phaseColor = mPaint.paintPhi(CWFPhaseList[frameId]);
        polyscope::getSurfaceMesh("CWF phase mesh")->translate({ 0, 0, 0 });
        polyscope::getSurfaceMesh("CWF phase mesh")->addVertexColorQuantity("vertex phi", phaseColor);
        polyscope::getSurfaceMesh("CWF phase mesh")->setEnabled(true);


        polyscope::registerSurfaceMesh("Linear phase mesh", upTriV, upTriF);
        mPaint.setNormalization(false);
        phaseColor = mPaint.paintPhi(LinearPhaseList[frameId]);
        polyscope::getSurfaceMesh("Linear phase mesh")->translate({ shiftx, 0, 0 });
        polyscope::getSurfaceMesh("Linear phase mesh")->addVertexColorQuantity("vertex phi", phaseColor);
        polyscope::getSurfaceMesh("Linear phase mesh")->setEnabled(true);

        polyscope::registerSurfaceMesh("Knoppel phase mesh", knoppelUpTriV, knoppelUpTriF);
        mPaint.setNormalization(false);
        phaseColor = mPaint.paintPhi(KnoppelPhaseList[frameId]);
        polyscope::getSurfaceMesh("Knoppel phase mesh")->translate({ 2 * shiftx, 0, 0 });
        polyscope::getSurfaceMesh("Knoppel phase mesh")->addVertexColorQuantity("vertex phi", phaseColor);
        polyscope::getSurfaceMesh("Knoppel phase mesh")->setEnabled(true);
    }
}

void updateFieldsInView(int frameId)
{
	std::cout << "update viewer. " << std::endl;
	registerMesh(frameId);
}


bool loadProblem(const std::string loadingPath, const std::string disType, int numFrames)
{
    std::cout << "load path: " << loadingPath << std::endl;
    if(disType == "comparison")
        displayType = comparison;
    else if (disType == "CWFRes")
        displayType = CWFRes;
    else
    {
        std::cerr << "invalid input type, valid types: comparison, CWFRes";
        return false;
    }

    nFrames = numFrames;
    curFrame = 0;
    loadPath = loadingPath;

    if(displayType == comparison)
    {
        std::string CWFFolder = loadingPath + "/CWF/render/";
        std::string knoppelFolder = loadingPath + "/Knoppel/render/";
        std::string linearFolder = loadingPath + "/Linear/render/";

        KnoppelPhaseList.resize(numFrames);
        LinearPhaseList.resize(numFrames);
        CWFPhaseList.resize(numFrames);

        igl::readOBJ(CWFFolder + "upmesh.obj", upTriV, upTriF);
        igl::readOBJ(knoppelFolder + "upmesh.obj", knoppelUpTriV, knoppelUpTriF);

        int nupverts = upTriV.rows();

//        for (uint32_t i = 0; i < numFrames; ++i)
//        {
//            loadCVSFile(CWFFolder + "upsampledPhase/upPhase" + std::to_string(i) + ".cvs", CWFPhaseList[i], nupverts);
//            loadCVSFile(knoppelFolder + "upsampledKnoppelPhase/upKnoppelPhase" + std::to_string(i) + ".cvs", KnoppelPhaseList[i], nupverts);
//            loadCVSFile(linearFolder + "upsampledPhase/upPhase" + std::to_string(i) + ".cvs", LinearPhaseList[i], nupverts);
//        }

        auto loadPerFrame = [&](const tbb::blocked_range<uint32_t>& range)
        {
            for (uint32_t i = range.begin(); i < range.end(); ++i)
            {
                loadCVSFile(CWFFolder + "upsampledPhase/upPhase" + std::to_string(i) + ".cvs", CWFPhaseList[i], nupverts);
                loadCVSFile(knoppelFolder + "upsampledPhase/upPhase" + std::to_string(i) + ".cvs", KnoppelPhaseList[i], nupverts);
                loadCVSFile(linearFolder + "upsampledPhase/upPhase" + std::to_string(i) + ".cvs", LinearPhaseList[i], nupverts);
            }
        };
        tbb::blocked_range<uint32_t> rangex(0u, (uint32_t)numFrames, GRAIN_SIZE);
        tbb::parallel_for(rangex, loadPerFrame);

    }
    else
    {
        std::string CWFFolder = loadingPath + "/CWF/render/";

        CWFPhaseList.resize(numFrames);
        CWFAmpList.resize(numFrames);
        wrinkleVList.resize(numFrames);
        wrinkleFList.resize(numFrames);

        igl::readOBJ(CWFFolder + "upmesh.obj", upTriV, upTriF);

        int nupverts = upTriV.rows();

        auto loadPerFrame = [&](const tbb::blocked_range<uint32_t>& range)
        {
            for (uint32_t i = range.begin(); i < range.end(); ++i)
            {
                loadCVSFile(CWFFolder + "upsampledPhase/upPhase" + std::to_string(i) + ".cvs", CWFPhaseList[i], nupverts);
                loadCVSFile(CWFFolder + "upsampledAmp/upAmp_" + std::to_string(i) + ".cvs", CWFAmpList[i], nupverts);
                igl::readOBJ(CWFFolder + "wrinkledMesh/wrinkledMesh_" + std::to_string(i) + ".obj", wrinkleVList[i], wrinkleFList[i]);
            }
        };
        tbb::blocked_range<uint32_t> rangex(0u, (uint32_t)numFrames, GRAIN_SIZE);
        tbb::parallel_for(rangex, loadPerFrame);
    }

	return true;
}

void callback() {
	ImGui::PushItemWidth(100);
	if (ImGui::CollapsingHeader("Frame Visualization Options", ImGuiTreeNodeFlags_DefaultOpen))
	{
		if (ImGui::SliderInt("current frame", &curFrame, 0, nFrames - 1))
		{
			curFrame = curFrame % nFrames;
			updateFieldsInView(curFrame);
		}
	}

	if (ImGui::Button("output images", ImVec2(-1, 0)))
	{
		std::cout << "save folder: " << loadPath << std::endl;
        std::string subFolder = "/polyimags/";
        mkdir(loadPath + subFolder);
        if(displayType == 0)
            subFolder += "comp";
        else
            subFolder += "CWFRes";
        mkdir(loadPath + subFolder);
		for (int i = 0; i < nFrames; i++)
		{
			updateFieldsInView(i);
			//polyscope::options::screenshotExtension = ".jpg";
			std::string name = loadPath + subFolder + "/output_" + std::to_string(i) + ".png";
			polyscope::screenshot(name);
		}
	}


	ImGui::PopItemWidth();
}


int main(int argc, char** argv)
{
    std::cout << argc << std::endl;
    if(argc != 4)
    {
        std::cerr << "usage:  ./loadSequece_bin [loading_path] [type (comparison or CWEFRes)] [num frames]" << std::endl;
        exit(EXIT_FAILURE);
    }
	if (!loadProblem(argv[1], argv[2], std::stoi(argv[3])))
	{
		std::cout << "failed to load file." << std::endl;
        exit(EXIT_FAILURE);
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

	updateFieldsInView(curFrame);
	// Show the gui
	polyscope::show();


	return 0;
}