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

std::vector<Eigen::MatrixXd> wrinkledVList;
std::vector<std::vector<std::complex<double>>> zList, upZList;
std::vector<Eigen::VectorXd> omegaList, ampList, upOmegaList, upPhiList, upAmpList;
std::vector<Eigen::MatrixXd> faceOmegaList;

std::vector<Eigen::VectorXd> sideVertexLinearPhiList;
std::vector<Eigen::VectorXd> ClouhTorcherPhiList;
std::vector<Eigen::VectorXd> sideVertexWojtanPhiList;
std::vector<Eigen::VectorXd> knoppelPhiList;
std::vector<Eigen::VectorXd> ZuenkoPhiList;

int upsamplingLevel = 2;
float wrinkleAmpScalingRatio = 1;
std::string workingFolder = "";
int numFrames = 20;
int curFrame = 0;

double globalAmpMin = 0;
double globalAmpMax = 1;
float vecratio = 0.01;
bool isUseV2 = false;
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

static void  getClouhTorcherUpsamplingPhi(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, const std::vector<Eigen::VectorXd>& edgeOmegaList, const std::vector<std::vector<std::complex<double>>>& zvalsList, Eigen::MatrixXd& upV, Eigen::MatrixXi& upF, std::vector<Eigen::VectorXd>& upPhiList, int upLevel)
{
	int nframes = edgeOmegaList.size();
	Eigen::SparseMatrix<double> mat;
	std::vector<int> facemap;
	std::vector<std::pair<int, Eigen::Vector3d>> bary;


	meshUpSampling(V, F, upV, upF, upLevel, &mat, &facemap, &bary);
	upPhiList.resize(nframes);

	MeshConnectivity mesh(F);
	MeshConnectivity upMesh;

	auto frameUpsampling = [&](const tbb::blocked_range<uint32_t>& range)
	{
		for (uint32_t i = range.begin(); i < range.end(); ++i)
		{
			getClouhTocherPhi(V, mesh, edgeOmegaList[i], zvalsList[i], bary, upPhiList[i]);
		}
	};

	tbb::blocked_range<uint32_t> rangex(0u, (uint32_t)nframes);
	tbb::parallel_for(rangex, frameUpsampling);
}

static void getSideVertexUpsamplingPhi(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, const std::vector<Eigen::VectorXd>& edgeOmegaList, const std::vector<std::vector<std::complex<double>>>& zvalsList, Eigen::MatrixXd& upV, Eigen::MatrixXi& upF, std::vector<Eigen::VectorXd>& upPhiListLinear, std::vector<Eigen::VectorXd>& upPhiListWojtan, int upLevel)
{
	int nframes = edgeOmegaList.size();
	Eigen::SparseMatrix<double> mat;
	std::vector<int> facemap;
	std::vector<std::pair<int, Eigen::Vector3d>> bary;


	meshUpSampling(V, F, upV, upF, upLevel, &mat, &facemap, &bary);
	upPhiListLinear.resize(nframes);
	upPhiListWojtan.resize(nframes);

	MeshConnectivity mesh(F);
	MeshConnectivity upMesh;

	auto frameUpsampling = [&](const tbb::blocked_range<uint32_t>& range)
	{
		for (uint32_t i = range.begin(); i < range.end(); ++i)
        {
			getSideVertexPhi(V, mesh, edgeOmegaList[i], zvalsList[i], bary, upPhiListLinear[i], 0);
			//getSideVertexPhi(V, mesh, edgeOmegaList[i], zvalsList[i], bary, upPhiListCubic[i], 1);
			getSideVertexPhi(V, mesh, edgeOmegaList[i], zvalsList[i], bary, upPhiListWojtan[i], 2);
		}
	};

	tbb::blocked_range<uint32_t> rangex(0u, (uint32_t)nframes);
	tbb::parallel_for(rangex, frameUpsampling);
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

static void getZuenkoUpsamplingPhi(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, const std::vector<Eigen::VectorXd>& edgeOmegaList, const std::vector<std::vector<std::complex<double>>>& zvalsList, Eigen::MatrixXd& upV, Eigen::MatrixXi& upF, std::vector<Eigen::VectorXd>& upPhiList, int upLevel)
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
		for (uint32_t n = range.begin(); n < range.end(); ++n)
		{
			int nupverts = upV.rows();
			upPhiList[n].resize(nupverts);

			for (uint32_t i = 0; i < nupverts; ++i)
			{
				int fid = bary[i].first;
				std::vector<std::complex<double>> vertzvals(3);
				Eigen::Vector3d edgews;
				for (int j = 0; j < 3; j++)
				{
					int vid = mesh.faceVertex(fid, j);
					int eid = mesh.faceEdge(fid, j);

					vertzvals[j] = zvalsList[n][vid];
					edgews(j) = edgeOmegaList[n](eid); // defined as mesh.edgeVertex(eid, 1) - mesh.edgeVertex(eid, 0)

					if (mesh.edgeVertex(eid, 1) == mesh.faceVertex(fid, (j + 1) % 3))
						edgews(j) *= -1;
				}

				std::complex<double> zval = IntrinsicFormula::getZvalsFromEdgeOmega(bary[i].second, vertzvals, edgews);
				upPhiList[n][i] = std::arg(zval);
			}
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
	getKnoppelUpsamplingPhi(triV, triF, omegaList, zList, upsampledTriV, upsampledTriF, knoppelPhiList, upsamplingLevel);
	getSideVertexUpsamplingPhi(triV, triF, omegaList, zList, upsampledTriV, upsampledTriF, sideVertexLinearPhiList, sideVertexWojtanPhiList, upsamplingLevel);
	getClouhTorcherUpsamplingPhi(triV, triF, omegaList, zList, upsampledTriV, upsampledTriF, ClouhTorcherPhiList, upsamplingLevel);
	getZuenkoUpsamplingPhi(triV, triF, omegaList, zList, upsampledTriV, upsampledTriF, ZuenkoPhiList, upsamplingLevel);

	upPhiList.resize(upZList.size());
	upAmpList.resize(upZList.size());
	faceOmegaList.resize(upZList.size());
	ampList.resize(upZList.size());

	parseSecMesh(upSecMesh, loopTriV, loopTriF);

	/*for (uint32_t i = 0; i < upZList.size(); ++i)
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
			globalAmpMin = std::min(ampList[i][j], globalAmpMin);
			globalAmpMax = std::max(ampList[i][j], globalAmpMax);
		}
	}*/

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
				globalAmpMin = std::min(ampList[i][j], globalAmpMin);
				globalAmpMax = std::max(ampList[i][j], globalAmpMax);
			}
		}
	};

	tbb::blocked_range<uint32_t> rangex(0u, (uint32_t)upZList.size());
	tbb::parallel_for(rangex, computeOursForVis);

	std::cout << "compute wrinkle meshes: " << std::endl;
	updateWrinkles(loopTriV, loopTriF, upZList, wrinkledVList, wrinkleAmpScalingRatio, isUseV2);
}

bool isFirstVis = true;
static void updateView(int frameId)
{
	double shiftx = 1.5 * (triV.col(0).maxCoeff() - triV.col(0).minCoeff());
	int n = 0;
	if (isFirstVis)
	{
		auto baseSurf = polyscope::registerSurfaceMesh("base mesh", triV, triF);
	}
   
    auto freqFields = polyscope::getSurfaceMesh("base mesh")->addFaceVectorQuantity("frequency field", vecratio * faceOmegaList[frameId], polyscope::VectorType::AMBIENT);
    auto initAmp = polyscope::getSurfaceMesh("base mesh")->addVertexScalarQuantity("amplitude", ampList[frameId]);
    initAmp->setMapRange(std::pair<double, double>(globalAmpMin, globalAmpMax));
    n++;

	if (isShowEveryThing)
	{
		// wrinkle mesh
		polyscope::registerSurfaceMesh("wrinkled mesh", wrinkledVList[frameId], loopTriF);
		polyscope::getSurfaceMesh("wrinkled mesh")->setSurfaceColor({ 80 / 255.0, 122 / 255.0, 91 / 255.0 });
		polyscope::getSurfaceMesh("wrinkled mesh")->translate({ n * shiftx, 0, 0 });
		n++;


		// amp pattern
		polyscope::registerSurfaceMesh("upsampled ampliude mesh", loopTriV, loopTriF);
		polyscope::getSurfaceMesh("upsampled ampliude mesh")->translate({ n * shiftx, 0, 0 });
		auto ampPatterns = polyscope::getSurfaceMesh("upsampled ampliude mesh")->addVertexScalarQuantity("vertex amplitude", upAmpList[frameId]);
		ampPatterns->setMapRange(std::pair<double, double>(globalAmpMin, globalAmpMax));
		ampPatterns->setEnabled(true);
		n++;
	}
	

	// ours phase pattern
	mPaint.setNormalization(false);
	if (isFirstVis)
	{
		polyscope::registerSurfaceMesh("upsampled phase mesh", loopTriV, loopTriF);
		polyscope::getSurfaceMesh("upsampled phase mesh")->translate({ n * shiftx, 0, 0 });
	}
		
	Eigen::MatrixXd phaseColor = mPaint.paintPhi(upPhiList[frameId]);
	auto ourPhasePatterns = polyscope::getSurfaceMesh("upsampled phase mesh")->addVertexColorQuantity("vertex phi", phaseColor);
	ourPhasePatterns->setEnabled(true);
	n++;

	// knoppel pahse pattern
	polyscope::registerSurfaceMesh("knoppel phase mesh", upsampledTriV, upsampledTriF);
	polyscope::getSurfaceMesh("knoppel phase mesh")->translate({ n * shiftx, 0, 0 });
	phaseColor = mPaint.paintPhi(knoppelPhiList[frameId]);
	auto knoppelPhasePatterns = polyscope::getSurfaceMesh("knoppel phase mesh")->addVertexColorQuantity("vertex phi", phaseColor);
	knoppelPhasePatterns->setEnabled(true);
	n++;


	// linear side vertex pahse pattern
	if (isFirstVis)
	{
		auto linearSideSurf = polyscope::registerSurfaceMesh("Linear-side phase mesh", upsampledTriV, upsampledTriF);
		linearSideSurf->setSmoothShade(true);
		polyscope::getSurfaceMesh("Linear-side phase mesh")->translate({ n * shiftx, 0, 0 });
	}
	phaseColor = mPaint.paintPhi(sideVertexLinearPhiList[frameId]);
	auto linearPhasePatterns = polyscope::getSurfaceMesh("Linear-side phase mesh")->addVertexColorQuantity("vertex phi", phaseColor);
	linearPhasePatterns->setEnabled(true);
	n++;


	// clouh torcher vertex pahse pattern
	if (isFirstVis)
	{
		auto ClouhTocherSurf = polyscope::registerSurfaceMesh("Clouh-Torcher phase mesh", upsampledTriV, upsampledTriF);
		ClouhTocherSurf->setSmoothShade(true);
		polyscope::getSurfaceMesh("Clouh-Torcher phase mesh")->translate({ n * shiftx, 0, 0 });
	}
	phaseColor = mPaint.paintPhi(ClouhTorcherPhiList[frameId]);
	auto cubicPhasePatterns = polyscope::getSurfaceMesh("Clouh-Torcher phase mesh")->addVertexColorQuantity("vertex phi", phaseColor);
	cubicPhasePatterns->setEnabled(true);
	n++;


	// wojtan side vertex pahse pattern
	if (isFirstVis)
	{
		auto WojtanSurf = polyscope::registerSurfaceMesh("Wojtan-side phase mesh", upsampledTriV, upsampledTriF);
		WojtanSurf->setSmoothShade(true);
		polyscope::getSurfaceMesh("Wojtan-side phase mesh")->translate({ n * shiftx, 0, 0 });
	}
	phaseColor = mPaint.paintPhi(sideVertexWojtanPhiList[frameId]);
	auto wojtanPhasePatterns = polyscope::getSurfaceMesh("Wojtan-side phase mesh")->addVertexColorQuantity("vertex phi", phaseColor);
	wojtanPhasePatterns->setEnabled(true);
	n++;

	// zuenko
	if (isFirstVis)
	{
		auto zuenkoSurf = polyscope::registerSurfaceMesh("zuenko phase mesh", upsampledTriV, upsampledTriF);
		zuenkoSurf->setSmoothShade(true);
		polyscope::getSurfaceMesh("zuenko phase mesh")->translate({ n * shiftx, 0, 0 });
	}
	phaseColor = mPaint.paintPhi(ZuenkoPhiList[frameId]);
	auto zuenkoPhasePatterns = polyscope::getSurfaceMesh("zuenko phase mesh")->addVertexColorQuantity("vertex phi", phaseColor);
	zuenkoPhasePatterns->setEnabled(true);
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
	isFirstVis = true;

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
	if (ImGui::Checkbox("Show wrinkles and Amp", &isShowEveryThing)) {}

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