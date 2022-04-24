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
#include "../../include/IntrinsicFormula/WrinkleEditingStaticEdgeModel.h"
#include "../../include/IntrinsicFormula/KnoppelStripePatternEdgeOmega.h"
#include "../../include/WrinkleFieldsEditor.h"
#include "../../include/SpherigonSmoothing.h"
#include "../../dep/SecStencils/types.h"
#include "../../dep/SecStencils/Subd.h"
#include "../../dep/SecStencils/utils.h"

#include "../../include/json.hpp"
#include "../../include/ComplexLoopNew.h"
#include "../../include/LoadSaveIO.h"
#include "../../include/SecMeshParsing.h"
#include "../../include/MeshLib/RegionEdition.h"

long long int selectFace(polyscope::SurfaceMesh* mesh) 
{
	using namespace polyscope;
	// Make sure we can see edges
	float oldEdgeWidth = mesh->getEdgeWidth();
	mesh->setEdgeWidth(1.);
	mesh->setEnabled(true);

	long long int returnFaceInd = -1;

	// Register the callback which creates the UI and does the hard work
	auto focusedPopupUI = [&]() {
		{ // Create a window with instruction and a close button.
			static bool showWindow = true;
			ImGui::SetNextWindowSize(ImVec2(300, 0), ImGuiCond_Once);
			ImGui::Begin("Select face", &showWindow);

			ImGui::PushItemWidth(300);
			ImGui::TextUnformatted("Hold ctrl and left-click to select a face");
			ImGui::Separator();

			// Pick by number
			ImGui::PushItemWidth(300);
			static int iF = -1;
			ImGui::InputInt("index", &iF);
			if (ImGui::Button("Select by index"))
			{
				if (iF >= 0 && (size_t)iF < mesh->nFaces()) 
				{
					returnFaceInd = iF;
					popContext();
				}
			}
			ImGui::PopItemWidth();

			ImGui::Separator();
			if (ImGui::Button("Abort")) {
				popContext();
			}
		}

		ImGuiIO& io = ImGui::GetIO();
		if (io.KeyCtrl && !io.WantCaptureMouse && ImGui::IsMouseClicked(0)) {

			ImGuiIO& io = ImGui::GetIO();

			// API is a giant mess..
			size_t pickInd;
			ImVec2 p = ImGui::GetMousePos();
			std::pair<Structure*, size_t> pickVal =
				pick::evaluatePickQuery(io.DisplayFramebufferScale.x * p.x, io.DisplayFramebufferScale.y * p.y);

			if (pickVal.first == mesh) {

				if (pickVal.second >= mesh->nVertices() && pickVal.second < mesh->nVertices() + mesh->nFaces()) 
				{
					returnFaceInd = pickVal.second - mesh->nVertices();
					popContext();
				}
			}
		}
	};


	// Pass control to the context we just created
	pushContext(focusedPopupUI);

	mesh->setEdgeWidth(oldEdgeWidth); // restore edge setting

	return returnFaceInd;
}

std::vector<VertexOpInfo> vertOpts;

Eigen::MatrixXd triV, upsampledTriV;
Eigen::MatrixXi triF, upsampledTriF;
MeshConnectivity triMesh;
Mesh secMesh, subSecMesh;

// initial information
Eigen::VectorXd initAmp;
Eigen::VectorXd initOmega;
std::vector<std::complex<double>> initZvals;

// base mesh information list
std::vector<Eigen::VectorXd> omegaList;
std::vector<Eigen::MatrixXd> faceOmegaList;
std::vector<std::vector<std::complex<double>>> zList;


// upsampled informations
std::vector<Eigen::VectorXd> subOmegaList;
std::vector<Eigen::MatrixXd> subFaceOmegaList;

std::vector<Eigen::VectorXd> phaseFieldsList;
std::vector<Eigen::VectorXd> ampFieldsList;
std::vector<std::vector<std::complex<double>>> upZList;
std::vector<Eigen::MatrixXd> wrinkledVList;

// reference amp and omega
std::vector<Eigen::VectorXd> refOmegaList;
std::vector<Eigen::VectorXd> refAmpList;

// region edition
RegionEdition regEdt;

int upsampleTimes = 2;

bool isForceOptimize = false;
bool isShowVectorFields = true;
bool isShowWrinkels = true;

PaintGeometry mPaint;

int numFrames = 20;
int curFrame = 0;

double globalAmpMax = 1;
double globalAmpMin = 0;

double dragSpeed = 0.5;

float vecratio = 0.001;

double gradTol = 1e-6;
double xTol = 0;
double fTol = 0;
int numIter = 1000;
int quadOrder = 4;
float wrinkleAmpScalingRatio = 1;

double spatialAmpRatio = 1;
double spatialEdgeRatio = 1;
double spatialKnoppelRatio = 1;

std::string workingFolder;

IntrinsicFormula::WrinkleEditingStaticEdgeModel editModel;

struct PickedFace
{
	int fid = -1;
	double ampChangeRatio = 1.;
	int effectiveRadius = 5;
	VecMotionType freqVecMotion = Enlarge;
	double freqVecChangeValue = 1.;
	bool isFreqAmpCoupled = false;

	std::vector<int> effectiveFaces = {};

	void buildEffectiveFaces(int nfaces)
	{
		if (fid == -1 || fid >= nfaces)
		{
			effectiveFaces = {};
			return;
		}
		else
		{
			Eigen::VectorXi curFaceFlags = Eigen::VectorXi::Zero(triF.rows());
			curFaceFlags(fid) = 1;
			Eigen::VectorXi curFaceFlagsNew = curFaceFlags;
			regEdt.faceDilation(curFaceFlagsNew, curFaceFlags, effectiveRadius);

			for (int i = 0; i < curFaceFlags.rows(); i++)
				if (curFaceFlags(i))
					effectiveFaces.push_back(i);
		}
	}
};

bool isSelectAll = false;
std::vector<PickedFace> pickFaces;
VecMotionType selectedMotion = Enlarge;

double selectedMotionValue = 2;
double selectedMagValue = 1;
bool isCoupled = false;

Eigen::VectorXi initSelectedFids;
Eigen::VectorXi selectedFids;
Eigen::VectorXi faceFlags;

int optTimes = 5;

bool isLoadOpt;

int clickedFid = -1;
int dilationTimes = 10;

bool isUseV2 = false;
bool isWarmStart = false;

// smoothing
int smoothingTimes = 3;
double smoothingRatio = 0.95;

void buildWrinkleMotions()
{
	int nverts = triV.rows();
	Eigen::VectorXi initSelectedVids;

	faceFlags2VertFlags(triMesh, nverts, initSelectedFids, initSelectedVids);

	int nselectedV = 0;
	for (int i = 0; i < nverts; i++)
		if (initSelectedVids(i))
			nselectedV++;
	std::cout << "num of selected vertices: " << nselectedV << std::endl;

	vertOpts.clear();
	vertOpts.resize(nverts, { None, isCoupled, 0, 1 });

	for (int i = 0; i < nverts; i++)
	{
		if (initSelectedVids(i))
			vertOpts[i] = { selectedMotion, isCoupled, selectedMotionValue, selectedMagValue };
	}

}

bool addSelectedFaces(const PickedFace face, Eigen::VectorXi& curFaceFlags)
{
	for (auto& f : face.effectiveFaces)
		if (curFaceFlags(f))
			return false;

	Eigen::VectorXi tmp = curFaceFlags;

	for (auto& f : face.effectiveFaces)
		curFaceFlags(f) = 1;


	return true;
}

void deleteSelectedFaces(const PickedFace face, Eigen::VectorXi& curFaceFlags)
{
	for (auto& f : face.effectiveFaces)
	{
		curFaceFlags(f) = 0;
	}
}

void updateSelectedRegionSetViz()
{
	regEdt.faceDilation(initSelectedFids, selectedFids, optTimes);

	int nfaces = triF.rows();
	Eigen::MatrixXd renderColor(triF.rows(), 3);
	renderColor.col(0).setConstant(1.0);
	renderColor.col(1).setConstant(1.0);
	renderColor.col(2).setConstant(1.0);

	for (int i = 0; i < nfaces; i++)
	{
		if (initSelectedFids(i) == 1)
			renderColor.row(i) << 1.0, 0, 0;
		else if (selectedFids(i) == 1)
			renderColor.row(i) << 0, 1.0, 0;
	}
	polyscope::getSurfaceMesh("base mesh")->addFaceColorQuantity("selected region", renderColor);
}

void addPickedFace(size_t ind)
{
	// Make sure not already used
	for (PickedFace& s : pickFaces)
	{
		if (s.fid == ind)
		{
			std::stringstream ss;
			ss << "Face " << ind;
			std::string vStr = ss.str();
			polyscope::warning("Face " + vStr + " is already picked");
			return;
		}
	}

	PickedFace newface;
	newface.fid = ind;

	newface.buildEffectiveFaces(triF.rows());
	std::cout << "num of new effective faces: " << newface.effectiveFaces.size() << ", dilation radius: " << newface.effectiveRadius << std::endl;


	if (addSelectedFaces(newface, initSelectedFids))
		pickFaces.push_back(newface);
	else
	{
		std::stringstream ss;
		ss << "Face " << ind;
		std::string vStr = ss.str();
		polyscope::warning("Face " + vStr + " is inside the effective domain");
		return;
	}
	updateSelectedRegionSetViz();
}

void buildFacesMenu()
{
	if (isSelectAll)
		return;

	bool anyChanged = false;

	ImGui::PushItemWidth(200);

	int id = 0;
	int eraseInd = -1;
	for (PickedFace& s : pickFaces)
	{
		std::stringstream ss;
		ss << "Face " << s.fid;
		std::string vStr = ss.str();
		ImGui::PushID(vStr.c_str());

		ImGui::TextUnformatted(vStr.c_str());

		ImGui::SameLine();
		if (ImGui::Button("delete"))
		{
			eraseInd = id;
			anyChanged = true;
		}
		ImGui::Indent();

		//int backupRadius = s.effectiveRadius;

		if (ImGui::InputInt("effective radius", &s.effectiveRadius))
		{
			anyChanged = true;
			/*deleteSelectedFaces(s, initSelectedFids);
			s.buildEffectiveFaces(triF.rows());
			if (!addSelectedFaces(s, initSelectedFids))
			{
				std::cout << "due to the overlap, failed to extend the effective radius, back to last effective one: " << backupRadius << std::endl;
				s.effectiveRadius = backupRadius;
				s.buildEffectiveFaces(triF.rows());
				assert(addSelectedFaces(s, initSelectedFids));
			}*/
		}
		ImGui::Combo("freq motion", (int*)&s.freqVecMotion, "Ratate\0Tilt\0Enlarge\0None\0");
		if (ImGui::InputDouble("freq change", &s.freqVecChangeValue)) anyChanged = true;
		if (ImGui::InputDouble("amp change", &s.ampChangeRatio)) anyChanged = true;
		if (ImGui::Checkbox("amp freq coupled", &s.isFreqAmpCoupled)) anyChanged = true;

		ImGui::Unindent();
		ImGui::PopID();
	}
	ImGui::PopItemWidth();

	// actually do erase, if requested
	if (eraseInd != -1)
	{
		deleteSelectedFaces(pickFaces[eraseInd], initSelectedFids);
		pickFaces.erase(pickFaces.begin() + eraseInd);
	}

	if (ImGui::Button("add face"))
	{
		long long int pickId = selectFace(polyscope::getSurfaceMesh("base mesh"));
		//int nverts = polyscope::getSurfaceMesh("base mesh")->nVertices();
		int nfaces = polyscope::getSurfaceMesh("base mesh")->nFaces();

		if (id >= 0 && id < nfaces)
		{
			addPickedFace(pickId);
			anyChanged = true;
		}
	}

	if (anyChanged)
	{
		updateSelectedRegionSetViz();
	}
}


void updateMagnitudePhase(const std::vector<Eigen::VectorXd>& wFrames, const			std::vector<std::vector<std::complex<double>>>& zFrames, 
	std::vector<Eigen::VectorXd>& magList, 
	std::vector<Eigen::VectorXd>& phaseList,
	std::vector<std::vector<std::complex<double>>>& upZFrames)
{
	magList.resize(wFrames.size());
	phaseList.resize(wFrames.size());
	upZFrames.resize(wFrames.size());

	subOmegaList.resize(wFrames.size());
	subFaceOmegaList.resize(wFrames.size());

	MeshConnectivity mesh(triF);


	auto computeMagPhase = [&](const tbb::blocked_range<uint32_t>& range)
	{
		for (uint32_t i = range.begin(); i < range.end(); ++i)
		{
			Mesh tmpMesh;
			Eigen::VectorXd edgeVec = swapEdgeVec(triF, wFrames[i], 0);
			SubdivideNew(secMesh, edgeVec, zFrames[i], subOmegaList[i], upZFrames[i], upsampleTimes, tmpMesh);

			subFaceOmegaList[i] = edgeVec2FaceVec(tmpMesh, subOmegaList[i]);

			magList[i].setZero(upZFrames[i].size());
			phaseList[i].setZero(upZFrames[i].size());

			for (int j = 0; j < magList[i].size(); j++)
			{
				magList[i](j) = std::abs(upZFrames[i][j]);
				phaseList[i](j) = std::arg(upZFrames[i][j]);
			}
		}
	};

	tbb::blocked_range<uint32_t> rangex(0u, (uint32_t)upZFrames.size(), GRAIN_SIZE);
	tbb::parallel_for(rangex, computeMagPhase);

}


void updateWrinkles(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, const std::vector<std::vector<std::complex<double>>>& zFrames, std::vector<Eigen::MatrixXd>& wrinkledVFrames, double scaleRatio, bool isUseV2)
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

	tbb::blocked_range<uint32_t> rangex(0u, (uint32_t)interpZList.size(), GRAIN_SIZE);
	tbb::parallel_for(rangex, computeWrinkles);


}

void initialization(const Eigen::MatrixXd& triV, const Eigen::MatrixXi& triF, Eigen::MatrixXd& upsampledTriV, Eigen::MatrixXi& upsampledTriF)
{
	triMesh = MeshConnectivity(triF);

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

	Subdivide(subSecMesh, upsampleTimes);
	subSecMesh.GetPos(upsampledTriV);

	upsampledTriF.resize(subSecMesh.GetFaceCount(), 3);
	for (int i = 0; i < upsampledTriF.rows(); i++)
	{
		upsampledTriF.row(i) << subSecMesh.GetFaceVerts(i)[0], subSecMesh.GetFaceVerts(i)[1], subSecMesh.GetFaceVerts(i)[2];
	}


	//meshUpSampling(triV, triF, upsampledTriV, upsampledTriF, upsampleTimes, NULL, NULL, &bary);


	selectedFids.setZero(triMesh.nFaces());
	initSelectedFids = selectedFids;
	regEdt = RegionEdition(triMesh, triV.rows());

}


void updateEditionDomain()
{
	selectedFids = initSelectedFids;

	int nselected0 = 0;
	for (int i = 0; i < initSelectedFids.rows(); i++)
	{
		if (initSelectedFids(i) == 1)
		{
			nselected0++;
		}
	}

	for (int i = 0; i < optTimes; i++)
	{
		std::cout << "dilation option to get interface, step: " << i << std::endl;
		Eigen::VectorXi selectedFidNew;
		regEdt.faceDilation(selectedFids, selectedFidNew);

		selectedFids = selectedFidNew;
	}
	faceFlags = initSelectedFids - selectedFids;

	int nselected = 0;
	for (int i = 0; i < selectedFids.rows(); i++)
	{
		if (selectedFids(i) == 1)
		{
			nselected++;
		}
	}

	int ninterfaces = 0;
	for (int i = 0; i < faceFlags.rows(); i++)
	{
		if (faceFlags(i) == -1)
			ninterfaces++;
	}
	std::cout << "initial selected faces: " << nselected0 << ", selected faces: " << nselected << ", num of interfaces: " << nselected - nselected0 << " " << ninterfaces << std::endl;

	std::cout << "build wrinkle motions. " << std::endl;
	buildWrinkleMotions();

}

void updatePaintingItems()
{
	// get interploated amp and phase frames
	std::cout << "compute upsampled phase: " << std::endl;
	updateMagnitudePhase(omegaList, zList, ampFieldsList, phaseFieldsList, upZList);

	std::cout << "compute wrinkle meshes: " << std::endl;
	updateWrinkles(upsampledTriV, upsampledTriF, upZList, wrinkledVList, wrinkleAmpScalingRatio, isUseV2);


	std::cout << "compute face vector fields:" << std::endl;
	faceOmegaList.resize(omegaList.size());
	for (int i = 0; i < omegaList.size(); i++)
	{
		faceOmegaList[i] = intrinsicEdgeVec2FaceVec(omegaList[i], triV, triMesh);
	}


	// update global maximum amplitude
	std::cout << "update max and min amp. " << std::endl;

	globalAmpMax = std::max(ampFieldsList[0].maxCoeff(), refAmpList[0].maxCoeff());
	globalAmpMin = std::min(ampFieldsList[0].minCoeff(), refAmpList[0].minCoeff());
	for (int i = 1; i < ampFieldsList.size(); i++)
	{
		globalAmpMax = std::max(globalAmpMax, std::max(ampFieldsList[i].maxCoeff(), refAmpList[i].maxCoeff()));
		globalAmpMin = std::min(globalAmpMin, std::min(ampFieldsList[i].minCoeff(), refAmpList[i].minCoeff()));
	}

	std::cout << "start to update viewer." << std::endl;
}

void solveKeyFrames(const Eigen::VectorXd& initAmp, const Eigen::VectorXd& initOmega, const Eigen::VectorXi& faceFlags, std::vector<Eigen::VectorXd>& wFrames, std::vector<std::vector<std::complex<double>>>& zFrames)
{
	editModel = IntrinsicFormula::WrinkleEditingStaticEdgeModel(triV, triMesh, vertOpts, faceFlags, quadOrder, spatialAmpRatio, spatialEdgeRatio, spatialKnoppelRatio);

	editModel.initialization(initZvals, initOmega, numFrames - 2);

	std::cout << "initilization finished!" << std::endl;
	Eigen::VectorXd x;
	std::cout << "convert list to variable." << std::endl;
	editModel.convertList2Variable(x);

	refOmegaList = editModel.getRefWList();
	refAmpList = editModel.getRefAmpList();


	if (isForceOptimize)
	{
		if (isWarmStart)
			editModel.warmstart();

		else
		{

			auto funVal = [&](const Eigen::VectorXd& x, Eigen::VectorXd* grad, Eigen::SparseMatrix<double>* hess, bool isProj) {
				Eigen::VectorXd deriv;
				Eigen::SparseMatrix<double> H;
				double E = editModel.computeEnergy(x, grad ? &deriv : NULL, hess ? &H : NULL, isProj);

				if (grad)
				{
					(*grad) = deriv;
				}

				if (hess)
				{
					(*hess) = H;
				}

				return E;
			};
			auto maxStep = [&](const Eigen::VectorXd& x, const Eigen::VectorXd& dir) {
				return 1.0;
			};

			auto getVecNorm = [&](const Eigen::VectorXd& x, double& znorm, double& wnorm) {
				editModel.getComponentNorm(x, znorm, wnorm);
			};



			OptSolver::testFuncGradHessian(funVal, x);

			auto x0 = x;
			OptSolver::newtonSolver(funVal, maxStep, x, numIter, gradTol, xTol, fTol, true, getVecNorm, &workingFolder);
			std::cout << "before optimization: " << x0.norm() << ", after optimization: " << x.norm() << std::endl;
		}
	}
	std::cout << "convert variable to list." << std::endl;
	editModel.convertVariable2List(x);
	std::cout << "get w list" << std::endl;
	wFrames = editModel.getWList();
	std::cout << "get z list" << std::endl;
	zFrames = editModel.getVertValsList();
}

void registerMesh(int frameId)
{
	int nverts = triV.rows();
	double shiftx = 1.5 * (triV.col(0).maxCoeff() - triV.col(0).minCoeff());
	polyscope::registerSurfaceMesh("base mesh", triV, triF);
	polyscope::getSurfaceMesh("base mesh")->addFaceVectorQuantity("frequency field", faceOmegaList[frameId]);
	updateSelectedRegionSetViz();


	Eigen::MatrixXd refFaceOmega = intrinsicEdgeVec2FaceVec(refOmegaList[frameId], triV, triMesh);
	polyscope::registerSurfaceMesh("reference mesh", triV, triF);
	polyscope::getSurfaceMesh("reference mesh")->translate({ shiftx, 0, 0 });
	polyscope::getSurfaceMesh("reference mesh")->addVertexScalarQuantity("reference amplitude", refAmpList[frameId]);
	polyscope::getSurfaceMesh("reference mesh")->addFaceVectorQuantity("reference frequency field", refFaceOmega);

	// phase pattern
	polyscope::registerSurfaceMesh("phase mesh", upsampledTriV, upsampledTriF);
	mPaint.setNormalization(false);
	Eigen::MatrixXd phaseColor = mPaint.paintPhi(phaseFieldsList[frameId]);
	polyscope::getSurfaceMesh("phase mesh")->translate({ 2 * shiftx, 0, 0 });
	polyscope::getSurfaceMesh("phase mesh")->addVertexColorQuantity("vertex phi", phaseColor);

	// amp pattern
	polyscope::registerSurfaceMesh("upsampled ampliude and frequency mesh", upsampledTriV, upsampledTriF);
	polyscope::getSurfaceMesh("upsampled ampliude and frequency mesh")->translate({ 3 * shiftx, 0, 0 });
	polyscope::getSurfaceMesh("upsampled ampliude and frequency mesh")->addVertexScalarQuantity("vertex amplitude", ampFieldsList[frameId]);
	polyscope::getSurfaceMesh("upsampled ampliude and frequency mesh")->addFaceVectorQuantity("subdivided frequency field", subFaceOmegaList[frameId]);

	// wrinkle mesh
	polyscope::registerSurfaceMesh("wrinkled mesh", wrinkledVList[frameId], upsampledTriF);
	polyscope::getSurfaceMesh("wrinkled mesh")->setSurfaceColor({ 80 / 255.0, 122 / 255.0, 91 / 255.0 });
	polyscope::getSurfaceMesh("wrinkled mesh")->translate({ 4 * shiftx, 0, 0 });
}

void updateFieldsInView(int frameId)
{
	std::cout << "update viewer. " << std::endl;
	registerMesh(frameId);
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
	std::string workingFolder = filePath.substr(0, id + 1);
	std::cout << "working folder: " << workingFolder << std::endl;

	json jval;
	inputJson >> jval;

	std::string meshFile = jval["mesh_name"];
	upsampleTimes = jval["upsampled_times"];


	meshFile = workingFolder + meshFile;
	igl::readOBJ(meshFile, triV, triF);
	triMesh = MeshConnectivity(triF);
	initialization(triV, triF, upsampledTriV, upsampledTriF);
	

	quadOrder = jval["quad_order"];
	numFrames = jval["num_frame"];

	isCoupled = jval["operation_details"]["amp_omega_coupling"];
	selectedMagValue = jval["operation_details"]["amp_operation_value"];
	selectedMotionValue = jval["operation_details"]["omega_operation_value"];

	std::string optype = jval["operation_details"]["omega_operation_type"];
	if (optype == "None")
		selectedMotion = None;
	else if (optype == "Enlarge")
		selectedMotion = Enlarge;
	else if (optype == "Rotate")
		selectedMotion = Rotate;
	else
		selectedMotion = None;

	isSelectAll = jval["region_details"]["select_all"];
	clickedFid = jval["region_details"]["selected_fid"];
	dilationTimes = jval["region_details"]["selected_domain_dilation"];
	optTimes = jval["region_details"]["interface_dilation"];

	pickFaces.clear();
	if (clickedFid != -1)
	{
		PickedFace pf;
		pf.fid = clickedFid;
		pf.effectiveRadius = dilationTimes;
		pf.isFreqAmpCoupled = isCoupled;
		pf.freqVecChangeValue = selectedMotionValue;
		pf.freqVecMotion = selectedMotion;
		pf.ampChangeRatio = selectedMagValue;

		pf.buildEffectiveFaces(triF.rows());
		pickFaces.push_back(pf);
		addSelectedFaces(pf, initSelectedFids);
	}
	
	updateEditionDomain();
	

	if (jval.contains(std::string_view{ "spatial_ratio" }))
	{
		if (jval["spatial_ratio"].contains(std::string_view{ "amp_ratio" }))
			spatialAmpRatio = jval["spatial_ratio"]["amp_ratio"];
		else
			spatialAmpRatio = 100;

		if (jval["spatial_ratio"].contains(std::string_view{ "edge_ratio" }))
			spatialEdgeRatio = jval["spatial_ratio"]["edge_ratio"];
		else
			spatialEdgeRatio = 100;

		if (jval["spatial_ratio"].contains(std::string_view{ "knoppel_ratio" }))
			spatialKnoppelRatio = jval["spatial_ratio"]["knoppel_ratio"];
		else
			spatialKnoppelRatio = 100;
	}
	else
	{
		spatialAmpRatio = 100;
		spatialEdgeRatio = 100;
		spatialKnoppelRatio = 100;
	}


	int nedges = triMesh.nEdges();
	int nverts = triV.rows();

	std::string initAmpPath = jval["init_amp"];
	std::string initOmegaPath = jval["init_omega"];
	std::string initZValsPath = "zvals.txt";
	if (jval.contains(std::string_view{ "init_zvals" }))
	{
		initZValsPath = jval["init_zvals"];
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


	std::string refAmp = jval["reference"]["ref_amp"];
	std::string refOmega = jval["reference"]["ref_omega"];

	std::string optZvals = jval["solution"]["opt_zvals"];
	std::string optOmega = jval["solution"]["opt_omega"];

	// edge omega List
	int iter = 0;
	bool isLoadRef = true;
	refAmpList.resize(numFrames);
	refOmegaList.resize(numFrames);

	for (uint32_t i = 0; i < numFrames; ++i) {

		if (!loadVertexAmp(workingFolder + refAmp + "/amp_" + std::to_string(i) + ".txt", triV.rows(), refAmpList[i]))
		{
			std::cout << "missing amp file: " << std::endl;
			isLoadRef = false;
			break;
		}

		std::string edgePath = workingFolder + refOmega + "/omega_" + std::to_string(i) + ".txt";
		if (!loadEdgeOmega(edgePath, nedges, refOmegaList[i])) {
			std::cout << "missing edge file." << std::endl;
			isLoadRef = false;
			break;
		}
	}


	isLoadOpt = true;
	zList.clear();
	omegaList.clear();
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

	if (isLoadOpt)
	{
		std::cout << "load zvals and omegas from file!" << std::endl;
	}
	if (!isLoadOpt || !isLoadRef)
	{
		editModel = IntrinsicFormula::WrinkleEditingStaticEdgeModel(triV, triMesh, vertOpts, faceFlags, quadOrder, spatialAmpRatio, spatialEdgeRatio, spatialKnoppelRatio);

		editModel.initialization(initZvals, initOmega, numFrames - 2);
		refAmpList = editModel.getRefAmpList();
		refOmegaList = editModel.getRefWList();

		if (!isLoadOpt)
		{
			zList = editModel.getVertValsList();
			omegaList = editModel.getWList();
		}
		
	}

	updatePaintingItems();

	curFrame = 0;

	return true;
}


bool saveProblem()
{
	std::string saveFileName = igl::file_dialog_save();

	std::string curOpt = "None";
	if (selectedMotion == Enlarge)
		curOpt = "Enlarge";
	else if (selectedMotion == Rotate)
		curOpt = "Rotate";

	using json = nlohmann::json;
	json jval =
	{
			{"mesh_name",         "mesh.obj"},
			{"num_frame",         zList.size()},
			{"quad_order",        quadOrder},
			{"spatial_ratio",     {
										   {"amp_ratio", spatialAmpRatio},
										   {"edge_ratio", spatialEdgeRatio},
										   {"knoppel_ratio", spatialKnoppelRatio}

								  }
			},
			{"upsampled_times",  upsampleTimes},
			{"init_omega",        "omega.txt"},
			{"init_amp",          "amp.txt"},
			{"init_zvls",         "zvals.txt"},
			{"region_global_details",	  {
										{"select_all", isSelectAll},
										{"omega_operation_motion", curOpt},
										{"omega_operation_value", selectedMotionValue},
										{"amp_omega_coupling", isCoupled},
										{"amp_operation_value", selectedMagValue}
								  }
			},
			{
			 "reference",         {
										  {"ref_amp", "/refAmp/"},
										  {"ref_omega", "/refOmega/"}
								  }
			},
			{
			 "solution",          {
										  {"opt_zvals", "/optZvals/"},
										  {"opt_omega", "/optOmega/"},
										  {"wrinkle_mesh", "/wrinkledMesh/"},
										  {"upsampled_amp", "/upsampledAmp/"},
										  {"upsampled_phase", "/upsampledPhase/"}
								  }
			}
	};

	for (int i = 0; i < pickFaces.size(); i++)
	{
		curOpt = "None";
		if (pickFaces[i].freqVecMotion == Enlarge)
			curOpt = "Enlarge";
		else if (pickFaces[i].freqVecMotion == Rotate)
			curOpt = "Rotate";
		json pfJval =
		{
			{"face_id", pickFaces[i].fid},
			{"effective_radius", pickFaces[i].effectiveRadius},
			{"omega_operation_motion", curOpt},
			{"omega_opereation_value", pickFaces[i].freqVecChangeValue},
			{"amp_operation_value", pickFaces[i].ampChangeRatio},
			{"amp_omega_coupling", pickFaces[i].isFreqAmpCoupled}
		};
		jval["region_local_details"].push_back(pfJval);
	}

	std::string filePath = saveFileName;
	std::replace(filePath.begin(), filePath.end(), '\\', '/'); // handle the backslash issue for windows
	int id = filePath.rfind("/");
	std::string workingFolder = filePath.substr(0, id + 1);

	std::ofstream iwfs(workingFolder + "omega.txt");
	iwfs << std::setprecision(std::numeric_limits<long double>::digits10 + 1) << initOmega << std::endl;

	std::ofstream iafs(workingFolder + "amp.txt");
	iafs << std::setprecision(std::numeric_limits<long double>::digits10 + 1) << initAmp << std::endl;

	std::ofstream izfs(workingFolder + "zvals.txt");
	for (int i = 0; i < initZvals.size(); i++)
	{
		izfs << std::setprecision(std::numeric_limits<long double>::digits10 + 1) << initZvals[i].real() << " " << initZvals[i].imag() << std::endl;
	}


	igl::writeOBJ(workingFolder + "mesh.obj", triV, triF);

	std::string outputFolder = workingFolder + "/optZvals/";
	mkdir(outputFolder);

	std::string omegaOutputFolder = workingFolder + "/optOmega/";
	mkdir(omegaOutputFolder);



	for (int i = 0; i < zList.size(); i++)
	{
		std::ofstream zfs(outputFolder + "zvals_" + std::to_string(i) + ".txt");
		std::ofstream wfs(omegaOutputFolder + "omega_" + std::to_string(i) + ".txt");
		wfs << std::setprecision(std::numeric_limits<long double>::digits10 + 1) << omegaList[i] << std::endl;
		for (int j = 0; j < zList[i].size(); j++)
		{
			zfs << std::setprecision(std::numeric_limits<long double>::digits10 + 1) << zList[i][j].real() << " " << zList[i][j].imag() << std::endl;
		}


	}

	Eigen::MatrixXd N;
	igl::per_vertex_normals(upsampledTriV, upsampledTriF, N);

	outputFolder = workingFolder + "/upsampledAmp/";
	mkdir(outputFolder);

	std::string outputFolderPhase = workingFolder + "/upsampledPhase/";
	mkdir(outputFolderPhase);

	std::string outputFolderWrinkles = workingFolder + "/wrinkledMesh/";
	mkdir(outputFolderWrinkles);

	for (int i = 0; i < ampFieldsList.size(); i++)
	{
		Eigen::MatrixXd wrinkledV = upsampledTriV;
		for (int j = 0; j < upsampledTriV.rows(); j++)
		{
			wrinkledV.row(j) = upsampledTriV.row(j) + ampFieldsList[i](j) * std::cos(phaseFieldsList[i][j]) * N.row(j);
		}

		igl::writeOBJ(outputFolderWrinkles + "wrinkledMesh_" + std::to_string(i) + ".obj", wrinkledV, upsampledTriF);
        laplacianSmoothing(wrinkledV, upsampledTriF, wrinkledV, smoothingRatio, smoothingTimes);
        igl::writeOBJ(outputFolderWrinkles + "wrinkledMeshSmoothed_" + std::to_string(i) + ".obj", wrinkledV, upsampledTriF);

		/*Eigen::MatrixXd upWrinkledV;
		Eigen::MatrixXi upWrinkledF;
		loopUpsampling(wrinkledV, upsampledTriF, upWrinkledV, upWrinkledF, 2);
		igl::writeOBJ(outputFolderWrinkles + "loopedWrinkledMesh_" + std::to_string(i) + ".obj", upWrinkledV, upWrinkledF);*/

		std::ofstream afs(outputFolder + "upAmp_" + std::to_string(i) + ".txt");
		std::ofstream pfs(outputFolderPhase + "upPhase" + std::to_string(i) + ".txt");
		afs << std::setprecision(std::numeric_limits<long double>::digits10 + 1) << ampFieldsList[i] << std::endl;
		pfs << std::setprecision(std::numeric_limits<long double>::digits10 + 1) << phaseFieldsList[i] << std::endl;
	}

	// save reference
	outputFolder = workingFolder + "/refAmp/";
	mkdir(outputFolder);
	for (int i = 0; i < refAmpList.size(); i++)
	{
		std::ofstream afs(outputFolder + "amp_" + std::to_string(i) + ".txt");
		afs << std::setprecision(std::numeric_limits<long double>::digits10 + 1) << refAmpList[i] << std::endl;
	}

	outputFolder = workingFolder + "/refOmega/";
	mkdir(outputFolder);
	for (int i = 0; i < refOmegaList.size(); i++)
	{
		std::ofstream wfs(outputFolder + "omega_" + std::to_string(i) + ".txt");
		wfs << std::setprecision(std::numeric_limits<long double>::digits10 + 1) << refOmegaList[i] << std::endl;
	}

	std::ofstream o(saveFileName);
	o << std::setw(4) << jval << std::endl;
	std::cout << "save file in: " << saveFileName << std::endl;

	return true;
}

void callback() {
	ImGui::PushItemWidth(100);
	float w = ImGui::GetContentRegionAvailWidth();
	float p = ImGui::GetStyle().FramePadding.x;
	if (ImGui::Button("Load", ImVec2(ImVec2((w - p) / 2.f, 0))))
	{
		loadProblem();
		updateMagnitudePhase(omegaList, zList, ampFieldsList, phaseFieldsList, upZList);
		updateFieldsInView(curFrame);
	}
	ImGui::SameLine(0, p);
	if (ImGui::Button("Save", ImVec2((w - p) / 2.f, 0)))
	{
		saveProblem();
	}
	if (ImGui::Button("Reset", ImVec2(-1, 0)))
	{
		curFrame = 0;
		updateFieldsInView(curFrame);
	}

	if (ImGui::InputInt("upsampled times", &upsampleTimes))
	{
		if (upsampleTimes >= 0)
		{
			initialization(triV, triF, upsampledTriV, upsampledTriF);
			if (isForceOptimize)	//already solve for the interp states
			{
				updateMagnitudePhase(omegaList, zList, ampFieldsList, phaseFieldsList, upZList);
				updateFieldsInView(curFrame);
			}
		}
	}
	if (ImGui::CollapsingHeader("Visualization Options", ImGuiTreeNodeFlags_DefaultOpen))
	{
		if (ImGui::Checkbox("is show vector fields", &isShowVectorFields))
		{
			updateFieldsInView(curFrame);
		}
		if (ImGui::Checkbox("is show wrinkled mesh", &isShowWrinkels))
		{
			updateFieldsInView(curFrame);
		}
		if (ImGui::Checkbox("is use v2", &isUseV2))
		{
			updateFieldsInView(curFrame);
		}
		if (ImGui::DragFloat("wrinkle amp scaling ratio", &wrinkleAmpScalingRatio, 0.0005, 0, 1))
        {
            if (wrinkleAmpScalingRatio >= 0)
                updateFieldsInView(curFrame);
        }
	}

    if (ImGui::CollapsingHeader("smoothing Options", ImGuiTreeNodeFlags_DefaultOpen))
    {
        if(ImGui::InputInt("smoothing times", &smoothingTimes))
        {
            smoothingTimes = smoothingTimes > 0 ? smoothingTimes : 0;
            updateFieldsInView(curFrame);
        }
        if(ImGui::InputDouble("smoothing ratio", &smoothingRatio))
        {
            smoothingRatio = smoothingRatio > 0 ? smoothingRatio : 0;
            updateFieldsInView(curFrame);
        }
    }

	if (ImGui::CollapsingHeader("Edition Options", ImGuiTreeNodeFlags_DefaultOpen))
	{
		ImGuiTabBarFlags tab_bar_flags = ImGuiTabBarFlags_None;
		if (ImGui::BeginTabBar("Selected Region", tab_bar_flags))
		{
			if (ImGui::BeginTabItem("Local"))
			{
				buildFacesMenu();
				ImGui::EndTabItem();
			}
			if (ImGui::BeginTabItem("Global"))
			{
				ImGui::Checkbox("Select all", &isSelectAll);
				ImGui::Combo("edition motion", (int*)&selectedMotion, "Ratate\0Tilt\0Enlarge\0None\0");
				if (ImGui::InputDouble("motion value", &selectedMotionValue))
				{
					if (selectedMotionValue < 0 && selectedMotion == Enlarge)
						selectedMotionValue = 0;
				}
				ImGui::Checkbox("vec mag coupled", &isCoupled);
				if (ImGui::InputDouble("mag motion value", &selectedMagValue))
				{
					if (selectedMagValue < 0)
						selectedMagValue = 1;
				}
				ImGui::EndTabItem();
			}
			ImGui::EndTabBar();
			/*ImGui::Checkbox("Select all", &isSelectAll);
			ImGui::InputInt("clicked face id", &clickedFid);

			if (ImGui::InputInt("dilation times", &dilationTimes))
			{
				if (dilationTimes < 0)
					dilationTimes = 3;
			}
			if (ImGui::InputInt("opt times", &optTimes))
			{
				if (optTimes < 0 || optTimes > 20)
					optTimes = 0;
			}*/
		}
	}
	

	if (ImGui::CollapsingHeader("Frame Visualization Options", ImGuiTreeNodeFlags_DefaultOpen))
	{
		if (ImGui::SliderInt("current frame", &curFrame, 0, numFrames - 1))
		{
			if (curFrame >= 0 && curFrame <= numFrames - 1)
				updateFieldsInView(curFrame);
		}
		if (ImGui::DragFloat("vec ratio", &(vecratio), 0.00005, 0, 1))
		{
			updateFieldsInView(curFrame);
		}
	}

	if (ImGui::CollapsingHeader("optimzation parameters", ImGuiTreeNodeFlags_DefaultOpen))
	{
		if (ImGui::InputInt("num of frames", &numFrames))
		{
			if (numFrames <= 0)
				numFrames = 10;
		}

		if (ImGui::InputInt("num iterations", &numIter))
		{
			if (numIter < 0)
				numIter = 1000;
		}
		if (ImGui::InputDouble("grad tol", &gradTol))
		{
			if (gradTol < 0)
				gradTol = 1e-6;
		}
		if (ImGui::InputDouble("x tol", &xTol))
		{
			if (xTol < 0)
				xTol = 0;
		}
		if (ImGui::InputDouble("f tol", &fTol))
		{
			if (fTol < 0)
				fTol = 0;
		}
		if (ImGui::InputInt("quad order", &quadOrder))
		{
			if (quadOrder <= 0 || quadOrder > 20)
				quadOrder = 4;
		}

		if (ImGui::InputDouble("spatial amp ratio", &spatialAmpRatio))
		{
			if (spatialAmpRatio < 0)
				spatialAmpRatio = 1;
		}

		if (ImGui::InputDouble("spatial edge ratio", &spatialEdgeRatio))
		{
			if (spatialEdgeRatio < 0)
				spatialEdgeRatio = 1;
		}

		if (ImGui::InputDouble("spatial knoppel ratio", &spatialKnoppelRatio))
		{
			if (spatialKnoppelRatio < 0)
				spatialKnoppelRatio = 1;
		}

		ImGui::Checkbox("warm start", &isWarmStart);

	}


	ImGui::Checkbox("Try Optimization", &isForceOptimize);

	if (ImGui::Button("comb fields & updates", ImVec2(-1, 0)))
	{

		updateEditionDomain();
		// solve for the path from source to target
		solveKeyFrames(initAmp, initOmega, faceFlags, omegaList, zList);
		updatePaintingItems();
		updateFieldsInView(curFrame);
	}

	if (ImGui::Button("output images", ImVec2(-1, 0)))
	{
		std::string curFolder = std::filesystem::current_path().string();
		std::cout << "save folder: " << curFolder << std::endl;
		for (int i = 0; i < ampFieldsList.size(); i++)
		{
			updateFieldsInView(i);
			//polyscope::options::screenshotExtension = ".jpg";
			std::string name = curFolder + "/output_" + std::to_string(i) + ".jpg";
			polyscope::screenshot(name);
		}
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

	// Register the mesh with Polyscope
//    polyscope::registerSurfaceMesh("input mesh", triV, triF);


	polyscope::view::upDir = polyscope::view::UpDir::ZUp;

	// Add the callback
	polyscope::state::userCallback = callback;

	polyscope::options::groundPlaneHeightFactor = 0.25; // adjust the plane height

	updateFieldsInView(curFrame);
	// Show the gui
	polyscope::show();


	return 0;
}