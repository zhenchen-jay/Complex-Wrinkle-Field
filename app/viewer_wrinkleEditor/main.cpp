#include "polyscope/polyscope.h"
#include "polyscope/pick.h"

#include <igl/PI.h>
#include <igl/avg_edge_length.h>
#include <igl/barycenter.h>
#include <igl/boundary_loop.h>
#include <igl/exact_geodesic.h>
#include <igl/gaussian_curvature.h>
#include <igl/invert_diag.h>
#include <igl/lscm.h>
#include <igl/massmatrix.h>
#include <igl/per_vertex_normals.h>
#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>
#include <igl/doublearea.h>
#include <igl/decimate.h>
#include <igl/file_dialog_open.h>
#include <igl/file_dialog_save.h>
#include <igl/boundary_loop.h>
#include <igl/cotmatrix_entries.h>
#include <igl/triangle/triangulate.h>
#include <igl/colormap.h>
#include <igl/cylinder.h>
#include <igl/principal_curvature.h>
#include <igl/heat_geodesics.h>
#include "polyscope/messages.h"
#include "polyscope/point_cloud.h"
#include "polyscope/surface_mesh.h"
#include "polyscope/view.h"

#include <iostream>
#include <fstream>
#include <filesystem>
#include <unordered_set>
#include <utility>

#include "../../include/CommonTools.h"
#include "../../include/MeshLib/MeshConnectivity.h"
#include "../../include/MeshLib/MeshUpsampling.h"
#include "../../include/Visualization/PaintGeometry.h"
#include "../../include/InterpolationScheme/VecFieldSplit.h"
#include "../../include/Optimization/NewtonDescent.h"
#include "../../include/Optimization/LBFGSSolver.h"
#include "../../include/Optimization/LinearConstrainedSolver.h"
#include "../../include/IntrinsicFormula/InterpolateZvalsFromEdgeOmega.h"
#include "../../include/IntrinsicFormula/AmpSolver.h"
#include "../../include/IntrinsicFormula/WrinkleEditingProcess.h"
#include "../../include/IntrinsicFormula/ComputeZdotFromHalfEdgeOmega.h"
#include "../../include/WrinkleFieldsEditor.h"
#include "../../include/IntrinsicFormula/IntrinsicKnoppelDrivenFormula.h"
#include "../../include/IntrinsicFormula/IntrinsicKeyFrameInterpolationFromHalfEdge.h"
#include "../../include/IntrinsicFormula/KnoppelStripePattern.h"
#include "../../include/json.hpp"
#include "../../include/MeshLib/RegionEdition.h"
#include <igl/cylinder.h>

enum FunctionType {
	Whirlpool = 0,
	PlaneWave = 1
};

enum RegionOpType
{
    Dilation = 0,
    Erosion = 1
};

Eigen::MatrixXd triV, upsampledTriV;
Eigen::MatrixXi triF, upsampledTriF;
MeshConnectivity triMesh, upsampledTriMesh;
std::vector<std::pair<int, Eigen::Vector3d>> bary;

std::vector<Eigen::MatrixXd> omegaList;
std::vector<Eigen::MatrixXd> vertexOmegaList;
std::vector<std::vector<std::complex<double>>> zList;


std::vector<Eigen::VectorXd> phaseFieldsList;
std::vector<Eigen::VectorXd> ampFieldsList;


// reference amp and omega
std::vector<Eigen::MatrixXd> refOmegaList;
std::vector<Eigen::VectorXd> refAmpList;


Eigen::MatrixXd dataV;
Eigen::MatrixXi dataF;
Eigen::MatrixXd dataVec;
Eigen::MatrixXd curColor;

int loopLevel = 2;

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
float wrinkleAmpScalingRatio = 0.01;

double triarea = 0.004;

std::string workingFolder;

IntrinsicFormula::WrinkleEditingProcess editModel;
VecMotionType selectedMotion = Rotate;
VecMotionType unselectedMotion = Enlarge;

double selectedMotionValue = M_PI / 2;
double unselectedMotionValue = 2;

Eigen::VectorXi initSelectedFids;
Eigen::VectorXi selectedFids;
Eigen::VectorXi faceFlags;

FunctionType refFunc = PlaneWave;

RegionOpType regOpType = Dilation;
int optTimes = 0;

//double bxmin = -0.25, bxmax = 0.25, bymin = -0.5, bymax = 0.5;
int clickedFid = -1;
int dilationTimes = 3;
double centerx = -0.5, centery = 0.25, dirx = 1, diry = 0;

void generateWhirlPool(double centerx, double centery, Eigen::MatrixXd& w, std::vector<std::complex<double>>& z, int pow = 1)
{
	z.resize(triV.rows());
	w.resize(triV.rows(), 2);
	std::cout << "whirl pool center: " << centerx << ", " << centery << std::endl;
	bool isnegative = false;
	if(pow < 0)
	{
		isnegative = true;
		pow *= -1;
	}

	for (int i = 0; i < z.size(); i++)
	{
		double x = triV(i, 0) - centerx;
		double y = triV(i, 1) - centery;
		double rsquare = x * x + y * y;

		if(isnegative)
		{
			z[i] = std::pow(std::complex<double>(x, -y), pow);

			if (std::abs(std::sqrt(rsquare)) < 1e-10)
				w.row(i) << 0, 0;
			else
				w.row(i) << pow * y / rsquare, -pow * x / rsquare;
		}
		else
		{
			z[i] = std::pow(std::complex<double>(x, y), pow);

			if (std::abs(std::sqrt(rsquare)) < 1e-10)
				w.row(i) << 0, 0;
			else
				//			w.row(i) << -y / rsquare, x / rsquare;
				w.row(i) << -pow * y / rsquare, pow * x / rsquare;
		}
	}
}

void generatePlaneWave(Eigen::Vector2d v, Eigen::MatrixXd& w, std::vector<std::complex<double>>& z)
{
	z.resize(triV.rows());
	w.resize(triV.rows(), 2);
	std::cout << "plane wave direction: " << v.transpose() << std::endl;

	for (int i = 0; i < z.size(); i++)
	{
		double theta = v.dot(triV.row(i).segment<2>(0));
		double x = std::cos(theta);
		double y = std::sin(theta);
		z[i] = std::complex<double>(x, y);
		w.row(i) = v;
	}
}

void getSelecteFids()
{
    selectedFids.setZero(triMesh.nFaces());
    initSelectedFids = selectedFids;
    if(clickedFid == -1)
        return;
    else
    {
        selectedFids(clickedFid) = 1;
        initSelectedFids = selectedFids;

        RegionEdition regEdt = RegionEdition(triMesh);

        for(int i = 0; i < dilationTimes; i++)
        {
            regEdt.faceDilation(initSelectedFids, selectedFids);
            initSelectedFids = selectedFids;
        }
    }
}

//void getSelecteFids(double minx, double miny, double maxx, double maxy)
//{
//	selectedFids.setZero(triF.rows());
//
//	for (int i = 0; i < triF.rows(); i++)
//	{
//		double centerx = 0;
//		double centery = 0;
//		for (int j = 0; j < 3; j++)
//		{
//			int vid = triMesh.faceVertex(i, j);
//			centerx += triV(vid, 0) / 3;
//			centery += triV(vid, 1) / 3;
//		}
//		if (centerx >= minx && centerx <= maxx && centery >= miny && centery <= maxy)
//			selectedFids(i) = 1;
//	}
//
//	initSelectedFids = selectedFids;
//}

void initialization()
{
	Eigen::SparseMatrix<double> S;
	std::vector<int> facemap;

	meshUpSampling(triV, triF, upsampledTriV, upsampledTriF, loopLevel, &S, &facemap, &bary);
	std::cout << "upsampling finished" << std::endl;

	triMesh = MeshConnectivity(triF);
	upsampledTriMesh = MeshConnectivity(upsampledTriF);
	selectedFids.setZero(triMesh.nFaces());
    initSelectedFids = selectedFids;
	
}


void buildWrinkleMotions(const Eigen::VectorXd& amp, const Eigen::MatrixXd& omega)
{
	int nverts = triV.rows();
	Eigen::VectorXi initSelectedVids;

	faceFlags2VertFlags(triMesh, nverts, initSelectedFids, initSelectedVids);

	double dt = 1.0 / (numFrames - 1);
	double initval0 = selectedMotion != Enlarge ? 0 : 1;
	double initval1 = unselectedMotion != Enlarge ? 0 : 1;

	refAmpList.resize(numFrames);
	refOmegaList.resize(numFrames);

	for (int f = 0; f < numFrames; f++)
	{
		std::vector<VertexOpInfo> vertexOpInfoList;
		vertexOpInfoList.resize(nverts, { None, 0 });

		double value0 = (selectedMotionValue - initval0) * dt * f + initval0;
		double value1 = (unselectedMotionValue - initval1) * dt * f + initval1;

		for (int i = 0; i < nverts; i++)
		{
			if (initSelectedVids(i))
				vertexOpInfoList[i] = { selectedMotion, value0 };
			else
				vertexOpInfoList[i] = { unselectedMotion, value1 };
		}

		Eigen::MatrixXd vertOmega = intrinsicHalfEdgeVec2VertexVec(omega, triV, triMesh);
		WrinkleFieldsEditor::editWrinkles(triV, triMesh, amp, vertOmega, vertexOpInfoList, refAmpList[f], vertOmega);
		refOmegaList[f] = vertexVec2IntrinsicHalfEdgeVec(vertOmega, triV, triMesh);
	}

	
}

void solveKeyFrames(std::vector<Eigen::MatrixXd>& wFrames, std::vector<std::vector<std::complex<double>>>& zFrames)
{
	Eigen::VectorXd faceArea;
	igl::doublearea(triV, triF, faceArea);
	faceArea /= 2;
	Eigen::MatrixXd cotEntries;
	igl::cotmatrix_entries(triV, triF, cotEntries);

	Eigen::VectorXi faceFlag;
	faceFlag.setConstant(triF.rows(), -1);


	editModel = IntrinsicFormula::WrinkleEditingProcess(triV, triMesh, faceFlags, quadOrder, 1.0);
	
	editModel.initialization(refAmpList, refOmegaList);

	std::cout << "initilization finished!" << std::endl;
	Eigen::VectorXd x;
	std::cout << "convert list to variable." << std::endl;
	editModel.convertList2Variable(x);

	
	if(isForceOptimize)
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

		auto postProcess = [&](Eigen::VectorXd& x)
		{
//            interpModel.postProcess(x);
		};


		OptSolver::testFuncGradHessian(funVal, x);

		auto x0 = x;
		OptSolver::newtonSolver(funVal, maxStep, x, numIter, gradTol, xTol, fTol, true, getVecNorm, &workingFolder, postProcess);
		std::cout << "before optimization: " << x0.norm() << ", after optimization: " << x.norm() << std::endl;
	}
	std::cout << "convert variable to list." << std::endl;
	editModel.convertVariable2List(x);
	std::cout << "get w list" << std::endl;
	wFrames = editModel.getWList();
	std::cout << "get z list" << std::endl;
	zFrames = editModel.getVertValsList();
}

void updateMagnitudePhase(const std::vector<Eigen::MatrixXd>& wFrames, const std::vector<std::vector<std::complex<double>>>& zFrames, std::vector<Eigen::VectorXd>& magList, std::vector<Eigen::VectorXd>& phaseList)
{
	std::vector<std::vector<std::complex<double>>> interpZList(wFrames.size());
	magList.resize(wFrames.size());
	phaseList.resize(wFrames.size());

	MeshConnectivity mesh(triF);

	auto computeMagPhase = [&](const tbb::blocked_range<uint32_t> &range)
	{
		for (uint32_t i = range.begin(); i < range.end(); ++i)
		{
			interpZList[i] = IntrinsicFormula::upsamplingZvals(mesh, zFrames[i], wFrames[i], bary);
			magList[i].setZero(interpZList[i].size());
			phaseList[i].setZero(interpZList[i].size());

			for (int j = 0; j < magList[i].size(); j++)
			{
				magList[i](j) = std::abs(interpZList[i][j]);
				phaseList[i](j) = std::arg(interpZList[i][j]);
			}
		}
	};

	tbb::blocked_range<uint32_t> rangex(0u, (uint32_t) interpZList.size(), GRAIN_SIZE);
	tbb::parallel_for(rangex, computeMagPhase);
}

void registerMeshByPart(const Eigen::MatrixXd& basePos, const Eigen::MatrixXi& baseF,
						const Eigen::MatrixXd& upPos, const Eigen::MatrixXi& upF, const double& shiftz, const double& ampMin,
						const double& ampMax,
						const Eigen::VectorXd ampVec, const Eigen::VectorXd& phaseVec, const Eigen::MatrixXd& omegaVec, 
						const Eigen::VectorXd& refAmp, const Eigen::MatrixXd& refOmega, const Eigen::VectorXi& vertFlag,
						Eigen::MatrixXd& renderV, Eigen::MatrixXi& renderF, Eigen::MatrixXd& renderVec, Eigen::MatrixXd& renderColor)
{
	int nverts = basePos.rows();
	int nfaces = baseF.rows();

	int nupverts = upPos.rows();
	int nupfaces = upF.rows();

	int ndataVerts = 2 * nverts + 2 * nupverts;
	int ndataFaces = 2 * nfaces + 2 * nupfaces;

	if(!isShowVectorFields)
	{
		ndataVerts = 2 * nupverts + nverts;
		ndataFaces = 2 * nupfaces + nverts;
	}
	if(isShowWrinkels)
	{
		ndataVerts += nupverts;
		ndataFaces += nupfaces;
	}

	double shiftx = 1.5 * (basePos.col(0).maxCoeff() - basePos.col(0).minCoeff());

	renderV.resize(ndataVerts, 3);
	renderVec.setZero(ndataVerts, 3);
	renderF.resize(ndataFaces, 3);
	renderColor.setZero(ndataVerts, 3);

	renderColor.col(0).setConstant(1.0);
	renderColor.col(1).setConstant(1.0);
	renderColor.col(2).setConstant(1.0);

	int curVerts = 0;
	int curFaces = 0;

	Eigen::MatrixXd shiftV = basePos;
	shiftV.col(0).setConstant(-shiftx);
	shiftV.col(1).setConstant(0);
	shiftV.col(2).setConstant(shiftz);

	for (int i = 0; i < nverts; i++)
	{
		if (vertFlag(i) == 1)
			renderColor.row(i) << 1.0, 0, 0;
		else if (vertFlag(i) == -1)
			renderColor.row(i) << 0, 1.0, 0;
	}
	renderV.block(curVerts, 0, nverts, 3) = basePos - shiftV;
	renderF.block(curFaces, 0, nfaces, 3) = baseF;
	if (isShowVectorFields)
	{
		for (int i = 0; i < nverts; i++)
			renderVec.row(i + curVerts) = refOmega.row(i);
	}

	curVerts += nverts;
	curFaces += nfaces;

	
	Eigen::MatrixXi shiftF = baseF;
	shiftF.setConstant(curVerts);
	shiftV.col(0).setConstant(0);
	shiftV.col(1).setConstant(0);
	shiftV.col(2).setConstant(shiftz);
	mPaint.setNormalization(false);
	Eigen::VectorXd normoalizedRefAmpVec = refAmp;
	for (int i = 0; i < normoalizedRefAmpVec.rows(); i++)
	{
		normoalizedRefAmpVec(i) = (refAmp(i) - ampMin) / (ampMax - ampMin);
	}
	std::cout << "ref amp (min, max): " << refAmp.minCoeff() << " " << refAmp.maxCoeff() << std::endl;
	Eigen::MatrixXd refColor = mPaint.paintAmplitude(normoalizedRefAmpVec);
	renderColor.block(curVerts, 0, nverts, 3) = refColor;
	renderV.block(curVerts, 0, nverts, 3) = basePos - shiftV;
	renderF.block(curFaces, 0, nfaces, 3) = baseF + shiftF;

	if(isShowVectorFields)
	{
		for (int i = 0; i < nverts; i++)
			renderVec.row(i + curVerts) = omegaVec.row(i);
	}
	curVerts += nverts;
	curFaces += nfaces;


	
	// interpolated amp
	shiftV = upPos;
	shiftV.col(0).setConstant(shiftx);
	shiftV.col(1).setConstant(0);
	shiftV.col(2).setConstant(shiftz);


	shiftF = upF;
	shiftF.setConstant(curVerts);

	// interpolated phase
	renderV.block(curVerts, 0, nupverts, 3) = upPos - shiftV;
	renderF.block(curFaces, 0, nupfaces, 3) = upF + shiftF;

	mPaint.setNormalization(false);
	Eigen::MatrixXd phiColor = mPaint.paintPhi(phaseVec);
	renderColor.block(curVerts, 0, nupverts, 3) = phiColor;

	curVerts += nupverts;
	curFaces += nupfaces;

	// interpolated amp
	shiftF.setConstant(curVerts);
	shiftV.col(0).setConstant(2 * shiftx);
	renderV.block(curVerts, 0, nupverts, 3) = upPos - shiftV;
	renderF.block(curFaces, 0, nupfaces, 3) = upF + shiftF;

	mPaint.setNormalization(false);
	Eigen::VectorXd normoalizedAmpVec = ampVec;
	for(int i = 0; i < normoalizedAmpVec.rows(); i++)
	{
		normoalizedAmpVec(i) = (ampVec(i) - ampMin) / (ampMax - ampMin);
	}
	Eigen::MatrixXd ampColor = mPaint.paintAmplitude(normoalizedAmpVec);
	renderColor.block(curVerts, 0, nupverts, 3) = ampColor;

	curVerts += nupverts;
	curFaces += nupfaces;

	// interpolated amp
	if(isShowWrinkels)
	{
		shiftF.setConstant(curVerts);
		shiftV.col(0).setConstant(3 * shiftx);
		Eigen::MatrixXd tmpV = upPos - shiftV;
		Eigen::MatrixXd tmpN;
		igl::per_vertex_normals(tmpV, upF, tmpN);

		Eigen::VectorXd ampCosVec(nupverts);

		for(int i = 0; i < nupverts; i++)
		{
			renderV.row(curVerts + i) = tmpV.row(i) + wrinkleAmpScalingRatio * ampVec(i) * std::cos(phaseVec(i)) * tmpN.row(i);
			ampCosVec(i) = normoalizedAmpVec(i) * std::cos(phaseVec(i));
		}
		renderF.block(curFaces, 0, nupfaces, 3) = upF + shiftF;

		mPaint.setNormalization(false);
		Eigen::MatrixXd ampCosColor = mPaint.paintAmplitude(ampCosVec);
		renderColor.block(curVerts, 0, nupverts, 3) = ampCosColor;

		curVerts += nupverts;
		curFaces += nupfaces;
	}

}

void registerMesh(int frameId)
{
	Eigen::MatrixXd sourceP, tarP, interpP;
	Eigen::MatrixXi sourceF, tarF, interpF;
	Eigen::MatrixXd sourceVec, tarVec, interpVec;
	Eigen::MatrixXd sourceColor, tarColor, interpColor;

	double shiftz = 1.5 * (triV.col(2).maxCoeff() - triV.col(2).minCoeff());
	int totalfames = ampFieldsList.size();
	Eigen::MatrixXd refVertOmega = intrinsicHalfEdgeVec2VertexVec(editModel.getRefWList()[frameId], triV, triMesh);

	Eigen::VectorXi selectedVids, initSelectedVids;
	faceFlags2VertFlags(triMesh, triV.rows(), selectedFids, selectedVids);
	faceFlags2VertFlags(triMesh, triV.rows(), initSelectedFids, initSelectedVids);

	for (int i = 0; i < selectedVids.rows(); i++)
	{
		if (selectedVids(i) && !initSelectedVids(i))
			selectedVids(i) = -1;
	}

	registerMeshByPart(triV, triF, upsampledTriV, upsampledTriF, 2 * shiftz, globalAmpMin, globalAmpMax,
		ampFieldsList[frameId], phaseFieldsList[frameId], vertexOmegaList[frameId], editModel.getRefAmpList()[frameId], refVertOmega, selectedVids, interpP, interpF, interpVec, interpColor);


	dataV = interpP;
	curColor = interpColor;
	dataVec = interpVec;
	dataF = interpF;

	polyscope::registerSurfaceMesh("input mesh", dataV, dataF);
}

void updateFieldsInView(int frameId)
{
	registerMesh(frameId);
	polyscope::getSurfaceMesh("input mesh")->addVertexColorQuantity("VertexColor", curColor);
	polyscope::getSurfaceMesh("input mesh")->getQuantity("VertexColor")->setEnabled(true);

	polyscope::getSurfaceMesh("input mesh")->addVertexVectorQuantity("vertex vector field", dataVec * vecratio, polyscope::VectorType::AMBIENT);
	polyscope::getSurfaceMesh("input mesh")->getQuantity("vertex vector field")->setEnabled(true);

}

int getSelectedFaceId()
{
    if(polyscope::pick::haveSelection())
    {
        unsigned long id = polyscope::pick::getSelection().second;
        int nverts = triV.rows();
        int nedges = triMesh.nEdges();
        int nfaces = triMesh.nFaces();

        if(id >= nverts && id < nfaces + nverts)
        {
            return id - nverts;
        }
        else
            return -1;
    }
    else
        return -1;
}


void callback() {
    clickedFid = getSelectedFaceId();
	ImGui::PushItemWidth(100);
    float w = ImGui::GetContentRegionAvailWidth();
    float p = ImGui::GetStyle().FramePadding.x;
//    if (ImGui::Button("Load", ImVec2(ImVec2((w - p) / 2.f, 0))))
//    {
//        std::string meshPath = igl::file_dialog_open();
//        igl::readOBJ(meshPath, triV, triF);
//        initialization();
//        // Initialize polyscope
//        polyscope::init();
//
//        // Register the mesh with Polyscope
//        polyscope::registerSurfaceMesh("input mesh", triV, triF);
//    }
//    ImGui::SameLine(0, p);
//    if (ImGui::Button("Save", ImVec2((w - p) / 2.f, 0)))
//    {
//        std::string saveFolder = igl::file_dialog_save();
//        //interpModel.save(saveFolder, triV, triF);
//    }
	if (ImGui::Button("Reset", ImVec2(-1, 0)))
	{
		curFrame = 0;
		updateFieldsInView(curFrame);
	}

	if (ImGui::InputInt("upsampled times", &loopLevel))
	{
		if (loopLevel >= 0)
		{
			initialization();
			if (isForceOptimize)	//already solve for the interp states
			{
				updateMagnitudePhase(omegaList, zList, ampFieldsList, phaseFieldsList);
				updateFieldsInView(curFrame);
			}
		}

	}
	if (ImGui::Checkbox("is show vector fields", &isShowVectorFields))
	{
		updateFieldsInView(curFrame);
	}
	if (ImGui::Checkbox("is show wrinkled mesh", &isShowWrinkels))
	{
		updateFieldsInView(curFrame);
	}
	if (ImGui::DragFloat("wrinkle amp scaling ratio", &wrinkleAmpScalingRatio, 0.0005, 0, 1))
	{
		if(wrinkleAmpScalingRatio >= 0)
			updateFieldsInView(curFrame);
	}

	if (ImGui::CollapsingHeader("Reference Wrinkle Fields", ImGuiTreeNodeFlags_DefaultOpen))
	{
		ImGui::Combo("ref func", (int*) &refFunc, "WhirlPool\0PlaneWave\0");
        ImGui::InputDouble("whirlpool center x", &centerx);
        ImGui::InputDouble("whirlpool center y", &centery);
        ImGui::InputDouble("plane wave dir x", &dirx);
        ImGui::InputDouble("plane wave dir y", &diry);
	}

	if (ImGui::CollapsingHeader("Underline Wrinkle Options", ImGuiTreeNodeFlags_DefaultOpen))
	{
		ImGui::Combo("underline motion", (int*)&unselectedMotion, "Ratate\0Tilt\0Enlarge\0None\0");
		if (ImGui::InputDouble("underline motion value", &unselectedMotionValue))
		{
			if (unselectedMotionValue < 0 && unselectedMotion == Enlarge)
				unselectedMotionValue = 0;
		}
	}
    if (ImGui::CollapsingHeader("Selected Region", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::InputInt("clicked face id", &clickedFid);

        if (ImGui::InputInt("dilation times", &dilationTimes))
        {
            if (dilationTimes < 0)
                dilationTimes = 3;
        }
    }
//	if (ImGui::CollapsingHeader("Selected BBox", ImGuiTreeNodeFlags_DefaultOpen))
//	{
//		if (ImGui::InputDouble("min x", &bxmin)) {}
//		ImGui::SameLine(0, p);
//		if (ImGui::InputDouble("max x", &bxmax))
//		{
//			if (bxmax < bxmin)
//				bxmax = bxmin + 0.1;
//		}
//
//		if (ImGui::InputDouble("min y", &bymin)) {}
//		ImGui::SameLine(0, p);
//		if (ImGui::InputDouble("max y", &bymax))
//		{
//			if (bymax < bymin)
//				bymax = bymin + 0.1;
//		}
//
//		if (ImGui::Button("update bbox", ImVec2(-1, 0)))
//		{
//			getSelecteFids(bxmin, bymin, bxmax, bymax);
//		}
//
//	}
	if (ImGui::CollapsingHeader("Wrinkle Edition Options", ImGuiTreeNodeFlags_DefaultOpen))
	{
		ImGui::Combo("edition motion", (int*)&selectedMotion, "Ratate\0Tilt\0Enlarge\0None\0");
		if (ImGui::InputDouble("motion value", &selectedMotionValue))
		{
			if (selectedMotionValue < 0 && selectedMotion == Enlarge)
				selectedMotionValue = 0;
		}
	}

	if (ImGui::CollapsingHeader("Frame Visualization Options", ImGuiTreeNodeFlags_DefaultOpen))
	{
		if (ImGui::SliderInt("current frame", &curFrame, 0, numFrames - 1))
		{
			if(curFrame >= 0 && curFrame <= numFrames - 1)
				updateFieldsInView(curFrame);
		}
		if (ImGui::DragFloat("vec ratio", &(vecratio), 0.0005, 0, 1))
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
	}
    ImGui::Combo("reg opt func", (int*) &regOpType, "Dilation\0Erosion\0");
    if (ImGui::InputInt("opt times", &optTimes))
    {
        if (optTimes < 0 || optTimes > 20)
            optTimes = 0;
    }

	ImGui::Checkbox("Try Optimization", &isForceOptimize);

	if (ImGui::Button("comb fields & updates", ImVec2(-1, 0)))
	{
		refAmpList.clear();
		refOmegaList.clear();
        getSelecteFids();
		
		Eigen::MatrixXd w;
		Eigen::VectorXd amp;
		std::vector<std::complex<double>> z;
		if(refFunc == Whirlpool)
		{
			generateWhirlPool(centerx, centery, w, z, 1);
		}
		else
		{
			Eigen::Vector2d dir;
			dir << dirx, diry;
            dir *= 2 * M_PI;
			generatePlaneWave(dir, w, z);
		}
		amp.setZero(triV.rows());
		for(int i = 0; i < amp.size(); i++)
			amp(i) = std::abs(z[i]);

		Eigen::MatrixXd vertFields3D(triV.rows(), 3);
		vertFields3D.block(0, 0, triV.rows(), 2) = w;
		vertFields3D.col(2).setZero();
		w = vertexVec2IntrinsicHalfEdgeVec(vertFields3D, triV, triMesh);

		
        RegionEdition regOpt(triMesh);
		selectedFids = initSelectedFids;
        for(int i = 0; i < optTimes; i++)
        {
            Eigen::VectorXi selectedFidNew;
            if(regOpType == Dilation)
                regOpt.faceDilation(selectedFids, selectedFidNew);

            else
                regOpt.faceErosion(selectedFids, selectedFidNew);


            selectedFids = selectedFidNew;
        }
		faceFlags = initSelectedFids - selectedFids;
		

		buildWrinkleMotions(amp, w);
		// solve for the path from source to target
		solveKeyFrames(omegaList, zList);
		// get interploated amp and phase frames
		std::cout << "compute upsampled phase: " << std::endl;
		updateMagnitudePhase(omegaList, zList, ampFieldsList, phaseFieldsList);
		std::cout << "compute upsampled phase finished!" << std::endl;
		vertexOmegaList.resize(omegaList.size());
		for(int i = 0; i < omegaList.size(); i++)
		{
			vertexOmegaList[i] = intrinsicHalfEdgeVec2VertexVec(omegaList[i], triV, triMesh);
		}
		
		// update global maximum amplitude
		globalAmpMax = std::max(ampFieldsList[0].maxCoeff(), editModel.getRefAmpList()[0].maxCoeff());
		globalAmpMin = std::min(ampFieldsList[0].minCoeff(), editModel.getRefAmpList()[0].minCoeff());
		for(int i = 1; i < ampFieldsList.size(); i++)
		{
			globalAmpMax = std::max(globalAmpMax, std::max(ampFieldsList[i].maxCoeff(), editModel.getRefAmpList()[i].maxCoeff()));
			globalAmpMin = std::min(globalAmpMin, std::min(ampFieldsList[i].minCoeff(), editModel.getRefAmpList()[i].minCoeff()));
		}
		updateFieldsInView(curFrame);
	}

	if (ImGui::Button("output images", ImVec2(-1, 0)))
	{
		for(int i = 0; i < ampFieldsList.size(); i++)
		{
			updateFieldsInView(i);
			//polyscope::options::screenshotExtension = ".jpg";
			std::string name = "output_" + std::to_string(i) + ".jpg";
			polyscope::screenshot(name);
		}
	}


	ImGui::PopItemWidth();
}

void generateSquare(double length, double width, double triarea, Eigen::MatrixXd& irregularV, Eigen::MatrixXi& irregularF)
{
	double area = length * width;
	int N = (0.25 * std::sqrt(area / triarea));
	N = N > 1 ? N : 1;
	double deltaX = length / (4.0 * N);
	double deltaY = width / (4.0 * N);

	Eigen::MatrixXd planeV;
	Eigen::MatrixXi planeE;

	int M = 2 * N + 1;
	planeV.resize(4 * M - 4, 2);
	planeE.resize(4 * M - 4, 2);

	for (int i = 0; i < M; i++)
	{
		planeV.row(i) << -length / 2, i * width / (M - 1) - width / 2;
	}
	for (int i = 1; i < M; i++)
	{
		planeV.row(M - 1 + i) << i * length / (M - 1)-length / 2, width / 2;
	}
	for (int i = 1; i < M; i++)
	{
		planeV.row(2 * (M - 1) + i) << length / 2, width/2 - i * width / (M - 1);
	}
	for (int i = 1; i < M - 1; i++)
	{
		planeV.row(3 * (M - 1) + i) << length / 2- i * length / (M - 1), - width / 2;
	}

	for (int i = 0; i < 4 * (M - 1); i++)
	{
		planeE.row(i) << i, (i + 1) % (4 * (M - 1));
	}

	Eigen::MatrixXd V2d;
	Eigen::MatrixXi F;
	Eigen::MatrixXi H(0, 2);
	std::cout << triarea << std::endl;
	// Create an output string stream
	std::ostringstream streamObj;
	//Add double to stream
	streamObj << triarea;
	const std::string flags = "q20a" + std::to_string(triarea);

	igl::triangle::triangulate(planeV, planeE, H, flags, V2d, F);
	irregularV.resize(V2d.rows(), 3);
	irregularV.setZero();
	irregularV.block(0, 0, irregularV.rows(), 2) = V2d.block(0, 0, irregularV.rows(), 2);
	irregularF = F;
	igl::writeOBJ("irregularPlane.obj", irregularV, irregularF);
}


int main(int argc, char** argv)
{
	generateSquare(2, 1, triarea, triV, triF);
	initialization();

	// Options
	polyscope::options::autocenterStructures = true;
	polyscope::view::windowWidth = 1024;
	polyscope::view::windowHeight = 1024;

	// Initialize polyscope
	polyscope::init();

	// Register the mesh with Polyscope
	polyscope::registerSurfaceMesh("input mesh", triV, triF);


	polyscope::view::upDir = polyscope::view::UpDir::ZUp;

	// Add the callback
	polyscope::state::userCallback = callback;

	polyscope::options::groundPlaneHeightFactor = 0.25; // adjust the plane height
	// Show the gui
	polyscope::show();

	return 0;
}