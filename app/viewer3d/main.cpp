#include "polyscope/polyscope.h"

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
#include <igl/cylinder.h>
#include <igl/principal_curvature.h>
#include <filesystem>
#include "polyscope/messages.h"
#include "polyscope/point_cloud.h"
#include "polyscope/surface_mesh.h"
#include "polyscope/view.h"

#include <iostream>
#include <fstream>
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
//#include "../../include/IntrinsicFormula/ComputeZdotFromEdgeOmega.h"
#include "../../include/IntrinsicFormula/ComputeZdotFromHalfEdgeOmega.h"
//#include "../../include/IntrinsicFormula/IntrinsicKeyFrameInterpolationFromEdge.h"
#include "../../include/IntrinsicFormula/IntrinsicKeyFrameInterpolationFromHalfEdge.h"
#include "../../include/IntrinsicFormula/KnoppelStripePattern.h"
#include <igl/cylinder.h>



Eigen::MatrixXd triV, upsampledTriV;
Eigen::MatrixXi triF, upsampledTriF;
MeshConnectivity triMesh, upsampledTriMesh;
std::vector<std::pair<int, Eigen::Vector3d>> bary;

Eigen::MatrixXd sourceOmegaFields, tarOmegaFields;
Eigen::MatrixXd sourceVertexOmegaFields, tarVertexOmegaFields;


std::vector<std::complex<double>> sourceZvals;
std::vector<std::complex<double>> tarZvals;


std::vector<Eigen::MatrixXd> omegaList;
std::vector<Eigen::MatrixXd> vertexOmegaList;
std::vector<std::vector<std::complex<double>>> zList;


std::vector<Eigen::VectorXd> phaseFieldsList;
std::vector<Eigen::VectorXd> ampFieldsList;


Eigen::MatrixXd dataV;
Eigen::MatrixXi dataF;
Eigen::MatrixXd dataVec;
Eigen::MatrixXd curColor;

int loopLevel = 2;

bool isForceOptimize = false;
bool isShowVectorFields = true;

PaintGeometry mPaint;

int numFrames = 50;
int curFrame = 0;

int numSourceWaves = 2;
int numTarWaves = 2;

double globalAmpMax = 1;

double dragSpeed = 0.5;

float vecratio = 0.001;

double gradTol = 1e-6;
double xTol = 0;
double fTol = 0;
int numIter = 1000;
int quadOrder = 4;
int numComb = 2;


enum InitializationType{
  Random = 0,
  Linear = 1,
};
enum DirectionType
{
    DIRPV1 = 0,
    DIRPV2 = 1,
    LOADFROMFILE = 2
};
enum OptSolverType
{
	Newton = 0,
	LBFGS = 1,
	Composite = 2	// use lbfgs to get a warm start
};

InitializationType initializationType = InitializationType::Linear;
DirectionType sourceDir = DIRPV1;
DirectionType tarDir = DIRPV2;
OptSolverType solverType = Newton;


bool loadEdgeOmega(const std::string& filename, const int &nlines, Eigen::MatrixXd& edgeOmega)
{
    std::ifstream infile(filename);
    if(!infile)
    {
        std::cerr << "invalid file name" << std::endl;
        return false;
    }
    else
    {
        edgeOmega.setZero(nlines, 2);
        for (int i = 0; i < nlines; i++)
        {
            std::string line;
            std::getline(infile, line);
            std::stringstream ss(line);

            std::string x, y;
            ss >> x;
            ss >> y;
            if (!ss)
            {
                edgeOmega.row(i) << std::stod(x), -std::stod(x);
            }
            else
                edgeOmega.row(i) << std::stod(x), std::stod(y);
        }
    }
    return true;
}

void initialization()
{
	Eigen::SparseMatrix<double> S;
	std::vector<int> facemap;

	meshUpSampling(triV, triF, upsampledTriV, upsampledTriF, loopLevel, &S, &facemap, &bary);
	std::cout << "upsampling finished" << std::endl;

	triMesh = MeshConnectivity(triF);
	upsampledTriMesh = MeshConnectivity(upsampledTriF);
}

void solveKeyFrames(const Eigen::MatrixXd& sourceVec, const Eigen::MatrixXd& tarVec, const std::vector<std::complex<double>>& sourceZvals, const std::vector<std::complex<double>>& tarZvals, const int numKeyFrames, std::vector<Eigen::MatrixXd>& wFrames, std::vector<std::vector<std::complex<double>>>& zFrames)
{
	Eigen::VectorXd faceArea;
	igl::doublearea(triV, triF, faceArea);
	faceArea /= 2;
	IntrinsicFormula::IntrinsicKeyFrameInterpolationFromHalfEdge interpModel = IntrinsicFormula::IntrinsicKeyFrameInterpolationFromHalfEdge(MeshConnectivity(triF), faceArea, numFrames, quadOrder, sourceZvals, sourceVec, tarZvals, tarVec);
	Eigen::VectorXd x;
	interpModel.convertList2Variable(x);        // linear initialization

    std::vector<std::complex<double>> testzvals;
    Eigen::MatrixXd cotEntries;
    igl::cotmatrix_entries(triV, triF, cotEntries);
    IntrinsicFormula::roundVertexZvalsFromHalfEdgeOmega(triMesh, sourceVec, faceArea, cotEntries, triV.rows(), testzvals);
    IntrinsicFormula::testRoundingEnergy(triMesh, sourceVec, faceArea, cotEntries, triV.rows(), testzvals);
	for (auto& z : testzvals)
	{
		Eigen::Vector2d rndvec;
		rndvec.setRandom();
		z = std::complex<double>(rndvec(0), rndvec(1));
	}
	IntrinsicFormula::testRoundingEnergy(triMesh, sourceVec, faceArea, cotEntries, triV.rows(), testzvals);

	//interpModel.testEnergy(x);
	//		std::cout << "starting energy: " << interpModel.computeEnergy(x) << std::endl;
	if (initializationType == InitializationType::Random)
	{
		x.setRandom();
		interpModel.convertVariable2List(x);
		interpModel.convertList2Variable(x);
	}
	else
	{
		// do nothing, since it is initialized as the linear interpolation.
	}

	auto initWFrames = interpModel.getWList();
	auto initZFrames = interpModel.getVertValsList();
	if (isForceOptimize)
	{
		auto funVal = [&](const Eigen::VectorXd& x, Eigen::VectorXd* grad, Eigen::SparseMatrix<double>* hess, bool isProj) {
			Eigen::VectorXd deriv;
			Eigen::SparseMatrix<double> H;
			double E = interpModel.computeEnergy(x, grad ? &deriv : NULL, hess ? &H : NULL, isProj);

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
			interpModel.getComponentNorm(x, znorm, wnorm);
		};



		OptSolver::testFuncGradHessian(funVal, x);

		auto x0 = x;
		if(solverType == Newton)
			OptSolver::newtonSolver(funVal, maxStep, x, numIter, gradTol, xTol, fTol, true, getVecNorm);
		else if (solverType == LBFGS)
			OptSolver::lbfgsSolver(funVal, maxStep, x, numIter, gradTol, xTol, fTol, true, getVecNorm);
		else if (solverType = Composite)
		{
			OptSolver::lbfgsSolver(funVal, maxStep, x, 1000, 1e-4, 1e-5, 1e-5, true, getVecNorm);
			OptSolver::newtonSolver(funVal, maxStep, x, numIter, gradTol, xTol, fTol, true, getVecNorm);
		}
		std::cout << "before optimization: " << x0.norm() << ", after optimization: " << x.norm() << ", difference: " << (x - x0).norm() << std::endl;
		std::cout << "x norm: " << x.norm() << std::endl;
	}
	interpModel.convertVariable2List(x);

	wFrames = interpModel.getWList();
	zFrames = interpModel.getVertValsList();

	for (int i = 0; i < wFrames.size() - 1; i++)
	{
		double zdotNorm = interpModel._zdotModel.computeZdotIntegration(zFrames[i], wFrames[i], zFrames[i + 1], wFrames[i + 1], NULL, NULL);

		double initZdotNorm = interpModel._zdotModel.computeZdotIntegration(initZFrames[i], initWFrames[i], initZFrames[i + 1], initWFrames[i + 1], NULL, NULL);

		std::cout << "frame " << i << ", before optimization: ||zdot||^2: " << initZdotNorm << ", after optimization, ||zdot||^2 = " << zdotNorm << std::endl;
	}


}

void updateMagnitudePhase(const std::vector<Eigen::MatrixXd>& wFrames, const std::vector<std::vector<std::complex<double>>>& zFrames, std::vector<Eigen::VectorXd>& magList, std::vector<Eigen::VectorXd>& phaseList)
{
	std::vector<std::vector<std::complex<double>>> interpZList(wFrames.size());
	magList.resize(wFrames.size());
	phaseList.resize(wFrames.size());

	MeshConnectivity mesh(triF);

	auto computeMagPhase = [&](const tbb::blocked_range<uint32_t>& range) {
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

	tbb::blocked_range<uint32_t> rangex(0u, (uint32_t)interpZList.size(), GRAIN_SIZE);
	tbb::parallel_for(rangex, computeMagPhase);
}

void registerMeshByPart(const Eigen::MatrixXd& basePos, const Eigen::MatrixXi& baseF,
	const Eigen::MatrixXd& upPos, const Eigen::MatrixXi& upF, const double& shiftz, const double& ampMax,
	Eigen::VectorXd ampVec, const Eigen::VectorXd& phaseVec, Eigen::MatrixXd* omegaVec,
	Eigen::MatrixXd& renderV, Eigen::MatrixXi& renderF, Eigen::MatrixXd& renderVec, Eigen::MatrixXd& renderColor)
{
	int nverts = basePos.rows();
	int nfaces = baseF.rows();

	int nupverts = upPos.rows();
	int nupfaces = upF.rows();

	int ndataVerts = nverts + 2 * nupverts;
	int ndataFaces = nfaces + 2 * nupfaces;

    if(!isShowVectorFields)
    {
        ndataVerts = 2 * nupverts;
        ndataFaces = 2 * nupfaces;
    }
	
	renderV.resize(ndataVerts, 3);
	renderVec.setZero(ndataVerts, 3);
	renderF.resize(ndataFaces, 3);
	renderColor.setZero(ndataVerts, 3);

	renderColor.col(0).setConstant(1.0);
	renderColor.col(1).setConstant(1.0);
	renderColor.col(2).setConstant(1.0);

	int curVerts = 0;
	int curFaces = 0;

	Eigen::VectorXd normalizedAmp = ampVec / ampMax; 


	Eigen::MatrixXd shiftV = basePos;
	shiftV.col(0).setConstant(0);
	shiftV.col(1).setConstant(0);
	shiftV.col(2).setConstant(shiftz);

    if(isShowVectorFields)
    {
        renderV.block(0, 0, nverts, 3) = basePos - shiftV;
        renderF.block(0, 0, nfaces, 3) = baseF;
        if (omegaVec)
        {
            for (int i = 0; i < nverts; i++)
                renderVec.row(i) = omegaVec->row(i);
        }

        curVerts += nverts;
        curFaces += nfaces;
    }


	double shiftx = 1.5 * (basePos.col(0).maxCoeff() - basePos.col(0).minCoeff());

	shiftV = upPos;
	shiftV.col(0).setConstant(shiftx);
	shiftV.col(1).setConstant(0);
	shiftV.col(2).setConstant(shiftz);


	Eigen::MatrixXi shiftF = upF;
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
	Eigen::MatrixXd ampColor = mPaint.paintAmplitude(ampVec / globalAmpMax);
	renderColor.block(curVerts, 0, nupverts, 3) = ampColor;

	curVerts += nupverts;
	curFaces += nupfaces;

}

void registerMesh(int frameId)
{
	Eigen::MatrixXd sourceP, tarP, interpP;
	Eigen::MatrixXi sourceF, tarF, interpF;
	Eigen::MatrixXd sourceVec, tarVec, interpVec;
	Eigen::MatrixXd sourceColor, tarColor, interpColor;

	double shiftx = 1.5 * (triV.col(0).maxCoeff() - triV.col(0).minCoeff());
	double shiftz = 1.5 * (triV.col(2).maxCoeff() - triV.col(2).minCoeff());
	int totalfames = ampFieldsList.size();
	registerMeshByPart(triV, triF, upsampledTriV, upsampledTriF, 0, globalAmpMax, ampFieldsList[0], phaseFieldsList[0], &sourceVertexOmegaFields, sourceP, sourceF, sourceVec, sourceColor);
	registerMeshByPart(triV, triF, upsampledTriV, upsampledTriF, shiftz, globalAmpMax, ampFieldsList[totalfames - 1], phaseFieldsList[totalfames - 1], &tarVertexOmegaFields, tarP, tarF, tarVec, tarColor);
	registerMeshByPart(triV, triF, upsampledTriV, upsampledTriF, 2 * shiftz, globalAmpMax, ampFieldsList[frameId], phaseFieldsList[frameId], &vertexOmegaList[frameId], interpP, interpF, interpVec, interpColor);

	
	Eigen::MatrixXi shifF = sourceF;

	int nPartVerts = sourceP.rows();
	int nPartFaces = sourceF.rows();

	dataV.setZero(3 * nPartVerts, 3);
	curColor.setZero(3 * nPartVerts, 3);
	dataVec.setZero(3 * nPartVerts, 3);
	dataF.setZero(3 * nPartFaces, 3);

	shifF.setConstant(nPartVerts);

	dataV.block(0, 0, nPartVerts, 3) = sourceP;
	dataVec.block(0, 0, nPartVerts, 3) = sourceVec;
	curColor.block(0, 0, nPartVerts, 3) = sourceColor;
	dataF.block(0, 0, nPartFaces, 3) = sourceF;

	dataV.block(nPartVerts, 0, nPartVerts, 3) = tarP;
	dataVec.block(nPartVerts, 0, nPartVerts, 3) = tarVec;
	curColor.block(nPartVerts, 0, nPartVerts, 3) = tarColor;
	dataF.block(nPartFaces, 0, nPartFaces, 3) = tarF + shifF;

	dataV.block(nPartVerts * 2, 0, nPartVerts, 3) = interpP;
	dataVec.block(nPartVerts * 2, 0, nPartVerts, 3) = interpVec;
	curColor.block(nPartVerts * 2, 0, nPartVerts, 3) = interpColor;
	dataF.block(nPartFaces * 2, 0, nPartFaces, 3) = interpF + 2 * shifF;

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


void callback() {
	ImGui::PushItemWidth(100);
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

	if (ImGui::CollapsingHeader("source Vector Fields Info", ImGuiTreeNodeFlags_DefaultOpen))
	{
        if (ImGui::Combo("source direction", (int*)&sourceDir, "PV1\0PV2\0Load From File\0"))
        {
            if(sourceDir == LOADFROMFILE)
            {
                std::string filename = igl::file_dialog_open();
                loadEdgeOmega(filename, triMesh.nEdges(), sourceOmegaFields);
                sourceVertexOmegaFields = intrinsicHalfEdgeVec2VertexVec(sourceOmegaFields, triV, triMesh);
            }
        }
		if (ImGui::InputInt("num source waves", &numSourceWaves))
		{
			if (numSourceWaves < 0)
				numSourceWaves = 2;
		}

	}
	if (ImGui::CollapsingHeader("target Vector Fields Info", ImGuiTreeNodeFlags_DefaultOpen))
	{
        if (ImGui::Combo("target direction", (int*)&tarDir, "PV1\0PV2\0Load From File\0"))
        {
            if(tarDir == LOADFROMFILE)
            {
                std::string filename = igl::file_dialog_open();
                loadEdgeOmega(filename, triMesh.nEdges(), tarOmegaFields);
                tarVertexOmegaFields = intrinsicHalfEdgeVec2VertexVec(tarOmegaFields, triV, triMesh);
            }
        }
		if (ImGui::InputInt("num target waves", &numTarWaves))
		{
			if (numTarWaves < 0)
				numTarWaves = 2;
		}
	}

	if (ImGui::CollapsingHeader("Frame Visualization Options", ImGuiTreeNodeFlags_DefaultOpen))
	{
		if (ImGui::InputDouble("drag speed", &dragSpeed))
		{
			if (dragSpeed <= 0)
				dragSpeed = 0.5;
		}
		if (ImGui::DragInt("current frame", &curFrame, dragSpeed, 0, numFrames + 1))
		{
			if(curFrame >= 0 && curFrame <= numFrames + 1)
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
        if (ImGui::InputInt("comb times", &numComb))
        {
            if (numComb < 0)
                numComb = 0;
        }
		if (ImGui::Combo("initialization types", (int*)&initializationType, "Random\0Linear\0Theoretical\0")) {}
		if (ImGui::Combo("Solver types", (int*)&solverType, "Newton\0L-BFGS\0Composite\0")) {}

	}
	

	ImGui::Checkbox("Try Optimization", &isForceOptimize);

	if (ImGui::Button("comb fields & updates", ImVec2(-1, 0)))
	{
		Eigen::MatrixXd PD1, PD2;
		Eigen::VectorXd PV1, PV2;
		igl::principal_curvature(triV, triF, PD1, PD2, PV1, PV2);

        for(int i = 0; i < numComb; i++)
        {
            combField(triF, PD1, PD1);
            combField(triF, PD2, PD2);
        }

        if(sourceDir == DirectionType::DIRPV1)
        {
            sourceOmegaFields = vertexVec2IntrinsicHalfEdgeVec(PD1, triV, triMesh);
            sourceVertexOmegaFields = PD1;
            sourceOmegaFields *= 2 * M_PI * numSourceWaves;
            sourceVertexOmegaFields *= 2 * M_PI * numSourceWaves;
        }

        else if(sourceDir == DirectionType::DIRPV2)
        {
            sourceOmegaFields = vertexVec2IntrinsicHalfEdgeVec(PD2, triV, triMesh);
            sourceVertexOmegaFields = PD2;
            sourceOmegaFields *= 2 * M_PI * numSourceWaves;
            sourceVertexOmegaFields *= 2 * M_PI * numSourceWaves;
        }


        if(tarDir == DirectionType::DIRPV1)
        {
            tarOmegaFields = vertexVec2IntrinsicHalfEdgeVec(PD1, triV, triMesh);
            tarVertexOmegaFields = PD1;
            tarOmegaFields *= 2 * M_PI * numTarWaves;
            tarVertexOmegaFields *= 2 * M_PI * numTarWaves;
        }

        else if(tarDir == DirectionType::DIRPV2)
        {
            tarOmegaFields = vertexVec2IntrinsicHalfEdgeVec(PD2, triV, triMesh);
            tarVertexOmegaFields = PD2;
            tarOmegaFields *= 2 * M_PI * numTarWaves;
            tarVertexOmegaFields *= 2 * M_PI * numTarWaves;
        }

		Eigen::VectorXd faceArea;
		Eigen::MatrixXd cotEntries;
		igl::doublearea(triV, triF, faceArea);
		faceArea /= 2;
		igl::cotmatrix_entries(triV, triF, cotEntries);
		int nverts = triV.rows();

		IntrinsicFormula::roundVertexZvalsFromHalfEdgeOmega(triMesh, sourceOmegaFields, faceArea, cotEntries, nverts, sourceZvals);
		IntrinsicFormula::roundVertexZvalsFromHalfEdgeOmega(triMesh, tarOmegaFields, faceArea, cotEntries, nverts, tarZvals);

		// solve for the path from source to target
		solveKeyFrames(sourceOmegaFields, tarOmegaFields, sourceZvals, tarZvals, numFrames, omegaList, zList);
		// get interploated amp and phase frames
		updateMagnitudePhase(omegaList, zList, ampFieldsList, phaseFieldsList);
        vertexOmegaList.resize(omegaList.size());
        for(int i = 0; i < omegaList.size(); i++)
        {
            vertexOmegaList[i] = intrinsicHalfEdgeVec2VertexVec(omegaList[i], triV, triMesh);
        }
		updateFieldsInView(curFrame);
	}

	ImGui::PopItemWidth();
}




int main(int argc, char** argv)
{
	std::string meshPath;
	if (argc < 2)
	{
		meshPath = "../../../data/bunny_lowres.obj";
	}
	else
		meshPath = argv[1];

	if (!igl::readOBJ(meshPath, triV, triF))
	{
		std::cerr << "mesh loading failed" << std::endl;
		exit(1);
	}
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

	Eigen::MatrixXd U;
	Eigen::MatrixXi G;
	//Eigen::VectorXi J;
	//igl::decimate(triV, triF, 1000, U, G, J);
	//igl::writeOBJ("test.obj", U, G);
    // igl::cylinder(40, 12, U, G);
    // U.col(0) *= 0.5;
    // U.col(1) *= 0.5;
    // U.col(2) *= 2;
    // igl::writeOBJ("test.obj", U, G);


    // Add the callback
    polyscope::state::userCallback = callback;

    polyscope::options::groundPlaneHeightFactor = 0.25; // adjust the plane height
    // Show the gui
    polyscope::show();

    return 0;
}