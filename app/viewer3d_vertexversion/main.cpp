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
#include <igl/colormap.h>
#include <igl/cylinder.h>
#include <igl/principal_curvature.h>
#include <filesystem>
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
#include "../../include/IntrinsicFormula/ComputeZdotFromHalfEdgeOmega.h"
#include "../../include/IntrinsicFormula/IntrinsicKeyFrameInterpolationFromHalfEdge.h"
#include "../../include/IntrinsicFormula/KnoppelStripePattern.h"
#include "../../include/EuclideanFormula/KeyFrameInterpolation.h"
#include "../../include/EuclideanFormula/ComputeZdotFromEuclideanOmega.h"
#include "../../include/EuclideanFormula/InterpolateZvalsFromEuclideanOmega.h"

#include <igl/cylinder.h>



Eigen::MatrixXd triV, upsampledTriV;
Eigen::MatrixXi triF, upsampledTriF;
MeshConnectivity triMesh, upsampledTriMesh;
std::vector<std::pair<int, Eigen::Vector3d>> bary;

Eigen::MatrixXd sourceVertexOmegaFields, tarVertexOmegaFields;


std::vector<std::complex<double>> sourceZvals;
std::vector<std::complex<double>> tarZvals;

std::vector<Eigen::MatrixXd> vertexOmegaList;
std::vector<std::vector<std::complex<double>>> zList;


std::vector<Eigen::VectorXd> phaseFieldsList;
std::vector<Eigen::VectorXd> ampFieldsList;

// baselines
std::vector<Eigen::VectorXd> KnoppelPhaseFieldsList;
std::vector<Eigen::VectorXd> linearPhaseFieldsList;
std::vector<Eigen::VectorXd> linearAmpFieldsList;


Eigen::MatrixXd dataV;
Eigen::MatrixXi dataF;
Eigen::MatrixXd dataVec;
Eigen::MatrixXd curColor;

int loopLevel = 2;

bool isForceOptimize = false;
bool isShowVectorFields = true;
bool isShowWrinkels = true;
bool isShowComparison = false;

PaintGeometry mPaint;

int numFrames = 50;
int curFrame = 0;

int numSourceWaves = 2;
int numTarWaves = 2;

double globalAmpMax = 1;
double globalAmpMin = 0;

double dragSpeed = 0.5;

float vecratio = 0.001;

double gradTol = 1e-6;
double xTol = 0;
double fTol = 0;
int numIter = 1000;
int quadOrder = 4;
int numComb = 2;
double wrinkleAmpScalingRatio = 0.0;

std::string workingFolder;
EuclideanFormula::KeyFrameInterpolation interpModel;

enum InitializationType {
	Random = 0,
	Linear = 1,
	Knoppel = 2
};
enum DirectionType
{
	DIRPV1 = 0,
	DIRPV2 = 1
};
enum OptSolverType
{
	Newton = 0,
	LBFGS = 1,
	Composite = 2,	// use lbfgs to get a warm start
	MultilevelNewton = 3 // use multilevel newton solver on the time
};

InitializationType initializationType = InitializationType::Linear;
DirectionType sourceDir = DIRPV1;
DirectionType tarDir = DIRPV2;
OptSolverType solverType = Newton;

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
	interpModel = EuclideanFormula::KeyFrameInterpolation(triV, MeshConnectivity(triF), faceArea, numFrames, quadOrder, sourceZvals, sourceVec, tarZvals, tarVec);
	Eigen::VectorXd x;
	interpModel.convertList2Variable(x);        // linear initialization
	if (initializationType == InitializationType::Random)
	{
		x.setRandom();
		interpModel.convertVariable2List(x);
		interpModel.convertList2Variable(x);
	}
	else if (initializationType == InitializationType::Knoppel)
	{
		double dt = 1.0 / (numFrames + 1);

		std::vector<Eigen::MatrixXd> wList;
		std::vector<std::vector<std::complex<double>>> zList;

		wList.resize(numFrames + 2);
		zList.resize(numFrames + 2);

		wList[0] = sourceVec;
		wList[numFrames + 1] = tarVec;

		zList[0] = sourceZvals;
		zList[numFrames + 1] = tarZvals;

		Eigen::VectorXd faceArea;
		Eigen::MatrixXd cotEntries;
		igl::doublearea(triV, triF, faceArea);
		faceArea /= 2;
		igl::cotmatrix_entries(triV, triF, cotEntries);
		int nverts = triV.rows();

		// linear interpolate in between
		for (int i = 1; i <= numFrames; i++)
		{
			double t = dt * i;
			wList[i] = (1 - t) * sourceVec + t * tarVec;
			Eigen::MatrixXd edgeW = vertexVec2IntrinsicHalfEdgeVec(wList[i], triV, triMesh);

			IntrinsicFormula::roundVertexZvalsFromHalfEdgeOmega(triMesh, edgeW, faceArea, cotEntries, nverts, zList[i]);

			Eigen::VectorXd curAmp;
			ampSolver(triV, triMesh, edgeW / 10, curAmp);

			for (int j = 0; j < triV.rows(); j++)
			{
				double curNorm = std::abs(zList[i][j]);
				if (curNorm)
					zList[i][j] = curAmp(j) / curNorm * zList[i][j];
			}
		}
		interpModel.setwzLists(zList, wList);
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
		if (solverType == Newton)
			OptSolver::newtonSolver(funVal, maxStep, x, numIter, gradTol, xTol, fTol, true, getVecNorm, &workingFolder);
		else if (solverType == LBFGS)
			OptSolver::lbfgsSolver(funVal, maxStep, x, numIter, gradTol, xTol, fTol, true, getVecNorm);
		else if (solverType == Composite)
		{
			OptSolver::lbfgsSolver(funVal, maxStep, x, 1000, 1e-4, 1e-5, 1e-5, true, getVecNorm);
			OptSolver::newtonSolver(funVal, maxStep, x, numIter, gradTol, xTol, fTol, true, getVecNorm, &workingFolder);
		}
		else if (solverType == MultilevelNewton)    // we use 4-level approach
		{
			int N = numKeyFrames + 2;
			int base = std::ceil(std::exp(std::log(N) / 2));
			std::cout << "base: " << base << std::endl;
			std::vector<std::vector<std::complex<double>>> zvalsList(2);
			zvalsList[0] = sourceZvals;
			zvalsList[1] = tarZvals;
			std::vector<Eigen::MatrixXd> omegaList(2);
			omegaList[0] = sourceVertexOmegaFields;
			omegaList[1] = tarVertexOmegaFields;

			for (int i = 1; i <= 2; i++)
			{
				interpModel = EuclideanFormula::KeyFrameInterpolation(triV, MeshConnectivity(triF), faceArea, base - 1, quadOrder, zvalsList, omegaList);
				interpModel.convertList2Variable(x);
				std::cout << "level: " << i << ", x size: " << x.size() << std::endl;
				OptSolver::newtonSolver(funVal, maxStep, x, numIter, gradTol, xTol, fTol, true, getVecNorm, &workingFolder);
				zvalsList = interpModel.getVertValsList();
				omegaList = interpModel.getWList();
			}

		}
		std::cout << "before optimization: " << x0.norm() << ", after optimization: " << x.norm() << std::endl;
	}
	interpModel.convertVariable2List(x);

	wFrames = interpModel.getWList();
	zFrames = interpModel.getVertValsList();

	for (int i = 0; i < wFrames.size() - 1; i++)
	{
		double zdotNorm = interpModel._zdotModel.computeZdotIntegration(zFrames[i], wFrames[i], zFrames[i + 1], wFrames[i + 1], NULL, NULL);

		if (solverType != MultilevelNewton)
		{
			double initZdotNorm = interpModel._zdotModel.computeZdotIntegration(initZFrames[i], initWFrames[i], initZFrames[i + 1], initWFrames[i + 1], NULL, NULL);

			std::cout << "frame " << i << ", before optimization: ||zdot||^2: " << initZdotNorm << ", after optimization, ||zdot||^2 = " << zdotNorm << std::endl;
		}
		else
			std::cout << "frame " << i << ", after optimization, ||zdot||^2 = " << zdotNorm << std::endl;

	}
	if (isForceOptimize)
		interpModel.save(workingFolder + "/data.json", triV, triF);


}

void updateMagnitudePhase(const std::vector<Eigen::MatrixXd>& wFrames, const std::vector<std::vector<std::complex<double>>>& zFrames, std::vector<Eigen::VectorXd>& magList, std::vector<Eigen::VectorXd>& phaseList)
{
	std::vector<std::vector<std::complex<double>>> interpZList(wFrames.size());
	magList.resize(wFrames.size());
	phaseList.resize(wFrames.size());

	auto computeMagPhase = [&](const tbb::blocked_range<uint32_t>& range) {
		for (uint32_t i = range.begin(); i < range.end(); ++i)
		{
			interpZList[i] = EuclideanFormula::upsamplingZvals(triV, triMesh, zFrames[i], wFrames[i], bary);
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
	const Eigen::MatrixXd& upPos, const Eigen::MatrixXi& upF, const double& shiftz, const double& ampMin, const double& ampMax,
	Eigen::VectorXd ampVec, const Eigen::VectorXd& phaseVec, Eigen::MatrixXd* omegaVec,
	Eigen::MatrixXd& renderV, Eigen::MatrixXi& renderF, Eigen::MatrixXd& renderVec, Eigen::MatrixXd& renderColor)
{
	int nverts = basePos.rows();
	int nfaces = baseF.rows();

	int nupverts = upPos.rows();
	int nupfaces = upF.rows();

	int ndataVerts = nverts + 2 * nupverts;
	int ndataFaces = nfaces + 2 * nupfaces;

	if (!isShowVectorFields)
	{
		ndataVerts = 2 * nupverts;
		ndataFaces = 2 * nupfaces;
	}
	if (isShowWrinkels)
	{
		ndataVerts += nupverts;
		ndataFaces += nupfaces;
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

	Eigen::MatrixXd shiftV = basePos;
	shiftV.col(0).setConstant(0);
	shiftV.col(1).setConstant(0);
	shiftV.col(2).setConstant(shiftz);

	if (isShowVectorFields)
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
	Eigen::VectorXd normoalizedAmpVec = ampVec;
	for (int i = 0; i < normoalizedAmpVec.rows(); i++)
	{
		normoalizedAmpVec(i) = (ampVec(i) - ampMin) / ampMax;
	}
	Eigen::MatrixXd ampColor = mPaint.paintAmplitude(normoalizedAmpVec);
	renderColor.block(curVerts, 0, nupverts, 3) = ampColor;

	curVerts += nupverts;
	curFaces += nupfaces;

	// interpolated amp
	if (isShowWrinkels)
	{
		shiftF.setConstant(curVerts);
		shiftV.col(0).setConstant(3 * shiftx);
		Eigen::MatrixXd tmpV = upPos - shiftV;
		Eigen::MatrixXd tmpN;
		igl::per_vertex_normals(tmpV, upF, tmpN);

		Eigen::VectorXd ampCosVec(nupverts);

		for (int i = 0; i < nupverts; i++)
		{
			renderV.row(curVerts + i) = tmpV.row(i) + wrinkleAmpScalingRatio * ampVec(i) * std::cos(phaseVec(i)) * tmpN.row(i);
			ampCosVec(i) = normoalizedAmpVec(i) * std::cos(phaseVec(i));
		}
		renderF.block(curFaces, 0, nupfaces, 3) = upF + shiftF;

		mPaint.setNormalization(false);
		Eigen::MatrixXd ampCosColor = mPaint.paintAmplitude(ampCosVec);
		renderColor.block(curVerts, 0, nupverts, 3) = ampCosColor;
		//    mPaint.setNormalization(false);
//    Eigen::RowVector3d rowcolor;
//
//    igl::colormap(igl::COLOR_MAP_TYPE_VIRIDIS, 4.0 / 9.0, rowcolor.data());
//    for(int i = 0; i < nupverts; i++)
//    {
//        renderColor.row(curVerts + i) = rowcolor;
//    }
		curVerts += nupverts;
		curFaces += nupfaces;
	}




}

void registerComparisonMeshByPart(Eigen::MatrixXd& upPos, Eigen::MatrixXi& upF, const double& shiftz, const double& ampMin, const double& ampMax, Eigen::VectorXd ampVec, const Eigen::VectorXd& phaseVec, Eigen::MatrixXd& renderV, Eigen::MatrixXi& renderF, Eigen::MatrixXd& renderColor)
{
	int nupverts = upPos.rows();
	int nupfaces = upF.rows();

	int ndataVerts = 3 * nupverts;
	int ndataFaces = 3 * nupfaces;

	renderV.resize(ndataVerts, 3);
	renderF.resize(ndataFaces, 3);
	renderColor.setZero(ndataVerts, 3);

	renderColor.col(0).setConstant(1.0);
	renderColor.col(1).setConstant(1.0);
	renderColor.col(2).setConstant(1.0);

	int curVerts = 0;
	int curFaces = 0;

	Eigen::MatrixXd shiftV = upPos;
	shiftV.col(0).setConstant(0);
	shiftV.col(1).setConstant(0);
	shiftV.col(2).setConstant(shiftz);

	double shiftx = 1.5 * (upPos.col(0).maxCoeff() - upPos.col(0).minCoeff());

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
	Eigen::VectorXd normoalizedAmpVec = ampVec;
	for (int i = 0; i < normoalizedAmpVec.rows(); i++)
	{
		normoalizedAmpVec(i) = (ampVec(i) - ampMin) / (ampMax - ampMin);
	}
	Eigen::MatrixXd ampColor = mPaint.paintAmplitude(normoalizedAmpVec);
	renderColor.block(curVerts, 0, nupverts, 3) = ampColor;

	curVerts += nupverts;
	curFaces += nupfaces;

	// interpolated amp
	shiftF.setConstant(curVerts);
	shiftV.col(0).setConstant(3 * shiftx);
	Eigen::MatrixXd tmpV = upPos - shiftV;
	Eigen::MatrixXd tmpN;
	igl::per_vertex_normals(tmpV, upF, tmpN);

	Eigen::VectorXd ampCosVec(nupverts);

	for (int i = 0; i < nupverts; i++)
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

void registerMesh(int frameId)
{
	if (!isShowComparison)
	{
		Eigen::MatrixXd sourceP, tarP, interpP;
		Eigen::MatrixXi sourceF, tarF, interpF;
		Eigen::MatrixXd sourceVec, tarVec, interpVec;
		Eigen::MatrixXd sourceColor, tarColor, interpColor;

		double shiftz = 1.5 * (triV.col(2).maxCoeff() - triV.col(2).minCoeff());
		int totalfames = ampFieldsList.size();
		registerMeshByPart(triV, triF, upsampledTriV, upsampledTriF, 0, globalAmpMin, globalAmpMax, ampFieldsList[0],
			phaseFieldsList[0], &sourceVertexOmegaFields, sourceP, sourceF, sourceVec, sourceColor);
		registerMeshByPart(triV, triF, upsampledTriV, upsampledTriF, shiftz, globalAmpMin, globalAmpMax,
			ampFieldsList[totalfames - 1], phaseFieldsList[totalfames - 1], &tarVertexOmegaFields, tarP,
			tarF, tarVec, tarColor);
		registerMeshByPart(triV, triF, upsampledTriV, upsampledTriF, 2 * shiftz, globalAmpMin, globalAmpMax, ampFieldsList[frameId],
			phaseFieldsList[frameId], &vertexOmegaList[frameId], interpP, interpF, interpVec,
			interpColor);


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
	else
	{
		Eigen::MatrixXd ourP, KnoppelP, linearP;
		Eigen::MatrixXi ourF, KnoppelF, linearF;
		Eigen::MatrixXd ourColor, KnoppelColor, linearColor;

		double shiftz = 1.5 * (triV.col(2).maxCoeff() - triV.col(2).minCoeff());
		Eigen::VectorXd KnoppelAmp = ampFieldsList[curFrame];
		KnoppelAmp.setConstant(globalAmpMax);
		registerComparisonMeshByPart(upsampledTriV, upsampledTriF, 0, globalAmpMin, globalAmpMax, ampFieldsList[curFrame], phaseFieldsList[curFrame], ourP, ourF, ourColor);
		registerComparisonMeshByPart(upsampledTriV, upsampledTriF, shiftz, globalAmpMin, globalAmpMax, linearAmpFieldsList[curFrame], linearPhaseFieldsList[curFrame], linearP, linearF, linearColor);
		registerComparisonMeshByPart(upsampledTriV, upsampledTriF, 2 * shiftz, globalAmpMin, globalAmpMax, KnoppelAmp, KnoppelPhaseFieldsList[curFrame], KnoppelP, KnoppelF, KnoppelColor);

		Eigen::MatrixXi shifF = ourF;

		int nPartVerts = ourP.rows();
		int nPartFaces = ourF.rows();

		dataV.setZero(3 * nPartVerts, 3);
		curColor.setZero(3 * nPartVerts, 3);
		dataVec.setZero(3 * nPartVerts, 3);
		dataF.setZero(3 * nPartFaces, 3);

		shifF.setConstant(nPartVerts);

		dataV.block(0, 0, nPartVerts, 3) = ourP;
		curColor.block(0, 0, nPartVerts, 3) = ourColor;
		dataF.block(0, 0, nPartFaces, 3) = ourF;

		dataV.block(nPartVerts, 0, nPartVerts, 3) = linearP;
		curColor.block(nPartVerts, 0, nPartVerts, 3) = linearColor;
		dataF.block(nPartFaces, 0, nPartFaces, 3) = linearF + shifF;

		dataV.block(nPartVerts * 2, 0, nPartVerts, 3) = KnoppelP;
		curColor.block(nPartVerts * 2, 0, nPartVerts, 3) = KnoppelColor;
		dataF.block(nPartFaces * 2, 0, nPartFaces, 3) = KnoppelF + 2 * shifF;

		polyscope::registerSurfaceMesh("input mesh", dataV, dataF);
	}

}

void updateFieldsInView(int frameId)
{
	registerMesh(frameId);
	polyscope::getSurfaceMesh("input mesh")->addVertexColorQuantity("VertexColor", curColor);
	polyscope::getSurfaceMesh("input mesh")->getQuantity("VertexColor")->setEnabled(true);

	if (!isShowComparison)
	{
		polyscope::getSurfaceMesh("input mesh")->addVertexVectorQuantity("vertex vector field", dataVec * vecratio, polyscope::VectorType::AMBIENT);
		polyscope::getSurfaceMesh("input mesh")->getQuantity("vertex vector field")->setEnabled(true);
	}

}


void callback() {
	ImGui::PushItemWidth(100);
	float w = ImGui::GetContentRegionAvailWidth();
	float p = ImGui::GetStyle().FramePadding.x;
	if (ImGui::Button("Load", ImVec2(ImVec2((w - p) / 2.f, 0))))
	{
		std::string loadJson = igl::file_dialog_open();
		if (interpModel.load(loadJson, triV, triF))
		{

			initialization();
			vertexOmegaList = interpModel.getWList();
			zList = interpModel.getVertValsList();

			numFrames = vertexOmegaList.size() - 2;
			updateMagnitudePhase(vertexOmegaList, zList, ampFieldsList, phaseFieldsList);

			globalAmpMax = ampFieldsList[0].maxCoeff();
			globalAmpMin = ampFieldsList[0].minCoeff();
			for (int i = 1; i < ampFieldsList.size(); i++)
			{
				globalAmpMax = std::max(globalAmpMax, ampFieldsList[i].maxCoeff());
				globalAmpMin = std::min(globalAmpMin, ampFieldsList[i].minCoeff());
			}
			sourceVertexOmegaFields = vertexOmegaList[0];
			tarVertexOmegaFields = vertexOmegaList[vertexOmegaList.size() - 1];

			sourceZvals = zList[0];
			tarZvals = zList[vertexOmegaList.size() - 1];

			KnoppelPhaseFieldsList.resize(ampFieldsList.size());
			Eigen::VectorXd faceArea;
			Eigen::MatrixXd cotEntries;
			igl::doublearea(triV, triF, faceArea);
			faceArea /= 2;
			igl::cotmatrix_entries(triV, triF, cotEntries);
			int nverts = triV.rows();

			Eigen::MatrixXd sourceEdgeW = vertexVec2IntrinsicHalfEdgeVec(sourceVertexOmegaFields, triV, triMesh);
			Eigen::MatrixXd tarEdgeW = vertexVec2IntrinsicHalfEdgeVec(tarVertexOmegaFields, triV, triMesh);

			for (int i = 0; i < ampFieldsList.size(); i++)
			{
				double t = 1.0 / (ampFieldsList.size() - 1) * i;
				Eigen::MatrixXd interpVecs = (1 - t) * sourceVertexOmegaFields + t * tarVertexOmegaFields;
				std::vector<std::complex<double>> interpZvals;

				Eigen::MatrixXd halfEdgeW = vertexVec2IntrinsicHalfEdgeVec(interpVecs, triV, triMesh);
				//Eigen::MatrixXd halfEdgeW = (1 - t) * sourceEdgeW + t * tarEdgeW;
				IntrinsicFormula::roundVertexZvalsFromHalfEdgeOmega(triMesh, halfEdgeW, faceArea, cotEntries, nverts, interpZvals);
				Eigen::VectorXd upTheta;
				IntrinsicFormula::getUpsamplingTheta(triMesh, halfEdgeW, interpZvals, bary, upTheta);
				KnoppelPhaseFieldsList[i] = upTheta;
			}

			// linear baseline
			auto tmpModel = IntrinsicFormula::IntrinsicKeyFrameInterpolationFromHalfEdge(MeshConnectivity(triF), faceArea, (ampFieldsList.size() - 2), quadOrder, sourceZvals, sourceVertexOmegaFields, tarZvals, tarVertexOmegaFields);
			updateMagnitudePhase(tmpModel.getWList(), tmpModel.getVertValsList(), linearAmpFieldsList, linearPhaseFieldsList);

			updateFieldsInView(curFrame);
		}
	}
	ImGui::SameLine(0, p);
	if (ImGui::Button("Save", ImVec2((w - p) / 2.f, 0)))
	{
		std::string saveFolder = igl::file_dialog_save();
		interpModel.save(saveFolder, triV, triF);
	}
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
				updateMagnitudePhase(vertexOmegaList, zList, ampFieldsList, phaseFieldsList);
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
	if (ImGui::Checkbox("is show comparison", &isShowComparison))
	{
		updateFieldsInView(curFrame);
	}
	if (ImGui::InputDouble("wrinkle amp scaling ratio", &wrinkleAmpScalingRatio))
	{
		if (wrinkleAmpScalingRatio >= 0)
			updateFieldsInView(curFrame);
	}

	if (ImGui::CollapsingHeader("source Vector Fields Info", ImGuiTreeNodeFlags_DefaultOpen))
	{
		if (ImGui::Combo("source direction", (int*)&sourceDir, "PV1\0PV2\0"))
		{
		}
		if (ImGui::InputInt("num source waves", &numSourceWaves))
		{
			if (numSourceWaves < 0)
				numSourceWaves = 2;
		}

	}
	if (ImGui::CollapsingHeader("target Vector Fields Info", ImGuiTreeNodeFlags_DefaultOpen))
	{
		if (ImGui::Combo("target direction", (int*)&tarDir, "PV1\0PV2\0"))
		{
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
			if (curFrame >= 0 && curFrame <= numFrames + 1)
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
		if (ImGui::Combo("initialization types", (int*)&initializationType, "Random\0Linear\0Knoppel\0")) {}
		if (ImGui::Combo("Solver types", (int*)&solverType, "Newton\0L-BFGS\0Composite\0Multilevel-Newton\0")) {}

	}


	ImGui::Checkbox("Try Optimization", &isForceOptimize);

	if (ImGui::Button("comb fields & updates", ImVec2(-1, 0)))
	{
		Eigen::MatrixXd PD1, PD2;
		Eigen::VectorXd PV1, PV2;
		igl::principal_curvature(triV, triF, PD1, PD2, PV1, PV2);

		for (int i = 0; i < numComb; i++)
		{
			combField(triF, PD1, PD1);
			combField(triF, PD2, PD2);
		}

		if (sourceDir == DirectionType::DIRPV1)
		{
			sourceVertexOmegaFields = PD1;
			sourceVertexOmegaFields *= 2 * M_PI * numSourceWaves;
		}

		else if (sourceDir == DirectionType::DIRPV2)
		{
			sourceVertexOmegaFields = PD2;
			sourceVertexOmegaFields *= 2 * M_PI * numSourceWaves;
		}


		if (tarDir == DirectionType::DIRPV1)
		{
			tarVertexOmegaFields = PD1;
			tarVertexOmegaFields *= 2 * M_PI * numTarWaves;
		}

		else if (tarDir == DirectionType::DIRPV2)
		{
			tarVertexOmegaFields = PD2;
			tarVertexOmegaFields *= 2 * M_PI * numTarWaves;
		}
		
		Eigen::VectorXd faceArea;
		Eigen::MatrixXd cotEntries;
		igl::doublearea(triV, triF, faceArea);
		faceArea /= 2;
		igl::cotmatrix_entries(triV, triF, cotEntries);
		int nverts = triV.rows();

		Eigen::MatrixXd sourceEdgeFields = vertexVec2IntrinsicHalfEdgeVec(sourceVertexOmegaFields, triV, triMesh);
		Eigen::MatrixXd tarEdgeFields = vertexVec2IntrinsicHalfEdgeVec(tarVertexOmegaFields, triV, triMesh);

		IntrinsicFormula::roundVertexZvalsFromHalfEdgeOmega(triMesh, sourceEdgeFields, faceArea, cotEntries, nverts, sourceZvals);
		IntrinsicFormula::roundVertexZvalsFromHalfEdgeOmega(triMesh, tarEdgeFields, faceArea, cotEntries, nverts, tarZvals);

		Eigen::VectorXd sourceAmp, tarAmp;
		ampSolver(triV, triMesh, sourceEdgeFields / 10, sourceAmp);
		ampSolver(triV, triMesh, tarEdgeFields / 10, tarAmp);

		for (int i = 0; i < triV.rows(); i++)
		{
			double sourceNorm = std::abs(sourceZvals[i]);
			double tarNorm = std::abs(tarZvals[i]);
			if (sourceNorm)
				sourceZvals[i] = sourceAmp(i) / sourceNorm * sourceZvals[i];
			if (tarNorm)
				tarZvals[i] = tarAmp(i) / tarNorm * tarZvals[i];
		}

		// solve for the path from source to target
		solveKeyFrames(sourceVertexOmegaFields, tarVertexOmegaFields, sourceZvals, tarZvals, numFrames, vertexOmegaList, zList);
		// get interploated amp and phase frames
		updateMagnitudePhase(vertexOmegaList, zList, ampFieldsList, phaseFieldsList);
		
		numFrames = vertexOmegaList.size() - 2;

		// update global maximum amplitude
		globalAmpMax = ampFieldsList[0].maxCoeff();
		globalAmpMin = ampFieldsList[0].minCoeff();
		for (int i = 1; i < ampFieldsList.size(); i++)
		{
			globalAmpMax = std::max(globalAmpMax, ampFieldsList[i].maxCoeff());
			globalAmpMin = std::min(globalAmpMin, ampFieldsList[i].minCoeff());
		}

		KnoppelPhaseFieldsList.resize(ampFieldsList.size());
		for (int i = 0; i < ampFieldsList.size(); i++)
		{
			double t = 1.0 / (ampFieldsList.size() - 1) * i;
			Eigen::MatrixXd interpVecs = (1 - t) * sourceVertexOmegaFields + t * tarVertexOmegaFields;
			Eigen::MatrixXd edgew = vertexVec2IntrinsicHalfEdgeVec(interpVecs, triV, triMesh);
			//edgew = (1 - t) * sourceEdgeFields + t * tarEdgeFields;
			std::vector<std::complex<double>> interpZvals;
			IntrinsicFormula::roundVertexZvalsFromHalfEdgeOmega(triMesh, edgew, faceArea, cotEntries, nverts, interpZvals);
			Eigen::VectorXd upTheta;
			IntrinsicFormula::getUpsamplingTheta(triMesh, edgew, interpZvals, bary, upTheta);
			KnoppelPhaseFieldsList[i] = upTheta;
		}

		// linear baseline
		auto tmpModel = EuclideanFormula::KeyFrameInterpolation(triV, MeshConnectivity(triF), faceArea, (ampFieldsList.size() - 2), quadOrder, sourceZvals, sourceVertexOmegaFields, tarZvals, tarVertexOmegaFields);
		updateMagnitudePhase(tmpModel.getWList(), tmpModel.getVertValsList(), linearAmpFieldsList, linearPhaseFieldsList);
		updateFieldsInView(curFrame);
	}

	if (ImGui::Button("output images", ImVec2(-1, 0)))
	{
		for (curFrame = 0; curFrame < ampFieldsList.size(); curFrame++)
		{
			updateFieldsInView(curFrame);
			polyscope::options::screenshotExtension = ".jpg";
			polyscope::screenshot();
		}
	}
	if (ImGui::Button("test functions", ImVec2(-1, 0)))
	{
		Eigen::VectorXd faceArea;
		igl::doublearea(triV, triF, faceArea);
		faceArea /= 2;

		Eigen::MatrixXd sourceVec = triV;
		sourceVec.setRandom();
		Eigen::MatrixXd tarVec = triV;
		tarVec.setRandom();

		std::vector<std::complex<double>> sourceZ(triV.rows()), tarZ(triV.rows());
		for (int i = 0; i < triV.rows(); i++)
		{
			Eigen::Vector2d randVec;
			randVec.setRandom();
			sourceZ[i] = std::complex<double>(randVec(0), randVec(1));

			randVec.setRandom();
			tarZ[i] = std::complex<double>(randVec(0), randVec(1));
		}

		auto testModel = EuclideanFormula::KeyFrameInterpolation(triV, MeshConnectivity(triF), faceArea, numFrames, quadOrder, sourceZ, sourceVec, tarZ, tarVec);
		Eigen::VectorXd x;
		testModel.convertList2Variable(x);
		testModel.testEnergy(x);
	}

	ImGui::PopItemWidth();
}




int main(int argc, char** argv)
{
	std::string meshPath;
	if (argc < 2)
	{
		meshPath = "../../../data/teddy/teddy_simulated.obj";
	}
	else
		meshPath = argv[1];
	int id = meshPath.rfind("/");
	workingFolder = meshPath.substr(0, id + 1); // include "/"
	//std::cout << workingFolder << std::endl;

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