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
#include <igl/heat_geodesics.h>
#include <igl/avg_edge_length.h>
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
//#include "../../include/IntrinsicFormula/ComputeZdotFromEdgeOmega.h"
#include "../../include/IntrinsicFormula/ComputeZdotFromHalfEdgeOmega.h"
//#include "../../include/IntrinsicFormula/IntrinsicKeyFrameInterpolationFromEdge.h"
#include "../../include/IntrinsicFormula/IntrinsicKnoppelDrivenFormula.h"
#include "../../include/IntrinsicFormula/IntrinsicKeyFrameInterpolationFromHalfEdge.h"
#include "../../include/IntrinsicFormula/KnoppelStripePattern.h"
#include "../../include/json.hpp"
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

// reference amp and omega
std::vector<Eigen::MatrixXd> refOmegaList;
std::vector<Eigen::VectorXd> refAmpList;


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

int numFrames = 20;
int curFrame = 0;

double sourceFreq = 1.0;
double tarFreq = 1.0;

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
float wrinkleAmpScalingRatio = 0.01;

int numSource = 1;
int numTarSource = 1;

std::string workingFolder;
//IntrinsicFormula::IntrinsicKeyFrameInterpolationFromHalfEdge interpModel;
IntrinsicFormula::IntrinsicKnoppelDrivenFormula interpModel;

enum InitializationType{
    Random = 0,
    Linear = 1,
    Knoppel = 2
};
enum DirectionType
{
    DIRPV1 = 0,
    DIRPV2 = 1,
    LOADFROMFILE = 2,
    GEODESIC = 3,
    GEODESICPERP = 4, 
    ROTATEDVEC = 5
};

InitializationType initializationType = InitializationType::Linear;
DirectionType sourceDir = DIRPV1;
DirectionType tarDir = DIRPV2;


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

void getDisFields(const Eigen::VectorXi& gamma, Eigen::VectorXd& d, Eigen::MatrixXd& disFields, Eigen::MatrixXd& disFieldsPerp)
{
    // compute geodesic
    igl::HeatGeodesicsData<double> data;
    double t = std::pow(igl::avg_edge_length(triV, triF), 2);
    const auto precompute = [&]()
    {
        if (!igl::heat_geodesics_precompute(triV, triF, t, data))
        {
            std::cerr << "Error: heat_geodesics_precompute failed." << std::endl;
            exit(EXIT_FAILURE);
        };
    };
    precompute();
    igl::heat_geodesics_solve(data, gamma, d);
    int nfaces = triMesh.nFaces();


    Eigen::MatrixXd faceN;
    igl::per_face_normals(triV, triF, faceN);
    disFields.setZero(nfaces, 3);
    disFieldsPerp.setZero(nfaces, 3);

    for (int i = 0; i < nfaces; i++)
    {
        Eigen::Vector3d ru = triV.row(triMesh.faceVertex(i, 1)) - triV.row(triMesh.faceVertex(i, 0));
        Eigen::Vector3d rv = triV.row(triMesh.faceVertex(i, 2)) - triV.row(triMesh.faceVertex(i, 0));

        Eigen::Matrix2d g, gInv;
        g << ru.dot(ru), ru.dot(rv), rv.dot(ru), rv.dot(rv);
        gInv = g.inverse();

        double u = d(triMesh.faceVertex(i, 1)) - d(triMesh.faceVertex(i, 0));
        double v = d(triMesh.faceVertex(i, 2)) - d(triMesh.faceVertex(i, 0));
        Eigen::Vector2d dVec(u, v);
        Eigen::Vector2d vec = gInv * dVec;

        Eigen::Vector3d n = faceN.row(i);
        n = n / n.norm();


        Eigen::Vector3d vec3D = vec(0) * ru + vec(1) * rv;             // for the norm adjustment
        Eigen::Vector3d vecPerp3D = n.cross(vec3D);  // for direction adjustment

        vecPerp3D *= 1 / (vecPerp3D.norm() * (d(triF(i, 0)) + d(triF(i, 1)) + d(triF(i, 2))) / 3.0);
        disFields.row(i) = vec3D;
        disFieldsPerp.row(i) = vecPerp3D;
    }
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
    Eigen::MatrixXd cotEntries;
    igl::cotmatrix_entries(triV, triF, cotEntries);

    refOmegaList.resize(numFrames + 2);
    refAmpList.resize(numFrames + 2, Eigen::VectorXd::Zero(sourceZvals.size()));

    double dt = 1.0 / (numFrames + 1);
    for (int i = 0; i < numFrames + 2; i++)
    {
        double t = dt * i;
        refOmegaList[i] = (1 - t) * sourceVec + t * tarVec;

        for (int j = 0; j < sourceZvals.size(); j++)
        {
            refAmpList[i](j) = (1 - t) * std::abs(sourceZvals[j]) + t * std::abs(tarZvals[j]);
        }
    }

    interpModel = IntrinsicFormula::IntrinsicKnoppelDrivenFormula(MeshConnectivity(triF), faceArea, cotEntries, refOmegaList, refAmpList, sourceZvals, tarZvals, refOmegaList[0], refOmegaList[refOmegaList.size() - 1], numFrames, 1.0, quadOrder);

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

        wList = refOmegaList;

        for (int i = 0; i < wList.size(); i++)
        {
            std::vector<std::complex<double>> zvals;
            IntrinsicFormula::roundVertexZvalsFromHalfEdgeOmegaVertexMag(triMesh, refOmegaList[i], refAmpList[i], faceArea, cotEntries, triV.rows(), zvals);
            zList[i] = zvals;
        }

        interpModel.setwzLists(zList, wList);
        interpModel.convertList2Variable(x);
    }
    else
    {
        // do nothing, since it is initialized as the linear interpolation.
    }
    Eigen::VectorXd deriv;
    Eigen::SparseMatrix<double> H;
    double E = interpModel.computeEnergy(x, &deriv, &H, false);

    std::cout << "optimization start!" << std::endl;
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

        auto postProcess = [&](Eigen::VectorXd& x)
        {
//            interpModel.postProcess(x);
        };


        OptSolver::testFuncGradHessian(funVal, x);

        auto x0 = x;
        OptSolver::newtonSolver(funVal, maxStep, x, numIter, gradTol, xTol, fTol, true, getVecNorm, &workingFolder, postProcess);
        std::cout << "before optimization: " << x0.norm() << ", after optimization: " << x.norm() << std::endl;
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
    /*if(isForceOptimize)
        interpModel.save(workingFolder + "/data.json", triV, triF);*/


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

    if(!isShowVectorFields)
    {
        ndataVerts = 2 * nupverts;
        ndataFaces = 2 * nupfaces;
    }
    if(isShowWrinkels)
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
    Eigen::VectorXd normoalizedAmpVec = ampVec;
    for(int i = 0; i < normoalizedAmpVec.rows(); i++)
    {
        normoalizedAmpVec(i) = (ampVec(i) - ampMin) / ampMax;
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
    for(int i = 0; i < normoalizedAmpVec.rows(); i++)
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

void registerMesh(int frameId)
{
    if(!isShowComparison)
    {
        Eigen::MatrixXd sourceP, tarP, interpP;
        Eigen::MatrixXi sourceF, tarF, interpF;
        Eigen::MatrixXd sourceVec, tarVec, interpVec;
        Eigen::MatrixXd sourceColor, tarColor, interpColor;

        double shiftz = 1.5 * (triV.col(2).maxCoeff() - triV.col(2).minCoeff());
        int totalfames = ampFieldsList.size();
        registerMeshByPart(triV, triF, upsampledTriV, upsampledTriF, 2 * shiftz, globalAmpMin, globalAmpMax, ampFieldsList[frameId],
                           phaseFieldsList[frameId], &vertexOmegaList[frameId], interpP, interpF, interpVec,
                           interpColor);


        dataV = interpP;
        curColor = interpColor;
        dataVec = interpVec;
        dataF = interpF;

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
        std::string meshPath = igl::file_dialog_open();
        igl::readOBJ(meshPath, triV, triF);
        // Initialize polyscope
        polyscope::init();

        // Register the mesh with Polyscope
        polyscope::registerSurfaceMesh("input mesh", triV, triF);

        //std::string loadJson = igl::file_dialog_open();
        //if (interpModel.load(loadJson, triV, triF))
        //{

        //    initialization();
        //    omegaList = interpModel.getWList();
        //    zList = interpModel.getVertValsList();

        //    numFrames = omegaList.size() - 2;
        //    updateMagnitudePhase(omegaList, zList, ampFieldsList, phaseFieldsList);

        //    globalAmpMax = ampFieldsList[0].maxCoeff();
        //    globalAmpMin = ampFieldsList[0].minCoeff();
        //    for(int i = 1; i < ampFieldsList.size(); i++)
        //    {
        //        globalAmpMax = std::max(globalAmpMax, ampFieldsList[i].maxCoeff());
        //        globalAmpMin = std::min(globalAmpMin, ampFieldsList[i].minCoeff());
        //    }

        //    vertexOmegaList.resize(omegaList.size());
        //    for (int i = 0; i < omegaList.size(); i++)
        //    {
        //        vertexOmegaList[i] = intrinsicHalfEdgeVec2VertexVec(omegaList[i], triV, triMesh);
        //    }

        //    sourceOmegaFields = omegaList[0];
        //    sourceVertexOmegaFields = vertexOmegaList[0];

        //    tarOmegaFields = omegaList[omegaList.size() - 1];
        //    tarVertexOmegaFields = vertexOmegaList[omegaList.size() - 1];

        //    sourceZvals = zList[0];
        //    tarZvals = zList[omegaList.size() - 1];

        //    KnoppelPhaseFieldsList.resize(ampFieldsList.size());
        //    Eigen::VectorXd faceArea;
        //    Eigen::MatrixXd cotEntries;
        //    igl::doublearea(triV, triF, faceArea);
        //    faceArea /= 2;
        //    igl::cotmatrix_entries(triV, triF, cotEntries);
        //    int nverts = triV.rows();
        //    for (int i = 0; i < ampFieldsList.size(); i++)
        //    {
        //        double t = 1.0 / (ampFieldsList.size() - 1) * i;
        //        Eigen::MatrixXd interpVecs = (1 - t) * sourceOmegaFields + t * tarOmegaFields;
        //        std::vector<std::complex<double>> interpZvals;
        //        IntrinsicFormula::roundVertexZvalsFromHalfEdgeOmega(triMesh, interpVecs, faceArea, cotEntries, nverts, interpZvals);
        //        Eigen::VectorXd upTheta;
        //        IntrinsicFormula::getUpsamplingTheta(triMesh, interpVecs, interpZvals, bary, upTheta);
        //        KnoppelPhaseFieldsList[i] = upTheta;
        //    }

        //    // linear baseline
        //    auto tmpModel = IntrinsicFormula::IntrinsicKeyFrameInterpolationFromHalfEdge(MeshConnectivity(triF), faceArea, (ampFieldsList.size() - 2), quadOrder, sourceZvals, sourceOmegaFields, tarZvals, tarOmegaFields);
        //    updateMagnitudePhase(tmpModel.getWList(), tmpModel.getVertValsList(), linearAmpFieldsList, linearPhaseFieldsList);

        //    updateFieldsInView(curFrame);
        //}
    }
    ImGui::SameLine(0, p);
    if (ImGui::Button("Save", ImVec2((w - p) / 2.f, 0)))
    {
        std::string saveFolder = igl::file_dialog_save();
        //interpModel.save(saveFolder, triV, triF);
    }

    if(ImGui::Button("Load reference", ImVec2(-1, 0)))
    {
        std::string loadFileName = igl::file_dialog_open();

        std::cout << "load file in: " << loadFileName << std::endl;
        using json = nlohmann::json;
        std::ifstream inputJson(loadFileName);
        if (!inputJson)
        {
            std::cerr << "missing json file in " << loadFileName << std::endl;
            return;
        }

        std::string filePath = loadFileName;
        std::replace(filePath.begin(), filePath.end(), '\\', '/'); // handle the backslash issue for windows
        int id = filePath.rfind("/");
        std::string workingFolder = filePath.substr(0, id + 1);
        std::cout << "working folder: " << workingFolder << std::endl;

        json jval;
        inputJson >> jval;

        std::string meshFile =jval["mesh_name"];
        meshFile = workingFolder + meshFile;
        igl::readOBJ(meshFile, triV, triF);
        triMesh = MeshConnectivity(triF);

        quadOrder = jval["quad_order"];
        numFrames = jval["num_frame"];

        // edge omega List
        int iter = 0;
        int nedges = triMesh.nEdges();
        int nverts = triV.rows();

        refAmpList.resize(numFrames + 2);
        refOmegaList.resize(numFrames + 2);

        auto loadAmpandOmega = [&](const tbb::blocked_range<uint32_t>& range) {
            for (uint32_t i = range.begin(); i < range.end(); ++i)
            {
                //std::cout << i << std::endl;
                std::ifstream afs(workingFolder + "amp_" + std::to_string(i) + ".txt");
                Eigen::VectorXd amp(nverts);

                for (int j = 0; j < nverts; j++)
                {
                    std::string line;
                    std::getline(afs, line);
                    std::stringstream ss(line);
                    std::string x;
                    ss >> x;
                    amp(j) = std::stod(x);
                }
                refAmpList[i] = amp;

                std::ifstream wfs(workingFolder + "halfEdgeOmega_" + std::to_string(i) + ".txt");
                Eigen::MatrixXd edgeW(nedges, 2);

                for (int j = 0; j < nedges; j++)
                {
                    std::string line;
                    std::getline(wfs, line);
                    std::stringstream ss(line);
                    std::string x, y;
                    ss >> x;
                    ss >> y;
                    edgeW.row(j) << std::stod(x), std::stod(y);
                }

                refOmegaList[i] = edgeW;
            }
        };

        tbb::blocked_range<uint32_t> rangex(0u, (uint32_t)(numFrames + 2), GRAIN_SIZE);
        tbb::parallel_for(rangex, loadAmpandOmega);
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
    if (ImGui::Checkbox("is show comparison", &isShowComparison))
    {
        updateFieldsInView(curFrame);
    }
    if (ImGui::DragFloat("wrinkle amp scaling ratio", &wrinkleAmpScalingRatio, 0.0005, 0, 1))
    {
        if(wrinkleAmpScalingRatio >= 0)
            updateFieldsInView(curFrame);
    }

    if (ImGui::CollapsingHeader("source Vector Fields Info", ImGuiTreeNodeFlags_DefaultOpen))
    {
        if (ImGui::Combo("source direction", (int*)&sourceDir, "PV1\0PV2\0Load From File\0Geodesic\0Geodesic Perp\0"))
        {
            if(sourceDir == LOADFROMFILE)
            {
                std::string filename = igl::file_dialog_open();
                loadEdgeOmega(filename, triMesh.nEdges(), sourceOmegaFields);
                sourceVertexOmegaFields = intrinsicHalfEdgeVec2VertexVec(sourceOmegaFields, triV, triMesh);
            }
        }
        if (ImGui::InputDouble("source frequency", &sourceFreq))
        {
            if (sourceFreq <= 0)
                sourceFreq = 1;
        }
        if (ImGui::InputInt("num of source heat sources", &numSource))
        {
            if (numSource < 0)
                numSource = 1;
        }

    }
    if (ImGui::CollapsingHeader("target Vector Fields Info", ImGuiTreeNodeFlags_DefaultOpen))
    {
        if (ImGui::Combo("target direction", (int*)&tarDir, "PV1\0PV2\0Load From File\0Geodesic\0Geodesic Perp\0Rotated source\0"))
        {
            if(tarDir == LOADFROMFILE)
            {
                std::string filename = igl::file_dialog_open();
                loadEdgeOmega(filename, triMesh.nEdges(), tarOmegaFields);
                tarVertexOmegaFields = intrinsicHalfEdgeVec2VertexVec(tarOmegaFields, triV, triMesh);
            }
        }
        if (ImGui::InputDouble("target frequency", &tarFreq))
        {
            if (tarFreq <= 0)
                tarFreq = 1;
        }
        if (ImGui::InputInt("num of target heat sources", &numTarSource))
        {
            if(numTarSource < 0)
                numTarSource = 1;
        }
    }

    if (ImGui::CollapsingHeader("Frame Visualization Options", ImGuiTreeNodeFlags_DefaultOpen))
    {
       /* if (ImGui::InputDouble("drag speed", &dragSpeed))
        {
            if (dragSpeed <= 0)
                dragSpeed = 0.5;
        }*/
        if (ImGui::SliderInt("current frame", &curFrame, 0, numFrames + 1))
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
        if (ImGui::Combo("initialization types", (int*)&initializationType, "Random\0Linear\0Knoppel\0")) {}
    }


    ImGui::Checkbox("Try Optimization", &isForceOptimize);

    if (ImGui::Button("comb fields & updates", ImVec2(-1, 0)))
    {
        Eigen::MatrixXd PD1, PD2;
        Eigen::VectorXd PV1, PV2;
        igl::principal_curvature(triV, triF, PD1, PD2, PV1, PV2);

        // compute geodesic
        igl::HeatGeodesicsData<double> data;
        double t = std::pow(igl::avg_edge_length(triV, triF), 2);
        const auto precompute = [&]()
        {
            if (!igl::heat_geodesics_precompute(triV, triF, t, data))
            {
                std::cerr << "Error: heat_geodesics_precompute failed." << std::endl;
                exit(EXIT_FAILURE);
            };
        };
        precompute();
        Eigen::VectorXd sD;

        Eigen::VectorXi sourceGamma(numSource);
        for (int i = 0; i < numSource; i++)
        {
            sourceGamma(i) = std::rand() % (triV.rows());
        }
        sourceGamma.resize(1);
        sourceGamma(0) = 5;
        std::cout << "source vertex id: " << sourceGamma.transpose() << std::endl;

        Eigen::MatrixXd sdisFields, sdisFieldsPerp; // per face vectors
        getDisFields(sourceGamma, sD, sdisFields, sdisFieldsPerp);

        for(int i = 0; i < numComb; i++)
        {
            combField(triF, PD1, PD1);
            combField(triF, PD2, PD2);
        }

        if(sourceDir == DirectionType::DIRPV1)
        {
            sourceOmegaFields = vertexVec2IntrinsicHalfEdgeVec(PD1, triV, triMesh);
            sourceVertexOmegaFields = PD1;
            sourceOmegaFields *= 2 * M_PI * sourceFreq;
            sourceVertexOmegaFields *= 2 * M_PI * sourceFreq;
        }

        else if(sourceDir == DirectionType::DIRPV2)
        {
            sourceOmegaFields = vertexVec2IntrinsicHalfEdgeVec(PD2, triV, triMesh);
            sourceVertexOmegaFields = PD2;
            sourceOmegaFields *= 2 * M_PI * sourceFreq;
            sourceVertexOmegaFields *= 2 * M_PI * sourceFreq;
        }
        else if (sourceDir == DirectionType::GEODESIC)
        {
            int nedges = triMesh.nEdges();
            sourceOmegaFields.setZero(nedges, 2);
            for (int i = 0; i < nedges; i++)
            {
                int vid0 = triMesh.edgeVertex(i, 0);
                int vid1 = triMesh.edgeVertex(i, 1);

                double d = sD(vid1) - sD(vid0);
                sourceOmegaFields.row(i) << d, -d;
            }
            sourceVertexOmegaFields = intrinsicHalfEdgeVec2VertexVec(sourceOmegaFields, triV, triMesh);

            sourceOmegaFields *= 2 * M_PI * sourceFreq;
            sourceVertexOmegaFields *= 2 * M_PI * sourceFreq;
        }
        else if (sourceDir == DirectionType::GEODESICPERP)
        {
            int nedges = triMesh.nEdges();
            int nfaces = triMesh.nFaces();
            sourceOmegaFields.setZero(nedges, 2);

            for (int i = 0; i < nfaces; i++)
            {
                Eigen::Vector3d vecPerp3D = sdisFieldsPerp.row(i);
                for (int j = 0; j < 3; j++)
                {
                    int eid = triMesh.faceEdge(i, j);

                    double factor = 2.0;
                    if (triMesh.edgeFace(eid, 0) == -1 || triMesh.edgeFace(eid, 1) == -1)
                        factor = 1.0;

                    Eigen::Vector3d v = triV.row(triMesh.edgeVertex(eid, 1)) - triV.row(triMesh.edgeVertex(eid, 0));
                    sourceOmegaFields(eid, 0) += v.dot(vecPerp3D) / factor;
                    sourceOmegaFields(eid, 1) += -v.dot(vecPerp3D) / factor;
                }

            }

            sourceVertexOmegaFields = intrinsicHalfEdgeVec2VertexVec(sourceOmegaFields, triV, triMesh);

            sourceOmegaFields *= 2 * M_PI * sourceFreq;
            sourceVertexOmegaFields *= 2 * M_PI * sourceFreq;
        }
        else if (sourceDir == DirectionType::ROTATEDVEC)
        {
            sourceOmegaFields *= sourceFreq;
            sourceVertexOmegaFields *= sourceFreq;
        }
        else
        {
            sourceOmegaFields *= sourceFreq;
            sourceVertexOmegaFields *= sourceFreq;
        }


        Eigen::VectorXd tD;

        Eigen::VectorXi tarGamma(numTarSource);
        for (int i = 0; i < numTarSource; i++)
        {
            tarGamma(i) = std::rand() % (triV.rows());
        }
        tarGamma.resize(1);
        tarGamma(0) = 189;

        std::cout << "source vertex id (target): " << tarGamma.transpose() << std::endl;
        Eigen::MatrixXd tdisFields, tdisFieldsPerp; // per face vectors
        getDisFields(tarGamma, tD, tdisFields, tdisFieldsPerp);

        if(tarDir == DirectionType::DIRPV1)
        {
            tarOmegaFields = vertexVec2IntrinsicHalfEdgeVec(PD1, triV, triMesh);
            tarVertexOmegaFields = PD1;
            tarOmegaFields *= 2 * M_PI * tarFreq;
            tarVertexOmegaFields *= 2 * M_PI * tarFreq;
        }

        else if(tarDir == DirectionType::DIRPV2)
        {
            tarOmegaFields = vertexVec2IntrinsicHalfEdgeVec(PD2, triV, triMesh);
            tarVertexOmegaFields = PD2;
            tarOmegaFields *= 2 * M_PI * tarFreq;
            tarVertexOmegaFields *= 2 * M_PI * tarFreq;
        }
        else if (tarDir == DirectionType::GEODESIC)
        {
            int nedges = triMesh.nEdges();
            tarOmegaFields.setZero(nedges, 2);
            for (int i = 0; i < nedges; i++)
            {
                int vid0 = triMesh.edgeVertex(i, 0);
                int vid1 = triMesh.edgeVertex(i, 1);

                double d = tD(vid1) - tD(vid0);
                tarOmegaFields.row(i) << d, -d;
            }
            tarVertexOmegaFields = intrinsicHalfEdgeVec2VertexVec(tarOmegaFields, triV, triMesh);

            tarOmegaFields *= 2 * M_PI * tarFreq;
            tarVertexOmegaFields *= 2 * M_PI * tarFreq;
        }
        else if (tarDir == DirectionType::GEODESICPERP)
        {
            int nedges = triMesh.nEdges();
            int nfaces = triMesh.nFaces();
            tarOmegaFields.setZero(nedges, 2);
            Eigen::MatrixXd faceN;
            igl::per_face_normals(triV, triF, faceN);

            for (int i = 0; i < nfaces; i++)
            {
                Eigen::Vector3d vecPerp3D = tdisFieldsPerp.row(i);
                for (int j = 0; j < 3; j++)
                {
                    int eid = triMesh.faceEdge(i, j);

                    double factor = 2.0;
                    if (triMesh.edgeFace(eid, 0) == -1 || triMesh.edgeFace(eid, 1) == -1)
                        factor = 1.0;

                    Eigen::Vector3d v = triV.row(triMesh.edgeVertex(eid, 1)) - triV.row(triMesh.edgeVertex(eid, 0));
                    tarOmegaFields(eid, 0) += v.dot(vecPerp3D) / factor;
                    tarOmegaFields(eid, 1) += -v.dot(vecPerp3D) / factor;
                }

            }

            tarVertexOmegaFields = intrinsicHalfEdgeVec2VertexVec(tarOmegaFields, triV, triMesh);

            tarOmegaFields *= 2 * M_PI * tarFreq;
            tarVertexOmegaFields *= 2 * M_PI * tarFreq;
        }
        else if (tarDir == DirectionType::ROTATEDVEC)
        {
            std::vector<RotateVertexInfo> rotVerts;
            for (int i = 0; i < triV.rows() / 2; i++)
            {
                rotVerts.push_back({ i, M_PI / 2.0 });
            }
            rotateIntrinsicVector(triV, triMesh, sourceOmegaFields, rotVerts, tarOmegaFields);
            tarVertexOmegaFields = intrinsicHalfEdgeVec2VertexVec(tarOmegaFields, triV, triMesh);
        }
        else
        {
            tarOmegaFields *= tarFreq;
            tarVertexOmegaFields *= tarFreq;
        }

        Eigen::VectorXd faceArea;
        Eigen::MatrixXd cotEntries;
        igl::doublearea(triV, triF, faceArea);
        faceArea /= 2;
        igl::cotmatrix_entries(triV, triF, cotEntries);
        int nverts = triV.rows();

        IntrinsicFormula::roundVertexZvalsFromHalfEdgeOmega(triMesh, sourceOmegaFields, faceArea, cotEntries, nverts, sourceZvals);
        IntrinsicFormula::roundVertexZvalsFromHalfEdgeOmega(triMesh, tarOmegaFields, faceArea, cotEntries, nverts, tarZvals);

        Eigen::VectorXd sourceAmp(nverts), tarAmp(nverts);
        if(sourceDir != GEODESICPERP)
            ampSolver(triV, triMesh, sourceOmegaFields / 10, sourceAmp);
        else
        {
            Eigen::VectorXd dPerpInverse = sD;
            dPerpInverse.setZero();
            Eigen::VectorXd numNeis = dPerpInverse;

            for(int i = 0; i < triF.rows(); i++)
            {
                for(int j = 0; j < 3; j++)
                {
                    int vid = triF(i, j);
                    dPerpInverse(vid) += sdisFieldsPerp.row(i).norm();
                    numNeis(vid)++;
                }
            }
            for(int i = 0; i < dPerpInverse.rows(); i++)
            {
                sourceAmp(i) = numNeis(i) / dPerpInverse(i);
            }
        }
        if(tarDir != GEODESICPERP)
            ampSolver(triV, triMesh, tarOmegaFields / 10, tarAmp);
        else
        {
            Eigen::VectorXd dPerpInverse = tD;
            dPerpInverse.setZero();
            Eigen::VectorXd numNeis = dPerpInverse;

            for(int i = 0; i < triF.rows(); i++)
            {
                for(int j = 0; j < 3; j++)
                {
                    int vid = triF(i, j);
                    dPerpInverse(vid) += tdisFieldsPerp.row(i).norm();
                    numNeis(vid)++;
                }
            }
            for(int i = 0; i < dPerpInverse.rows(); i++)
            {
                tarAmp(i) = numNeis(i) / dPerpInverse(i);
            }
        }

        for (int i = 0; i < triV.rows(); i++)
        {
            double sourceNorm = std::abs(sourceZvals[i]);
            double tarNorm = std::abs(tarZvals[i]);
            if(sourceNorm)
                sourceZvals[i] = sourceAmp(i) / sourceNorm * sourceZvals[i];
            if(tarNorm)
                tarZvals[i] = tarAmp(i) / tarNorm * tarZvals[i];
        }

        // solve for the path from source to target
        solveKeyFrames(sourceOmegaFields, tarOmegaFields, sourceZvals, tarZvals, numFrames, omegaList, zList);
        // get interploated amp and phase frames
        updateMagnitudePhase(omegaList, zList, ampFieldsList, phaseFieldsList);
        vertexOmegaList.resize(omegaList.size());
        for(int i = 0; i < omegaList.size(); i++)
        {
            vertexOmegaList[i] = intrinsicHalfEdgeVec2VertexVec(omegaList[i], triV, triMesh);
        }
        numFrames = omegaList.size() - 2;

        // update global maximum amplitude
        globalAmpMax = ampFieldsList[0].maxCoeff();
        globalAmpMin = ampFieldsList[0].minCoeff();
        for(int i = 1; i < ampFieldsList.size(); i++)
        {
            globalAmpMax = std::max(globalAmpMax, ampFieldsList[i].maxCoeff());
            globalAmpMin = std::min(globalAmpMin, ampFieldsList[i].minCoeff());
        }

        KnoppelPhaseFieldsList.resize(ampFieldsList.size());
        for(int i = 0; i < ampFieldsList.size(); i++)
        {
            double t = 1.0 / (ampFieldsList.size() - 1) * i;
            Eigen::MatrixXd interpVecs = (1 - t) * sourceOmegaFields + t * tarOmegaFields;
            std::vector<std::complex<double>> interpZvals;
            IntrinsicFormula::roundVertexZvalsFromHalfEdgeOmega(triMesh, interpVecs, faceArea, cotEntries, nverts, interpZvals);
            Eigen::VectorXd upTheta;
            IntrinsicFormula::getUpsamplingTheta(triMesh, interpVecs, interpZvals, bary, upTheta);
            KnoppelPhaseFieldsList[i] = upTheta;
        }

        // linear baseline
        auto tmpModel = IntrinsicFormula::IntrinsicKeyFrameInterpolationFromHalfEdge(MeshConnectivity(triF), faceArea, (ampFieldsList.size() - 2), quadOrder, sourceZvals, sourceOmegaFields, tarZvals, tarOmegaFields);
        updateMagnitudePhase(tmpModel.getWList(), tmpModel.getVertValsList(), linearAmpFieldsList, linearPhaseFieldsList);
        updateFieldsInView(curFrame);
    }

    if (ImGui::Button("output images", ImVec2(-1, 0)))
    {
        for(int i = 0; i < ampFieldsList.size(); i++)
        {
            updateFieldsInView(curFrame);
            //polyscope::options::screenshotExtension = ".jpg";
            std::string name = "output_" + std::to_string(i) + ".jpg";
            polyscope::screenshot(name);
        }
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

    // compute geodesic
    Eigen::VectorXi sourceGamma(numSource);
    for (int i = 0; i < numSource; i++)
    {
        sourceGamma(i) = std::rand() % (triV.rows());
    }
    sourceGamma.resize(1);
    sourceGamma(0) = 5;

    Eigen::VectorXd d;
    Eigen::MatrixXd disFields, disFieldsPerp;
    getDisFields(sourceGamma, d, disFields, disFieldsPerp);

    Eigen::VectorXd dPerpInverse = d;
    dPerpInverse.setZero();
    Eigen::VectorXd numNeis = dPerpInverse;

    for(int i = 0; i < triF.rows(); i++)
    {
        for(int j = 0; j < 3; j++)
        {
            int vid = triF(i, j);
            dPerpInverse(vid) += disFieldsPerp.row(i).norm();
            numNeis(vid)++;
        }
    }
    for(int i = 0; i < dPerpInverse.rows(); i++)
    {
        dPerpInverse(i) = numNeis(i) / dPerpInverse(i);
    }

    polyscope::getSurfaceMesh("input mesh")->addFaceVectorQuantity("distance field", disFields);
    polyscope::getSurfaceMesh("input mesh")->addFaceVectorQuantity("distance field perp", disFieldsPerp);
//    polyscope::getSurfaceMesh("input mesh")->addVertexDistanceQuantity("distance values", d);
    polyscope::getSurfaceMesh("input mesh")->addVertexDistanceQuantity("distance inverse values", dPerpInverse);


    polyscope::view::upDir = polyscope::view::UpDir::ZUp;

    // Add the callback
    polyscope::state::userCallback = callback;

    polyscope::options::groundPlaneHeightFactor = 0.25; // adjust the plane height
    // Show the gui
    polyscope::show();

    return 0;
}