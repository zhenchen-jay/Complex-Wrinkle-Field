#include "polyscope/polyscope.h"

#include <igl/PI.h>
#include <igl/avg_edge_length.h>
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
#include "../../include/IntrinsicFormula/IntrinsicKnoppelDrivenFormula.h"
#include "../../include/IntrinsicFormula/IntrinsicKeyFrameInterpolationFromHalfEdge.h"
#include "../../include/IntrinsicFormula/KnoppelStripePattern.h"
#include "../../include/json.hpp"

Eigen::MatrixXd triV, upsampledTriV;
Eigen::MatrixXi triF, upsampledTriF;
MeshConnectivity triMesh, upsampledTriMesh;

std::vector<Eigen::MatrixXd> omegaList;
std::vector<std::vector<std::complex<double>>> zList;

// reference amp and omega
std::vector<Eigen::MatrixXd> refOmegaList;
std::vector<Eigen::VectorXd> refAmpList;


std::vector<Eigen::VectorXd> phaseFieldsList;
std::vector<Eigen::VectorXd> ampFieldsList;


Eigen::MatrixXd dataV;
Eigen::MatrixXi dataF;
Eigen::MatrixXd dataVec;
Eigen::MatrixXd curColor;

int loopLevel = 2;

bool isForceOptimize = false;
bool isShowVectorFields = true;
bool isShowWrinkels = true;

PaintGeometry mPaint;

int numFrames = 50;
int curFrame = 0;


double globalAmpMax = 1;
double globalAmpMin = 0;

float vecratio = 0.1;

bool isLoadOpt = false;
bool isReinitialize = false;

double gradTol = 1e-6;
double xTol = 0;
double fTol = 0;
int numIter = 1000;

double spatialRatio = 1;
int quadOrder = 4;
double wrinkleAmpScalingRatio = 1.0;

std::string workingFolder;
IntrinsicFormula::IntrinsicKnoppelDrivenFormula interpModel;
std::vector<std::pair<int, Eigen::Vector3d>> bary;
std::vector<std::complex<double>> _initZvals, _tarZvals;

enum InitializationType{
    Random = 0,
    Linear = 1,
    Knoppel = 2
};

InitializationType initializationType = InitializationType::Linear;


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

void initialization(const Eigen::MatrixXd& triV, const Eigen::MatrixXi& triF)
{
    triMesh = MeshConnectivity(triF);

    Eigen::SparseMatrix<double> S;
    std::vector<int> facemap;

    meshUpSampling(triV, triF, upsampledTriV, upsampledTriF, loopLevel, &S, &facemap, &bary);
    std::cout << "upsampling finished" << std::endl;
    upsampledTriMesh = MeshConnectivity(upsampledTriF);
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
    meshFile = workingFolder + meshFile;
    igl::readOBJ(meshFile, triV, triF);
    initialization(triV, triF);

    quadOrder = jval["quad_order"];
    numFrames = jval["num_frame"];

    std::string refAmp = jval["ref_amp"];
    std::string refOmega = jval["ref_omega"];
    std::string optSol = jval["opt_sol"];

    // edge omega List
    int iter = 0;
    int nedges = triMesh.nEdges();
    int nverts = triV.rows();

    refAmpList.resize(numFrames + 2);
    refOmegaList.resize(numFrames + 2);

    for (uint32_t i = 0; i < numFrames + 2; ++i) {
        //std::cout << i << std::endl;
        std::ifstream afs(workingFolder + refAmp + "/amp_" + std::to_string(i) + ".txt");

        if (!afs) {
            std::cout << "missing amp file: " << std::endl;
            return false;
        }

        Eigen::VectorXd amp(nverts);

        for (int j = 0; j < nverts; j++) {
            std::string line;
            std::getline(afs, line);
            std::stringstream ss(line);
            std::string x;
            ss >> x;
            amp(j) = std::stod(x);
        }
        refAmpList[i] = amp;

        std::string edgePath = workingFolder + refOmega + "/omega_" + std::to_string(i) + ".txt";
        Eigen::MatrixXd edgeW;
        if (!loadEdgeOmega(edgePath, nedges, edgeW)) {
            std::cout << "missing edge file." << std::endl;
            return false;
        }
        refOmegaList[i] = edgeW;
    }

    isLoadOpt = true;
    zList.clear();
    omegaList.clear();
    for(int i = 0; i < numFrames + 2; i++)
    {
        std::string zvalFile = workingFolder + optSol + "/zvals_" + std::to_string(i) + ".txt";
        std::string halfEdgeOmegaFile = workingFolder + optSol + "/halfEdgeOmega_" + std::to_string(i) + ".txt";

        std::ifstream zfs(zvalFile);
        if(!zfs)
        {
            isLoadOpt = false;
            break;
        }

        std::vector<std::complex<double>> zvals(nverts);

        for (int j = 0; j < nverts; j++) {
            std::string line;
            std::getline(zfs, line);
            std::stringstream ss(line);
            std::string x, y;
            ss >> x;
            ss >> y;
            zvals[j] = std::complex<double>(std::stod(x), std::stod(y));
        }


        Eigen::MatrixXd edgeOmega;
        if (!loadEdgeOmega(halfEdgeOmegaFile, nedges, edgeOmega)) {
           isLoadOpt = false;
           break;
        }

        zList.push_back(zvals);
        omegaList.push_back(edgeOmega);
    }

    if(isLoadOpt)
    {
        std::cout << "load zvals and omegas from file!" << std::endl;
    }
    else
    {
        std::cout << "failed to load zvals and omegas from file, set them to be random values!" << std::endl;
        zList.resize(numFrames + 2);
        omegaList.resize(numFrames + 2);

        for(int i = 0; i < numFrames + 2; i++)
        {
            omegaList[i].setRandom(nedges, 2);
            Eigen::Vector2d rnd = Eigen::Vector2d::Random();
            zList[i].resize(nverts, std::complex<double>(rnd(0), rnd(1)));
        }
    }

    globalAmpMin = std::numeric_limits<double>::infinity();
    globalAmpMax = -std::numeric_limits<double>::infinity();

    for(int j = 0; j < zList[0].size(); j++)
    {
        globalAmpMin = std::min(globalAmpMin, std::abs(zList[0][j]));
        globalAmpMax = std::max(globalAmpMax, std::abs(zList[0][j]));
    }


    for (int i = 0; i < zList.size(); i++)
    {
        for(int j = 0; j < zList[i].size(); j++)
        {
            globalAmpMin = std::min(globalAmpMin, std::abs(zList[i][j]));
            globalAmpMax = std::max(globalAmpMax, std::abs(zList[i][j]));
        }
    }
    std::cout << "global amp range: " << globalAmpMin << ", " << globalAmpMax << std::endl;

    // for(int i = 1; i < numFrames + 2; i++)
    // {
    //     refOmegaList[i] = (1 + 1.5 / (numFrames + 1) * i) * refOmegaList[0];
    //     refAmpList[i] =  refAmpList[0] / (1 + 1.5 / (numFrames + 1) * i);
    // }
    return true;
}


bool saveProblem()
{
    std::string saveFileName = igl::file_dialog_save();

    using json = nlohmann::json;
    json jval;
    jval["mesh_name"] = "mesh.obj";
    jval["num_frame"] = zList.size() - 2;
    jval["quad_order"] = quadOrder;
    jval["ref_amp"] = "/amp/";
    jval["ref_omega"] = "/omega/";
    jval["opt_sol"] = "/optSol/";

    std::string filePath = saveFileName;
    std::replace(filePath.begin(), filePath.end(), '\\', '/'); // handle the backslash issue for windows
    int id = filePath.rfind("/");
    std::string workingFolder = filePath.substr(0, id + 1);

    igl::writeOBJ(workingFolder + "mesh.obj", triV, triF);

    std::string outputFolder = workingFolder + "optSol/";
    if (!std::filesystem::exists(outputFolder))
    {
        std::cout << "create directory: " << outputFolder << std::endl;
        if (!std::filesystem::create_directories(outputFolder))
        {
            std::cout << "create folder failed." << outputFolder << std::endl;
            exit(1);
        }
    }

    for (int i = 0; i < zList.size(); i++)
    {
        std::ofstream zfs(outputFolder + "zvals_" + std::to_string(i) + ".txt");
        std::ofstream wfs(outputFolder + "halfEdgeOmega_" + std::to_string(i) + ".txt");
        wfs << std::setprecision(std::numeric_limits<long double>::digits10 + 1) << omegaList[i] << std::endl;
        for (int j = 0; j < zList[i].size(); j++)
        {
            zfs << std::setprecision(std::numeric_limits<long double>::digits10 + 1) << zList[i][j].real() << " " << zList[i][j].imag() << std::endl;
        }
    }

    // save reference
    outputFolder = workingFolder + "/amp/";
    if (!std::filesystem::exists(outputFolder))
    {
        std::cout << "create directory: " << outputFolder << std::endl;
        if (!std::filesystem::create_directories(outputFolder))
        {
            std::cout << "create folder failed." << outputFolder << std::endl;
            exit(1);
        }
    }
    for (int i = 0; i < refAmpList.size(); i++)
    {
        std::ofstream afs(outputFolder + "amp_" + std::to_string(i) + ".txt");
        afs << std::setprecision(std::numeric_limits<long double>::digits10 + 1) << refAmpList[i] << std::endl;
    }

    outputFolder = workingFolder + "/omega/";
    if (!std::filesystem::exists(outputFolder))
    {
        std::cout << "create directory: " << outputFolder << std::endl;
        if (!std::filesystem::create_directories(outputFolder))
        {
            std::cout << "create folder failed." << outputFolder << std::endl;
            exit(1);
        }
    }
    for (int i = 0; i < refOmegaList.size(); i++)
    {
        std::ofstream wfs(outputFolder + "omega_" + std::to_string(i) + ".txt");
        wfs << std::setprecision(std::numeric_limits<long double>::digits10 + 1) << refOmegaList[i] << std::endl;
    }

    std::ofstream o(workingFolder + "data.json");
    o << std::setw(4) << jval << std::endl;
    std::cout << "save file in: " << workingFolder + "data.json" << std::endl;

    return true;
}

void solveKeyFrames(std::vector<Eigen::MatrixXd>& wFrames, std::vector<std::vector<std::complex<double>>>& zFrames)
{
    Eigen::VectorXd faceArea;
    igl::doublearea(triV, triF, faceArea);
    faceArea /= 2;
    Eigen::MatrixXd cotEntries;
    igl::cotmatrix_entries(triV, triF, cotEntries);

    using complex = std::complex<double>;
    std::vector<complex> initZvals, tarZvals;

    IntrinsicFormula::roundVertexZvalsFromHalfEdgeOmegaVertexMag(triMesh, refOmegaList[0], refAmpList[0], faceArea, cotEntries, triV.rows(), initZvals);
    IntrinsicFormula::roundVertexZvalsFromHalfEdgeOmegaVertexMag(triMesh, refOmegaList[refOmegaList.size() - 1], refAmpList[refOmegaList.size() - 1], faceArea, cotEntries, triV.rows(), tarZvals);

    interpModel = IntrinsicFormula::IntrinsicKnoppelDrivenFormula(MeshConnectivity(triF), faceArea, cotEntries, refOmegaList, refAmpList, initZvals, tarZvals, refOmegaList[0], refOmegaList[refOmegaList.size() - 1], numFrames, spatialRatio, quadOrder);

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

        for(int i = 0; i < wList.size(); i++)
        {
            std::vector<complex> zvals;
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

    globalAmpMin = std::numeric_limits<double>::infinity();
    globalAmpMax = -std::numeric_limits<double>::infinity();

    for(int j = 0; j < zFrames[0].size(); j++)
    {
        globalAmpMin = std::min(globalAmpMin, std::abs(zFrames[0][j]));
        globalAmpMax = std::max(globalAmpMax, std::abs(zFrames[0][j]));
    }



    for (int i = 0; i < wFrames.size() - 1; i++)
    {
        double zdotNorm = interpModel._zdotModel.computeZdotIntegration(zFrames[i], wFrames[i], zFrames[i + 1], wFrames[i + 1], NULL, NULL);

        std::cout << "frame " << i << ", after optimization, ||zdot||^2 = " << zdotNorm << std::endl;
        for(int j = 0; j < zFrames[i].size(); j++)
        {
            globalAmpMin = std::min(globalAmpMin, std::abs(zFrames[i][j]));
            globalAmpMax = std::max(globalAmpMax, std::abs(zFrames[i][j]));
        }
    }
    std::cout << "global amp range: " << globalAmpMin << ", " << globalAmpMax << std::endl;
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
            Eigen::MatrixXd vertOmega = intrinsicHalfEdgeVec2VertexVec(*omegaVec, triV, triMesh);

            for (int i = 0; i < nverts; i++)
                renderVec.row(i) = vertOmega.row(i);
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


void registerMesh(int frameId) {

    Eigen::MatrixXd interpP;
    Eigen::MatrixXi interpF;
    Eigen::MatrixXd interpVec;
    Eigen::MatrixXd interpColor;

    double shiftz = 1.5 * (triV.col(2).maxCoeff() - triV.col(2).minCoeff());

    registerMeshByPart(triV, triF, upsampledTriV, upsampledTriF, 2 * shiftz, globalAmpMin, globalAmpMax,
                       ampFieldsList[frameId],
                       phaseFieldsList[frameId], &omegaList[frameId], interpP, interpF, interpVec,
                       interpColor);

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

    if (isShowVectorFields)
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
        loadProblem();
        updateMagnitudePhase(omegaList, zList, ampFieldsList, phaseFieldsList);
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

    if (ImGui::InputInt("upsampled times", &loopLevel))
    {
        if (loopLevel >= 0)
        {
            initialization(triV, triF);
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

    if (ImGui::InputDouble("wrinkle amp scaling ratio", &wrinkleAmpScalingRatio))
    {
        if(wrinkleAmpScalingRatio >= 0)
            updateFieldsInView(curFrame);
    }

    if (ImGui::CollapsingHeader("Frame Visualization Options", ImGuiTreeNodeFlags_DefaultOpen))
    {
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

        if (ImGui::Combo("initialization types", (int*)&initializationType, "Random\0Linear\0Knoppel\0")) {}
        if (ImGui::InputDouble("Spatial Ratio", &spatialRatio))
        {
            if(spatialRatio < 0)
                spatialRatio = 1.0;
        }
    }


    ImGui::Checkbox("Try Optimization", &isForceOptimize);
    ImGui::Checkbox("Reinitialize", &isReinitialize);

    if(ImGui::Button("update viewer", ImVec2(-1, 0)))
    {
        if(!isLoadOpt || isForceOptimize || isReinitialize)
            solveKeyFrames(omegaList, zList);
        updateMagnitudePhase(omegaList, zList, ampFieldsList, phaseFieldsList);
        updateFieldsInView(curFrame);
    }

    if (ImGui::Button("output images", ImVec2(-1, 0)))
    {
        for(curFrame = 0; curFrame < ampFieldsList.size(); curFrame++)
        {
            updateFieldsInView(curFrame);
            polyscope::options::screenshotExtension = ".jpg";
            polyscope::screenshot();
        }
        curFrame = 0;
    }

    ImGui::PopItemWidth();
}




int main(int argc, char** argv)
{
    if(!loadProblem())
    {
        std::cout << "failed to load file." << std::endl;
        return 1;
    }

    initialization(triV, triF);

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