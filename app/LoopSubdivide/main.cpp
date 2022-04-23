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
#include "../../dep/SecStencils/utils.h"

#include "../../include/json.hpp"
#include "../../include/ComplexLoop.h"
#include "../../include/ComplexLoopNew.h"
#include "../../include/MeshLib/RegionEdition.h"

Eigen::MatrixXd triV, loopTriV, NV, newN;
Eigen::MatrixXi triF, loopTriF, NF;

MeshConnectivity triMesh;
Mesh secMesh, subSecMesh;
std::vector<std::pair<int, Eigen::Vector3d>> bary;

Eigen::VectorXd initAmp;
Eigen::VectorXd initOmega;
std::vector<std::complex<double>> initZvals;

Eigen::VectorXd loopedAmp, loopedOmega, loopedPhase;
std::vector<std::complex<double>> loopedZvals;
std::vector<int> badfaces;

Eigen::MatrixXd faceOmega;
Eigen::MatrixXd loopedFaceOmega;

Eigen::MatrixXd dataV;
Eigen::MatrixXi dataF;
Eigen::MatrixXd dataVec;
Eigen::MatrixXd curColor;

int loopLevel = 1;

bool isShowVectorFields = true;
bool isShowWrinkels = true;

PaintGeometry mPaint;

float vecratio = 0.001;
float wrinkleAmpScalingRatio = 1;

std::string workingFolder;
double ampTol = 0.1, curlTol = 1e-4;

// smoothing
int smoothingTimes = 0;
double smoothingRatio = 0.95;


std::map<std::pair<int, int>, int> he2Edge(const Eigen::MatrixXi& faces)
{
    std::map< std::pair<int, int>, int > heToEdge;
    std::vector< std::vector<int> > edgeToVert;
    for (int face = 0; face < faces.rows(); ++face)
    {
        for (int i = 0; i < 3; ++i)
        {
            int vi = faces(face, i);
            int vj = faces(face, (i + 1) % 3);
            assert(vi != vj);

            std::pair<int, int> he = std::make_pair(vi, vj);
            if (he.first > he.second) std::swap(he.first, he.second);
            if (heToEdge.find(he) != heToEdge.end()) continue;

            heToEdge[he] = edgeToVert.size();
            edgeToVert.push_back(std::vector<int>(2));
            edgeToVert.back()[0] = he.first;
            edgeToVert.back()[1] = he.second;
        }
    }
    return heToEdge;
}

std::map<std::pair<int, int>, int> he2Edge(const std::vector< std::vector<int>>& edgeToVert)
{
    std::map< std::pair<int, int>, int > heToEdge;
    for (int i = 0; i < edgeToVert.size(); i++)
    {
        std::pair<int, int> he = std::make_pair(edgeToVert[i][0], edgeToVert[i][1]);
        heToEdge[he] = i;
    }
    return heToEdge;
}

Eigen::VectorXd swapEdgeVec(const std::vector< std::vector<int>>& edgeToVert, const Eigen::VectorXd& edgeVec, int flag)
{
    Eigen::VectorXd  edgeVecSwap = edgeVec;
    std::map< std::pair<int, int>, int > heToEdge = he2Edge(edgeToVert);

    int idx = 0;
    for (auto it : heToEdge)
    {
        if (flag == 0)   // ours to secstencils
            edgeVecSwap(it.second) = edgeVec(idx);
        else
            edgeVecSwap(idx) = edgeVec(it.second);
        idx++;
    }
    return edgeVecSwap;
}

Eigen::VectorXd swapEdgeVec(const Eigen::MatrixXi& faces, const Eigen::VectorXd& edgeVec, int flag)
{
    Eigen::VectorXd  edgeVecSwap = edgeVec;
    std::map< std::pair<int, int>, int > heToEdge = he2Edge(faces);

    int idx = 0;
    for (auto it : heToEdge)
    {
        if (flag == 0)   // ours to secstencils
            edgeVecSwap(it.second) = edgeVec(idx);
        else
            edgeVecSwap(idx) = edgeVec(it.second);
        idx++;
    }
    return edgeVecSwap;
}

std::vector<std::vector<int>> swapEdgeIndices(const Eigen::MatrixXi& faces, const std::vector<std::vector<int>>& edgeIndices, int flag)
{
    std::vector<std::vector<int>> edgeIndicesSwap = edgeIndices;
    std::map< std::pair<int, int>, int > heToEdge = he2Edge(faces);

    int idx = 0;
    for (auto it : heToEdge)
    {
        if (flag == 0)   // ours to secstencils
        {
            edgeIndicesSwap[it.second] = edgeIndices[idx];
        }
        else
        {
            edgeIndicesSwap[idx] = edgeIndices[it.second];
        }
        idx++;
    }

    return edgeIndicesSwap;
}

Eigen::MatrixXd edgeVec2FaceVec(const Mesh& mesh, Eigen::VectorXd& edgeVec)
{
    int nfaces = mesh.GetFaceCount();
    int nedges = mesh.GetEdgeCount();
    Eigen::MatrixXd fVec(nfaces, 3);
    fVec.setZero();

    for (int f = 0; f < nfaces; f++)
    {
        std::vector<int> faceEdges = mesh.GetFaceEdges(f);
        std::vector<int> faceVerts = mesh.GetFaceVerts(f);
        for (int j = 0; j < 3; j++)
        {
            int vid = faceVerts[j];
            int eid0 = faceEdges[j];
            int eid1 = faceEdges[(j + 2) % 3];

            Eigen::Vector3d e0 = mesh.GetVertPos(faceVerts[(j + 1) % 3]) - mesh.GetVertPos(vid);
            Eigen::Vector3d e1 = mesh.GetVertPos(faceVerts[(j + 2) % 3]) - mesh.GetVertPos(vid);

            int flag0 = 1, flag1 = 1;
            Eigen::Vector2d rhs;

            if (mesh.GetEdgeVerts(eid0)[0] == vid)
            {
                flag0 = 1;
            }
            else
            {
                flag0 = -1;
            }


            if (mesh.GetEdgeVerts(eid1)[0] == vid)
            {
                flag1 = 1;
            }
            else
            {
                flag1 = -1;
            }
            rhs(0) = flag0 * edgeVec(eid0);
            rhs(1) = flag1 * edgeVec(eid1);

            Eigen::Matrix2d I;
            I << e0.dot(e0), e0.dot(e1), e1.dot(e0), e1.dot(e1);
            Eigen::Vector2d sol = I.inverse() * rhs;

            fVec.row(f) += (sol(0) * e0 + sol(1) * e1) / 3;
        }
    }
    return fVec;
}

bool loadEdgeOmega(const std::string& filename, const int& nlines, Eigen::VectorXd& edgeOmega)
{
    std::ifstream infile(filename);
    if (!infile)
    {
        std::cerr << "invalid edge omega file name" << std::endl;
        return false;
    }
    else
    {
        Eigen::MatrixXd halfEdgeOmega(nlines, 2);
        edgeOmega.setZero(nlines);
        for (int i = 0; i < nlines; i++)
        {
            std::string line;
            std::getline(infile, line);
            std::stringstream ss(line);

            std::string x, y;
            ss >> x;
            if (!ss)
                return false;
            ss >> y;
            if (!ss)
            {
                halfEdgeOmega.row(i) << std::stod(x), -std::stod(x);
            }
            else
                halfEdgeOmega.row(i) << std::stod(x), std::stod(y);
        }
        edgeOmega = (halfEdgeOmega.col(0) - halfEdgeOmega.col(1)) / 2;
    }
    return true;
}

bool loadVertexZvals(const std::string& filePath, const int& nlines, std::vector<std::complex<double>>& zvals)
{
    std::ifstream zfs(filePath);
    if (!zfs)
    {
        std::cerr << "invalid zvals file name" << std::endl;
        return false;
    }

    zvals.resize(nlines);

    for (int j = 0; j < nlines; j++) {
        std::string line;
        std::getline(zfs, line);
        std::stringstream ss(line);
        std::string x, y;
        ss >> x;
        ss >> y;
        zvals[j] = std::complex<double>(std::stod(x), std::stod(y));
    }
    return true;
}

bool loadVertexAmp(const std::string& filePath, const int& nlines, Eigen::VectorXd& amp)
{
    std::ifstream afs(filePath);

    if (!afs)
    {
        std::cerr << "invalid ref amp file name" << std::endl;
        return false;
    }

    amp.setZero(nlines);

    for (int j = 0; j < nlines; j++)
    {
        std::string line;
        std::getline(afs, line);
        std::stringstream ss(line);
        std::string x;
        ss >> x;
        if (!ss)
            return false;
        amp(j) = std::stod(x);
    }
    return true;
}
void updateMagnitudePhase(const Eigen::VectorXd& omega, const std::vector<std::complex<double>>& zvals, Eigen::VectorXd& omegaNew, std::vector<std::complex<double>>& upZvals, Eigen::VectorXd& upsampledAmp, Eigen::VectorXd& upsampledPhase)
{
    NV = triV;
    NF = triF;
    Eigen::MatrixXd VN;
    igl::per_vertex_normals(triV, triF, VN);

    meshUpSampling(triV, triF, NV, NF, loopLevel, NULL, NULL, &bary);
    curvedPNTriangleUpsampling(triV, triF, VN, bary, NV, newN);

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

    Eigen::VectorXd edgeVec = swapEdgeVec(triF, omega, 0);

    Subdivide(secMesh, edgeVec, zvals, omegaNew, upZvals, loopLevel, subSecMesh, ampTol, curlTol, &badfaces);
    

    subSecMesh.GetPos(loopTriV);
    loopTriF.resize(subSecMesh.GetFaceCount(), 3);
    for (int i = 0; i < loopTriF.rows(); i++)
    {
        for (int j = 0; j < 3; j++)
            loopTriF(i, j) = subSecMesh.GetFaceVerts(i)[j];
    }

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

void updatePaintingItems()
{
    // get interploated amp and phase frames
    std::cout << "compute upsampled phase: " << std::endl;
    updateMagnitudePhase(initOmega, initZvals, loopedOmega, loopedZvals, loopedAmp, loopedPhase);

    std::cout << "compute face vector fields:" << std::endl;
    faceOmega = intrinsicEdgeVec2FaceVec(initOmega, triV, triMesh);

    loopedFaceOmega = edgeVec2FaceVec(subSecMesh, loopedOmega);

    std::cout << "start to update viewer." << std::endl;
}

void updateFieldsInView()
{
    std::cout << "update viewer. " << std::endl;

    polyscope::registerSurfaceMesh("input mesh", triV, triF);
    polyscope::getSurfaceMesh("input mesh")->addVertexScalarQuantity("amp color", initAmp);
    polyscope::getSurfaceMesh("input mesh")->getQuantity("amp color")->setEnabled(false);

    if (isShowVectorFields)
    {
        polyscope::getSurfaceMesh("input mesh")->addFaceVectorQuantity("vector field", faceOmega);
    }


    polyscope::registerSurfaceMesh("looped amp mesh", loopTriV, loopTriF);
    double shiftx = 1.5 * (triV.col(0).maxCoeff() - triV.col(0).minCoeff());
    double shiftz = 1.5 * (loopTriV.col(2).maxCoeff() - loopTriV.col(2).minCoeff());

    polyscope::getSurfaceMesh("looped amp mesh")->translate(glm::vec3(shiftx, 0, 2 * shiftz));
    polyscope::getSurfaceMesh("looped amp mesh")->setEnabled(false);

    polyscope::getSurfaceMesh("looped amp mesh")->addVertexScalarQuantity("amp color", loopedAmp);
    polyscope::getSurfaceMesh("looped amp mesh")->getQuantity("amp color")->setEnabled(true);

    if (isShowVectorFields)
    {
        polyscope::getSurfaceMesh("looped amp mesh")->addFaceVectorQuantity("upsampled vector field", loopedFaceOmega);
    }

    polyscope::registerSurfaceMesh("looped phase mesh", loopTriV, loopTriF);
    polyscope::getSurfaceMesh("looped phase mesh")->translate(glm::vec3(shiftx, 0, shiftz));
    polyscope::getSurfaceMesh("looped phase mesh")->setEnabled(true);

    mPaint.setNormalization(false);
    Eigen::MatrixXd phiColor = mPaint.paintPhi(loopedPhase);
    polyscope::getSurfaceMesh("looped phase mesh")->addVertexColorQuantity("phase color", phiColor);
    polyscope::getSurfaceMesh("looped phase mesh")->getQuantity("phase color")->setEnabled(true);


    Eigen::MatrixXd wrinkledTriV = loopTriV;
    Eigen::MatrixXd normal;
    igl::per_vertex_normals(loopTriV, loopTriF, normal);

    for(int i = 0; i < loopTriV.rows(); i++)
    {
        wrinkledTriV.row(i) += wrinkleAmpScalingRatio * loopedZvals[i].real() * normal.row(i);
    }

    Eigen::MatrixXd faceColors(loopTriF.rows(), 3);
    for (int i = 0; i < faceColors.rows(); i++)
    {
        if (badfaces[i])
            faceColors.row(i) << 1, 1, 1;
        else
            faceColors.row(i) << 80 / 255.0, 122 / 255.0, 91 / 255.0;
    }

    igl::writeOBJ(workingFolder + "wrinkledMesh_loop.obj", wrinkledTriV, loopTriF);

    polyscope::registerSurfaceMesh("loop wrinkled mesh", wrinkledTriV, loopTriF);
    polyscope::getSurfaceMesh("loop wrinkled mesh")->setSurfaceColor({ 80 / 255.0, 122 / 255.0, 91 / 255.0 });
    polyscope::getSurfaceMesh("loop wrinkled mesh")->addFaceColorQuantity("wrinkled color", faceColors);
    polyscope::getSurfaceMesh("loop wrinkled mesh")->translate(glm::vec3(shiftx, 0, 0));
    polyscope::getSurfaceMesh("loop wrinkled mesh")->setEnabled(true);

    Eigen::VectorXd edgeVecSwapped;
    std::vector<std::vector<int>> edge2verts;
    for(int i = 0; i < subSecMesh.GetEdgeCount(); i++)
    {
        edge2verts.push_back(subSecMesh.GetEdgeVerts(i));
    }
    edgeVecSwapped = swapEdgeVec(edge2verts, loopedOmega, 1);
    Eigen::VectorXd edgeArea = getEdgeArea(loopTriV, loopTriF);
    Eigen::VectorXd vertArea = getVertArea(loopTriV, loopTriF);

    std::vector<std::complex<double>> tmpZvals;
    Eigen::VectorXd edgeVec = swapEdgeVec(triF, initOmega, 0);

    Eigen::VectorXd loopedOmegaNew = loopedOmega, loopedAmpNew = loopedAmp, loopedPhaseNew = loopedPhase;

    SubdivideNew(secMesh, edgeVec, initZvals, loopedOmegaNew, tmpZvals, loopLevel, subSecMesh);

    Eigen::MatrixXd wrinkledTrivNew = loopTriV;

    for(int i = 0; i < loopTriV.rows(); i++)
    {
        loopedPhaseNew(i) = std::arg(tmpZvals[i]);
        loopedAmpNew(i) = std::abs(tmpZvals[i]);
    }

    for (int i = 0; i < loopTriV.rows(); i++)
    {
       wrinkledTrivNew.row(i) += wrinkleAmpScalingRatio * loopedAmpNew(i) * std::cos(loopedPhaseNew(i)) * normal.row(i);
    }

    laplacianSmoothing(wrinkledTrivNew, loopTriF, wrinkledTrivNew, smoothingRatio, smoothingTimes);
    igl::writeOBJ(workingFolder + "wrinkledMesh_loop_Zuenko.obj", wrinkledTrivNew, loopTriF);

    polyscope::registerSurfaceMesh("Loop-Zuenko amp mesh", loopTriV, loopTriF);
    polyscope::getSurfaceMesh("Loop-Zuenko amp mesh")->translate(glm::vec3(2 * shiftx, 0, 2 * shiftz));
    polyscope::getSurfaceMesh("Loop-Zuenko amp mesh")->setEnabled(false);

    polyscope::getSurfaceMesh("Loop-Zuenko amp mesh")->addVertexScalarQuantity("amp color", loopedAmpNew);
    polyscope::getSurfaceMesh("Loop-Zuenko amp mesh")->getQuantity("amp color")->setEnabled(true);

    if (isShowVectorFields)
    {
        Eigen::MatrixXd loopedFaceOmegaNew = edgeVec2FaceVec(subSecMesh, loopedOmegaNew);
        polyscope::getSurfaceMesh("Loop-Zuenko amp mesh")->addFaceVectorQuantity("upsampled vector field", loopedFaceOmegaNew);
    }

    polyscope::registerSurfaceMesh("Loop-Zuenko phase mesh", loopTriV, loopTriF);
    polyscope::getSurfaceMesh("Loop-Zuenko phase mesh")->translate(glm::vec3(2 * shiftx, 0, shiftz));
    polyscope::getSurfaceMesh("Loop-Zuenko phase mesh")->setEnabled(true);

    mPaint.setNormalization(false);
    phiColor = mPaint.paintPhi(loopedPhaseNew);
    polyscope::getSurfaceMesh("Loop-Zuenko phase mesh")->addVertexColorQuantity("phase color", phiColor);
    polyscope::getSurfaceMesh("Loop-Zuenko phase mesh")->getQuantity("phase color")->setEnabled(true);

    polyscope::registerSurfaceMesh("Loop-Zuenko mesh", wrinkledTrivNew, loopTriF);
    polyscope::getSurfaceMesh("Loop-Zuenko mesh")->setSurfaceColor({ 80 / 255.0, 122 / 255.0, 91 / 255.0 });
    polyscope::getSurfaceMesh("Loop-Zuenko mesh")->translate(glm::vec3(2 * shiftx, 0, 0));
    polyscope::getSurfaceMesh("Loop-Zuenko mesh")->setEnabled(true);

    meshUpSampling(triV, triF, NV, NF, loopLevel, NULL, NULL, &bary);
    std::vector<std::complex<double>> zuenkoZvals = IntrinsicFormula::upsamplingZvals(triMesh, initZvals, initOmega, bary);

    Eigen::MatrixXd zuenkoNormal;
    igl::per_vertex_normals(NV, NF, zuenkoNormal);

    Eigen::MatrixXd zuenkoNV = NV;
    Eigen::VectorXd zuenkoAmp = loopedAmp, zuenkoPhase = loopedPhase;

    for(int i = 0; i < loopTriV.rows(); i++)
    {
        zuenkoNV.row(i) += wrinkleAmpScalingRatio * zuenkoZvals[i].real() * zuenkoNormal.row(i);
        zuenkoPhase(i) = std::arg(zuenkoZvals[i]);
        zuenkoAmp(i) = std::abs(zuenkoZvals[i]);
    }

    polyscope::registerSurfaceMesh("Zuenko amp mesh", NV, NF);
    polyscope::getSurfaceMesh("Zuenko amp mesh")->translate(glm::vec3(3 * shiftx, 0, 2 * shiftz));
    polyscope::getSurfaceMesh("Zuenko amp mesh")->setEnabled(false);


    polyscope::getSurfaceMesh("Zuenko amp mesh")->addVertexScalarQuantity("amp color", zuenkoAmp);
    polyscope::getSurfaceMesh("Zuenko amp mesh")->getQuantity("amp color")->setEnabled(true);

    polyscope::registerSurfaceMesh("Zuenko phase mesh", NV, NF);
    polyscope::getSurfaceMesh("Zuenko phase mesh")->translate(glm::vec3(3 * shiftx, 0, shiftz));
    polyscope::getSurfaceMesh("Zuenko phase mesh")->setEnabled(true);

    mPaint.setNormalization(false);
    phiColor = mPaint.paintPhi(zuenkoPhase);
    polyscope::getSurfaceMesh("Zuenko phase mesh")->addVertexColorQuantity("phase color", phiColor);
    polyscope::getSurfaceMesh("Zuenko phase mesh")->getQuantity("phase color")->setEnabled(true);


    laplacianSmoothing(zuenkoNV, NF, zuenkoNV, smoothingRatio, smoothingTimes);

    igl::writeOBJ(workingFolder + "wrinkledMesh_Zuenko.obj", zuenkoNV, NF);

    polyscope::registerSurfaceMesh("Zuenko mesh", zuenkoNV, NF);
    polyscope::getSurfaceMesh("Zuenko mesh")->setSurfaceColor({ 80 / 255.0, 122 / 255.0, 91 / 255.0 });
    polyscope::getSurfaceMesh("Zuenko mesh")->translate(glm::vec3(3 * shiftx, 0, 0));
    polyscope::getSurfaceMesh("Zuenko mesh")->setEnabled(true);


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
    loopLevel = jval["upsampled_times"];

    meshFile = workingFolder + meshFile;


    igl::readOBJ(meshFile, triV, triF);
    triMesh = MeshConnectivity(triF);
    initialization(triV, triF, loopTriV, loopTriF);

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

    updateMagnitudePhase(initOmega, initZvals, loopedOmega, loopedZvals, loopedAmp, loopedPhase);

    updatePaintingItems();

    return true;
}


void callback() {
    ImGui::PushItemWidth(100);
    float w = ImGui::GetContentRegionAvailWidth();
    float p = ImGui::GetStyle().FramePadding.x;
    if (ImGui::Button("Load", ImVec2(-1, 0)))
    {
        loadProblem();
        updateFieldsInView();
    }

    if (ImGui::InputInt("underline loop level", &loopLevel))
    {
        if (loopLevel < 0)
            loopLevel = 0;
    }

    if (ImGui::CollapsingHeader("Visualization Options", ImGuiTreeNodeFlags_DefaultOpen))
    {
        if (ImGui::Checkbox("is show vector fields", &isShowVectorFields))
        {
            updateFieldsInView();
        }

        /*if(ImGui::DragFloat("vector ratio", &vecratio, 0.00005, 0, 10))
        {
            updateFieldsInView();
        }*/

        if (ImGui::DragFloat("wrinkle amp scaling ratio", &wrinkleAmpScalingRatio, 0.0005, 0, 1))
        {
            if (wrinkleAmpScalingRatio >= 0)
                updateFieldsInView();
        }

    }

    if (ImGui::CollapsingHeader("smoothing Options", ImGuiTreeNodeFlags_DefaultOpen))
    {
        if(ImGui::InputInt("smoothing times", &smoothingTimes))
        {
            smoothingTimes = smoothingTimes > 0 ? smoothingTimes : 0;
        }
        if(ImGui::InputDouble("smoothing ratio", &smoothingRatio))
        {
            smoothingRatio = smoothingRatio > 0 ? smoothingRatio : 0;
        }
    }

    if (ImGui::CollapsingHeader("Bad faces threshold", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::InputDouble("magnitude tol", &ampTol);
        ImGui::InputDouble("curl free tol", &curlTol);
    }

    if (ImGui::Button("comb fields & updates", ImVec2(-1, 0)))
    {
        // solve for the path from source to target
        updatePaintingItems();
        updateFieldsInView();
    }

    if (ImGui::Button("display informations", ImVec2(-1, 0)))
    {
        std::cout << "initial mesh: " << std::endl;
        std::cout << "\nvertex info: " << std::endl;
        for(int i = 0; i < triV.rows(); i++)
        {
            std::cout << "v_" << i << ": " << triV.row(i) << ", zvals: (" << initZvals[i].real() << ", " << initZvals[i].imag() << ")" << std::endl;
        }

        std::cout << "\nedge info: " << std::endl;
        for(int i = 0; i < triMesh.nEdges(); i++)
        {
            std::cout << "e_" << i << ": " << "edge vertex: " << triMesh.edgeVertex(i, 0) << " " << triMesh.edgeVertex(i, 1) << ", w: " << initOmega(i) << std::endl;
        }

        std::cout << "\nface info: " << std::endl;
        for(int i = 0; i < triF.rows(); i++)
        {
            std::cout << "f_" << i << ": " << triF.row(i) << std::endl;
        }

        std::cout << "\nloop level: " << loopLevel << std::endl;
        std::cout << "\nvertex info: " << std::endl;
        for(int i = 0; i < loopTriV.rows(); i++)
        {
            std::cout << "v_" << i << ": " << loopTriV.row(i) << ", theta: " << loopedPhase(i) << ", amp: " << loopedAmp(i) << ", zvals: (" << loopedZvals[i].real() << ", " << loopedZvals[i].imag() << ")" << std::endl;
        }

        std::cout << "\nedge info: " << std::endl;
        for(int i = 0; i < subSecMesh.GetEdgeCount(); i++)
        {
            std::cout << "e_" << i << ": " << "edge vertex: " << subSecMesh.GetEdgeVerts(i)[0] << " " << subSecMesh.GetEdgeVerts(i)[1] << ", w: " << loopedOmega(i) << std::endl;
        }

        std::cout << "\nface info: " << std::endl;
        for(int i = 0; i < loopTriF.rows(); i++)
        {
            std::cout << "f_" << i << ": " << loopTriF.row(i) << std::endl;
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

    updateFieldsInView();
    // Show the gui
    polyscope::show();


    return 0;
}