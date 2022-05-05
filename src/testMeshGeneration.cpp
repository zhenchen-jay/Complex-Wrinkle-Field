#include "../include/testMeshGeneration.h"
#include <iostream>
#include <igl/boundary_loop.h>
#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>
#include <igl/triangle/triangulate.h>

bool mapPlane2Cylinder(Eigen::MatrixXd planeV, Eigen::MatrixXi planeF, Eigen::MatrixXd& cylinderV, Eigen::MatrixXi& cylinderF, Eigen::VectorXi* rest2cylinder)
{
    // assume the clamped vertices are aligned.

    std::vector<Eigen::Vector3d> V;
    int nverts = planeV.rows();
    int nfaces = planeF.rows();
    Eigen::VectorXi vertsMap(nverts);
    vertsMap.setConstant(-1);

    Eigen::VectorXi boundaryLoop;
    igl::boundary_loop(planeF, boundaryLoop);

    double minY = planeV.col(1).minCoeff();
    double maxY = planeV.col(1).maxCoeff();

    Eigen::Vector4i corners;

    std::map<int, double> top;
    std::map<int, double> bot;
    std::map<int, double> left;
    std::map<int, double> right;


    for (int i = 0; i < boundaryLoop.size(); i++)
    {
        int vid = boundaryLoop(i);
        if (planeV(vid, 1) == minY)
            bot[vid] = planeV(vid, 0);
        else if (planeV(vid, 1) == maxY)
            top[vid] = planeV(vid, 0);
    }

    double min = std::numeric_limits<double>::infinity();
    double max = -std::numeric_limits<double>::infinity();
    int flagMin = -1;
    int flagMax = -1;

    for (auto& it : bot)
    {
        if (planeV(it.first, 0) < min)
        {
            flagMin = it.first;
            min = planeV(it.first, 0);
        }
        if (planeV(it.first, 0) > max)
        {
            flagMax = it.first;
            max = planeV(it.first, 0);
        }
    }
    corners(0) = flagMin;
    corners(1) = flagMax;

    min = std::numeric_limits<double>::infinity();
    max = -std::numeric_limits<double>::infinity();
    flagMin = -1;
    flagMax = -1;

    for (auto& it : top)
    {
        if (planeV(it.first, 0) < min)
        {
            flagMin = it.first;
            min = planeV(it.first, 0);
        }
        if (planeV(it.first, 0) > max)
        {
            flagMax = it.first;
            max = planeV(it.first, 0);
        }
    }

    corners(2) = flagMax;
    corners(3) = flagMin;

    int startid = -1;
    for (int i = 0; i < boundaryLoop.size(); i++)
    {
        int vid = boundaryLoop(i);
        if (vid == corners(1))
        {
            startid = i;
            break;
        }
    }

    for (int i = 0; i < boundaryLoop.size(); i++)
    {
        int vid = boundaryLoop((startid + i) % boundaryLoop.size());
        right[vid] = planeV(vid, 1);
        if (vid == corners(2))
            break;
    }

    startid = -1;
    for (int i = 0; i < boundaryLoop.size(); i++)
    {
        int vid = boundaryLoop(i);
        if (vid == corners(3))
        {
            startid = i;
            break;
        }
    }

    for (int i = 0; i < boundaryLoop.size(); i++)
    {
        int vid = boundaryLoop((startid + i) % boundaryLoop.size());
        left[vid] = planeV(vid, 1);
        if (vid == corners(0))
            break;
    }

    double r = (planeV(corners(1), 0) - planeV(corners(0), 0)) / (2 * M_PI);


    int index = 0;
    for (int i = 0; i < nverts; i++)
    {
        if (left.find(i) != left.end() || right.find(i) != right.end())
        {
            continue;
        }
        double theta = (planeV(i, 0) - planeV(corners(0), 0)) / r;
        V.push_back(Eigen::Vector3d(r * std::cos(theta), r * std::sin(theta), planeV(i, 1)));
        vertsMap(i) = index;
        index++;
    }

    for (auto& it : left)
    {
        double theta = (planeV(it.first, 0) - planeV(corners(0), 0)) / r;
        V.push_back(Eigen::Vector3d(r * std::cos(theta), r * std::sin(theta), planeV(it.first, 1)));
        vertsMap(it.first) = index;
        index++;
    }


    for (auto& it : right)
    {
        for (auto& lit : left)
        {
            if (std::abs(it.second - lit.second) < 1e-6)
            {
                vertsMap(it.first) = vertsMap(lit.first);
                break;
            }
        }
        if (vertsMap(it.first) == -1)
        {
            std::cout << "mismatching!" << std::endl;
            std::cout << "mismatched vid: "<<it.first<<std::endl;
            return false;
        }
    }
    cylinderV.resize(index, 3);
    std::cout << "num of vertices: " << index << ", before: " << planeV.rows() << std::endl;

    for (int i = 0; i < index; i++)
    {
        cylinderV.row(i) = V[i].transpose();
    }
    cylinderF.resize(nfaces, 3);
    for (int i = 0; i < nfaces; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            cylinderF(i, j) = vertsMap(planeF(i, j));
        }
    }
    if(rest2cylinder)
        *rest2cylinder = vertsMap;
    return true;
}

void generateCylinder(double radius, double height, double triarea)
{
    Eigen::MatrixXd checkBoxV, halfRegularV, irregularV, checkBoxCV, halfRegularCV, irregularCV;
    Eigen::MatrixXi checkBoxF, halfRegularF, irregularF, checkBoxCF, halfRegularCF, irregularCF;

    double l = 2 * M_PI * radius;
    double w = height;
    double area = l * w;
    int N = (0.5 * std::sqrt(area / triarea));
    N = N > 1 ? N : 2;
    N = 2;
    double deltaX = l / (2.0 * N);
    double deltaY = w / (2.0 * N);

    int index = 0;
    checkBoxV.resize((2 * N + 1) * (2 * N + 1), 3);
    for (int i = 0; i <= 2 * N; i++)
        for (int j = 0; j <= 2 * N; j++)
        {
            checkBoxV.row(index) << j * deltaX, i* deltaY, 0;
            index++;
        }
    int M = 2 * N + 1;
    checkBoxF.resize(8 * N * N, 3);
    index = 0;
    for (int i = 0; i < 2 * N; i++)
        for (int j = 0; j < 2 * N; j++)
        {
            checkBoxF.row(index) << i * M + j, i* M + j + 1, (1 + i)* M + j + 1;
            checkBoxF.row(index + 1) << (1 + i) * M + j + 1, (1 + i)* M + j, i* M + j;
            index += 2;
        }
    if (mapPlane2Cylinder(checkBoxV, checkBoxF, checkBoxCV, checkBoxCF, NULL))
    {
        for (int i = 0; i < checkBoxCV.rows(); i++)
        {
            checkBoxCV(i, 2) -= height / 2.0;
        }
        for (int i = 0; i < checkBoxV.rows(); i++)
        {
            double x = checkBoxV(i, 0);
            double y = checkBoxV(i, 1);
            checkBoxV(i, 0) = 0;
            checkBoxV(i, 1) = x;
            checkBoxV(i, 2) = y;
        }
        igl::writeOBJ("checkBoxCylinder.obj", checkBoxCV, checkBoxCF);
        igl::writeOBJ("checkBoxPlane.obj", checkBoxV, checkBoxF);
    }

    Eigen::MatrixXd regularV = checkBoxV, regularCV;
    Eigen::MatrixXi regularF = checkBoxF, regularCF = checkBoxCF;
    double theta = M_PI / 3;
    index = 0;
    for (int i = 0; i <= 2 * N; i++)
        for (int j = 0; j <= 2 * N; j++)
        {
            regularV.row(index) << j * deltaX - 1.0 / tan(theta) * (i * deltaY), i* deltaY, 0;
            index++;
        }
    if (mapPlane2Cylinder(regularV, regularF, regularCV, regularCF, NULL))
    {
        for (int i = 0; i < regularCV.rows(); i++)
        {
            regularCV(i, 2) -= height / 2.0;
        }
        for (int i = 0; i < regularV.rows(); i++)
        {
            double x = regularV(i, 0);
            double y = regularV(i, 1);
            regularV(i, 0) = 0;
            regularV(i, 1) = x;
            regularV(i, 2) = y;
        }
        igl::writeOBJ("regularCylinder.obj", regularCV, regularCF);
        igl::writeOBJ("regularPlane.obj", regularV, regularF);
    }

    Eigen::MatrixXd planeV(2 * M, 2);
    Eigen::MatrixXi planeE(2 * M + M - 2, 2);

    for (int i = 0; i < M; i++)
    {
        planeV.row(i) << 0, i* w / (M - 1);
    }
    for (int i = 0; i < M; i++)
    {
        planeV.row(M + i) << l, w - i * w / (M - 1);
    }


    for (int i = 0; i < 2 * M; i++)
    {
        planeE.row(i) << i, (i + 1) % (2 * M);
    }

    for (int i = 0; i < M - 2; i++)
    {
        planeE.row(2 * M + i) << i + 1, 2 * M - 2 - i;
    }



    Eigen::MatrixXd V2d1;
    Eigen::MatrixXi F1;
    Eigen::MatrixXi H1(0, 2);
    std::stringstream ss;
    ss << "a" << triarea << "q";
    igl::triangle::triangulate(planeV, planeE, H1, ss.str(), V2d1, F1);
    halfRegularV.resize(V2d1.rows(), 3);
    halfRegularV.setZero();
    halfRegularV.block(0, 0, halfRegularV.rows(), 2) = V2d1;
    halfRegularF = F1;

    if (mapPlane2Cylinder(halfRegularV, halfRegularF, halfRegularCV, halfRegularCF, NULL))
    {
        for (int i = 0; i < halfRegularCV.rows(); i++)
        {
            halfRegularCV(i, 2) -= height / 2.0;
        }
        for (int i = 0; i < halfRegularV.rows(); i++)
        {
            double x = halfRegularV(i, 0);
            double y = halfRegularV(i, 1);
            halfRegularV(i, 0) = 0;
            halfRegularV(i, 1) = x;
            halfRegularV(i, 2) = y;
        }
        igl::writeOBJ("halfRegularCylinder.obj", halfRegularCV, halfRegularCF);
        igl::writeOBJ("halfRegularPlane.obj", halfRegularV, halfRegularF);
    }

    planeV.resize(4 * M - 4, 2);
    planeE.resize(4 * M - 4, 2);

    for (int i = 0; i < M; i++)
    {
        planeV.row(i) << 0, i* w / (M - 1);
    }
    for (int i = 1; i < M; i++)
    {
        planeV.row(M - 1 + i) << i * l / (M - 1), w;
    }
    for (int i = 1; i < M; i++)
    {
        planeV.row(2 * (M - 1) + i) << l, w - i * w / (M - 1);
    }
    for (int i = 1; i < M - 1; i++)
    {
        planeV.row(3 * (M - 1) + i) << l - i * l / (M - 1), 0;
    }

    for (int i = 0; i < 4 * (M - 1); i++)
    {
        planeE.row(i) << i, (i + 1) % (4 * (M - 1));
    }

    Eigen::MatrixXd V2d;
    Eigen::MatrixXi F;
    Eigen::MatrixXi H(0, 2);
    const std::string flags = "q20a" + std::to_string(triarea);
    igl::triangle::triangulate(planeV, planeE, H, flags, V2d, F);
    irregularV.resize(V2d.rows(), 3);
    irregularV.setZero();
    irregularV.block(0, 0, irregularV.rows(), 2) = V2d;
    irregularF = F;

    igl::writeOBJ("irregularPlane.obj", irregularV, irregularF);
    if (mapPlane2Cylinder(irregularV, irregularF, irregularCV, irregularCF, NULL))
    {
        for (int i = 0; i < irregularCV.rows(); i++)
        {
            irregularCV(i, 2) -= height / 2.0;
        }
        for (int i = 0; i < irregularV.rows(); i++)
        {
            double x = irregularV(i, 0);
            double y = irregularV(i, 1);
            irregularV(i, 0) = 0;
            irregularV(i, 1) = x;
            irregularV(i, 2) = y;
        }
        igl::writeOBJ("irregularCylinder.obj", irregularCV, irregularCF);
        igl::writeOBJ("irregularPlane.obj", irregularV, irregularF);
    }

}

void generateCylinderWaves(const Eigen::MatrixXd& restV, const MeshConnectivity& restMesh, const Eigen::MatrixXd& cylinderV, const MeshConnectivity& cylinderMesh, double numWaves, double ampMag, Eigen::VectorXd& amp, Eigen::VectorXd& edgeOmega, std::vector<std::complex<double>>& zvals)
{
    int nrestverts = restV.rows();
    int nverts = cylinderV.rows();
    int nedges = cylinderMesh.nEdges();
    int nfaces = cylinderMesh.nFaces();

    Eigen::VectorXd restPhi(nrestverts);
    for(int i = 0; i < nrestverts; i++)
    {
        double x = restV(i, 0);
        restPhi(i) = numWaves * x;
    }
    amp.setOnes(nverts);
    amp *= ampMag;

    edgeOmega.setZero(nedges);
    zvals.resize(nverts, 0);
    for(int i = 0; i < nfaces; i++)
    {
        for(int j = 0; j < 3; j++)
        {
            int eid = cylinderMesh.faceEdge(i, j);
            int vid = cylinderMesh.faceVertex(i, j);

            double w = restPhi(restMesh.faceVertex(i, (j + 2) % 3)) - restPhi(restMesh.faceVertex(i, (j+1)%3));
            if(cylinderMesh.faceVertex(i, (j+1)%3) > cylinderMesh.faceVertex(i, (j+2) % 3))
                w *= -1;
            edgeOmega(eid) = w;

            double theta = restPhi(restMesh.faceVertex(i, j));
            zvals[vid] = ampMag * std::complex<double>(std::cos(theta), std::sin(theta));
        }
    }

}

void generateCylinderWaves(double radius, double height, double triarea, double numWaves, double ampMag, Eigen::MatrixXd& cylinderV, Eigen::MatrixXi& cylinderF, Eigen::VectorXd& amp, Eigen::VectorXd& dphi, std::vector<std::complex<double>>& zvals)
{
    Eigen::MatrixXd irregularV, irregularCV;
    Eigen::MatrixXi irregularF, irregularCF;

    double l = 2 * M_PI * radius;
    double w = height;
    double area = l * w;

    int N = (0.5 * std::sqrt(area / triarea));
    N = N > 1 ? N : 2;
    int M = 2 * N + 1;

    Eigen::MatrixXd planeV(2 * M, 2);
    Eigen::MatrixXi planeE(2 * M + M - 2, 2);
    planeV.resize(4 * M - 4, 2);
    planeE.resize(4 * M - 4, 2);

    for (int i = 0; i < M; i++)
    {
        planeV.row(i) << 0, i * w / (M - 1);
    }
    for (int i = 1; i < M; i++)
    {
        planeV.row(M - 1 + i) << i * l / (M - 1), w;
    }
    for (int i = 1; i < M; i++)
    {
        planeV.row(2 * (M - 1) + i) << l, w - i * w / (M - 1);
    }
    for (int i = 1; i < M - 1; i++)
    {
        planeV.row(3 * (M - 1) + i) << l - i * l / (M - 1), 0;
    }

    for (int i = 0; i < 4 * (M - 1); i++)
    {
        planeE.row(i) << i, (i + 1) % (4 * (M - 1));
    }

    Eigen::MatrixXd V2d;
    Eigen::MatrixXi F;
    Eigen::MatrixXi H(0, 2);
    const std::string flags = "q20a" + std::to_string(triarea);
    igl::triangle::triangulate(planeV, planeE, H, flags, V2d, F);
    irregularV.resize(V2d.rows(), 3);
    irregularV.setZero();
    irregularV.block(0, 0, irregularV.rows(), 2) = V2d;
    irregularF = F;

    igl::writeOBJ("irregularPlane.obj", irregularV, irregularF);
    Eigen::VectorXi rest2cylinder;
    if (mapPlane2Cylinder(irregularV, irregularF, irregularCV, irregularCF, &rest2cylinder))
    {
        for (int i = 0; i < irregularCV.rows(); i++)
        {
            irregularCV(i, 2) -= height / 2.0;
        }
        igl::writeOBJ("irregularCylinder.obj", irregularCV, irregularCF);
        igl::writeOBJ("irregularPlane.obj", irregularV, irregularF);

        cylinderV = irregularCV;
        cylinderF = irregularCF;

        generateCylinderWaves(irregularV, MeshConnectivity(irregularF), irregularCV, MeshConnectivity(irregularCF), numWaves, ampMag, amp, dphi, zvals);
    }
}