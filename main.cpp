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
#include <igl/file_dialog_open.h>
#include <igl/file_dialog_save.h>
#include <igl/boundary_loop.h>
#include <igl/triangle/triangulate.h>
#include <filesystem>
#include "polyscope/messages.h"
#include "polyscope/point_cloud.h"
#include "polyscope/surface_mesh.h"

#include <iostream>
#include <fstream>
#include <unordered_set>
#include <utility>


#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

#ifndef GRAIN_SIZE
#define GRAIN_SIZE 10
#endif


#include "include/InterpolationScheme/PhaseInterpolation.h"
#include "include/InterpolationScheme/PlaneWaveExtraction.h"
#include "include/MeshLib/MeshConnectivity.h"
#include "include/MeshLib/MeshUpsampling.h"
#include "include/Visualization/PaintGeometry.h"
#include "include/InterpolationScheme/VecFieldSplit.h"
#include "include/Optimization/NewtonDescent.h"
#include "include/DynamicInterpolation/GetInterpolatedValues.h"


Eigen::MatrixXd triV2D, triV3D, upsampledTriV2D, upsampledTriV3D, wrinkledV;
Eigen::MatrixXi triF2D, triF3D, upsampledTriF2D, upsampledTriF3D;
std::vector<std::pair<int, Eigen::Vector3d>> bary;

std::vector<std::complex<double>> zvals;
std::vector<std::complex<double>> theoZVals;
Eigen::MatrixXd omegaFields;
Eigen::MatrixXd theoOmega;

Eigen::MatrixXd planeFields;
Eigen::MatrixXd whirlFields;

Eigen::VectorXd phaseField(0);
Eigen::VectorXd ampField(0);


Eigen::MatrixXd dataV;
Eigen::MatrixXi dataF;
Eigen::MatrixXd dataVec;
Eigen::MatrixXd curColor;

int loopLevel = 1;

bool isVisualizePhase = false;
bool isVisualizeAmp = false;
bool isVisualizeWrinkles = false;
bool isShowOnlyWhirlPool = false;
bool isShowOnlyPlaneWave = false;

bool isVisualizeVertexOmega = false;
bool isFixed = true;

std::vector<Eigen::MatrixXd> wrinkledVs;
std::vector<Eigen::VectorXd> phaseFieldList;
std::vector<Eigen::VectorXd> ampFieldList;
std::vector<Eigen::MatrixXd> omegaFieldsList;
std::vector<std::vector<std::complex<double>>> zvalsList;

PhaseInterpolation model;
PaintGeometry mPaint;

int numFrames = 1000;
int curFrame = 0;
int sigIndex1 = 1;
int sigIndex2 = 1;
double triarea = 0.1;

double fixedx = 0;
double fixedy = 0;
Eigen::Vector2d fixedv(1.0, -0.5);

enum MotionType
{
	MT_LINEAR = 0,
	MT_ENTIRE_LINEAR = 1,
	MT_ROTATION = 2,
	MT_SINEWAVE = 3,
	MT_COMPLICATE = 4,
	MT_SPIRAL = 5
};


enum TargetType {
	Whirlpool = 0,
	PlaneWave = 1,
	Summation = 2,
	YShape = 3,
	TwoWhirlPool,
	Random = 4
};

enum InterpolationType {
	PureWhirlpool = 0,
	PurePlaneWave = 1,
	NaiveSplit = 2,
	NewSplit = 3,
	JustLinear = 4
};

TargetType tarType = TargetType::PlaneWave;
InterpolationType interType = InterpolationType::NewSplit;

MotionType motionType = MotionType::MT_LINEAR;

void generateSquare(double length, double width, double triarea, Eigen::MatrixXd& irregularV, Eigen::MatrixXi& irregularF)
{
	double area = length * width;
	int N = (0.25 * std::sqrt(area / triarea));
	N = N > 1 ? N : 1;
	double deltaX = length / (4.0 * N);
	double deltaY = width / (4.0 * N);

	Eigen::MatrixXd planeV;
	Eigen::MatrixXi planeE;

	planeV.resize(10, 2);
	planeE.resize(10, 2);

	for (int i = -2; i <= 2; i++)
	{
		planeV.row(i + 2) << length / 4.0 * i, -width / 2.0;
	}

	for (int i = 2; i >= -2; i--)
	{
		planeV.row(5 + 2 - i) << length / 4.0 * i, width / 2.0;
	}

	for (int i = 0; i < 10; i++)
	{
		planeE.row(i) << i, (i + 1) % 10;
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

//	V2d.resize(4, 3);
//	V2d << 0, 0, 0,
//	1, 0, 0,
//	1, 1, 0,
//	0, 1, 0;
//
//	F.resize(2, 3);
//	F << 0, 1, 2,
//	2, 3, 0;

	irregularV.resize(V2d.rows(), 3);
	irregularV.setZero();
	irregularV.block(0, 0, irregularV.rows(), 2) = V2d.block(0, 0, irregularV.rows(), 2);
	irregularF = F;
}


void generateWhirlPool(double centerx, double centery, Eigen::MatrixXd& w, std::vector<std::complex<double>>& z, int pow = 1, std::vector<std::complex<double>> *upsampledZ = NULL, Eigen::MatrixXd *upOmega = NULL)
{
	z.resize(triV2D.rows());
	w.resize(triV2D.rows(), 2);
	std::cout << "whirl pool center: " << centerx << ", " << centery << std::endl;

	for (int i = 0; i < z.size(); i++)
	{
		double x = triV2D(i, 0) - centerx;
		double y = triV2D(i, 1) - centery;
		double rsquare = x * x + y * y;

		z[i] = std::pow(std::complex<double>(x, y), pow);

		if (std::abs(std::sqrt(rsquare)) < 1e-10)
			w.row(i) << 0, 0;
		else
//			w.row(i) << -y / rsquare, x / rsquare;
            w.row(i) << -pow * y / rsquare, pow * x / rsquare;
	}

	if(upsampledZ)
	{
	    upsampledZ->resize(upsampledTriV2D.rows());
	    for(int i = 0; i < upsampledZ->size(); i++)
	    {
	        double x = upsampledTriV2D(i, 0) - centerx;
	        double y = upsampledTriV2D(i, 1) - centery;
	        double rsquare = x * x + y * y;

			upsampledZ->at(i) = std::pow(std::complex<double>(x, y), pow);
	    }
	}
	if(upOmega)
	{
	    upOmega->resize(upsampledTriV2D.rows(), 2);
	    for(int i = 0; i < upsampledZ->size(); i++)
	    {
	        double x = upsampledTriV2D(i, 0) - centerx;
	        double y = upsampledTriV2D(i, 1) - centery;
	        double rsquare = x * x + y * y;

	        if (std::abs(std::sqrt(rsquare)) < 1e-10)
	            upOmega->row(i) << 0, 0;
	        else
	            upOmega->row(i) << -pow * y / rsquare, pow * x / rsquare;
	    }

	}
}

void generatePlaneWave(Eigen::Vector2d v, Eigen::MatrixXd& w, std::vector<std::complex<double>>& z, std::vector<std::complex<double>> *upsampledZ = NULL, Eigen::MatrixXd *upOmega = NULL)
{
    z.resize(triV2D.rows());
    w.resize(triV2D.rows(), 2);
    std::cout << "plane wave direction: " << v.transpose() << std::endl;

    for (int i = 0; i < z.size(); i++)
    {
        double theta = v.dot(triV2D.row(i).segment<2>(0));
        double x = std::cos(theta);
        double y = std::sin(theta);
        z[i] = std::complex<double>(x, y);
        w.row(i) = v;
    }

    if(upsampledZ)
    {
        upsampledZ->resize(upsampledTriV2D.rows());
        for(int i = 0; i < upsampledZ->size(); i++)
        {
            double theta = v.dot(upsampledTriV2D.row(i).segment<2>(0));
            double x = std::cos(theta);
            double y = std::sin(theta);
            upsampledZ->at(i) = std::complex<double>(x, y);
        }
    }
    if(upOmega)
    {
        upOmega->resize(upsampledTriV2D.rows(), 2);
        for(int i = 0; i < upsampledZ->size(); i++)
        {
            upOmega->row(i) = v;
        }
    }
}

void generateTwoWhirlPool(double centerx0, double centery0, double centerx1, double centery1, Eigen::MatrixXd& w, std::vector<std::complex<double>>& z, int n0 = 1, int n1 = 1, std::vector<std::complex<double>>* upsampledZ = NULL, Eigen::MatrixXd *upOmega = NULL)
{
	Eigen::MatrixXd w0, w1, upOmega0, upOmega1;
	std::vector<std::complex<double>> z0, z1, upsampledZ0, upsampledZ1;

	generateWhirlPool(centerx0, centery0, w0, z0, n0, upsampledZ ? &upsampledZ0 : NULL, upOmega ? &upOmega0 : NULL);
	generateWhirlPool(centerx1, centery1, w1, z1, n1, upsampledZ ? &upsampledZ1 : NULL, upOmega ? &upOmega1 : NULL);

	std::cout << "whirl pool center: " << centerx0 << ", " << centery0 << std::endl;
	std::cout << "whirl pool center: " << centerx1 << ", " << centery1 << std::endl;

	z.resize(triV2D.rows());
	w.resize(triV2D.rows(), 2);

	w = w0 + w1;

	for (int i = 0; i < z.size(); i++)
	{
		z[i] = z0[i] * z1[i];
	}

	if (upsampledZ)
	{
		upsampledZ->resize(upsampledTriV2D.rows());
		for (int i = 0; i < upsampledZ->size(); i++)
		{
			upsampledZ->at(i) = upsampledZ0[i] * upsampledZ1[i];
		}
	}

	if(upOmega)
	{
	    *upOmega = upOmega0 + upOmega1;
	}
}

void generatePlaneSumWhirl(double centerx, double centery, Eigen::Vector2d v, Eigen::MatrixXd& w, std::vector<std::complex<double>>& z, std::vector<std::complex<double>> *upsampledZ = NULL, Eigen::MatrixXd* upOmega = NULL)
{
    z.resize(triV2D.rows());
    w.resize(triV2D.rows(), 2);
    std::cout << "whirl pool center: " << centerx << ", " << centery << std::endl;
    std::cout << "plane wave direction: " << v.transpose() << std::endl;

    for (int i = 0; i < z.size(); i++)
    {
        double x = triV2D(i, 0) - centerx;
        double y = triV2D(i, 1) - centery;
        double rsquare = x * x + y * y;

        double theta = v.dot(triV2D.row(i).segment<2>(0));

        z[i] = std::complex<double>(x, y) * std::complex<double>(std::cos(theta), std::sin(theta));

        if (std::abs(std::sqrt(rsquare)) < 1e-10)
            w.row(i) << 0, 0;
        else
            w.row(i) << -y / rsquare, x / rsquare;
        w.row(i) += v;
    }

    if(upsampledZ)
    {
        upsampledZ->resize(upsampledTriV2D.rows());
        if(upOmega)
            upOmega->resize(upsampledTriV2D.rows(), 2);

        for(int i = 0; i < upsampledZ->size(); i++)
        {
            double x = upsampledTriV2D(i, 0) - centerx;
            double y = upsampledTriV2D(i, 1) - centery;

            double rsquare = x * x + y * y;

            double theta = v.dot(upsampledTriV2D.row(i).segment<2>(0));

            upsampledZ->at(i) = std::complex<double>(x, y) * std::complex<double>(std::cos(theta), std::sin(theta));

            if(upOmega)
            {
                if (std::abs(std::sqrt(rsquare)) < 1e-10)
                    upOmega->row(i) << 0, 0;
                else
                    upOmega->row(i) << -y / rsquare, x / rsquare;
                upOmega->row(i) += v;
            }

        }
    }
}

void generateYshape(Eigen::Vector2d w1, Eigen::Vector2d w2, Eigen::MatrixXd &w, std::vector<std::complex<double>> &z, std::vector<std::complex<double>> *upsampledZ, Eigen::MatrixXd* upOmega = NULL)
{
    z.resize(triV2D.rows());
    w.resize(triV2D.rows(), 2);

    std::cout << "w1: " << w1.transpose() << std::endl;
    std::cout << "w2: " << w2.transpose() << std::endl;

    double ymax = triV2D.col(1).maxCoeff();
    double ymin = triV2D.col(1).minCoeff();

    for (int i = 0; i < z.size(); i++)
    {
        double theta = w1.dot(triV2D.row(i).segment<2>(0));
        double x = std::cos(theta);
        double y = std::sin(theta);
        std::complex<double> z1 = std::complex<double>(x, y);

        theta = w2.dot(triV2D.row(i).segment<2>(0));
        x = std::cos(theta);
        y = std::sin(theta);
        std::complex<double> z2 = std::complex<double>(x, y);

        double weight = (triV2D(i, 1) - triV2D.col(1).minCoeff()) / (triV2D.col(1).maxCoeff() - triV2D.col(1).minCoeff());
        z[i] = (1 - weight) * z1 + weight * z2;

        std::complex<double> z1x = z1 * std::complex<double>(0, w1(0));
        std::complex<double> z1y = z1 * std::complex<double>(0, w1(1));

        std::complex<double> z2x = z2 * std::complex<double>(0, w2(0));
        std::complex<double> z2y = z2 * std::complex<double>(0, w2(1));

        double wx = 0;
        double wy = 1 / (ymax - ymin);

        w(i, 0) = (std::complex<double>(z[i].real(), -z[i].imag()) * ((1 - weight) * z1x + weight * z2x + wx * (z2 - z1))).imag();
        w(i, 1) = (std::complex<double>(z[i].real(), -z[i].imag()) * ((1 - weight) * z1y + weight * z2y + wy * (z2 - z1))).imag();

    }

    if(upsampledZ)
    {
        upsampledZ->resize(upsampledTriV2D.rows());
        if(upOmega)
            upOmega->resize(upsampledTriV2D.rows(), 2);
        for(int i = 0; i < upsampledZ->size(); i++)
        {
            double theta = w1.dot(upsampledTriV2D.row(i).segment<2>(0));
            double x = std::cos(theta);
            double y = std::sin(theta);
            std::complex<double> z1 = std::complex<double>(x, y);

            theta = w2.dot(upsampledTriV2D.row(i).segment<2>(0));
            x = std::cos(theta);
            y = std::sin(theta);
            std::complex<double> z2 = std::complex<double>(x, y);

            double weight = (upsampledTriV2D(i, 1) - upsampledTriV2D.col(1).minCoeff()) / (upsampledTriV2D.col(1).maxCoeff() - upsampledTriV2D.col(1).minCoeff());
            upsampledZ->at(i) = (1 - weight) * z1 + weight * z2;

            if(upOmega)
            {
                std::complex<double> z1x = z1 * std::complex<double>(0, w1(0));
                std::complex<double> z1y = z1 * std::complex<double>(0, w1(1));

                std::complex<double> z2x = z2 * std::complex<double>(0, w2(0));
                std::complex<double> z2y = z2 * std::complex<double>(0, w2(1));

                double wx = 0;
                double wy = 1 / (ymax - ymin);

                (*upOmega)(i, 0) = (std::complex<double>(upsampledZ->at(i).real(), -upsampledZ->at(i).imag()) * ((1 - weight) * z1x + weight * z2x + wx * (z2 - z1))).imag();
                (*upOmega)(i, 1) = (std::complex<double>(upsampledZ->at(i).real(), -upsampledZ->at(i).imag()) * ((1 - weight) * z1y + weight * z2y + wy * (z2 - z1))).imag();
            }
        }
    }
}

void generateRandom(Eigen::MatrixXd& w, std::vector<std::complex<double>>& z, std::vector<std::complex<double>> *upsampledZ = NULL, Eigen::MatrixXd *upOmega = NULL)
{
    z.resize(triV2D.rows());
    w.resize(triV2D.rows(), 2);
    for(int i=0; i < w.rows(); i++)
    {
        w.row(i).setRandom();

        Eigen::Vector2d v = Eigen::Vector2d::Random();
        z[i] = std::complex<double>(v(0), v(1));
    }
    if(upsampledZ)
    {
        upsampledZ->resize(upsampledTriV2D.rows());
        if(upOmega)
            upOmega->setRandom(upsampledTriV2D.rows(), 2);
        for(int i = 0; i < upsampledZ->size(); i++)
        {
            Eigen::Vector2d v = Eigen::Vector2d::Random();
            upsampledZ->at(i) = std::complex<double>(v(0), v(1));
        }
    }
}

void generateSingularity(double& x0, double& y0, double t, MotionType motion)
{
	double r = 0.8;
	if (motion == MotionType::MT_LINEAR)
	{
		x0 = -r + 2 * r * t;
		y0 = 0;
	}
	else if (motion == MotionType::MT_ENTIRE_LINEAR)
	{
		x0 = -1.0 + 2 * t;
		y0 = 0;
	}
	else if (motion == MotionType::MT_ROTATION)
	{
		double theta = t * 2 * M_PI;
		x0 = r * std::cos(theta);
		y0 = r * std::sin(theta);
	}
	else if (motion == MotionType::MT_SINEWAVE)
	{
		x0 = -r + 2 * r * t;
		y0 = r * std::sin(M_PI / r * x0);
	}
	else if (motion == MotionType::MT_COMPLICATE)
	{
		double theta = t * 4 * M_PI;
		x0 = r * std::cos(theta);

		double p = -r + 2 * r * t;
		y0 = r * std::sin(4 * M_PI / r * p);
	}
	else if (motion == MotionType::MT_SPIRAL)
	{
		double curR = (1 - t) * r;
		double theta = t * 6 * M_PI;
		x0 = curR * std::cos(theta);
		y0 = curR * std::sin(theta);
	}
	else
	{
		std::cout << "undefined motion type!" << std::endl;
		exit(1);
	}
}

void registerVecMesh()
{
    int ndataVerts = triV2D.rows();
    int ndataFaces = triF2D.rows();

    int nverts = ndataVerts;
    int nfaces = ndataFaces;

	int nupverts = upsampledTriV2D.rows();
	int nupfaces = upsampledTriF2D.rows();

    ndataVerts = 3 * nverts + 4 * nupverts;
    ndataFaces = 3 * nfaces + 4 * nupfaces;
//    ndataVerts = 3 * nverts + 6 * nupverts;
//    ndataFaces = 3 * nfaces + 6 * nupfaces;


    int currentDataVerts = nverts;
    int currentDataFaces = nfaces;

	// vector fields
    dataV.resize(ndataVerts, 3);
    dataF.resize(ndataFaces, 3);

	dataV.block(0, 0, nverts, 3) = triV2D;
	dataF.block(0, 0, nfaces, 3) = triF2D;

	Eigen::MatrixXd shiftV = triV2D;
	double shiftx = 1.5 * (triV2D.col(0).maxCoeff() - triV2D.col(0).minCoeff());
	double shifty = 1.5 * (triV2D.col(1).maxCoeff() - triV2D.col(1).minCoeff());
	shiftV.col(0).setConstant(0.5 * shiftx);
	shiftV.col(1).setConstant(shifty);
	shiftV.col(2).setConstant(0);


	Eigen::MatrixXi shiftF = triF2D;
	shiftF.setConstant(currentDataVerts);

	dataV.block(currentDataVerts, 0, nverts, 3) = triV2D - shiftV;
	dataF.block(currentDataFaces, 0, nfaces, 3) = triF2D + shiftF;

	currentDataVerts += nverts;
	currentDataFaces += nfaces;


	shiftV.col(0).setConstant(-0.5 * shiftx);
	shiftV.col(1).setConstant(shifty);
	shiftV.col(2).setConstant(0);
	shiftF.setConstant(currentDataVerts);
	dataV.block(currentDataVerts, 0, nverts, 3) = triV2D - shiftV;
	dataF.block(currentDataFaces, 0, nfaces, 3) = triF2D + shiftF;
	currentDataVerts += nverts;
	currentDataFaces += nfaces;

    curColor.resize(ndataVerts, 3);


	dataVec = dataV;
	dataVec.setZero();

	for (int i = 0; i < triV2D.rows(); i++)
	{
		dataVec.row(i) << omegaFields(i, 0), omegaFields(i, 1), 0;
		dataVec.row(i + nverts) << planeFields(i, 0), planeFields(i, 1), 0;
		dataVec.row(i + 2 * nverts) << whirlFields(i, 0), whirlFields(i, 1), 0;
	}
//	for (int i = 0; i < upsampledTriV2D.rows(); i++)
//	{
//	    dataVec.row(i + 3 * nverts) << theoOmega(i, 0), theoOmega(i, 1), 0;
//	    dataVec.row(i + 3 * nverts + nupverts) << upsampledOmega(i, 0), upsampledOmega(i, 1), 0;
//	}
//	shiftV = upsampledTriV2D;
//	shiftV.col(0).setConstant(2 * shiftx);
//	shiftV.col(1).setConstant(0);
//	shiftV.col(2).setConstant(0);
//
//	shiftF = upsampledTriF2D;
//	shiftF.setConstant(currentDataVerts);
//
//	dataV.block(currentDataVerts, 0, nupverts, 3) = upsampledTriV2D - shiftV;
//	dataF.block(currentDataFaces, 0, nupfaces, 3) = upsampledTriF2D + shiftF;
//
//
//	currentDataVerts += nupverts;
//	currentDataFaces += nupfaces;
//
//	shiftV = upsampledTriV2D;
//	shiftV.col(0).setConstant(2 * shiftx);
//	shiftV.col(1).setConstant(shifty);
//	shiftV.col(2).setConstant(0);
//
//	shiftF = upsampledTriF2D;
//	shiftF.setConstant(currentDataVerts);
//
//	dataV.block(currentDataVerts, 0, nupverts, 3) = upsampledTriV2D - shiftV;
//	dataF.block(currentDataFaces, 0, nupfaces, 3) = upsampledTriF2D + shiftF;
//
//	currentDataVerts += nupverts;
//	currentDataFaces += nupfaces;
//


    curColor.col(0).setConstant(1.0);
    curColor.col(1).setConstant(1.0);
    curColor.col(2).setConstant(1.0);



	// theo zvals
	shiftV = upsampledTriV2D;
	shiftV.col(0).setConstant(2 * shiftx);
	shiftV.col(1).setConstant(0);
	shiftV.col(2).setConstant(0);

	shiftF = upsampledTriF2D;
	shiftF.setConstant(currentDataVerts);

	dataV.block(currentDataVerts, 0, nupverts, 3) = upsampledTriV2D - shiftV;
	dataF.block(currentDataFaces, 0, nupfaces, 3) = upsampledTriF2D + shiftF;


	Eigen::VectorXd theoTheta(nupverts);
	for (int i = 0; i < nupverts; i++)
	{
		theoTheta(i) = std::arg(theoZVals[i]);
	}
	mPaint.setNormalization(false);
	Eigen::MatrixXd phiColor;
	phiColor = mPaint.paintPhi(theoTheta);

	curColor.block(currentDataVerts, 0, nupverts, 3) = phiColor;

	currentDataVerts += nupverts;
	currentDataFaces += nupfaces;

	shiftV = upsampledTriV2D;
	shiftV.col(0).setConstant(3 * shiftx);
	shiftV.col(1).setConstant(0);
	shiftV.col(2).setConstant(0);

	shiftF = upsampledTriF2D;
	shiftF.setConstant(currentDataVerts);

	dataV.block(currentDataVerts, 0, nupverts, 3) = upsampledTriV2D - shiftV;
	dataF.block(currentDataFaces, 0, nupfaces, 3) = upsampledTriF2D + shiftF;

	Eigen::VectorXd theoAmp(nupverts);
	for (int i = 0; i < nupverts; i++)
	{
		theoAmp(i) = std::abs(theoZVals[i]);
	}

	double ampMax = std::max(theoAmp.maxCoeff(), ampField.maxCoeff());
	mPaint.setNormalization(false);
	Eigen::MatrixXd ampColor;
	ampColor = mPaint.paintAmplitude(theoAmp / ampMax);

	curColor.block(currentDataVerts, 0, nupverts, 3) = ampColor;

	currentDataVerts += nupverts;
	currentDataFaces += nupfaces;

	// interpolated part
	shiftV = upsampledTriV2D;
	shiftV.col(0).setConstant(2 * shiftx);
	shiftV.col(1).setConstant(shifty);
	shiftV.col(2).setConstant(0);

	shiftF = upsampledTriF2D;
	shiftF.setConstant(currentDataVerts);

	dataV.block(currentDataVerts, 0, nupverts, 3) = upsampledTriV2D - shiftV;
	dataF.block(currentDataFaces, 0, nupfaces, 3) = upsampledTriF2D + shiftF;

	mPaint.setNormalization(false);
	phiColor = mPaint.paintPhi(phaseField);

	curColor.block(currentDataVerts, 0, nupverts, 3) = phiColor;
	currentDataVerts += nupverts;
	currentDataFaces += nupfaces;

	shiftV = upsampledTriV2D;
	shiftV.col(0).setConstant(3 * shiftx);
	shiftV.col(1).setConstant(shifty);
	shiftV.col(2).setConstant(0);

	shiftF = upsampledTriF2D;
	shiftF.setConstant(currentDataVerts);

	dataV.block(currentDataVerts, 0, nupverts, 3) = upsampledTriV2D - shiftV;
	dataF.block(currentDataFaces, 0, nupfaces, 3) = upsampledTriF2D + shiftF;

	mPaint.setNormalization(false);
	ampColor = mPaint.paintAmplitude(ampField / ampMax);

	curColor.block(currentDataVerts, 0, nupverts, 3) = ampColor;
	currentDataVerts += nupverts;
	currentDataFaces += nupfaces;

    
    polyscope::registerSurfaceMesh("input mesh", dataV, dataF);
}

void updateVecFieldsInView()
{
    std::cout << "update view" << std::endl;
    registerVecMesh();
    polyscope::getSurfaceMesh("input mesh")->addVertexColorQuantity("VertexColor", curColor);
    polyscope::getSurfaceMesh("input mesh")->getQuantity("VertexColor")->setEnabled(true);

    polyscope::getSurfaceMesh("input mesh")->addVertexVectorQuantity("vertex vector field", dataVec);
    polyscope::getSurfaceMesh("input mesh")->getQuantity("vertex vector field")->setEnabled(true);
}

void registerMesh(int frameId)
{
	if (frameId < 0 || frameId >= numFrames)
	{
		std::cout << "out of range, reset to the initial." << std::endl;
		frameId = 0;
	}

	int ndataVerts = upsampledTriV3D.rows();
	int ndataFaces = upsampledTriF3D.rows();

	int nupsampledVerts = ndataVerts;
	int nupsampledFaces = ndataFaces;

	if (isVisualizeWrinkles)
	{
		ndataVerts += nupsampledVerts;
		ndataFaces += nupsampledFaces;
	}

	if (isVisualizeAmp)
	{
		ndataVerts += nupsampledVerts;
		ndataFaces += nupsampledFaces;
	}

	if (isVisualizeVertexOmega)
	{
		ndataVerts += triV3D.rows();
		ndataFaces += triF3D.rows();
	}

	int currentDataVerts = nupsampledVerts;
	int currentDataFaces = nupsampledFaces;

	dataV.resize(ndataVerts, 3);
	dataF.resize(ndataFaces, 3);
	curColor.resize(ndataVerts, 3);

	curColor.col(0).setConstant(1.0);
	curColor.col(1).setConstant(1.0);
	curColor.col(2).setConstant(1.0);

	if (isVisualizePhase)
	{
		Eigen::MatrixXd phiColor;
		mPaint.setNormalization(false);
		phiColor = mPaint.paintPhi(phaseFieldList[frameId]);
		curColor.block(0, 0, nupsampledVerts, 3) = phiColor;
	}
	else
	{
		curColor.block(0, 0, nupsampledVerts, 3).col(0).setConstant(1.0);
		curColor.block(0, 0, nupsampledVerts, 3).col(1).setConstant(1.0);
		curColor.block(0, 0, nupsampledVerts, 3).col(2).setConstant(1.0);
	}

	if (isVisualizeAmp)
	{
		mPaint.setNormalization(true);
		Eigen::MatrixXd ampColor;
		ampColor = mPaint.paintAmplitude(ampFieldList[frameId]);
		curColor.block(currentDataVerts, 0, nupsampledVerts, 3) = ampColor;
	}

	dataV.block(0, 0, nupsampledVerts, 3) = upsampledTriV3D;
	dataF.block(0, 0, nupsampledFaces, 3) = upsampledTriF3D;

	Eigen::MatrixXd shiftV = upsampledTriV3D;
	double shiftAmount = 1.5 * (upsampledTriV3D.col(0).maxCoeff() - upsampledTriV3D.col(0).minCoeff());
	shiftV.col(0).setConstant(shiftAmount);
	shiftV.col(1).setConstant(0);
	shiftV.col(2).setConstant(0);


	Eigen::MatrixXi shiftF = upsampledTriF3D;
	shiftF.setConstant(currentDataVerts);

	if (isVisualizeAmp)
	{
		dataV.block(currentDataVerts, 0, nupsampledVerts, 3) = upsampledTriV3D - shiftV;
		dataF.block(nupsampledFaces, 0, nupsampledFaces, 3) = upsampledTriF3D + shiftF;

		currentDataVerts += nupsampledVerts;
		currentDataFaces += nupsampledFaces;
	}

	shiftF.setConstant(currentDataVerts);
	if (isVisualizeWrinkles)
	{
		dataV.block(currentDataVerts, 0, nupsampledVerts, 3) = wrinkledVs[frameId] + shiftV;
		dataF.block(currentDataFaces, 0, nupsampledFaces, 3) = upsampledTriF3D + shiftF;

		currentDataVerts += nupsampledVerts;
		currentDataFaces += nupsampledFaces;
	}

	if (isVisualizeVertexOmega)
	{
		shiftV = triV3D;
		shiftAmount = 1.5 * (upsampledTriV3D.col(1).maxCoeff() - upsampledTriV3D.col(1).minCoeff());
		shiftV.col(0).setConstant(0);
		shiftV.col(1).setConstant(shiftAmount);
		shiftV.col(2).setConstant(0);

		shiftF = triF3D;
		shiftF.setConstant(currentDataVerts);

		dataV.block(currentDataVerts, 0, triV3D.rows(), 3) = triV3D + shiftV;
		dataF.block(currentDataFaces, 0, triF3D.rows(), 3) = triF3D + shiftF;


		dataVec = dataV;
		dataVec.setZero();

		for (int i = 0; i < triV2D.rows(); i++)
		{
			dataVec.row(currentDataVerts + i) << omegaFieldsList[frameId](i, 0), omegaFieldsList[frameId](i, 1), 0;
		}
	}
	polyscope::registerSurfaceMesh("input mesh", dataV, dataF);
	

}

void updateFieldsInView(int frameId)
{
	std::cout << "update view" << std::endl;
	registerMesh(frameId);
	polyscope::getSurfaceMesh("input mesh")->addVertexColorQuantity("VertexColor", curColor);
	if (isVisualizeVertexOmega)
	{
		polyscope::getSurfaceMesh("input mesh")
			->addVertexVectorQuantity("vertex vector field", dataVec);
		polyscope::getSurfaceMesh("input mesh")->getQuantity("vertex vector field")->setEnabled(true);
	}
}

void computeAmpTheta(const Eigen::MatrixXd& w, const std::vector<std::complex<double>>& z, Eigen::VectorXd &amp, Eigen::VectorXd& theta, Eigen::MatrixXd& wrinkledPos)
{
	Eigen::VectorXd upsampledAmp, upsampledTheta;
	std::vector<std::complex<double>> upsampledPhase;

	Eigen::MatrixXd faceField;
	PlaneWaveExtraction extractModel(triV2D, MeshConnectivity(triF2D), w);
	extractModel.extractPlaneWave(faceField);

	Eigen::MatrixXd planeOmega, waterpoolOmega;
	planeOmega = w;
	planeOmega.setZero();

	Eigen::VectorXd verCounts(triV2D.rows());
	verCounts.setZero();

	for (int i = 0; i < triF2D.rows(); i++)
	{
		for (int j = 0; j < 3; j++)
		{
			int vid = triF2D(i, j);
			verCounts(vid)++;
			planeOmega.row(vid) += faceField.row(i);
		}
	}

	for (int i = 0; i < verCounts.rows(); i++)
	{
		planeOmega.row(i) /= verCounts(i);
	}
	waterpoolOmega = w - planeOmega;

	model.estimatePhase(planeOmega, waterpoolOmega, z, upsampledPhase);
	model.getAngleMagnitude(upsampledPhase, upsampledTheta, upsampledAmp);

	wrinkledPos = upsampledTriV3D;

	Eigen::MatrixXd normals;
	igl::per_vertex_normals(upsampledTriV3D, upsampledTriF3D, normals);

	for (int i = 0; i < upsampledTriV3D.rows(); i++)
	{
		wrinkledPos.row(i) += 0.05 * upsampledAmp(i) * std::cos(upsampledTheta(i)) * normals.row(i);
	}
	
	theta = upsampledTheta;
	amp = upsampledAmp;
}


void computeAmpThetaInSequence()
{
	ampFieldList.resize(numFrames);
	phaseFieldList.resize(numFrames);
	wrinkledVs.resize(numFrames);
	omegaFieldsList.resize(numFrames);
	zvalsList.resize(numFrames);

	/*auto computeAmpThetaPerFame = [&](const tbb::blocked_range<uint32_t>& range)
	{
		for (uint32_t i = range.begin(); i < range.end(); ++i)
		{
			double t = i * 1.0 / numFrames;
			double x0, y0;
			generateSingularity(x0, y0, t, motionType);
			std::cout << "x0: " << x0 << ", y0: " << y0 << std::endl;
			Eigen::MatrixXd w;
			std::vector<std::complex<double>> z;
			generateWhirlPool(x0, y0, w, z);

			Eigen::VectorXd amp, theta;
			Eigen::MatrixXd wrinkledPos;
			computeAmpTheta(w, z, amp, theta, wrinkledPos);


			ampFieldList[i] = amp;
			phaseFieldList[i] = theta;
			wrinkledVs[i] = wrinkledPos;
			zvalsList[i] = z;
			omegaFieldsList[i] = w;
		}
	};

	tbb::blocked_range<uint32_t> rangex(0u, (uint32_t)numFrames, GRAIN_SIZE);
	tbb::parallel_for(rangex, computeAmpThetaPerFame);*/

	for (uint32_t i = 0; i < numFrames; ++i)
	{
		double t = i * 1.0 / numFrames;
		double x0, y0;
		generateSingularity(x0, y0, t, motionType);
		std::cout << "x0: " << x0 << ", y0: " << y0 << std::endl;
		Eigen::MatrixXd w;
		std::vector<std::complex<double>> z;
		generateWhirlPool(x0, y0, w, z);

		Eigen::VectorXd amp, theta;
		Eigen::MatrixXd wrinkledPos;
		computeAmpTheta(w, z, amp, theta, wrinkledPos);


		ampFieldList[i] = amp;
		phaseFieldList[i] = theta;
		wrinkledVs[i] = wrinkledPos;
		zvalsList[i] = z;
		omegaFieldsList[i] = w;
	}

	ampField = ampFieldList[numFrames - 1];
	phaseField = phaseFieldList[numFrames - 1];
	wrinkledV = wrinkledVs[numFrames - 1];
	zvals = zvalsList[numFrames - 1];
	omegaFields = omegaFieldsList[numFrames - 1];
}

void callback() {
	ImGui::PushItemWidth(100);
	if (ImGui::Button("Reset", ImVec2(-1, 0)))
	{
		curFrame = 0;
		updateFieldsInView(curFrame);
	}
	
	if (ImGui::InputInt("upsampling level", &loopLevel))
	{
		if (loopLevel <= 0)
			loopLevel = 2;
		//loopWithCorners(triV, triF, corners, upsampledTriV, upsampledTriF, loopLevel);

		meshUpSampling(triV2D, triF2D, upsampledTriV2D, upsampledTriF2D, loopLevel);
		meshUpSampling(triV3D, triF3D, upsampledTriV3D, upsampledTriF3D, loopLevel);

		MeshConnectivity mesh3D(triF3D), upsampledMesh3D(upsampledTriF3D);
		MeshConnectivity mesh2D(triF2D), upsampledMesh2D(upsampledTriF2D);

		model = PhaseInterpolation(triV2D, mesh2D, upsampledTriV2D, upsampledMesh2D, triV3D, mesh3D, upsampledTriV3D, upsampledMesh3D);

		computeAmpThetaInSequence();
		updateFieldsInView(curFrame);
	}
	if (ImGui::Checkbox("visualize phase", &isVisualizePhase))
	{
		updateFieldsInView(curFrame);
	}
	if (ImGui::Checkbox("visualize amp", &isVisualizeAmp))
	{
		updateFieldsInView(curFrame);
	}
	if (ImGui::Checkbox("visualize wrinkles", &isVisualizeWrinkles))
	{
		updateFieldsInView(curFrame);
	}
	if (ImGui::Checkbox("Visualize vertex omega", &isVisualizeVertexOmega))
	{
		updateFieldsInView(curFrame);
	}
	
	if (ImGui::Combo("Motion types", (int*)&motionType, "linear\0entire linear\0rotate\0sin-wave\0complicate\0spiral\0\0"))
	{
		curFrame = 0;
		computeAmpThetaInSequence();
		updateFieldsInView(curFrame);
	}

	if (ImGui::DragInt("current frame", &curFrame, 0.5, 0, numFrames - 1))
	{
		updateFieldsInView(curFrame);
	}

	ImGui::PopItemWidth();
}

void doSplit(const Eigen::MatrixXd& vecfields, Eigen::MatrixXd& planePart, Eigen::MatrixXd& whirlPoolPart)
{
	if (interType == InterpolationType::PurePlaneWave)
	{
		std::cout << "pure plane wave" << std::endl;
		planePart = vecfields;
		whirlPoolPart = vecfields;
		whirlPoolPart.setZero();
	}
	else if (interType == InterpolationType::PureWhirlpool)
	{
		std::cout << "pure whirl pool" << std::endl;
		planePart = vecfields;
		whirlPoolPart = vecfields;
		planePart.setZero();
	}
	else if (interType == InterpolationType::NaiveSplit)
	{
		std::cout << "naive split" << std::endl;
		Eigen::Vector2d aveVec;
		aveVec.setZero();

		for (int i = 0; i < vecfields.rows(); i++)
		{
			aveVec += vecfields.row(i);
		}
		aveVec /= vecfields.rows();

		planePart = vecfields;
		for (int i = 0; i < vecfields.rows(); i++)
		{
			planePart.row(i) = aveVec;
		}

		whirlPoolPart = vecfields - planePart;

	}
	else if (interType == InterpolationType::NewSplit)
	{
		std::cout << "new split" << std::endl;
		VecFieldsSplit testModel = VecFieldsSplit(triV2D, MeshConnectivity(triF2D), vecfields);
		Eigen::Vector2d aveVec;
		aveVec.setZero();

		for (int i = 0; i < vecfields.rows(); i++)
			aveVec += vecfields.row(i);
		aveVec /= vecfields.rows();

		Eigen::VectorXd x(2 * vecfields.rows());
		x.setRandom();

		/*for(int i = 0; i < vecfields.rows(); i++)
			x.segment<2>(2 * i) = aveVec;*/

		auto funVal = [&](const Eigen::VectorXd& x, Eigen::VectorXd* grad, Eigen::SparseMatrix<double>* hess, bool isProj) {
			Eigen::VectorXd deriv;
			Eigen::SparseMatrix<double> H;
			double E = testModel.optEnergy(x, grad ? &deriv : NULL, hess ? &H : NULL, isProj);

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

		OptSolver::newtonSolver(funVal, maxStep, x, 1000, 1e-6, 0, 0, true);

		planePart = vecfields;
		for (int i = 0; i < planePart.rows(); i++)
		{
			planePart.row(i) = x.segment<2>(2 * i);
		}

		whirlPoolPart = vecfields - planePart;
	}
	else
	{
		planePart = vecfields;
		whirlPoolPart = vecfields;
		whirlPoolPart.setZero();
		planePart.setZero();
	}
	if(isShowOnlyPlaneWave)
	    whirlPoolPart.setZero();
	else if(isShowOnlyWhirlPool)
	    planePart.setZero();
}


void initialization()
{
	generateSquare(2.0, 2.0, triarea, triV2D, triF2D);

	triV3D = triV2D;
	triF3D = triF2D;

	Eigen::SparseMatrix<double> S;
	std::vector<int> facemap;

	meshUpSampling(triV2D, triF2D, upsampledTriV2D, upsampledTriF2D, loopLevel, &S, &facemap, &bary);
	meshUpSampling(triV3D, triF3D, upsampledTriV3D, upsampledTriF3D, loopLevel);
	std::cout << "upsampling finished" << std::endl;

	MeshConnectivity mesh3D(triF3D), upsampledMesh3D(upsampledTriF3D);
	MeshConnectivity mesh2D(triF2D), upsampledMesh2D(upsampledTriF2D);

	model = PhaseInterpolation(triV2D, mesh2D, upsampledTriV2D, upsampledMesh2D, triV3D, mesh3D, upsampledTriV3D, upsampledMesh3D, &bary);
}

void generateTargetVals()
{
	if (tarType == TargetType::Whirlpool)
	{
		Eigen::Vector2d center = Eigen::Vector2d::Random();
		if(isFixed)
		    center << fixedx, fixedy;
		generateWhirlPool(center(0), center(1), omegaFields, zvals, sigIndex1, &theoZVals);
		doSplit(omegaFields, planeFields, whirlFields);
	}
	else if (tarType == TargetType::PlaneWave)
	{
		Eigen::Vector2d v = Eigen::Vector2d::Random();
		if(isFixed)
		    v = fixedv;
		generatePlaneWave(v, omegaFields, zvals, &theoZVals);
		doSplit(omegaFields, planeFields, whirlFields);
	}
	else if (tarType == TargetType::Summation)
	{
		Eigen::Vector2d center = Eigen::Vector2d::Random();
		Eigen::Vector2d v = Eigen::Vector2d::Random();
		if(isFixed)
		{
		    v = fixedv;
		    center << fixedx, fixedy;
		}


		generatePlaneSumWhirl(center(0), center(1), v, omegaFields, zvals, &theoZVals);
		doSplit(omegaFields, planeFields, whirlFields);
	}
	else if (tarType == TargetType::YShape)
	{
		Eigen::Vector2d w1(1, 0);
		Eigen::Vector2d w2(1, 0);

		w1(0) = 2 * 3.1415926;
		w2(0) = 4 * 3.1415926;
		generateYshape(w1, w2, omegaFields, zvals, &theoZVals);
		doSplit(omegaFields, planeFields, whirlFields);
	}
	else if (tarType == TargetType::TwoWhirlPool)
	{
		Eigen::Vector2d center0 = Eigen::Vector2d::Random();
		Eigen::Vector2d center1 = Eigen::Vector2d::Random();
		if(isFixed)
		{
		    center0 << fixedx, fixedy;
		    center1 << 0.8, -0.3;
		}
		generateTwoWhirlPool(center0(0), center0(1), center1(0), center1(1), omegaFields, zvals, sigIndex1, sigIndex2, &theoZVals);
		doSplit(omegaFields, planeFields, whirlFields);
	}
	else
	{
		generateRandom(omegaFields, zvals, &theoZVals);
		doSplit(omegaFields, planeFields, whirlFields);
	}
	std::vector<std::complex<double>> upsampledZvals;
	model.estimatePhase(planeFields, whirlFields, zvals, upsampledZvals);
	model.getAngleMagnitude(upsampledZvals, phaseField, ampField);
}

void vecCallback() {
    ImGui::PushItemWidth(100);
	if (ImGui::InputDouble("triangle area", &triarea))
	{
	    if(triarea > 0)
		    initialization();
	}
	if (ImGui::InputInt("upsampled times", &loopLevel))
	{
	    if(loopLevel > 0)
		    initialization();
	}
	if (ImGui::InputInt("Singularity index 1", &sigIndex1))
	{}
	if (ImGui::InputInt("Singularity index 2", &sigIndex2))
	{}
    if (ImGui::Combo("vec types", (int*)&tarType, "Whirl pool\0plane wave\0sum\0Y shape\0Two Whirl Pool\0random\0\0")){}
    if (ImGui::Combo("interpolation types", (int*)&interType, "Pure Whirl pool\0Pure plane wave\0Naive Split\0New Split\0Just linear\0\0")) {}
    if (ImGui::Checkbox("Show Only Plane wave", &isShowOnlyPlaneWave)){}
    if(ImGui::Checkbox("Show Only Whirl pool", &isShowOnlyWhirlPool)){}
    if(ImGui::Checkbox("Fixed center and dir", &isFixed)){}
	if (ImGui::Button("Test", ImVec2(-1, 0)))
	{
		Eigen::Vector2d v = Eigen::Vector2d::Random();
		if (isFixed)
			v = fixedv;
		generatePlaneWave(v, omegaFields, zvals, &theoZVals);
		planeFields = omegaFields;
		whirlFields = omegaFields;
		planeFields.setRandom();
		whirlFields.setZero();

		std::vector<std::complex<double>> upsampledZvals, upsampledZvals1;
		model.estimatePhase(planeFields, whirlFields, zvals, upsampledZvals);
		model.getAngleMagnitude(upsampledZvals, phaseField, ampField);

		GetInterpolatedValues testmodel = GetInterpolatedValues(triV2D, triF2D, upsampledTriV2D, upsampledTriF2D, bary);
		upsampledZvals1 = testmodel.getZValues(planeFields, zvals, NULL, NULL);

		for (int i = 0; i < upsampledZvals.size(); i++)
		{
			if (std::abs(upsampledZvals[i] - upsampledZvals1[i]) > 1e-6)
			{
				std::cout << "error in vertex: " << i << ", " << upsampledZvals[i] << ", " << upsampledZvals1[i] << std::endl;
			}
		}

		int vid = std::rand() % upsampledTriV2D.rows();
//		testmodel.testPlaneWaveValue(planeFields, zvals, vid);
		
		omegaFields.setRandom();
		auto zvals1 = zvals;
		for(auto& z: zvals1)
		{
		    Eigen::Vector2d randz = Eigen::Vector2d::Random();
		    z = std::complex<double>(randz(0), randz(1));
		}
//		testmodel.testPlaneWaveValueDot(planeFields, omegaFields, zvals, zvals1, 0.1, vid);
		
		testmodel.testZDotSquarePerVertex(planeFields, omegaFields, zvals, zvals1, 0.1, 4);

		testmodel.testZDotSquareIntegration(planeFields, omegaFields, zvals, zvals1, 0.1);


	}
	if (ImGui::Button("update viewer", ImVec2(-1, 0)))
	{
		generateTargetVals();
		updateVecFieldsInView();
	}

    ImGui::PopItemWidth();
}

void testFunction(Eigen::MatrixXd pos, Eigen::MatrixXi face)
{
    Eigen::MatrixXd vecfields(pos.rows(), 2);
    vecfields.setRandom();

    VecFieldsSplit testModel(pos, MeshConnectivity(face), vecfields);
    std::cout << "test plane wave part." << std::endl;
    testModel.testPlaneWaveSmoothnessPerface(vecfields, 0);
    testModel.testPlaneWaveSmoothness(vecfields);

    std::cout << "\ntest whirl pool part." << std::endl;
    testModel.testWhirlpoolSmoothnessPerface(vecfields, 0);
    testModel.testPlaneWaveSmoothness(vecfields);

    std::vector<std::complex<double>> z;
    generateWhirlPool(0.1, 0.2, vecfields, z);
    std::cout << "test whirl energy: (expected 0) " << testModel.whirlpoolSmoothness(vecfields) << std::endl;

    Eigen::Vector2d v = Eigen::Vector2d::Random();
    for(int i = 0; i < vecfields.rows(); i++)
        vecfields.row(i) = v;
    std::cout << "test plane energy: (expected 0) " << testModel.planeWaveSmoothness(vecfields) << std::endl;

    std::cout << "test opt energy: " << std::endl;
    Eigen::VectorXd x = Eigen::VectorXd::Random(2 * pos.rows());
    testModel.testOptEnergy(x);


    // test newton solver:
    generateWhirlPool(0.1, 0.2, vecfields, z);
    testModel = VecFieldsSplit(pos, MeshConnectivity(face), vecfields);

    Eigen::Vector2d aveVec;
    aveVec.setZero();

    for(int i = 0; i < vecfields.rows(); i++)
       aveVec += vecfields.row(i);
    aveVec /= vecfields.rows();

    for(int i = 0; i < vecfields.rows(); i++)
        x.segment<2>(2 * i) = aveVec;

    auto funVal = [&](const Eigen::VectorXd& x, Eigen::VectorXd* grad, Eigen::SparseMatrix<double>* hess, bool isProj){
        Eigen::VectorXd deriv;
        Eigen::SparseMatrix<double> H;
        double E = testModel.optEnergy(x, grad ? &deriv : NULL, hess ? &H : NULL, isProj);

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
    auto maxStep = [&](const Eigen::VectorXd& x, const Eigen::VectorXd& dir){
        return 1.0;
    };

    OptSolver::newtonSolver(funVal, maxStep, x, 1000, 1e-6, 0, 0, true);
    std::cout << "Norm(x) (expected 0): " << x.norm() << std::endl;

    v = Eigen::Vector2d::Random();
    Eigen::VectorXd planeX = x;
    for(int i = 0; i < vecfields.rows(); i++)
    {
        vecfields.row(i) = v;
        planeX.segment<2>(2 * i) = v;
    }
    testModel = VecFieldsSplit(pos, MeshConnectivity(face), vecfields);
    x.setRandom();

    OptSolver::newtonSolver(funVal, maxStep, x, 10000, 1e-6, 0, 0, true);
    std::cout << "Norm(x - xtheo) (expected 0): " << (x - planeX).norm() << std::endl;
}

//int main(int argc, char** argv)
//{
//	generateSquare(2.0, 2.0, 0.1, triV2D, triF2D);
//
//	triV3D = triV2D;
//	triF3D = triF2D;
//
//	meshUpSampling(triV2D, triF2D, upsampledTriV2D, upsampledTriF2D, loopLevel);
//	meshUpSampling(triV3D, triF3D, upsampledTriV3D, upsampledTriF3D, loopLevel);
//	std::cout << "upsampling finished" << std::endl;
//
//	MeshConnectivity mesh3D(triF3D), upsampledMesh3D(upsampledTriF3D);
//	MeshConnectivity mesh2D(triF2D), upsampledMesh2D(upsampledTriF2D);
//
//	model = PhaseInterpolation(triV2D, mesh2D, upsampledTriV2D, upsampledMesh2D, triV3D, mesh3D, upsampledTriV3D, upsampledMesh3D);
//
//	//generateWhirlPool(0, 0, omegaFields, zvals);
//	computeAmpThetaInSequence();
//
//	/*computePhaseInSequence(numofFrames);
//	std::cout << "compute finished" << std::endl;*/
//
//	// Options
//	polyscope::options::autocenterStructures = true;
//	polyscope::view::windowWidth = 1024;
//	polyscope::view::windowHeight = 1024;
//
//	// Initialize polyscope
//	polyscope::init();
//
//
//	// Register the mesh with Polyscope
//	polyscope::registerSurfaceMesh("input mesh", upsampledTriV3D, upsampledTriF3D);
//
//	// Add the callback
//	polyscope::state::userCallback = callback;
//
//	// Show the gui
//	polyscope::show();
//
//	return 0;
//}





int main(int argc, char** argv)
{
	initialization();
    
	// Options
    polyscope::options::autocenterStructures = true;
    polyscope::view::windowWidth = 1024;
    polyscope::view::windowHeight = 1024;

    // Initialize polyscope
    polyscope::init();


    // Register the mesh with Polyscope
    polyscope::registerSurfaceMesh("input mesh", triV2D, triF2D);

    // Add the callback
    polyscope::state::userCallback = vecCallback;
//    polyscope::options::groundPlaneMode = polyscope::GroundPlaneMode::None;

    polyscope::options::groundPlaneHeightFactor = 0.25; // adjust the plane height
    // Show the gui
    polyscope::show();

    return 0;
}