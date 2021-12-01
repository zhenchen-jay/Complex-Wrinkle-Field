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


#include "include/InterpolationScheme/PhaseInterpolation.h"
#include "include/InterpolationScheme/PlaneWaveExtraction.h"
#include "include/MeshLib/MeshConnectivity.h"
#include "include/MeshLib/MeshUpsampling.h"
#include "include/Visualization/PaintGeometry.h"


Eigen::MatrixXd triV2D, triV3D, upsampledTriV2D, upsampledTriV3D, wrinkledV;
Eigen::MatrixXi triF2D, triF3D, upsampledTriF2D, upsampledTriF3D;

std::vector<std::complex<double>> zvals;
Eigen::MatrixXd omegaFields;

Eigen::VectorXd phaseField(0);
Eigen::VectorXd ampField(0);

Eigen::VectorXd ampFieldNormalized(0);
Eigen::VectorXd phaseFieldNormalized(0);

Eigen::MatrixXd dataV;
Eigen::MatrixXi dataF;
Eigen::MatrixXd dataVec;
Eigen::MatrixXd curColor;

int loopLevel = 4;

bool isVisualizePhase = false;
bool isVisualizeAmp = false;
bool isVisualizeWrinkles = false;

bool isVisualizeVertexOmega = false;


std::vector<Eigen::MatrixXd> wrinkledVs;

PhaseInterpolation model;
PaintGeometry mPaint;

void updateFieldsInView()
{
	std::cout << "update view" << std::endl;
	if (isVisualizeVertexOmega)
	{
		polyscope::getSurfaceMesh("input mesh")
			->addVertexVectorQuantity("vertex vector field", dataVec);
		polyscope::getSurfaceMesh("input mesh")->getQuantity("vertex vector field")->setEnabled(true);
	}
}

void computeAmpTheta()
{
	Eigen::VectorXd upsampledAmp, upsampledTheta;
	std::vector<std::complex<double>> upsampledPhase;

	Eigen::MatrixXd faceField;
	PlaneWaveExtraction extractModel(triV2D, MeshConnectivity(triF2D), omegaFields);
	extractModel.extractPlaneWave(faceField);

	Eigen::MatrixXd planeOmega, waterpoolOmega;
	planeOmega = omegaFields;
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
	waterpoolOmega = omegaFields - planeOmega;

	model.estimatePhase(planeOmega, waterpoolOmega, zvals, upsampledPhase);
	model.getAngleMagnitude(upsampledPhase, upsampledTheta, upsampledAmp);

	wrinkledV = upsampledTriV3D;

	Eigen::MatrixXd normals;
	igl::per_vertex_normals(upsampledTriV3D, upsampledTriF3D, normals);

	for (int i = 0; i < upsampledTriV3D.rows(); i++)
	{
		wrinkledV.row(i) += 0.05 * upsampledAmp(i) * std::cos(upsampledTheta(i)) * normals.row(i);
	}
	
	phaseField = upsampledTheta;
	ampField = upsampledAmp;
}


void callback() {
	ImGui::PushItemWidth(100);
	if (ImGui::Button("Reset", ImVec2(-1, 0)))
	{
		updateFieldsInView();
	}
	
	if (ImGui::InputInt("upsampling level", &loopLevel))
	{
		if (loopLevel <= 0)
			loopLevel = 2;
		//loopWithCorners(triV, triF, corners, upsampledTriV, upsampledTriF, loopLevel);

		meshUpSampling(triV2D, triF2D, upsampledTriV2D, upsampledTriF2D, loopLevel);
		meshUpSampling(triV3D, triF3D, upsampledTriV3D, upsampledTriF3D, loopLevel);
	}
	ImGui::Checkbox("visualize phase", &isVisualizePhase);
	ImGui::Checkbox("visualize amp", &isVisualizeAmp);
	ImGui::Checkbox("visualize wrinkles", &isVisualizeWrinkles);
	ImGui::Checkbox("Visualize vertex omega", &isVisualizeVertexOmega);

	if (ImGui::Button("register mesh", ImVec2(-1, 0)))
	{
		MeshConnectivity mesh3D(triF3D), upsampledMesh3D(upsampledTriF3D);
		MeshConnectivity mesh2D(triF2D), upsampledMesh2D(upsampledTriF2D);

		model = PhaseInterpolation(triV2D, mesh2D, upsampledTriV2D, upsampledMesh2D, triV3D, mesh3D, upsampledTriV3D, upsampledMesh3D);
		computeAmpTheta();

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
			phiColor = mPaint.paintPhi(phaseField);
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
			ampColor = mPaint.paintAmplitude(ampField);
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
			dataV.block(currentDataVerts, 0, nupsampledVerts, 3) = upsampledTriV3D + shiftV;
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
				dataVec.row(currentDataVerts + i) << omegaFields(i, 0), omegaFields(i, 1), 0;
			}
		}



		polyscope::registerSurfaceMesh("input mesh", dataV, dataF);
		polyscope::getSurfaceMesh("input mesh")->addVertexColorQuantity("VertexColor", curColor);
		updateFieldsInView();
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
	irregularV.resize(V2d.rows(), 3);
	irregularV.setZero();
	irregularV.block(0, 0, irregularV.rows(), 2) = V2d;
	irregularF = F;
}

void testFunction()
{
	Eigen::MatrixXd irregularV;
	Eigen::MatrixXi irregularF;

	double length = 1.0;
	double width = 1.0;
	double triarea = 0.001;

	generateSquare(length, width, triarea, irregularV, irregularF);
	igl::writeOBJ("planeMesh.obj", irregularV, irregularF);

	MeshConnectivity meshF(irregularF);

	Eigen::MatrixXd vertFields(irregularV.rows(), 2);
	vertFields.setRandom();
	
	/*for (int i = 0; i < irregularV.rows(); i++)
		vertFields.row(i) << 1, 0;*/

	PlaneWaveExtraction model(irregularV, meshF, vertFields);

	int nfaces = meshF.nFaces();
	Eigen::VectorXd x(3 * nfaces);
	x.setRandom();
	//x.setConstant(1.0 / 3);


	int nedges = meshF.nEdges();

	int eid = std::rand() % nedges;

	while (meshF.edgeFace(eid, 0) == -1 || meshF.edgeFace(eid, 1) == -1)
	{
		eid = std::rand() % nedges;
	}
	std::cout << eid << std::endl;
	model.testOptEnergyPerEdge(x, eid);
	model.testOptEnergy(x);
}


int main(int argc, char** argv)
{
	generateSquare(1.0, 1.0, 0.1, triV2D, triF2D);

	triV3D = triV2D;
	triF3D = triF2D;

	meshUpSampling(triV2D, triF2D, upsampledTriV2D, upsampledTriF2D, loopLevel);
	meshUpSampling(triV3D, triF3D, upsampledTriV3D, upsampledTriF3D, loopLevel);
	std::cout << "upsampling finished" << std::endl;

	zvals.resize(triV3D.rows());
	omegaFields.resize(triV3D.rows(), 2);

	for (int i = 0; i < zvals.size(); i++)
	{
		double x = triV2D(i, 0);
		double y = triV2D(i, 1);
		double rsquare = x * x + y * y;

		zvals[i] = std::complex<double>(x, y);

		if (std::abs(std::sqrt(rsquare)) < 1e-10)
			omegaFields.row(i) << 0, 0;
		else
			omegaFields.row(i) << -y / rsquare, x / rsquare;
	}

	/*computePhaseInSequence(numofFrames);
	std::cout << "compute finished" << std::endl;*/

	// Options
	polyscope::options::autocenterStructures = true;
	polyscope::view::windowWidth = 1024;
	polyscope::view::windowHeight = 1024;

	// Initialize polyscope
	polyscope::init();

	
	// Register the mesh with Polyscope
	polyscope::registerSurfaceMesh("input mesh", upsampledTriV3D, upsampledTriF3D);

	// Add the callback
	polyscope::state::userCallback = callback;

	// Show the gui
	polyscope::show();

	return 0;
}
