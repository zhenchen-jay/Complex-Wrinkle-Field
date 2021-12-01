// #include "polyscope/polyscope.h"

// #include <igl/PI.h>
// #include <igl/avg_edge_length.h>
// #include <igl/barycenter.h>
// #include <igl/boundary_loop.h>
// #include <igl/exact_geodesic.h>
// #include <igl/gaussian_curvature.h>
// #include <igl/invert_diag.h>
// #include <igl/lscm.h>
// #include <igl/massmatrix.h>
// #include <igl/per_vertex_normals.h>
// #include <igl/readOBJ.h>
// #include <igl/writeOBJ.h>
// #include <igl/doublearea.h>
// #include <igl/file_dialog_open.h>
// #include <igl/file_dialog_save.h>
// #include <igl/boundary_loop.h>
// //#include <igl/triangle/triangulate.h>
// #include <filesystem>
// #include "polyscope/messages.h"
// #include "polyscope/point_cloud.h"
// #include "polyscope/surface_mesh.h"

// #include <iostream>
// #include <fstream>
// #include <unordered_set>
// #include <utility>


// #include "include/InterpolationScheme/PhaseInterpolation.h"
// #include "include/MeshLib/MeshConnectivity.h"
// #include "include/MeshLib/MeshUpsampling.h"



// Eigen::MatrixXd triV2D, triV3D, upsampledTriV2D, upsampledTriV3D, wrinkledV;
// Eigen::MatrixXi triF2D, triF3D, upsampledTriF2D, upsampledTriF3D;

// Eigen::VectorXd phaseField(0);
// Eigen::VectorXd ampField(0);

// Eigen::VectorXd ampFieldNormalized(0);
// Eigen::VectorXd phaseFieldNormalized(0);

// std::vector<Eigen::VectorXd> ampFieldLists;
// std::vector<Eigen::VectorXd> ampFieldNormalizedLists;

// std::vector<Eigen::VectorXd> phaseFieldLists;
// std::vector<Eigen::VectorXd> phaseFieldNormalizedLists;

// double curAmpMin = 0;
// double curAmpMax = 1;

// double globalAmpMin = 0;
// double globalAmpMax = 1;

// Eigen::MatrixXd dataV;
// Eigen::MatrixXi dataF;
// Eigen::MatrixXd edgeStartV, edgeEndV;

// int numofFrames = 1000;
// int loopLevel = 2;
// double dampingRatio = 1;

// bool isNormalize = false;
// bool isVisualizePhase = false;
// bool isVisualizeAmp = false;
// bool isVisualizeWrinkles = false;

// bool isAnimated = true;
// bool isOnlyChangeOne = false;

// Eigen::MatrixXd curColor;
// int currentFrame = 0;
// int fps = 30;
// bool isPaused = true;
// bool isQuaticAmp = false;

// int numWaves = 8;
// int interpType = 0;

// std::vector<Eigen::MatrixXd> wrinkledVs;


// void normalizeCurrentFrame(const Eigen::VectorXd& input, Eigen::VectorXd& normalizedScalarField, double* min = NULL, double* max = NULL)
// {
// 	double curMin = input.minCoeff();
// 	double curMax = input.maxCoeff();

// 	if (min)
// 		curMin = *min;
// 	if (max)
// 		curMax = *max;

// 	if (curMin > curMax)
// 	{
// 		std::cout << "some errors in the input min and max values." << std::endl;

// 		curMin = input.minCoeff();
// 		curMax = input.maxCoeff();
// 	}

// 	normalizedScalarField = input;
// 	for (int i = 0; i < input.size(); i++)
// 	{
// 		normalizedScalarField(i) = (input(i) - curMin) / (curMax - curMin);
// 	}

// }

// void computePhaseInSequence(const int& numofFrames)
// {
// 	double PI = 3.1415926535898;
// 	MeshConnectivity mesh3D(triF3D), upsampledMesh3D(upsampledTriF3D);
// 	MeshConnectivity mesh2D(triF2D), upsampledMesh2D(upsampledTriF2D);

// 	phaseFieldLists.resize(numofFrames);
// 	ampFieldLists.resize(numofFrames);

// 	phaseFieldNormalizedLists.resize(numofFrames);
// 	ampFieldNormalizedLists.resize(numofFrames);

// 	wrinkledVs.resize(numofFrames);

// 	PhaseInterpolation model(triV2D, mesh2D, upsampledTriV2D, upsampledMesh2D, triV3D, mesh3D, upsampledTriV3D, upsampledMesh3D);

// 	Eigen::MatrixXd normals;
// 	igl::per_vertex_normals(upsampledTriV3D, upsampledTriF3D, normals);

// 	std::vector<std::complex<double>> initTheta(triV3D.rows());

// 	for (int i = 0; i < triV3D.rows(); i++)
// 		initTheta[i] = std::complex<double>(1, 0);

// 	/*initTheta.row(0) << -std::sqrt(3) / 2, -1.0 / 2;
// 	initTheta.row(1) << std::sqrt(3) / 2, -1.0 / 2;
// 	initTheta.row(2) << 0, 1.0;*/

// 	for (int i = 0; i < mesh3D.nEdges(); i++)
// 	{
// 		std::cout << "edge id: " << i << std::endl;
// 		std::cout << "vertex id: " << mesh3D.edgeVertex(i, 0) << ", value: " << triV3D.row(mesh3D.edgeVertex(i, 0)) << std::endl;
// 		std::cout << "vertex id: " << mesh3D.edgeVertex(i, 1) << ", value: " << triV3D.row(mesh3D.edgeVertex(i, 1)) << std::endl;
// 	}

// 	for (uint32_t n = 0; n < numofFrames; ++n)
// 	{
// 		Eigen::MatrixXd vertexOmega(triV2D.rows(), 3);

// 		vertexOmega.setZero();

// 		double frequency = (numWaves) * 1.0 / numofFrames * (n + 1);


// 		vertexOmega.row(0) << 2 * frequency * PI, 0, 0;
// 		vertexOmega.row(1) << 2 * frequency * PI, 0, 0;
// 		vertexOmega.row(2) << 4 * PI, 0, 0;

	
// 		Eigen::VectorXd upsampledAmp, upsampledTheta;

// 		std::vector<std::complex<double>> upsampledPhase;

// 		/*if (n == 250)
// 		{
// 			std::cout << "\n";
// 		}*/

// 		model.estimatePhase(vertexOmega, initTheta, upsampledPhase, interpType);
// 		model.getAngleMagnitude(upsampledPhase, upsampledTheta, upsampledAmp);

// 		/*if (n == 249 || n == 250 || n == 251)
// 		{
// 			if (loopLevel == 2)
// 			{
// 				std::cout << "\nframe: " << n << ", freq: " << frequency << std::endl;
// 				std::cout << "vertex omega: \n" << vertexOmega << std::endl;
// 				std::cout << "upsampled z: " << std::endl;
// 				for (int i = 0; i < upsampledTriV3D.rows(); i++)
// 				{
// 					std::cout << "vertix pos: " << upsampledTriV3D.row(i) << ", \tbary: " << model._baryCoords[i].second.transpose() << ", \tz: " << upsampledPhase[i] << std::endl;
// 				}
				
// 			}
// 		}*/

// 		phaseFieldLists[n] = upsampledTheta;
// 		ampFieldLists[n] = upsampledAmp;

// 		wrinkledVs[n] = upsampledTriV3D;

// 		for (int i = 0; i < upsampledTriV3D.rows(); i++)
// 		{
// 			wrinkledVs[n].row(i) += 0.05 * upsampledAmp(i) * std::cos(upsampledTheta(i)) * normals.row(i);
// 		}

// 		ampField = upsampledAmp;
// 		phaseField = upsampledTheta;

// 		globalAmpMin = std::min(globalAmpMin, upsampledAmp.minCoeff());
// 		globalAmpMax = std::max(globalAmpMax, upsampledAmp.maxCoeff());

// 	}
// }

// void updateFieldsInView()
// {
// 	std::cout << "update view" << std::endl;
// 	ampField = ampFieldLists[currentFrame];
// 	phaseField = phaseFieldLists[currentFrame];

// 	if (isNormalize)
// 	{
// 		normalizeCurrentFrame(ampField, ampFieldNormalized, &globalAmpMin, &globalAmpMax);
// 		polyscope::getSurfaceMesh("input mesh")
// 			->addVertexScalarQuantity("vertex amp field", ampFieldNormalized,
// 				polyscope::DataType::SYMMETRIC);
// 		polyscope::getSurfaceMesh("input mesh")->getQuantity("vertex amp field")->setEnabled(true);
// 	}
// 	else
// 	{
// 		polyscope::getSurfaceMesh("input mesh")
// 			->addVertexScalarQuantity("vertex amp field", ampField,
// 				polyscope::DataType::SYMMETRIC);
// 		polyscope::getSurfaceMesh("input mesh")->getQuantity("vertex amp field")->setEnabled(true);
// 	}
// }

// void callback() {
// 	ImGui::PushItemWidth(100);
// 	if (ImGui::Button("Reset", ImVec2(-1, 0)))
// 	{
// 		currentFrame = 0;
// 		updateFieldsInView();
// 	}
// 	if (ImGui::Button("update the dynamic vector field"))
// 	{
// 		computePhaseInSequence(numofFrames);
// 		updateFieldsInView();
// 	}
// 	ImGui::SameLine();
// 	if (ImGui::InputInt("total Frames ", &numofFrames))
// 	{
// 		if (numofFrames <= 0)
// 			numofFrames = 1000;
// 	}

// 	if (ImGui::DragInt("current frame", &currentFrame, 0.5, 0, numofFrames))
// 	{
// 		std::cout << "current frame: " << currentFrame << std::endl;
// 		if (currentFrame < ampFieldLists.size())
// 			updateFieldsInView();
// 	}

// 	if (ImGui::Combo("interp Type", &interpType, "composite\0plane wave\0water pool\0\0"))
// 	{
// 		computePhaseInSequence(numofFrames);
// 		currentFrame = 0;
// 		updateFieldsInView();
// 	}

// 	if (ImGui::InputInt("upsampling level", &loopLevel))
// 	{
// 		if (loopLevel <= 0)
// 			loopLevel = 8;
// 		//loopWithCorners(triV, triF, corners, upsampledTriV, upsampledTriF, loopLevel);

// 		meshUpSampling(triV2D, triF2D, upsampledTriV2D, upsampledTriF2D, loopLevel);
// 		meshUpSampling(triV3D, triF3D, upsampledTriV3D, upsampledTriF3D, loopLevel);
// 		polyscope::registerSurfaceMesh("input mesh", upsampledTriV3D, upsampledTriF3D);

// 		computePhaseInSequence(numofFrames);
// 		updateFieldsInView();
// 	}

// 	if (ImGui::InputInt("num of waves", &numWaves))
// 	{
// 		if (numWaves <= 0)
// 			numWaves = 1;
// 		computePhaseInSequence(numofFrames);
// 	}

// 	ImGui::PopItemWidth();
// }

// int main(int argc, char** argv)
// {
// 	triV2D.resize(3, 3);
// 	triV2D << -std::sqrt(3) / 2, -1.0 / 2, 0,
// 		std::sqrt(3) / 2, -1.0 / 2, 0,
// 		0, 1, 0;
// 	triF2D.resize(1, 3);
// 	triF2D << 0, 1, 2;

// 	triV3D = triV2D;
// 	triF3D = triF2D;

// 	meshUpSampling(triV2D, triF2D, upsampledTriV2D, upsampledTriF2D, loopLevel);
// 	meshUpSampling(triV3D, triF3D, upsampledTriV3D, upsampledTriF3D, loopLevel);
// 	std::cout << "upsampling finished" << std::endl;

	
// 	computePhaseInSequence(numofFrames);
// 	std::cout << "compute finished" << std::endl;

// 	// Options
// 	polyscope::options::autocenterStructures = true;
// 	polyscope::view::windowWidth = 1024;
// 	polyscope::view::windowHeight = 1024;

// 	// Initialize polyscope
// 	polyscope::init();

	
// 	// Register the mesh with Polyscope
// 	polyscope::registerSurfaceMesh("input mesh", upsampledTriV3D, upsampledTriF3D);

// 	// Add the callback
// 	polyscope::state::userCallback = callback;

// 	// Show the gui
// 	polyscope::show();

// 	return 0;
// }


//#include <nasoq.h>
#include <iostream>

/*
 * Minimizing 1/2 x^THx + q^Tx + C; Cx <= d
 * H and C are sparse CSC matrices
 * q and d are dense arrays
 */

int main(int argc, char *argv[]){
 /// Declaring inputs
// size_t sizeH;
// size_t nnzH;
// size_t CRows; size_t CCols; size_t nnzA;
// sizeH = 2; nnzH = 2;
// CRows = 4; CCols = 2; nnzA = 8;
// auto *q = new double[sizeH];
// auto *Hp = new int[sizeH+1];
// auto *Hi = new int[nnzH];
// auto *Hx = new double[nnzH];
// auto *Cp = new int[CCols+1];
// auto *Ci = new int[nnzA];
// auto *Cx = new double[nnzA];
// auto *d = new double[CRows];
//
// q[0] = -4; q[1] = -4;
//
// Hp[0]=0;Hp[1]=1;Hp[2]=2;
// Hi[0]=0;Hi[1]=1;
// Hx[0]=2;Hx[1]=2;
//
// Cp[0]=0;Cp[1]=4;Cp[2]=8;
// Ci[0]=0;Ci[1]=1;Ci[2]=2;Ci[3]=3;
// Ci[4]=0;Ci[5]=1;Ci[6]=2;Ci[7]=3;
// Cx[0]=2;Cx[1]=1;Cx[2]=-1;Cx[3]=-2;
// Cx[4]=1;Cx[5]=-1;Cx[6]=-1;Cx[7]=1;
//
// d[0]=2;d[1]=1;d[2]=1;d[3]=2;
//
// /// Solving the QP pronlem
// nasoq::Nasoq *qm;
// qm = new nasoq::Nasoq(sizeH,Hp,Hi,Hx,
//                       q,CRows,CCols,Cp, Ci, Cx, d);
// qm->diag_perturb=1e-9;
// qm->eps_abs=1e-3;
// qm->max_iter = 0;
// qm->variant = nasoq::PREDET;
// int converged = qm->solve();
//
// /// Printing results
// if(converged)
//  std::cout<<"The problem is converged\n";
//
// // expected x={0.4,1.2};
// auto *x = qm->primal_vars;
// std::cout<<"Primal variables: ";
// for (int i = 0; i < sizeH; ++i) {
//  std::cout<<x[i]<<",";
// }
//
// // expected z = {1.6,0,0,0}
// std::cout<<"\nDual variables: ";
// auto *z = qm->dual_vars;
// for (int i = 0; i < CRows; ++i) {
//  std::cout<<z[i]<<",";
// }
//
// delete qm;
// delete []Hp; delete []Hi; delete []Hx; delete []q;
// delete []Cp;  delete []Ci; delete []Cx; delete []d;
 return 0;
}
