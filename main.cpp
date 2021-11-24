#include <igl/opengl/glfw/Viewer.h>
#include <igl/writePLY.h>
#include <igl/hausdorff.h>
#include <iostream>
#include <random>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <igl/signed_distance.h>
#include <igl/cotmatrix.h>
#include <igl/per_vertex_normals.h>
#include <igl/barycenter.h>
#include <igl/per_vertex_normals.h>

#include <Eigen/CholmodSupport>
#include <imgui/imgui.h>

#include "../../include/InterpolationScheme/PhaseInterpolation.h"
#include "../../include/MeshLib/MeshConnectivity.h"
#include "../../include/MeshLib/MeshUpsampling.h"
#include "../../include/Visualization/PaintGeometry.h"



Eigen::MatrixXd triV2D, triV3D, upsampledTriV2D, upsampledTriV3D, wrinkledV;
Eigen::MatrixXi triF2D, triF3D, upsampledTriF2D, upsampledTriF3D;


Eigen::MatrixXd dataV;
Eigen::MatrixXi dataF;
Eigen::MatrixXd edgeStartV, edgeEndV;

Eigen::VectorXd amp, edgeDphi;

Eigen::MatrixXd complexPhase;

Eigen::VectorXd upsampledAmp, upsampledTheta;

int loopLevel = 2;
double dampingRatio = 1;

bool isNormalize = false;
bool isVisualizePhase = false;
bool isVisualizeAmp = false;
bool isVisualizeWrinkles = false;

bool isAnimated = true;
bool isOnlyChangeOne = false;

std::vector<Eigen::MatrixXd> thetaColors;
std::vector<Eigen::MatrixXd> ampColors;
Eigen::MatrixXd curColor;
int currentFrame = 0;
int fps = 30;
bool isPaused = true;
bool isQuaticAmp = false;

int numWaves = 8;
int interpType = 0;

std::vector<Eigen::MatrixXd> wrinkledVs;
std::vector<PhaseInterpolation::CutEdge> seamEdges;


PaintGeometry mPaint;


void repaint(igl::opengl::glfw::Viewer& viewer)
{
	viewer.data().clear();
	viewer.data().clear();
	viewer.data().set_face_based(false);

	if (isVisualizePhase || isVisualizeWrinkles)
	{
		//std::cout << "current frame: " << currentFrame << ", current color norm: " << curColor.norm() << std::endl;
		viewer.core().is_animating = isAnimated;
		viewer.core().animation_max_fps = fps;
	}

	viewer.data().set_mesh(dataV, dataF);
	viewer.data().set_colors(curColor);
}

void locateTheSeam(const MeshConnectivity& mesh2D, const MeshConnectivity& mesh3D, std::vector<PhaseInterpolation::CutEdge>& seam)
{
	seam.clear();
	int nfaces = mesh3D.nFaces();

	assert(nfaces == mesh2D.nFaces());

	std::vector<bool> isVisited(mesh3D.nEdges(), false);


	for (int i = 0; i < nfaces; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			int restEid = mesh2D.faceEdge(i, j);
			int refEid = mesh3D.faceEdge(i, j);

			if (mesh2D.edgeFace(restEid, 0) == -1 || mesh2D.edgeFace(restEid, 1) == -1) // this edge is on the boundary of the rest shape
			{
				if (mesh3D.edgeFace(refEid, 0) != -1 && mesh3D.edgeFace(refEid, 1) != -1) // but it is not on the boundary of the reference shape, then it is a cut edge
				{
					if (!isVisited[refEid]) // if this edge is not visited, we add it to the cut edge vector
					{
						PhaseInterpolation::CutEdge cut;
						cut.refEid = refEid;
						cut.restEid = restEid;

						cut.restAdjFid = i;
						cut.refAdjFid(0) = mesh3D.edgeFace(refEid, 0);
						cut.refAdjFid(1) = mesh3D.edgeFace(refEid, 1);

						seam.push_back(cut);
						isVisited[refEid] = true;
					}
				}
			}
		}
	}
	std::cout << "Find " << seam.size() << " cut edges. " << std::endl;
}

void computePhaseInSequence(const int& numofFrames)
{
	double PI = 3.1415926535898;
	MeshConnectivity mesh3D(triF3D), upsampledMesh3D(upsampledTriF3D);
	MeshConnectivity mesh2D(triF2D), upsampledMesh2D(upsampledTriF2D);

	thetaColors.resize(numofFrames);
	ampColors.resize(numofFrames);
	wrinkledVs.resize(numofFrames);

	PhaseInterpolation model(triV2D, mesh2D, upsampledTriV2D, upsampledMesh2D, triV3D, mesh3D, upsampledTriV3D, upsampledMesh3D);

	Eigen::MatrixXd normals;
	igl::per_vertex_normals(upsampledTriV3D, upsampledTriF3D, normals);

	std::vector<std::complex<double>> initTheta(triV3D.rows());

	for (int i = 0; i < triV3D.rows(); i++)
		initTheta[i] = std::complex<double>(1, 0);

	/*initTheta.row(0) << -std::sqrt(3) / 2, -1.0 / 2;
	initTheta.row(1) << std::sqrt(3) / 2, -1.0 / 2;
	initTheta.row(2) << 0, 1.0;*/

	for (int i = 0; i < mesh3D.nEdges(); i++)
	{
		std::cout << "edge id: " << i << std::endl;
		std::cout << "vertex id: " << mesh3D.edgeVertex(i, 0) << ", value: " << triV3D.row(mesh3D.edgeVertex(i, 0)) << std::endl;
		std::cout << "vertex id: " << mesh3D.edgeVertex(i, 1) << ", value: " << triV3D.row(mesh3D.edgeVertex(i, 1)) << std::endl;
	}

	for (uint32_t n = 0; n < numofFrames; ++n)
	{
		Eigen::MatrixXd vertexOmega(triV2D.rows(), 3);

		vertexOmega.setZero();

		double frequency = (numWaves) * 1.0 / numofFrames * (n + 1);
		
	
		vertexOmega.row(0) << 2 * frequency * PI, 0, 0;
		vertexOmega.row(1) << 2 * frequency * PI, 0, 0;
		vertexOmega.row(2) << 4 * PI, 0, 0;

		/*edgeOmega(0) = 2 * PI;
		edgeOmega(1) = -2 * PI;
		edgeOmega(2) = 2 * frequency * PI;*/

		/*edgeOmega(0) = 2 * frequency * PI;
		edgeOmega(1) = 2 * frequency * PI;
		edgeOmega(2) = 0;
		edgeOmega(3) = 0;
		edgeOmega(4) = -2 * frequency * PI;*/

		/*edgeOmega(0) = 2 * frequency * PI;
		edgeOmega(1) = 2 * frequency * PI;
		edgeOmega(2) = 0;
		edgeOmega(3) = 0;
		edgeOmega(4) = -2 * PI;*/

		/*if (!isOnlyChangeOne)
		{
			double frequency = numWaves + 5 * numWaves * 1.0 / numofFrames * n;
			edgeOmega(0) = 2 * frequency * PI;
			edgeOmega(1) = 2 * frequency * PI;
			edgeOmega(2) = 0;
			edgeOmega(3) = 0;
			edgeOmega(4) = -2 * frequency * PI;
		}
		else
		{
			double frequency = numWaves + 5 * numWaves * 1.0 / numofFrames * n;
			edgeOmega(0) = 2 * frequency * PI;
			edgeOmega(1) = 2 * (frequency + numWaves) / 2.0 * PI;
			edgeOmega(2) = 0;
			edgeOmega(3) = 0;
			edgeOmega(4) = -2 * numWaves * PI;
		}*/

		/*double frequency = numWaves + 5 * numWaves * 1.0 / numofFrames * n;
		edgeOmega << 2 * frequency * PI,
			-2 * frequency * PI,
			0,
			-2 * frequency * PI,
			2 * frequency * PI,
			-2 * frequency * PI,
			0,
			2 * frequency * PI,
			-2 * frequency * PI,
			0,
			-2 * frequency * PI,
			0,
			2 * frequency * PI,
			-2 * frequency * PI,
			2 * frequency * PI,
			2 * frequency * PI;*/

			/*edgeOmega.setZero();
			edgeOmega(0) = 2 * frequency * PI;
			*/


		Eigen::VectorXd upsampledAmp;

		std::vector<std::complex<double>> upsampledPhase;

		model.estimatePhase(vertexOmega, initTheta, upsampledPhase, interpType);
		model.getAngleMagnitude(upsampledPhase, upsampledTheta, upsampledAmp);

		Eigen::MatrixXd phiColor, ampColor;
		mPaint.setNormalization(false);
		phiColor = mPaint.paintPhi(upsampledTheta);

		mPaint.setNormalization(isNormalize);
		ampColor = mPaint.paintAmplitude(upsampledAmp);

		thetaColors[n] = phiColor;
		ampColors[n] = ampColor;

		wrinkledVs[n] = upsampledTriV3D;

		for (int i = 0; i < upsampledTriV3D.rows(); i++)
		{
			wrinkledVs[n].row(i) += 0.05 * upsampledAmp(i) * std::cos(upsampledTheta(i)) * normals.row(i);
		}
	}
}

int main(int argc, char* argv[])
{
	//triV.resize(8, 3);
	//triV << 0, 0, 0,
	//	1, 0, 0,
	//	1, 1, 0,
	//	0, 1, 0,
	//	0, 0, 1,
	//	1, 0, 1,
	//	1, 1, 1,
	//	0, 1, 1;
	//triF.resize(12, 3);
	//triF << 0, 3, 1,
	//	1, 3, 2,
	//	5, 6, 4,
	//	4, 6, 7,
	//	0, 1, 4,
	//	4, 1, 5,
	//	2, 3, 6,
	//	6, 3, 7,
	//	2, 6, 1,
	//	1, 6, 5, 
	//	0, 4, 3,
	//	3, 4, 7;

	triV2D.resize(3, 3);
	triV2D << -std::sqrt(3) / 2, -1.0 / 2, 0,
		std::sqrt(3) / 2, -1.0 / 2, 0,
		0, 1, 0;
	triF2D.resize(1, 3);
	triF2D << 0, 1, 2;

	triV3D = triV2D;
	triF3D = triF2D;

	/*triV.resize(4, 3);
	triV << 0, 0, 0,
		1, 0, 0,
		1, 1, 0,
		0, 1, 0;
	triF.resize(2, 3);
	triF << 0, 1, 2,
		2, 3, 0;*/

		/*triV2D.resize(10, 3);
		triV2D << 0, 0, 0,
			1, 0, 0,
			2, 0, 0,
			3, 0, 0,
			0, 1, 0,
			1, 1, 0,
			2, 1, 0,
			3, 1, 0,
			4, 0, 0,
			4, 1, 0;

		triF2D.resize(8, 3);
		triF2D << 0, 1, 4,
			4, 1, 5,
			1, 2, 5,
			5, 2, 6,
			2, 3, 6,
			6, 3, 7,
			3, 8, 7,
			7, 8, 9;

		triV3D.resize(8, 3);
		triV3D << 1, 0, 0,
			1, 1, 0,
			0, 1, 0,
			0, 0, 0,
			1, 0, 1,
			1, 1, 1,
			0, 1, 1,
			0, 0, 1;

		triF3D.resize(8, 3);
		triF3D << 0, 1, 4,
			4, 1, 5,
			1, 2, 5,
			5, 2, 6,
			2, 3, 6,
			6, 3, 7,
			3, 0, 7,
			7, 0, 4;

		igl::writeOBJ("restCube.obj", triV2D, triF2D);
		igl::writeOBJ("cube.obj", triV3D, triF3D);*/

		/*std::set<int> corners2D, corners3D;
		for (int i = 0; i < 10; i++)
		{
			corners2D.insert(i);
			if (i < 8)
				corners3D.insert(i);
		}*/

	meshUpSampling(triV2D, triF2D, upsampledTriV2D, upsampledTriF2D, loopLevel);
	meshUpSampling(triV3D, triF3D, upsampledTriV3D, upsampledTriF3D, loopLevel);
	std::cout << "upsampling finished" << std::endl;

	MeshConnectivity upsampledMesh2D = MeshConnectivity(upsampledTriF2D);
	MeshConnectivity upsampledMesh3D = MeshConnectivity(upsampledTriF3D);


	//locateTheSeam(upsampledMesh2D, upsampledMesh3D, seamEdges);

	igl::writeOBJ("test_upsampled2D.obj", upsampledTriV2D, upsampledTriF2D);
	igl::writeOBJ("test_upsampled3D.obj", upsampledTriV3D, upsampledTriF3D);

	// get the wrinkled mesh and phase value

	int numofFrames = 1000;

	/*std::set<int> corners;

	std::vector<std::vector<int>> bnds;
	igl::boundary_loop(triF, bnds);
	for (auto& bnd : bnds)
	{
		for (auto& vid : bnd)
		{
			if (corners.count(vid) == 0)
				corners.insert(vid);
		}
	}*/

	std::cout << "argc: " << argc << ", argv: " << std::endl;
	for (int i = 0; i < argc; i++)
	{
		std::cout << argv[i] << std::endl;
	}
	//system("pause");

	/*loopWithCorners(triV, triF, corners, upsampledTriV, upsampledTriF, loopLevel);

	igl::writeOBJ("test_upsampled.obj", upsampledTriV, upsampledTriF);*/


	computePhaseInSequence(numofFrames);

	igl::opengl::glfw::Viewer viewer;

	// Attach a menu plugin
	igl::opengl::glfw::imgui::ImGuiMenu menu;
	viewer.plugins.push_back(&menu);



	// Add content to the default menu window

	menu.callback_draw_custom_window = [&]()
	{
		// Define next window position + size
		ImGui::SetNextWindowPos(ImVec2(180.f * menu.menu_scaling(), 10), ImGuiCond_FirstUseEver);
		ImGui::SetNextWindowSize(ImVec2(0.0, 0.0), ImGuiCond_FirstUseEver);
		ImGui::Begin(
			"phase viewer", nullptr,
			ImGuiWindowFlags_NoSavedSettings
		);

		if (ImGui::CollapsingHeader("Viewing Options", ImGuiTreeNodeFlags_DefaultOpen))
		{
			if (ImGui::Button("Center object", ImVec2(-1, 0)))
			{
				viewer.core().align_camera_center(viewer.data().V, viewer.data().F);
			}
			if (ImGui::Button("Snap canonical view", ImVec2(-1, 0)))
			{
				viewer.snap_to_canonical_quaternion();
			}

			// Select rotation type
			int rotation_type = static_cast<int>(viewer.core().rotation_type);
			//int rotation_type = 0;
			static Eigen::Quaternionf trackball_angle = Eigen::Quaternionf::Identity();
			static bool orthographic = true;
			if (ImGui::Combo("Camera Type", &rotation_type, "Trackball\0Two Axes\0 2D Mode\0\0"))
			{
				using RT = igl::opengl::ViewerCore::RotationType;
				auto new_type = static_cast<RT>(rotation_type);
				if (new_type != viewer.core().rotation_type)
				{
					if (new_type == RT::ROTATION_TYPE_NO_ROTATION)
					{
						trackball_angle = viewer.core().trackball_angle;
						orthographic = viewer.core().orthographic;
						viewer.core().trackball_angle = Eigen::Quaternionf::Identity();
						viewer.core().orthographic = true;
					}
					else if (viewer.core().rotation_type == RT::ROTATION_TYPE_NO_ROTATION)
					{
						viewer.core().trackball_angle = trackball_angle;
						viewer.core().orthographic = orthographic;
					}
					viewer.core().set_rotation_type(new_type);
				}
			}

			// Orthographic view
			ImGui::Checkbox("Orthographic view", &(viewer.core().orthographic));
			//            ImGui::PopItemWidth();
		}
		// Draw options
		if (ImGui::CollapsingHeader("Draw Options", ImGuiTreeNodeFlags_DefaultOpen))
		{
			if (ImGui::InputInt("upsampling level", &loopLevel))
			{
				if (loopLevel <= 0)
					loopLevel = 8;
				//loopWithCorners(triV, triF, corners, upsampledTriV, upsampledTriF, loopLevel);

				meshUpSampling(triV2D, triF2D, upsampledTriV2D, upsampledTriF2D, loopLevel);
				meshUpSampling(triV3D, triF3D, upsampledTriV3D, upsampledTriF3D, loopLevel);


				upsampledMesh2D = MeshConnectivity(upsampledTriF2D);
				upsampledMesh3D = MeshConnectivity(upsampledTriF3D);

				locateTheSeam(upsampledMesh2D, upsampledMesh3D, seamEdges);

				computePhaseInSequence(numofFrames);
			}

			if (ImGui::InputInt("num of waves", &numWaves))
			{
				if (numWaves <= 0)
					numWaves = 1;
				computePhaseInSequence(numofFrames);
			}

			if (ImGui::InputInt("FPS", &fps))
			{
				repaint(viewer);
			}

			ImGui::Checkbox("Visualize Wrinkles", &isVisualizeWrinkles);
			ImGui::Checkbox("isnormalize", &isNormalize);
			ImGui::Checkbox("Visualize Phase", &isVisualizePhase);
			ImGui::Checkbox("Visualize Amp", &isVisualizeAmp);
			
			if (ImGui::Checkbox("Changing one-face w", &(isOnlyChangeOne)))
			{
				computePhaseInSequence(numofFrames);
				currentFrame = 0;
			}

			if (ImGui::Checkbox("Invert normals", &(viewer.data().invert_normals)))
			{
				viewer.data().dirty |= igl::opengl::MeshGL::DIRTY_NORMAL;
			}
			ImGui::ColorEdit4("Background", viewer.core().background_color.data(),
				ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_PickerHueWheel);
			ImGui::ColorEdit4("Line color", viewer.data().line_color.data(),
				ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_PickerHueWheel);
			ImGui::PushItemWidth(ImGui::GetWindowWidth() * 0.3f);
			ImGui::DragFloat("Shininess", &(viewer.data().shininess), 0.05f, 0.0f, 100.0f);

			if (ImGui::Checkbox("Pause", &(isPaused)))
			{
				viewer.core().is_animating = !isAnimated;
				//std::cout << "current frame: " << currentFrame << std::endl;
				repaint(viewer);
			}

			if (ImGui::Combo("interp Type", &interpType, "composite\0plane wave\0water pool\0\0"))
			{
				computePhaseInSequence(numofFrames);
				currentFrame = 0;
				repaint(viewer);
			}

			if (ImGui::Button("Reset", ImVec2(-1, 0)))
			{
				currentFrame = 0;
				repaint(viewer);
			}

			if (ImGui::Button("update viewer", ImVec2(-1, 0)))
			{
				repaint(viewer);
			}
		}

		// Overlays
		auto make_checkbox = [&](const char* label, unsigned int& option)
		{
			return ImGui::Checkbox(label,
				[&]() { return viewer.core().is_set(option); },
				[&](bool value) { return viewer.core().set(option, value); }
			);
		};

		if (ImGui::CollapsingHeader("Overlays", ImGuiTreeNodeFlags_DefaultOpen))
		{
			make_checkbox("Wireframe", viewer.data().show_lines);
			make_checkbox("Fill", viewer.data().show_faces);
			bool showvertid = viewer.data().show_vertex_labels != 0;
			if (ImGui::Checkbox("Show vertex labels", &showvertid))
			{
				viewer.data().show_vertex_labels = (showvertid ? 1 : 0);
			}
			bool showfaceid = viewer.data().show_face_labels != 0;
			if (ImGui::Checkbox("Show faces labels", &showfaceid))
			{
				viewer.data().show_face_labels = showfaceid;
			}
		}
		ImGui::End();
	};

	viewer.callback_pre_draw = [&](igl::opengl::glfw::Viewer& viewer)->bool
	{
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

		int currentDataVerts = nupsampledVerts;
		int currentDataFaces = nupsampledFaces;

		dataV.resize(ndataVerts, 3);
		dataF.resize(ndataFaces, 3);
		curColor.resize(ndataVerts, 3);

		curColor.col(0).setConstant(1.0);
		curColor.col(1).setConstant(0.0);
		curColor.col(2).setConstant(0.0);

		if (isVisualizePhase)
		{
			curColor.block(0, 0, nupsampledVerts, 3) = thetaColors[currentFrame];
		}
		else
		{
			curColor.block(0, 0, nupsampledVerts, 3).col(0).setConstant(1.0);
			curColor.block(0, 0, nupsampledVerts, 3).col(1).setConstant(1.0);
			curColor.block(0, 0, nupsampledVerts, 3).col(2).setConstant(0.0);
		}

		if (isVisualizeAmp)
		{
			curColor.block(currentDataVerts, 0, nupsampledVerts, 3) = ampColors[currentFrame];
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
			dataV.block(currentDataVerts, 0, nupsampledVerts, 3) = wrinkledVs[currentFrame] + shiftV;
			dataF.block(currentDataFaces, 0, nupsampledFaces, 3) = upsampledTriF3D + shiftF;

			currentDataVerts += nupsampledVerts;
			currentDataFaces += nupsampledFaces;
		}


		viewer.data().set_face_based(false);

		//std::cout << "set data done!" << std::endl;
		repaint(viewer);
		//std::cout << "Repaint done!" << std::endl;

		if (!isPaused)
			currentFrame = (currentFrame + 1) % numofFrames;
		return false;
	};

	viewer.launch();
	return 0;

}
