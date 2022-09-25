#include "polyscope/polyscope.h"
#include "polyscope/pick.h"

#include <igl/invert_diag.h>
#include <igl/per_vertex_normals.h>
#include <igl/readOBJ.h>
#include <igl/readOFF.h>
#include <igl/writeOBJ.h>
#include <igl/boundary_loop.h>
#include <igl/doublearea.h>
#include <igl/loop.h>
#include <igl/decimate.h>
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
#include "../../include/IntrinsicFormula/KnoppelStripePatternEdgeOmega.h"
#include "../../include/testMeshGeneration.h"
#include "../../include/SpherigonSmoothing.h"

#include "../../include/json.hpp"
#include "../../include/ComplexLoop/ComplexLoop.h"
#include "../../include/ComplexLoop/ComplexLoopAmpPhase.h"
#include "../../include/ComplexLoop/ComplexLoopAmpPhaseEdgeJump.h"
#include "../../include/ComplexLoop/ComplexLoopReIm.h"
#include "../../include/ComplexLoop/ComplexLoopZuenko.h"

#include "../../include/LoadSaveIO.h"
#include "../../include/SecMeshParsing.h"
#include "../../include/OtherApproaches/KnoppelAlgorithm.h"
#include "../../include/OtherApproaches/TFWAlgorithm.h"
#include "../../include/OtherApproaches/ZuenkoAlgorithm.h"

Eigen::MatrixXd triV, G0V, G1V, midPointV;
Eigen::MatrixXi triF, G0F, G1F, midPointF;

Eigen::MatrixXd triN, triT;

int upsamplingLevel = 2;
std::string prefix = "";


static void callback() {
	ImGui::PushItemWidth(100);
	float w = ImGui::GetContentRegionAvailWidth();
	float p = ImGui::GetStyle().FramePadding.x;

	if (ImGui::Button("load", ImVec2(-1, 0)))
	{
		std::string filePath = igl::file_dialog_open();
		std::replace(filePath.begin(), filePath.end(), '\\', '/'); // handle the backslash issue for windows
		int id = filePath.rfind(".");
		prefix = filePath.substr(0, id);
		triN.resize(0, 3);
		igl::readOBJ(filePath, triV, triT, triN, triF, triF, triF);
		if (triN.rows() == 0 || triN.rows() != triV.rows())
		{
			igl::per_vertex_normals(triV, triF, triN);
		}
		polyscope::registerSurfaceMesh("initial mesh", triV, triF);
	}
	if (ImGui::Button("save", ImVec2(-1, 0)))
	{
		igl::writeOBJ(prefix + "_midpoint.obj", midPointV, midPointF);
		igl::writeOBJ(prefix + "_G0mesh.obj", G0V, G0F);
		igl::writeOBJ(prefix + "_G1mesh.obj", G1V, G1F);
	}

	if (ImGui::InputInt("underline upsampling level", &upsamplingLevel))
	{
		if (upsamplingLevel < 0)
			upsamplingLevel = 2;
	}
	
	if (ImGui::Button("recompute", ImVec2(-1, 0)))
	{
		std::vector<std::pair<int, Eigen::Vector3d>> bary;
		meshUpSampling(triV, triF, midPointV, midPointF, upsamplingLevel, NULL, NULL, &bary);
		MeshConnectivity mesh(triF);
		Eigen::MatrixXd upN;
		ZuenkoAlg::spherigonSmoothing(triV, mesh, triN, bary, G0V, upN, false);
		ZuenkoAlg::spherigonSmoothing(triV, mesh, triN, bary, G1V, upN, true);
		G0F = midPointF;
		G1F = midPointF;
		polyscope::registerSurfaceMesh("midpoint mesh", midPointV, midPointF);
		polyscope::getSurfaceMesh("midpoint mesh")->setEnabled(false);
		polyscope::registerSurfaceMesh("Spherigon-G0 mesh", G0V, midPointF);
		polyscope::getSurfaceMesh("Spherigon-G0 mesh")->setEnabled(false);
		polyscope::registerSurfaceMesh("Spherigon-G1 mesh", G1V, midPointF);
	}


	ImGui::PopItemWidth();
}


int main(int argc, char** argv)
{
	std::string filePath = "";
	if(argc < 2)
	{
		filePath = igl::file_dialog_open();
	}
	else
	{
		filePath = argv[2];
	}
	std::replace(filePath.begin(), filePath.end(), '\\', '/'); // handle the backslash issue for windows
	int id = filePath.rfind(".");
	prefix = filePath.substr(0, id);

	igl::readOBJ(filePath, triV, triT, triN, triF, triF, triF);
	if (triN.rows() == 0)
	{
		igl::per_vertex_normals(triV, triF, triN);
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
	polyscope::registerSurfaceMesh("initial mesh", triV, triF);
	// Show the gui
	polyscope::show();


	return 0;
}