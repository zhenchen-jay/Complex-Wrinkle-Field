#include "geometrycentral/surface/halfedge_mesh.h"
#include "geometrycentral/surface/surface_mesh.h"
#include "geometrycentral/surface/heat_method_distance.h"
#include "geometrycentral/surface/meshio.h"
#include "geometrycentral/surface/surface_centers.h"
#include "geometrycentral/surface/vector_heat_method.h"
#include "geometrycentral/surface/vertex_position_geometry.h"
#include "geometrycentral/surface/direction_fields.h"

#include "polyscope/point_cloud.h"
#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"

#include "../../dep/SecStencils/Mesh.h"
#include "../../include/ComplexLoop/ComplexLoop.h"
#include "../../include/ComplexLoop/ComplexLoopAmpPhase.h"
#include "../../include/ComplexLoop/ComplexLoopAmpPhaseEdgeJump.h"
#include "../../include/ComplexLoop/ComplexLoopReIm.h"
#include "../../include/ComplexLoop/ComplexLoopZuenko.h"
#include "../../include/SecMeshParsing.h"

#include "imgui.h"
#include "../../include/MeshLib/RegionEdition.h"
#include "../../include/CommonTools.h"
#include "../../include/AmpExtraction.h"
#include "../../include/IntrinsicFormula/KnoppelStripePatternEdgeOmega.h"
#include "../../include/IntrinsicFormula/InterpolateZvals.h"
#include "../../include/MeshLib/MeshUpsampling.h"
#include "../../include/Visualization/PaintGeometry.h"
#include "../../include/Optimization/NewtonDescent.h"
#include "../../include/json.hpp"
#include "../../include/LoadSaveIO.h"
#include "../../include/testMeshGeneration.h"

#include <igl/doublearea.h>
#include <igl/cotmatrix_entries.h>
#include <igl/cotmatrix.h>
#include <igl/per_vertex_normals.h>
#include <igl/file_dialog_open.h>
#include <igl/file_dialog_save.h>
#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>
#include <igl/decimate.h>

#include <sstream>

using namespace geometrycentral;
using namespace geometrycentral::surface;

// Manage a list of sources
struct SourceVert {
	Vertex vertex;
	double scalarVal = 1.;
	double vectorMag = 100.;
	float vectorAngleRad = 0.;
};

// smoothing
int smoothingTimes = 3;
double smoothingRatio = 0.95;
bool isUseLoop = false;

std::unique_ptr<VectorHeatMethodSolver> solver;

void convertGeoCentralMesh(HalfedgeMesh& geoMesh, VertexPositionGeometry& geoPos, Eigen::MatrixXd& triV, MeshConnectivity& mesh)
{
	int nverts = geoPos.inputVertexPositions.size();
	triV.resize(nverts, 3);
	for (int i = 0; i < nverts; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			triV(i, j) = geoPos.inputVertexPositions[i][j];
		}
	}

	Eigen::MatrixXi triF;
	int nfaces = geoMesh.nFaces();
	triF.resize(nfaces, 3);
	for (int i = 0; i < nfaces; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			triF(i, j) = geoMesh.getFaceVertexList()[i][j];
		}
	}
	mesh = MeshConnectivity(triF);
}

void getVecTransport(VertexPositionGeometry& geoPos, std::vector<SourceVert> sourcePoints, Eigen::MatrixXd& vertVecs)
{
	std::vector<std::tuple<SurfacePoint, geometrycentral::Vector2>> points;
	for (SourceVert& s : sourcePoints) {
		points.emplace_back(s.vertex, geometrycentral::Vector2::fromAngle(s.vectorAngleRad) * s.vectorMag);
	}
	VertexData<geometrycentral::Vector2> vectorExtension = solver->transportTangentVectors(points);

	int nverts = geoPos.inputVertexPositions.size();

	vertVecs.setZero(nverts, 3);


	for (size_t iV = 0; iV < nverts; iV++) {

        geometrycentral::Vector3 normal = geoPos.vertexNormals[iV];
        geometrycentral::Vector3 basisX = geoPos.vertexTangentBasis[iV][0];
        geometrycentral::Vector3 basisY = geoPos.vertexTangentBasis[iV][1];

		std::complex<double> angle = std::complex<double>(vectorExtension[iV][0], vectorExtension[iV][1]);

        geometrycentral::Vector3 vec3 = basisX * (float)angle.real() + basisY * (float)angle.imag();
		vertVecs.row(iV) << vec3.x, vec3.y, vec3.z;
	}
}

void getSmoothestVec(VertexPositionGeometry& geoPos, Eigen::MatrixXd& vertVecs)
{
	VertexData<geometrycentral::Vector2> vectorExtension;
	if(geoPos.boundaryLoopIndices.size())
		vectorExtension = computeSmoothestBoundaryAlignedVertexDirectionField(geoPos);
	else
		vectorExtension = computeSmoothestVertexDirectionField(geoPos);

	int nverts = geoPos.inputVertexPositions.size();

	vertVecs.setZero(nverts, 3);


	for (size_t iV = 0; iV < nverts; iV++) {

		geometrycentral::Vector3 normal = geoPos.vertexNormals[iV];
		geometrycentral::Vector3 basisX = geoPos.vertexTangentBasis[iV][0];
		geometrycentral::Vector3 basisY = geoPos.vertexTangentBasis[iV][1];

		std::complex<double> angle = std::complex<double>(vectorExtension[iV][0], vectorExtension[iV][1]);

		geometrycentral::Vector3 vec3 = basisX * (float)angle.real() + basisY * (float)angle.imag();
		vertVecs.row(iV) << vec3.x, vec3.y, vec3.z;
	}
}


// == Geometry-central data
std::unique_ptr<HalfedgeMesh> mesh;
std::unique_ptr<VertexPositionGeometry> geometry;

Eigen::MatrixXd triV;
MeshConnectivity triMesh;
int upsampleTimes = 2;
bool isFixBnd = false;

// Polyscope visualization handle, to quickly add data to the surface
polyscope::SurfaceMesh* psMesh;

// Some algorithm parameters
float tCoef = 1.0;
double sCoef = 1.0;
double vFreq = 2 * M_PI;
double vAmp = 1.0;
int vertexInd = 0;
int pCenter = 2;

std::vector<SourceVert> sourcePoints;
Eigen::VectorXd edgeVec;
Eigen::VectorXd vertAmp;
std::vector<std::complex<double>> vertZvals;

void loadBaseMesh(const std::string& meshPath)
{
	std::tie(mesh, geometry) = loadMesh(meshPath);
	geometry->requireVertexTangentBasis();
	geometry->requireVertexNormals();
	geometry->requireFaceTangentBasis();
	geometry->requireVertexIndices();

	// Register the mesh with polyscope
	polyscope::registerSurfaceMesh("base mesh", geometry->inputVertexPositions, mesh->getFaceVertexList(), polyscopePermutations(*mesh));

	convertGeoCentralMesh(*mesh, *geometry, triV, triMesh);
    
	double shiftx = 1.2 * (triV.col(0).maxCoeff() - triV.col(0).minCoeff());
	Eigen::MatrixXd shiftedV = triV;
	shiftedV.setZero();
	shiftedV.col(0).setConstant(shiftx);
	psMesh = polyscope::registerSurfaceMesh("extended mesh", triV + shiftedV, triMesh.faces());

	// Set vertex tangent spaces
	VertexData<geometrycentral::Vector3> vBasisX(*mesh);
	for (Vertex v : mesh->vertices()) {
		vBasisX[v] = geometry->vertexTangentBasis[v][0];
	}
	polyscope::getSurfaceMesh("base mesh")->setVertexTangentBasisX(vBasisX);



	// Set face tangent spaces
	FaceData<geometrycentral::Vector3> fBasisX(*mesh);
	for (Face f : mesh->faces()) {
		fBasisX[f] = geometry->faceTangentBasis[f][0];
	}
	polyscope::getSurfaceMesh("base mesh")->setFaceTangentBasisX(fBasisX);
}

void load()
{
	std::string loadFileName = igl::file_dialog_open();

	std::cout << "load file in: " << loadFileName << std::endl;
	using json = nlohmann::json;
	std::ifstream inputJson(loadFileName);
	if (!inputJson) {
		std::cerr << "missing json file in " << loadFileName << std::endl;
		exit(EXIT_FAILURE);
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

	loadBaseMesh(meshFile);
	std::cout << "load base mesh finished!" << std::endl;
	upsampleTimes = 0;
	tCoef = 1;
	sCoef = 1;

	if (jval.contains(std::string_view{ "upsampled_times" }))
	{
		upsampleTimes = jval["upsampled_times"];
	}
	if (jval.contains(std::string_view{ "vector_heat_timeStep" }))
	{
		tCoef = jval["vector_heat_timeStep"];
	}
	if (jval.contains(std::string_view{ "amplitude_extend_coeff" }))
	{
		sCoef = jval["amplitude_extend_coeff"];
	}

	std::cout << "t coef: " << tCoef << ", s coef: " << sCoef << std::endl;

	if (jval.contains(std::string_view{ "laplacian_smooth" }))
	{
		if(jval["laplacian_smooth"].contains(std::string_view{ "smoothing_ratio" }))
			smoothingRatio = jval["laplacian_smooth"]["smoothing_ratio"];
		if (jval["laplacian_smooth"].contains(std::string_view{ "smoothing_times" }))
			smoothingTimes = jval["laplacian_smooth"]["smoothing_times"];
	}
	
	
	
	sourcePoints.clear();

	Eigen::VectorXi vertFlags = Eigen::VectorXi::Zero(mesh->nVertices());

	if (jval.contains(std::string_view{ "source_points" }))
	{
		std::cout << "num of source points: " << jval["source_points"].size() << std::endl;

		for (int i = 0; i < jval["source_points"].size(); i++)
		{
			int ind = jval["source_points"][i]["vert_id"];
			Vertex v = mesh->vertex(ind);

			if (vertFlags[ind])
			{
				std::stringstream ss;
				ss << "Vertex " << v;
				std::string vStr = ss.str();
				polyscope::warning("Vertex " + vStr + " is already a source");
				return;
			}

			SourceVert newV;
			newV.vertex = v;
			newV.vectorMag = jval["source_points"][i]["vector_magnitude"];
			newV.vectorAngleRad = jval["source_points"][i]["vector_radiusAngle"];
			newV.scalarVal = jval["source_points"][i]["scalar_value"];
			
			sourcePoints.push_back(newV);
		}
		std::cout << "loading finished!" << std::endl;
	}
}

void save()
{
	std::string filePath = igl::file_dialog_save();
	
	std::replace(filePath.begin(), filePath.end(), '\\', '/'); // handle the backslash issue for windows
	int id = filePath.rfind("/");
	std::string workingFolder = filePath.substr(0, id + 1);

	std::ofstream iwfs(workingFolder + "omega.txt");
	iwfs << std::setprecision(std::numeric_limits<long double>::digits10 + 1) << edgeVec << std::endl;

	std::ofstream iafs(workingFolder + "amp.txt");
	iafs << std::setprecision(std::numeric_limits<long double>::digits10 + 1) << vertAmp << std::endl;

	std::ofstream izfs(workingFolder + "zvals.txt");
	for (int i = 0; i < vertZvals.size(); i++)
	{
		izfs << std::setprecision(std::numeric_limits<long double>::digits10 + 1) << vertZvals[i].real() << " " << vertZvals[i].imag() << std::endl;
	}

	igl::writeOBJ(workingFolder + "mesh.obj", triV, triMesh.faces());

	using json = nlohmann::json;
	json jval =
	{
			{"mesh_name",         "mesh.obj"},
			{"num_frame",         10},
			{"quad_order",        4},
			{"spatial_ratio",     {
										   {"amp_ratio", 10},
										   {"edge_ratio", 10},
										   {"knoppel_ratio", 10}

								  }
			},
			{"upsampled_times", upsampleTimes},
			{"init_omega",        "omega.txt"},
			{"init_amp",          "amp.txt"},
			{"init_zvls",         "zvals.txt"},
			{
			 "region_global_details", 
								  {
										  {"select_all", false},
										  {"omega_operation_motion", "None"},
										  {"omega_operation_value", 1},
										  {"amp_omega_coupling", false},
										  {"amp_operation_value", 1}
								  }
			},
			{
			 "reference",         {
										  {"ref_amp", "/refAmp/"},
										  {"ref_omega", "/refOmega/"}
								  }
			},
			{
			 "solution",          {
										  {"opt_zvals", "/optZvals/"},
										  {"opt_omega", "/optOmega/"},
										  {"wrinkle_mesh", "/wrinkledMesh/"},
										  {"upsampled_amp", "/upsampledAmp/"},
										  {"upsampled_phase", "/upsampledPhase/"}
								  }
			}
	};

	std::string saveDataFileName = workingFolder + "/data.json";
	std::ofstream o(saveDataFileName);
	o << std::setw(4) << jval << std::endl;
	std::cout << "save file in: " << saveDataFileName << std::endl;

	json vecJval =
	{
			{"mesh_name",         "mesh.obj"},
			{"upsampled_times", upsampleTimes},
			{"laplacian_smooth",
									{
										{"smoothing_times", smoothingTimes},
										{"smoothing_ratio", smoothingRatio}
									}
			},
			{"vector_heat_timeStep", tCoef},
			{"amplitude_extend_coeff", sCoef}
	};
	for (int i = 0; i < sourcePoints.size(); i++)
	{
		json spJval = 
		{
			{"vert_id", sourcePoints[i].vertex.getIndex()},
			{"vector_magnitude", sourcePoints[i].vectorMag},
			{"vector_radiusAngle", sourcePoints[i].vectorAngleRad},
			{"scalar_value", sourcePoints[i].scalarVal}
		};
		vecJval["source_points"].push_back(spJval);
	}
	
	std::ofstream wo(filePath);
	wo << std::setw(4) << vecJval << std::endl;
	std::cout << "save wrinkle design file in: " << filePath << std::endl;
}

bool vizFirstRun = true;
void updateSourceSetViz() 
{

	// Scalar balls around sources
	std::vector<std::pair<size_t, double>> sourcePairs;
	for (SourceVert& s : sourcePoints) {
		size_t ind = geometry->vertexIndices[s.vertex];
		sourcePairs.emplace_back(ind, s.scalarVal);
	}
	auto scalarQ = polyscope::getSurfaceMesh("base mesh")->addVertexIsolatedScalarQuantity("source scalars", sourcePairs);

	scalarQ->setColorMap("reds");

	if (vizFirstRun) 
	{
		scalarQ->setEnabled(true);
	}

	// Vectors at sources
	VertexData<geometrycentral::Vector2> sourceVectors(*mesh, geometrycentral::Vector2::zero());
	Eigen::VectorXd ampList = Eigen::VectorXd::Zero(mesh->nVertices());
	for (SourceVert& s : sourcePoints) 
	{
		sourceVectors[s.vertex] = geometrycentral::Vector2::fromAngle(s.vectorAngleRad) * s.vectorMag;
	}

	auto vectorQ = polyscope::getSurfaceMesh("base mesh")->addVertexIntrinsicVectorQuantity("source vectors", sourceVectors);
	vectorQ->setVectorLengthScale(.05);
	vectorQ->setVectorRadius(.005);
	vectorQ->setVectorColor(glm::vec3{ 227 / 255., 52 / 255., 28 / 255. });
	if (vizFirstRun) 
	{
		vectorQ->setEnabled(true);
	}

	vizFirstRun = false;
}

void addVertexSource(size_t ind) 
{
	Vertex v = mesh->vertex(ind);

	// Make sure not already used
	for (SourceVert& s : sourcePoints) 
	{
		if (s.vertex == v) 
		{
			std::stringstream ss;
			ss << "Vertex " << v;
			std::string vStr = ss.str();
			polyscope::warning("Vertex " + vStr + " is already a source");
			return;
		}
	}

	SourceVert newV;
	newV.vertex = v;
	sourcePoints.push_back(newV);
	updateSourceSetViz();
}


void vectorTransport() {
	if (solver == nullptr) {
		solver.reset(new VectorHeatMethodSolver(*geometry, tCoef));
	}

	if (sourcePoints.size() == 0) {
		polyscope::warning("no source points set");
		return;
	}

	Eigen::MatrixXd vertVec;
	getVecTransport(*geometry, sourcePoints, vertVec);

	auto psVec = psMesh->addVertexVectorQuantity("vector extension", vertVec);
	psVec->setEnabled(true);
}

void smoothestVector()
{
	Eigen::MatrixXd vertVec;
	getSmoothestVec(*geometry, vertVec);

	auto psVec = psMesh->addVertexVectorQuantity("smoothest vector", vertVec);
	psVec->setEnabled(true);
}

void stripePatternExtraction(const Eigen::MatrixXd& vertVec)
{
	edgeVec = vertexVec2IntrinsicVec(vertVec, triV, triMesh);
	Eigen::MatrixXd faceVec = intrinsicEdgeVec2FaceVec(edgeVec, triV, triMesh);
	auto fVec = psMesh->addFaceVectorQuantity("face vector", faceVec);
	fVec->setEnabled(true);

	vertAmp.setOnes(triV.rows());
	vertAmp *= vAmp;

	Eigen::VectorXd edgeArea, vertArea;
	edgeArea = getEdgeArea(triV, triMesh);
	vertArea = getVertArea(triV, triMesh);

	std::vector<std::complex<double>> upsampledZvals;
	vertZvals.resize(triV.rows(), 0);

	IntrinsicFormula::roundZvalsFromEdgeOmega(triMesh, edgeVec, edgeArea, vertArea, triV.rows(), vertZvals);

	for (int i = 0; i < vertZvals.size(); i++)
	{
		double theta = std::arg(vertZvals[i]);
		vertZvals[i] = vertAmp(i) * std::complex<double>(std::cos(theta), std::sin(theta));
	}


	Eigen::MatrixXd upsampledTriV, wrinkledV;
	Eigen::MatrixXi upsampledTriF;

	Mesh secMesh = convert2SecMesh(triV, triMesh.faces());
	Mesh subSecMesh = secMesh;
	Eigen::VectorXd secEdgeVec = swapEdgeVec(triMesh.faces(), edgeVec, 0);
	Eigen::VectorXd subEdgeVec;

	std::shared_ptr<ComplexLoop> complexLoopOpt = std::make_shared<ComplexLoopZuenko>();
	complexLoopOpt->setBndFixFlag(true);
	complexLoopOpt->SetMesh(secMesh);
	complexLoopOpt->Subdivide(secEdgeVec, vertZvals, subEdgeVec, upsampledZvals, upsampleTimes);
	subSecMesh = complexLoopOpt->GetMesh();
	parseSecMesh(subSecMesh, upsampledTriV, upsampledTriF);

	Eigen::VectorXd mag(upsampledTriV.rows()), phase(upsampledTriV.rows());
	wrinkledV = upsampledTriV;
	/*Eigen::MatrixXd vertNormals;
	igl::per_vertex_normals(upsampledTriV, upsampledTriF, vertNormals);*/

	std::vector<std::vector<int>> vertNeiEdges, vertNeiFaces;
	buildVertexNeighboringInfo(MeshConnectivity(upsampledTriF), upsampledTriV.rows(), vertNeiEdges, vertNeiFaces);

	getWrinkledMesh(upsampledTriV, upsampledTriF, upsampledZvals, &vertNeiFaces, wrinkledV, 1.0, true);

	for (int i = 0; i < upsampledTriV.rows(); i++)
	{
		mag(i) = std::abs(upsampledZvals[i]);
		phase(i) = std::arg(upsampledZvals[i]);
		//wrinkledV.row(i) += upsampledZvals[i].real() * vertNormals.row(i);
	}

	PaintGeometry mpaint;

	mpaint.setNormalization(false);
	Eigen::MatrixXd phiColor = mpaint.paintPhi(phase);

	int nupverts = upsampledTriV.rows();
	int nupfaces = upsampledTriF.rows();

	double shiftx = 1.2 * (triV.col(0).maxCoeff() - triV.col(0).minCoeff());

	std::cout << "shift x: " << shiftx << std::endl;

	Eigen::MatrixXd wrinkleColor = wrinkledV;
	for (int i = 0; i < nupverts; i++)
		wrinkleColor.row(i) << 80 / 255.0, 122 / 255.0, 91 / 255.0;

	Eigen::MatrixXd shiftV = Eigen::MatrixXd::Zero(nupverts, 3);
	shiftV.col(0).setConstant(shiftx);

	polyscope::registerSurfaceMesh("wrinkled mesh", wrinkledV + 2 * shiftV, upsampledTriF);
	polyscope::getSurfaceMesh("wrinkled mesh")->addVertexColorQuantity("VertexColor", wrinkleColor);
	polyscope::getSurfaceMesh("wrinkled mesh")->getQuantity("VertexColor")->setEnabled(true);


	polyscope::registerSurfaceMesh("phase mesh", upsampledTriV + 3 * shiftV, upsampledTriF);
	polyscope::getSurfaceMesh("phase mesh")->addVertexColorQuantity("VertexColor", phiColor);
	polyscope::getSurfaceMesh("phase mesh")->getQuantity("VertexColor")->setEnabled(true);

	polyscope::registerSurfaceMesh("ampliude mesh", upsampledTriV + 4 * shiftV, upsampledTriF);
	polyscope::getSurfaceMesh("ampliude mesh")->addVertexScalarQuantity("VertexColor", mag);
	polyscope::getSurfaceMesh("ampliude mesh")->getQuantity("VertexColor")->setEnabled(true);


}

void buildPointsMenu() 
{

	bool anyChanged = false;

	ImGui::PushItemWidth(200);

	int id = 0;
	int eraseInd = -1;
	for (SourceVert& s : sourcePoints) 
	{
		std::stringstream ss;
		ss << "Vertex " << s.vertex;
		std::string vStr = ss.str();
		ImGui::PushID(vStr.c_str());

		ImGui::TextUnformatted(vStr.c_str());

		ImGui::SameLine();
		if (ImGui::Button("delete")) 
		{
			eraseInd = id;
			anyChanged = true;
		}
		ImGui::Indent();

		if (ImGui::InputDouble("scalar value", &s.scalarVal)) anyChanged = true;
		if (ImGui::InputDouble("vector mag", &s.vectorMag)) anyChanged = true;
		if (ImGui::SliderAngle("vector angle", &s.vectorAngleRad)) anyChanged = true;

		ImGui::Unindent();
		ImGui::PopID();
		id++;
	}
	ImGui::PopItemWidth();

	// actually do erase, if requested
	if (eraseInd != -1) 
	{
		sourcePoints.erase(sourcePoints.begin() + eraseInd);
	}

	if (ImGui::Button("add point")) 
	{
		long long int pickVert = polyscope::getSurfaceMesh("base mesh")->selectVertex();
		if (pickVert >= 0 && pickVert < triV.rows()) 
		{
			addVertexSource(pickVert);
			anyChanged = true;
		}
	}

	if (anyChanged)
	{
		updateSourceSetViz();
	}
}

void myCallback() {
	ImGui::PushItemWidth(100);
	float w = ImGui::GetContentRegionAvailWidth();
	float p = ImGui::GetStyle().FramePadding.x;
	if (ImGui::Button("Load", ImVec2(ImVec2((w - p) / 2.f, 0))))
	{
		load();
		updateSourceSetViz();
	}
	ImGui::SameLine(0, p);
	if (ImGui::Button("Save", ImVec2((w - p) / 2.f, 0)))
	{
		save();
	}

	ImGuiTabBarFlags tab_bar_flags = ImGuiTabBarFlags_None;
	if (ImGui::BeginTabBar("Visualization Options", tab_bar_flags))
	{
		if (ImGui::BeginTabItem("Wrinkle Mesh Smoothing"))
		{
			if (ImGui::InputInt("smoothing times", &smoothingTimes))
			{
				smoothingTimes = smoothingTimes > 0 ? smoothingTimes : 0;
			}
			if (ImGui::InputDouble("smoothing ratio", &smoothingRatio))
			{
				smoothingRatio = smoothingRatio > 0 ? smoothingRatio : 0;
			}
			ImGui::EndTabItem();
		}
		if (ImGui::BeginTabItem("Wrinkle Mesh Upsampling"))
		{
			ImGui::Checkbox("use loop", &isUseLoop);
			ImGui::Checkbox("bnd fix", &isFixBnd);
			if (ImGui::InputInt("upsampled level", &upsampleTimes))
			{
				if (upsampleTimes < 0)
					upsampleTimes = 2;
			}
			ImGui::EndTabItem();
		}
		ImGui::EndTabBar();
	}
	if (ImGui::BeginTabBar("Wrinkle Design", tab_bar_flags)) 
	{
		if (ImGui::BeginTabItem("Vector Heat Algorithm")) 
		{
			ImGui::TextUnformatted("Algorithm options:");
			ImGui::PushItemWidth(100);
			if (ImGui::InputFloat("tCoef", &tCoef))
			{
				solver.reset();
			}
			ImGui::PopItemWidth();
			// Build the list of source points
			if (ImGui::TreeNode("select source points")) 
			{
				buildPointsMenu();
				ImGui::TreePop();
			}
			ImGui::EndTabItem();
		}

		if (ImGui::BeginTabItem("Smoothest Vector fields Algorithm"))
		{
			if (ImGui::InputDouble("freq", &vFreq))
			{
				if (vFreq < 0)
				{
					vFreq = 2 * M_PI;
				}
			}
			if (ImGui::InputDouble("amp", &vAmp))
			{
				if (vAmp < 0)
				{
					vAmp = 1;
				}
			}
		}

		ImGui::EndTabBar();
	}

	if (ImGui::Button("run vector heat method")) 
	{
		vectorTransport();
	}

	if (ImGui::Button("run smoothest vector field"))
	{
		Eigen::MatrixXd vertVec;
		getSmoothestVec(*geometry, vertVec);
		vertVec *= vFreq;
		stripePatternExtraction(vertVec);
	}
}

int main(int argc, char** argv)
{    
	Eigen::MatrixXd testV;
	Eigen::MatrixXi testF;
	Eigen::VectorXi map;
	igl::readOBJ("G:/WrinkleEdition_dataset/meshes/eight.obj", testV, testF);
	igl::decimate(testV, testF, 2000, testV, testF, map);
	igl::writeOBJ("G:/WrinkleEdition_dataset/meshes/eight_decimate.obj", testV, testF);


	// Initialize polyscope
	polyscope::init();

	// Set the callback function
	polyscope::state::userCallback = myCallback;
	load();
	updateSourceSetViz();



	polyscope::view::upDir = polyscope::view::UpDir::ZUp;

	// Give control to the polyscope gui
	polyscope::show();

	return EXIT_SUCCESS;
}
