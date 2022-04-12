#include "geometrycentral/surface/halfedge_mesh.h"
#include "geometrycentral/surface/heat_method_distance.h"
#include "geometrycentral/surface/meshio.h"
#include "geometrycentral/surface/surface_centers.h"
#include "geometrycentral/surface/vector_heat_method.h"
#include "geometrycentral/surface/vertex_position_geometry.h"

#include "polyscope/point_cloud.h"
#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"

#include "imgui.h"
#include "../../include/MeshLib/RegionEdition.h"
#include "../../include/CommonTools.h"
#include "../../include/AmpExtraction.h"

#include <sstream>

using namespace geometrycentral;
using namespace geometrycentral::surface;

// Manage a list of sources
struct SourceVert {
	Vertex vertex;
	double scalarVal = 1.;
	int effectiveRadius = 5;
	double vectorMag = 1.;
	float vectorAngleRad = 0.;
};

std::unique_ptr<VectorHeatMethodSolver> solver;

void convertPolyScopeMesh(const polyscope::SurfaceMesh& psMesh, Eigen::MatrixXd& triV, MeshConnectivity& mesh)
{
	int nverts = psMesh.vertices.size();
	triV.resize(nverts, 3);
	for (int i = 0; i < nverts; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			triV(i, j) = psMesh.vertices[i][j];
		}
	}

	Eigen::MatrixXi triF;
	int nfaces = psMesh.faces.size();
	triF.resize(nfaces, 3);
	for (int i = 0; i < nfaces; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			triF(i, j) = psMesh.faces[i][j];
		}
	}
	mesh = MeshConnectivity(triF);
}

void getVecTransport(const polyscope::SurfaceMesh& psMesh, std::vector<SourceVert> sourcePoints, Eigen::MatrixXd& vertVecs)
{
	std::vector<std::tuple<SurfacePoint, Vector2>> points;
	for (SourceVert& s : sourcePoints) {
		points.emplace_back(s.vertex, Vector2::fromAngle(s.vectorAngleRad) * s.vectorMag);
	}
	VertexData<Vector2> vectorExtension = solver->transportTangentVectors(points);

	vertVecs.setZero(psMesh.nVertices(), 3);



	for (size_t iV = 0; iV < psMesh.nVertices(); iV++) {

		glm::vec3 normal = psMesh.vertexNormals[iV];
		glm::vec3 basisX = psMesh.vertexTangentSpaces[iV][0];
		glm::vec3 basisY = psMesh.vertexTangentSpaces[iV][1];

		std::complex<double> angle = std::complex<double>(vectorExtension[iV][0], vectorExtension[iV][1]);

		glm::vec3 vec3 = basisX * (float)angle.real() + basisY * (float)angle.imag();
		vertVecs.row(iV) << vec3.x, vec3.y, vec3.z;
	}
}

void buildMask(const MeshConnectivity& mesh, int nverts, std::vector<SourceVert> sourcePoints, Eigen::VectorXi& vertFlags, Eigen::VectorXi& edgeFlags, Eigen::VectorXi& faceFlags)
{
	int nfaces = mesh.nFaces();
	int nedges = mesh.nEdges();

	faceFlags.setZero(nfaces);

	RegionEdition regEd(mesh);

	std::vector<std::vector<int>> vertNeiEdges, vertNeiFaces;

	buildVertexNeighboringInfo(mesh, nverts, vertNeiEdges, vertNeiFaces);
	for (SourceVert& s : sourcePoints) 
	{
		if (s.effectiveRadius)
		{
			Eigen::VectorXi curfaceFlags, curfaceFlagsNew;

			curfaceFlags.setZero(nfaces);
			curfaceFlagsNew.setZero(nfaces);

			int vid = s.vertex.getIndex();

			for (int i = 0; i < vertNeiFaces[vid].size(); i++)
			{
				curfaceFlags(vertNeiFaces[vid][i]) = 1;
			}
			curfaceFlagsNew = curfaceFlags;
			
			for (int i = 0; i < s.effectiveRadius - 1; i++)
			{
				regEd.faceDilation(curfaceFlags, curfaceFlagsNew);
				curfaceFlags = curfaceFlagsNew;
			}

			faceFlags += curfaceFlags;
		}	
	}

	vertFlags.setZero(nverts);
	edgeFlags.setZero(nedges);

	for (int i = 0; i < nfaces; i++)
	{
		if (faceFlags(i))
		{
			faceFlags(i) = 1;
			for (int j = 0; j < 3; j++)
			{
				int eid = mesh.faceEdge(i, j);
				int vid = mesh.faceVertex(i, j);

				vertFlags(vid) = 1;
				edgeFlags(eid) = 1;
			}
		}
			
	}


}

// == Geometry-central data
std::unique_ptr<HalfedgeMesh> mesh;
std::unique_ptr<VertexPositionGeometry> geometry;

Eigen::MatrixXd triV;
MeshConnectivity triMesh;

// Polyscope visualization handle, to quickly add data to the surface
polyscope::SurfaceMesh* psMesh;

// Some algorithm parameters
float tCoef = 1.0;
int vertexInd = 0;
int pCenter = 2;

std::vector<SourceVert> sourcePoints;


bool vizFirstRun = true;
void updateSourceSetViz() {

	// Scalar balls around sources
	std::vector<std::pair<size_t, double>> sourcePairs;
	for (SourceVert& s : sourcePoints) {
		size_t ind = geometry->vertexIndices[s.vertex];
		sourcePairs.emplace_back(ind, s.scalarVal);
	}
	auto scalarQ = polyscope::getSurfaceMesh()->addVertexIsolatedScalarQuantity("source scalars", sourcePairs);

	scalarQ->setColorMap("reds");

	if (vizFirstRun) {
		scalarQ->setEnabled(true);
	}

	// Vectors at sources
	VertexData<Vector2> sourceVectors(*mesh, Vector2::zero());
	for (SourceVert& s : sourcePoints) {
		sourceVectors[s.vertex] = Vector2::fromAngle(s.vectorAngleRad) * s.vectorMag;
	}
	auto vectorQ = polyscope::getSurfaceMesh()->addVertexIntrinsicVectorQuantity("source vectors", sourceVectors);
	vectorQ->setVectorLengthScale(.05);
	vectorQ->setVectorRadius(.005);
	vectorQ->setVectorColor(glm::vec3{ 227 / 255., 52 / 255., 28 / 255. });
	if (vizFirstRun) {
		vectorQ->setEnabled(true);
	}

	vizFirstRun = false;
}

void addVertexSource(size_t ind) {
	Vertex v = mesh->vertex(ind);

	// Make sure not already used
	for (SourceVert& s : sourcePoints) {
		if (s.vertex == v) {
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

	Eigen::VectorXi faceFlags, vertFlags, edgeFlags;
	buildMask(triMesh, triV.rows(), sourcePoints, vertFlags, edgeFlags, faceFlags);

	Eigen::MatrixXd vertVec;
	getVecTransport(*psMesh, sourcePoints, vertVec);

	for (int i = 0; i < triV.rows(); i++)
		vertVec.row(i) *= vertFlags(i);

	auto psVec = psMesh->addVertexVectorQuantity("vector extension", vertVec);
	psVec->setEnabled(true);

	/*Eigen::VectorXd edgeVec = vertexVec2IntrinsicVec(vertVec, triV, triMesh);

	Eigen::MatrixXd faceVec = intrinsicEdgeVec2FaceVec(edgeVec, triV, triMesh);
	for (int i = 0; i < triMesh.nFaces(); i++)
		faceVec.row(i) *= faceFlags(i);

	auto fVec = psMesh->addFaceVectorQuantity("face extension", faceVec);
	fVec->setEnabled(true);*/
}

void wrinkleExtraction()
{
	if (solver == nullptr) {
		solver.reset(new VectorHeatMethodSolver(*geometry, tCoef));
	}

	if (sourcePoints.size() == 0) {
		polyscope::warning("no source points set");
		return;
	}

	Eigen::VectorXi faceFlags, vertFlags, edgeFlags;
	buildMask(triMesh, triV.rows(), sourcePoints, vertFlags, edgeFlags, faceFlags);

	Eigen::MatrixXd vertVec;
	getVecTransport(*psMesh, sourcePoints, vertVec);

	for (int i = 0; i < triV.rows(); i++)
		vertVec.row(i) *= vertFlags(i);

	Eigen::VectorXd edgeVec = vertexVec2IntrinsicVec(vertVec, triV, triMesh);
	for (int i = 0; i < triMesh.nEdges(); i++)
		edgeVec(i) *= edgeFlags(i);

	Eigen::MatrixXd faceVec = intrinsicEdgeVec2FaceVec(edgeVec, triV, triMesh);
	for (int i = 0; i < triMesh.nFaces(); i++)
		faceVec.row(i) *= faceFlags(i);

	std::vector<std::pair<int, double>> clampedAmps;
	for (int i = 0; i < vertVec.rows(); i++)
	{
		if (vertVec(i) == 0)
			clampedAmps.push_back({ i, 0 });
	}

	for (SourceVert& s : sourcePoints)
	{
		int vid = s.vertex.getIndex();
		clampedAmps.push_back({ vid, s.scalarVal });
	}

	Eigen::VectorXd amp;

	ampExtraction(triV, triMesh, edgeVec, clampedAmps, amp);

	auto fVec = psMesh->addFaceVectorQuantity("face vector", faceVec);
	fVec->setEnabled(true);

	auto pScal = psMesh->addVertexScalarQuantity("amp", amp);
	//std::cout << amp(0) << std::endl;
	pScal->setEnabled(true);
}

void buildPointsMenu() {

	bool anyChanged = false;

	ImGui::PushItemWidth(200);

	int id = 0;
	int eraseInd = -1;
	for (SourceVert& s : sourcePoints) {
		std::stringstream ss;
		ss << "Vertex " << s.vertex;
		std::string vStr = ss.str();
		ImGui::PushID(vStr.c_str());

		ImGui::TextUnformatted(vStr.c_str());

		ImGui::SameLine();
		if (ImGui::Button("delete")) {
			eraseInd = id;
			anyChanged = true;
		}
		ImGui::Indent();

		if (ImGui::InputDouble("scalar value", &s.scalarVal)) anyChanged = true;
		if (ImGui::InputDouble("vector mag", &s.vectorMag)) anyChanged = true;
		if (ImGui::InputInt("effective radius", &s.effectiveRadius)) anyChanged = true;
		if (ImGui::SliderAngle("vector angle", &s.vectorAngleRad)) anyChanged = true;

		ImGui::Unindent();
		ImGui::PopID();
	}
	ImGui::PopItemWidth();

	// actually do erase, if requested
	if (eraseInd != -1) {
		sourcePoints.erase(sourcePoints.begin() + eraseInd);
	}

	if (ImGui::Button("add point")) {
		long long int pickVert = polyscope::getSurfaceMesh()->selectVertex();
		if (pickVert >= 0) {
			addVertexSource(pickVert);
			anyChanged = true;
		}
	}

	if (anyChanged) {
		updateSourceSetViz();
	}
}

void myCallback() {

	ImGuiTabBarFlags tab_bar_flags = ImGuiTabBarFlags_None;
	if (ImGui::BeginTabBar("MyTabBar", tab_bar_flags)) {
		if (ImGui::BeginTabItem("Basic algorithm")) {

			ImGui::TextUnformatted("Algorithm options:");
			ImGui::PushItemWidth(100);
			if (ImGui::InputFloat("tCoef", &tCoef)) {
				solver.reset();
			}
			ImGui::PopItemWidth();

			// Build the list of source points
			if (ImGui::TreeNode("select source points")) {
				buildPointsMenu();
				ImGui::TreePop();
			}

			if (ImGui::Button("run vector transport")) {
				vectorTransport();
			}
			if (ImGui::Button("run wrinkle extraction")) {
				wrinkleExtraction();
			}

			ImGui::EndTabItem();
		}
		ImGui::EndTabBar();
	}
}

int main(int argc, char** argv) {

	// Make sure a mesh name was given
	if (argc != 2) {
		std::cerr << "Please specify a mesh file as argument" << std::endl;
		return EXIT_FAILURE;
	}

	// Initialize polyscope
	polyscope::init();

	// Set the callback function
	polyscope::state::userCallback = myCallback;

	// Load mesh
	std::tie(mesh, geometry) = loadMesh(argv[1]);

	// Register the mesh with polyscope
	psMesh = polyscope::registerSurfaceMesh(polyscope::guessNiceNameFromPath(argv[1]),
		geometry->inputVertexPositions, mesh->getFaceVertexList(),
		polyscopePermutations(*mesh));

	convertPolyScopeMesh(*psMesh, triV, triMesh);


	// Set vertex tangent spaces
	geometry->requireVertexTangentBasis();
	VertexData<Vector3> vBasisX(*mesh);
	for (Vertex v : mesh->vertices()) {
		vBasisX[v] = geometry->vertexTangentBasis[v][0];
	}
	polyscope::getSurfaceMesh()->setVertexTangentBasisX(vBasisX);

	// Set face tangent spaces
	geometry->requireFaceTangentBasis();
	FaceData<Vector3> fBasisX(*mesh);
	for (Face f : mesh->faces()) {
		fBasisX[f] = geometry->faceTangentBasis[f][0];
	}
	polyscope::getSurfaceMesh()->setFaceTangentBasisX(fBasisX);

	// To start, pick two vertices as sources
	geometry->requireVertexIndices();
	addVertexSource(0);
	addVertexSource(mesh->nVertices() / 2);
	sourcePoints[1].scalarVal = 3.0;


	polyscope::view::upDir = polyscope::view::UpDir::ZUp;

	// Give control to the polyscope gui
	polyscope::show();

	return EXIT_SUCCESS;
}
