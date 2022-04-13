#include "geometrycentral/surface/halfedge_mesh.h"
#include "geometrycentral/surface/surface_mesh.h"
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
#include "../../include/IntrinsicFormula/KnoppelStripePatternEdgeOmega.h"
#include "../../include/IntrinsicFormula/InterpolateZvals.h"
#include "../../include/MeshLib/MeshUpsampling.h"
#include "../../include/Visualization/PaintGeometry.h"

#include <igl/doublearea.h>
#include <igl/cotmatrix_entries.h>
#include <igl/per_vertex_normals.h>

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
	std::vector<std::tuple<SurfacePoint, Vector2>> points;
	for (SourceVert& s : sourcePoints) {
		points.emplace_back(s.vertex, Vector2::fromAngle(s.vectorAngleRad) * s.vectorMag);
	}
	VertexData<Vector2> vectorExtension = solver->transportTangentVectors(points);

	int nverts = geoPos.inputVertexPositions.size();

	vertVecs.setZero(nverts, 3);


	for (size_t iV = 0; iV < nverts; iV++) {

		Vector3 normal = geoPos.vertexNormals[iV];
		Vector3 basisX = geoPos.vertexTangentBasis[iV][0];
		Vector3 basisY = geoPos.vertexTangentBasis[iV][1];

		std::complex<double> angle = std::complex<double>(vectorExtension[iV][0], vectorExtension[iV][1]);

		Vector3 vec3 = basisX * (float)angle.real() + basisY * (float)angle.imag();
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
int loopLevel = 2;

// Polyscope visualization handle, to quickly add data to the surface
polyscope::SurfaceMesh* psMesh;

// Some algorithm parameters
float tCoef = 1.0;
int vertexInd = 0;
int pCenter = 2;

std::vector<SourceVert> sourcePoints;

Eigen::MatrixXd cotEntries;
Eigen::VectorXd faceArea;


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
	VertexData<Vector2> sourceVectors(*mesh, Vector2::zero());
	for (SourceVert& s : sourcePoints) 
	{
		sourceVectors[s.vertex] = Vector2::fromAngle(s.vectorAngleRad) * s.vectorMag;
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

	Eigen::VectorXi faceFlags, vertFlags, edgeFlags;
	buildMask(triMesh, triV.rows(), sourcePoints, vertFlags, edgeFlags, faceFlags);

	Eigen::MatrixXd vertVec;
	getVecTransport(*geometry, sourcePoints, vertVec);

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
	getVecTransport(*geometry, sourcePoints, vertVec);

	Eigen::VectorXd edgeVec = vertexVec2IntrinsicVec(vertVec, triV, triMesh);
	/*for (int i = 0; i < triMesh.nEdges(); i++)
		edgeVec(i) *= edgeFlags(i);*/

	Eigen::MatrixXd faceVec = intrinsicEdgeVec2FaceVec(edgeVec, triV, triMesh);
	for (int i = 0; i < triMesh.nFaces(); i++)
		faceVec.row(i) *= faceFlags(i);

	std::vector<std::pair<int, double>> clampedAmps;
	for (int i = 0; i < vertFlags.rows(); i++)
	{
		if (vertFlags(i) == 0)
			clampedAmps.push_back({ i, 0 });
	}

	for (SourceVert& s : sourcePoints)
	{
		int vid = s.vertex.getIndex();
		clampedAmps.push_back({ vid, s.scalarVal });
	}

	Eigen::VectorXd amp;

	ampExtraction(triV, triMesh, edgeVec, clampedAmps, amp);
	//amp.setConstant(sourcePoints[0].scalarVal);

	std::cout << "amp range: " << amp.minCoeff() << " " << amp.maxCoeff() << std::endl;

	auto fVec = psMesh->addFaceVectorQuantity("face vector", faceVec);
	fVec->setEnabled(true);

	auto pScal = psMesh->addVertexScalarQuantity("amp", amp);
	pScal->setEnabled(true);

	/*for (int i = 0; i < triV.rows(); i++)
		if(!vertFlags(i))
			amp(i) = 0;*/

	std::vector<std::complex<double>> zvals, upsampledZvals;
	IntrinsicFormula::roundZvalsFromEdgeOmegaVertexMag(triMesh, edgeVec, amp, faceArea, cotEntries, amp.rows(), zvals);


	Eigen::MatrixXd upsampledTriV, wrinkledV;
	Eigen::MatrixXi upsampledTriF;
	std::vector<std::pair<int, Eigen::Vector3d>> bary;
	
	meshUpSampling(triV, triMesh.faces(), upsampledTriV, upsampledTriF, loopLevel, NULL, NULL, &bary);

	upsampledZvals = IntrinsicFormula::upsamplingZvals(triMesh, zvals, edgeVec, bary);

	Eigen::VectorXd mag(upsampledTriV.rows()), phase(upsampledTriV.rows());
	wrinkledV = upsampledTriV;
	Eigen::MatrixXd vertNormals;
	igl::per_vertex_normals(upsampledTriV, upsampledTriF, vertNormals);

	for (int i = 0; i < upsampledTriV.rows(); i++)
	{
		mag(i) = std::abs(upsampledZvals[i]);
		phase(i) = std::arg(upsampledZvals[i]);
		wrinkledV.row(i) += upsampledZvals[i].real() * vertNormals.row(i);
	}

	PaintGeometry mpaint;
	mpaint.setNormalization(true);
	Eigen::MatrixXd ampColor = mpaint.paintAmplitude(mag);

	std::cout << "upsampled mag range: " << mag.minCoeff() << " " << mag.maxCoeff() << std::endl;

	mpaint.setNormalization(false);
	Eigen::MatrixXd phiColor = mpaint.paintPhi(phase);

	int nupverts = upsampledTriV.rows();
	int nupfaces = upsampledTriF.rows();

	double shiftx = 1.5 * (triV.col(0).maxCoeff() - triV.col(0).minCoeff());

	Eigen::MatrixXd shiftV = Eigen::MatrixXd::Zero(nupverts, 3);
	shiftV.col(0).setConstant(shiftx);
	Eigen::MatrixXi shiftF = upsampledTriF;
	shiftF.setConstant(nupverts);

	Eigen::MatrixXd dataV;
	Eigen::MatrixXi dataF;
	Eigen::MatrixXd dataColor;

	dataV.setZero(nupverts * 3, 3);
	dataColor.setZero(nupverts * 3, 3);
	dataF.setZero(nupfaces * 3, 3);
	
	dataV.block(0, 0, nupverts, 3) = wrinkledV - shiftV;
	dataF.block(0, 0, nupfaces, 3) = upsampledTriF;
	for (int i = 0; i < nupverts; i++)
		dataColor.row(i) << 80 / 255.0, 122 / 255.0, 91 / 255.0;

	dataV.block(nupverts, 0, nupverts, 3) = upsampledTriV - 2 * shiftV;
	dataV.block(2 * nupverts, 0, nupverts, 3) = upsampledTriV - 3 * shiftV;
	dataF.block(nupfaces, 0, nupfaces, 3) = upsampledTriF + shiftF;
	dataF.block(2 * nupfaces, 0, nupfaces, 3) = upsampledTriF + 2 * shiftF;

	dataColor.block(nupverts, 0, nupverts, 3) = phiColor;
	dataColor.block(2 * nupverts, 0, nupverts, 3) = ampColor;

	polyscope::registerSurfaceMesh("wrinkled mesh", dataV, dataF); 
	polyscope::getSurfaceMesh("wrinkled mesh")->addVertexColorQuantity("VertexColor", dataColor);
	polyscope::getSurfaceMesh("wrinkled mesh")->getQuantity("VertexColor")->setEnabled(true);
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
		if (ImGui::InputInt("effective radius", &s.effectiveRadius)) anyChanged = true;
		if (ImGui::SliderAngle("vector angle", &s.vectorAngleRad)) anyChanged = true;

		ImGui::Unindent();
		ImGui::PopID();
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

	ImGuiTabBarFlags tab_bar_flags = ImGuiTabBarFlags_None;
	if (ImGui::BeginTabBar("MyTabBar", tab_bar_flags)) {
		if (ImGui::BeginTabItem("Basic algorithm")) {

			ImGui::TextUnformatted("Algorithm options:");
			ImGui::PushItemWidth(100);
			if (ImGui::InputFloat("tCoef", &tCoef)) {
				solver.reset();
			}
			ImGui::PopItemWidth();
			if (ImGui::InputInt("upsampled level", &loopLevel))
			{
				if (loopLevel < 0)
					loopLevel = 2;
			}
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
	geometry->requireVertexTangentBasis();
	geometry->requireVertexNormals();
	geometry->requireFaceTangentBasis();
	geometry->requireVertexIndices();

	// Register the mesh with polyscope
	psMesh = polyscope::registerSurfaceMesh("base mesh", geometry->inputVertexPositions, mesh->getFaceVertexList(), polyscopePermutations(*mesh));

	convertGeoCentralMesh(*mesh, *geometry, triV, triMesh);


	// Set vertex tangent spaces
	VertexData<Vector3> vBasisX(*mesh);
	for (Vertex v : mesh->vertices()) {
		vBasisX[v] = geometry->vertexTangentBasis[v][0];
	}
	polyscope::getSurfaceMesh("base mesh")->setVertexTangentBasisX(vBasisX);


	
	// Set face tangent spaces
	FaceData<Vector3> fBasisX(*mesh);
	for (Face f : mesh->faces()) {
		fBasisX[f] = geometry->faceTangentBasis[f][0];
	}
	polyscope::getSurfaceMesh("base mesh")->setFaceTangentBasisX(fBasisX);

	// To start, pick two vertices as sources
	geometry->requireVertexIndices();
	addVertexSource(mesh->nVertices() / 2);


	igl::doublearea(triV, triMesh.faces(), faceArea);
	faceArea /= 2;
	igl::cotmatrix_entries(triV, triMesh.faces(), cotEntries);


	polyscope::view::upDir = polyscope::view::UpDir::ZUp;

	// Give control to the polyscope gui
	polyscope::show();

	return EXIT_SUCCESS;
}
