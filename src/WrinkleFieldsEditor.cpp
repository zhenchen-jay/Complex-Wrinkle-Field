#include "../include/WrinkleFieldsEditor.h"
#include <igl/per_vertex_normals.h>

void WrinkleFieldsEditor::editWrinkles(const Eigen::MatrixXd &pos, const MeshConnectivity &mesh,
									   const Eigen::VectorXd &amp, const Eigen::MatrixXd &omega,
									   const std::vector<VertexOpInfo> &vertInfo, Eigen::VectorXd &ampNew,
									   Eigen::MatrixXd &omegaNew)
{
	ampNew = amp;
	omegaNew = omega;

	int nverts = pos.rows();
	Eigen::MatrixXd normals;
	igl::per_vertex_normals(pos, mesh.faces(), normals);

	//auto vertEditor = [&](const tbb::blocked_range<uint32_t>& range) {
		//for (uint32_t i = range.begin(); i < range.end(); ++i)
	for (uint32_t i = 0; i < nverts; ++i)
		{
			Eigen::Vector3d omegaVert;
			editWrinklesPerVertex(pos, mesh, normals, amp, omega, vertInfo, i, ampNew(i), omegaVert);
			omegaNew.row(i) = omegaVert;
		}
	//};

	//tbb::blocked_range<uint32_t> rangex(0u, (uint32_t)(nverts), GRAIN_SIZE);
	//tbb::parallel_for(rangex, vertEditor);
}

void WrinkleFieldsEditor::editWrinklesPerVertex(const Eigen::MatrixXd &pos, const MeshConnectivity &mesh,
												const Eigen::MatrixXd &vertNormals, const Eigen::VectorXd &amp,
												const Eigen::MatrixXd &omega, const std::vector<VertexOpInfo> &vertInfo,
												int vid, double &ampNew, Eigen::Vector3d &omegaNew)
{
	ampNew = amp(vid);
	omegaNew = omega.row(vid);
	Eigen::Vector3d axis = vertNormals.row(vid);
	if(vertInfo[vid].vecOptType == Rotate)
	{
		// first normalize axis
		double ux = axis(0) / axis.norm(), uy = axis(1) / axis.norm(), uz = axis(2) / axis.norm();
		Eigen::Matrix3d rotMat;

		double c = std::cos(vertInfo[vid].vecOptValue);
		double s = std::sin(vertInfo[vid].vecOptValue);
		rotMat << c + ux * ux * (1 - c), ux* uy* (1 - c) - uz * s, ux* uz* (1 - c) + uy * s,
				uy* ux* (1 - c) + uz * s, c + uy * uy * (1 - c), uy* uz* (1 - c) - ux * s,
				uz* ux* (1 - c) - uy * s, uz* uy* (1 - c) + ux * s, c + uz * uz * (1 - c);

		omegaNew = rotMat * omegaNew;
	}
	else if (vertInfo[vid].vecOptType == Enlarge)
	{
		omegaNew *= vertInfo[vid].vecOptValue;
		ampNew /= vertInfo[vid].vecOptValue;
	}
	else if (vertInfo[vid].vecOptType == Tilt)
	{
		double ux = axis(0) / axis.norm(), uy = axis(1) / axis.norm(), uz = axis(2) / axis.norm();
		Eigen::Matrix3d rotMat;

		double c = std::cos(vertInfo[vid].vecOptValue);
		double s = std::sin(vertInfo[vid].vecOptValue);
		rotMat << c + ux * ux * (1 - c), ux* uy* (1 - c) - uz * s, ux* uz* (1 - c) + uy * s,
				uy* ux* (1 - c) + uz * s, c + uy * uy * (1 - c), uy* uz* (1 - c) - ux * s,
				uz* ux* (1 - c) - uy * s, uz* uy* (1 - c) + ux * s, c + uz * uz * (1 - c);

		omegaNew = rotMat * omegaNew;
		omegaNew /= omegaNew.norm();
		omegaNew *= omega.row(vid).norm() / c;
		ampNew *= c;
	}


}

void WrinkleFieldsEditor::edgeBasedWrinkleEdition(const Eigen::MatrixXd& pos, const MeshConnectivity& mesh, const Eigen::VectorXd& amp, const Eigen::VectorXd& omega, const std::vector<VertexOpInfo>& vertInfo, Eigen::VectorXd& ampNew, Eigen::VectorXd& omegaNew)
{
	int nverts = pos.rows();
	int nfaces = mesh.nFaces();
	int nedges = mesh.nEdges();

	std::vector<std::vector<int>> vertNeiFaces, vertNeiEdges;
	buildVertexNeighboringInfo(mesh, nverts, vertNeiEdges, vertNeiFaces);

	ampNew = amp;
	omegaNew = omega;


	Eigen::VectorXi edgeFlags;
	edgeFlags.setZero(nedges);

	Eigen::VectorXi vertexFlags;
	vertexFlags.setZero(nverts);

	Eigen::MatrixXd faceNormals;
	igl::per_face_normals(pos, mesh.faces(), faceNormals);

	for (int i = 0; i < nfaces; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			int vid = mesh.faceVertex(i, j);

			if (!vertInfo[vid].isMagOptCoupled && !vertexFlags(vid))
				ampNew(vid) = vertInfo[vid].vecMagValue * amp(vid);

			if (vertInfo[vid].vecOptType == None)
				continue;

			int eid0 = mesh.faceEdge(i, (j + 1) % 3);
			int eid1 = mesh.faceEdge(i, (j + 2) % 3);

			Eigen::Vector3d e0 = pos.row(mesh.faceVertex(i, (j + 2) % 3)) - pos.row(vid);
			Eigen::Vector3d e1 = pos.row(mesh.faceVertex(i, (j + 1) % 3)) - pos.row(vid);

			int flag0 = 1, flag1 = 1;
			Eigen::Vector2d rhs;

			if (mesh.edgeVertex(eid0, 0) == vid)
			{
				flag0 = 1;
			}
			else
			{
				flag0 = -1;
			}


			if (mesh.edgeVertex(eid1, 0) == vid)
			{
				flag1 = 1;
			}
			else
			{
				flag1 = -1;
			}
			rhs(0) = flag0 * omega(eid0);
			rhs(1) = flag1 * omega(eid1);

			Eigen::Matrix2d I;
			I << e0.dot(e0), e0.dot(e1), e1.dot(e0), e1.dot(e1);
			Eigen::Vector2d sol = I.inverse() * rhs;

			Eigen::Vector3d w = sol(0) * e0 + sol(1) * e1;

			Eigen::Vector3d axis = faceNormals.row(i);
			if (vertInfo[vid].vecOptType == Rotate)
			{
				// first normalize axis
				double ux = axis(0) / axis.norm(), uy = axis(1) / axis.norm(), uz = axis(2) / axis.norm();
				Eigen::Matrix3d rotMat;

				double c = std::cos(vertInfo[vid].vecOptValue);
				double s = std::sin(vertInfo[vid].vecOptValue);
				rotMat << c + ux * ux * (1 - c), ux* uy* (1 - c) - uz * s, ux* uz* (1 - c) + uy * s,
					uy* ux* (1 - c) + uz * s, c + uy * uy * (1 - c), uy* uz* (1 - c) - ux * s,
					uz* ux* (1 - c) - uy * s, uz* uy* (1 - c) + ux * s, c + uz * uz * (1 - c);

				w = rotMat * w;
			}
			else if (vertInfo[vid].vecOptType == Enlarge)
			{
				w *= vertInfo[vid].vecOptValue;
				if (vertInfo[vid].isMagOptCoupled && !vertexFlags(vid))
					ampNew(vid) /= vertInfo[vid].vecOptValue;
			}
			else if (vertInfo[vid].vecOptType == Tilt)
			{
				double ux = axis(0) / axis.norm(), uy = axis(1) / axis.norm(), uz = axis(2) / axis.norm();
				Eigen::Matrix3d rotMat;

				double c = std::cos(vertInfo[vid].vecOptValue);
				double s = std::sin(vertInfo[vid].vecOptValue);
				rotMat << c + ux * ux * (1 - c), ux* uy* (1 - c) - uz * s, ux* uz* (1 - c) + uy * s,
					uy* ux* (1 - c) + uz * s, c + uy * uy * (1 - c), uy* uz* (1 - c) - ux * s,
					uz* ux* (1 - c) - uy * s, uz* uy* (1 - c) + ux * s, c + uz * uz * (1 - c);

				double wn = w.norm();
				w = rotMat * w;
				w /= wn;
				w *= wn / c;

				if (vertInfo[vid].isMagOptCoupled && !vertexFlags(vid))
					ampNew(vid) *= c;
			}

			if (edgeFlags(eid0))
			{
				omegaNew(eid0) += flag0 * w.dot(e0);
				omegaNew(eid0) /= 2;
			}
			else
			{
				omegaNew(eid0) = flag0 * w.dot(e0);
			}
			
			if (edgeFlags(eid1))
			{
				omegaNew(eid1) += flag1 * w.dot(e1);
				omegaNew(eid1) /= 2;
			}
			else
			{
				omegaNew(eid1) = flag1 * w.dot(e1);
			}


			edgeFlags(eid0)++;
			edgeFlags(eid1)++;

			vertexFlags(vid) = 1;
		}
	}
}

void WrinkleFieldsEditor::halfEdgeBasedWrinkleEdition(const Eigen::MatrixXd& pos, const MeshConnectivity& mesh, const Eigen::VectorXd& amp, const Eigen::MatrixXd& omega, const std::vector<VertexOpInfo>& vertInfo, Eigen::VectorXd& ampNew, Eigen::MatrixXd& omegaNew)
{
	int nverts = pos.rows();
	int nfaces = mesh.nFaces();
	int nedges = mesh.nEdges();

	std::vector<std::vector<int>> vertNeiFaces, vertNeiEdges;
	buildVertexNeighboringInfo(mesh, nverts, vertNeiEdges, vertNeiFaces);

	ampNew = amp;
	omegaNew = omega;

	Eigen::VectorXi vertexFlags;
	vertexFlags.setZero(nverts);

	Eigen::MatrixXd faceNormals;
	igl::per_face_normals(pos, mesh.faces(), faceNormals);

	Eigen::MatrixXi visitedTimes(nedges, 2);
	visitedTimes.setZero();

	for (int i = 0; i < nfaces; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			int vid = mesh.faceVertex(i, j);

			if (!vertInfo[vid].isMagOptCoupled && !vertexFlags(vid))
				ampNew(vid) = vertInfo[vid].vecMagValue * amp(vid);

			if (vertInfo[vid].vecOptValue == None)
				continue;

			int eid0 = mesh.faceEdge(i, (j + 1) % 3);
			int eid1 = mesh.faceEdge(i, (j + 2) % 3);

			Eigen::Vector3d e0 = pos.row(mesh.faceVertex(i, (j + 2) % 3)) - pos.row(vid);
			Eigen::Vector3d e1 = pos.row(mesh.faceVertex(i, (j + 1) % 3)) - pos.row(vid);

			int flag0 = 0, flag1 = 0;
			Eigen::Vector2d rhs;

			if (mesh.edgeVertex(eid0, 0) == vid)
			{
				flag0 = 0;
				rhs(0) = omega(eid0, 0);
			}
			else
			{
				flag0 = 1;
				rhs(0) = omega(eid0, 1);
			}
				

			if (mesh.edgeVertex(eid1, 0) == vid)
			{
				flag1 = 0;
				rhs(1) = omega(eid1, 0);
			}
			else
			{
				flag1 = 1;
				rhs(1) = omega(eid1, 1);
			}
				
			Eigen::Matrix2d I;
			I << e0.dot(e0), e0.dot(e1), e1.dot(e0), e1.dot(e1);
			Eigen::Vector2d sol = I.inverse() * rhs;

			Eigen::Vector3d w = sol(0) * e0 + sol(1) * e1;

			Eigen::Vector3d axis = faceNormals.row(i);
			if (vertInfo[vid].vecOptType == Rotate)
			{
				double angle = vertInfo[vid].vecOptValue / 180.0 * M_PI;
				// first normalize axis
				double ux = axis(0) / axis.norm(), uy = axis(1) / axis.norm(), uz = axis(2) / axis.norm();
				Eigen::Matrix3d rotMat;

				double c = std::cos(angle);
				double s = std::sin(angle);
				rotMat << c + ux * ux * (1 - c), ux* uy* (1 - c) - uz * s, ux* uz* (1 - c) + uy * s,
					uy* ux* (1 - c) + uz * s, c + uy * uy * (1 - c), uy* uz* (1 - c) - ux * s,
					uz* ux* (1 - c) - uy * s, uz* uy* (1 - c) + ux * s, c + uz * uz * (1 - c);

				w = rotMat * w;
			}
			else if (vertInfo[vid].vecOptType == Enlarge)
			{
				w *= vertInfo[vid].vecOptValue;
				if(vertInfo[vid].isMagOptCoupled && !vertexFlags(vid))
					ampNew(vid) /= vertInfo[vid].vecOptValue;
			}
			else if (vertInfo[vid].vecOptType == Tilt)
			{
				double angle = vertInfo[vid].vecOptValue / 180.0 * M_PI;
				// first normalize axis
				double ux = axis(0) / axis.norm(), uy = axis(1) / axis.norm(), uz = axis(2) / axis.norm();
				Eigen::Matrix3d rotMat;

				double c = std::cos(angle);
				double s = std::sin(angle);
				rotMat << c + ux * ux * (1 - c), ux* uy* (1 - c) - uz * s, ux* uz* (1 - c) + uy * s,
					uy* ux* (1 - c) + uz * s, c + uy * uy * (1 - c), uy* uz* (1 - c) - ux * s,
					uz* ux* (1 - c) - uy * s, uz* uy* (1 - c) + ux * s, c + uz * uz * (1 - c);

				double wn = w.norm();
				w = rotMat * w;
				w /= wn;
				w *= wn / c;

				if (vertInfo[vid].isMagOptCoupled && !vertexFlags(vid))
					ampNew(vid) *= c;
			}

			
			if (visitedTimes(eid0, flag0))
			{
				omegaNew(eid0, flag0) += w.dot(e0);
			}
				
			else
				omegaNew(eid0, flag0) = w.dot(e0);
			
			if (visitedTimes(eid1, flag1))
			{
				omegaNew(eid1, flag1) += w.dot(e1);
			}
				
			else
				omegaNew(eid1, flag1) = w.dot(e1);
				
			
			visitedTimes(eid0, flag0)++;
			visitedTimes(eid1, flag1)++;

			vertexFlags(vid) = 1;
		}
	}

	for (int i = 0; i < nedges; i++)
	{
		if(visitedTimes(i, 0))
			omegaNew(i, 0) /= visitedTimes(i, 0);
		if(visitedTimes(i, 1))
			omegaNew(i, 1) /= visitedTimes(i, 1);
			
	}
}