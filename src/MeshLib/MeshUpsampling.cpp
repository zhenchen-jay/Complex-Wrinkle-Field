#include "../../include/MeshLib/MeshUpsampling.h"
#include <igl/adjacency_list.h>
#include <igl/triangle_triangle_adjacency.h>
#include <igl/unique.h>
#include <igl/per_vertex_normals.h>
#include <igl/writeOBJ.h>
#include <igl/boundary_loop.h>
#include <igl/cotmatrix.h>

static void midPoint(const int n_verts, const Eigen::MatrixXi& F, Eigen::SparseMatrix<double>& S, Eigen::MatrixXi& NF, std::vector<int>* faceTracing)
{

	typedef Eigen::SparseMatrix<double> SparseMat;
	typedef Eigen::Triplet<double> Triplet_t;

	//Ref. igl::loop
	Eigen::MatrixXi FF, FFi;
	igl::triangle_triangle_adjacency(F, FF, FFi);
	std::vector<std::vector<typename Eigen::MatrixXi::Scalar>> adjacencyList;
	igl::adjacency_list(F, adjacencyList, true);
	//Compute the number and positions of the vertices to insert (on edges)
	Eigen::MatrixXi NI = Eigen::MatrixXi::Constant(FF.rows(), FF.cols(), -1);
	Eigen::MatrixXi NIdoubles = Eigen::MatrixXi::Zero(FF.rows(), FF.cols());
	Eigen::VectorXi vertIsOnBdry = Eigen::VectorXi::Zero(n_verts);
	int counter = 0;
	for (int i = 0; i < FF.rows(); ++i)
	{
		for (int j = 0; j < 3; ++j)
		{
			if (NI(i, j) == -1)
			{
				NI(i, j) = counter;
				NIdoubles(i, j) = 0;
				if (FF(i, j) != -1)
				{
					//If it is not a boundary
					NI(FF(i, j), FFi(i, j)) = counter;
					NIdoubles(i, j) = 1;
				}
				else
				{
					//Mark boundary vertices for later
					vertIsOnBdry(F(i, j)) = 1;
					vertIsOnBdry(F(i, (j + 1) % 3)) = 1;
				}
				++counter;
			}
		}
	}

	const int& n_odd = n_verts;
	const int& n_even = counter;
	const int n_newverts = n_odd + n_even;

	//Construct vertex positions
	std::vector<Triplet_t> tripletList;
	for (int i = 0; i < n_odd; ++i)
	{
		//Old vertices
		tripletList.emplace_back(i, i, 1.);
	}
	for (int i = 0; i < FF.rows(); ++i)
	{
		//New vertices
		for (int j = 0; j < 3; ++j)
		{
			if (NIdoubles(i, j) == 0)
			{
				if (FF(i, j) == -1)
				{
					//Boundary vertex
					tripletList.emplace_back(NI(i, j) + n_odd, F(i, j), 1. / 2.);
					tripletList.emplace_back(NI(i, j) + n_odd, F(i, (j + 1) % 3), 1. / 2.);
				}
				else
				{
					//                    tripletList.emplace_back(NI(i,j) + n_odd, F(i,j), 1./4.);
					//                    tripletList.emplace_back(NI(i,j) + n_odd, F(i, (j+1)%3), 1./4.);
					//                    tripletList.emplace_back(NI(i,j) + n_odd, F(i, (j+2)%3), 1./4.);
					//                    tripletList.emplace_back(NI(i,j) + n_odd, F(FF(i,j), (FFi(i,j)+2)%3), 1./4.);
					tripletList.emplace_back(NI(i, j) + n_odd, F(i, j), 1. / 2.);
					tripletList.emplace_back(NI(i, j) + n_odd, F(i, (j + 1) % 3), 1. / 2.);
				}
			}
		}
	}
	S.resize(n_newverts, n_verts);
	S.setFromTriplets(tripletList.begin(), tripletList.end());

	// Build the new topology (Every face is replaced by four)
	if (faceTracing)
		faceTracing->resize(F.rows() * 4);
	NF.resize(F.rows() * 4, 3);
	for (int i = 0; i < F.rows(); ++i)
	{
		Eigen::VectorXi VI(6);
		VI << F(i, 0), F(i, 1), F(i, 2), NI(i, 0) + n_odd, NI(i, 1) + n_odd, NI(i, 2) + n_odd;

		Eigen::VectorXi f0(3), f1(3), f2(3), f3(3);
		f0 << VI(0), VI(3), VI(5);
		f1 << VI(1), VI(4), VI(3);
		f2 << VI(3), VI(4), VI(5);
		f3 << VI(4), VI(2), VI(5);

		NF.row((i * 4) + 0) = f0;
		NF.row((i * 4) + 1) = f1;
		NF.row((i * 4) + 2) = f2;
		NF.row((i * 4) + 3) = f3;

		if (faceTracing)
			for (int j = 0; j < 4; j++)
				(*faceTracing)[4 * i + j] = i;
	}
}


void meshUpSampling(Eigen::MatrixXd V, Eigen::MatrixXi F, Eigen::MatrixXd &NV, Eigen::MatrixXi &NF, int numSubdivs, Eigen::SparseMatrix<double> *mat, std::vector<int>* facemap)
{
	NV = V;
	NF = F;
	// midpoint subdivision
	if (mat)
	{
		mat->resize(V.rows(), V.rows());
		mat->setIdentity();
	}

	std::vector<int> tmpfacemap;
	if (facemap)
	{
		facemap->resize(F.rows());
		for (int i = 0; i < F.rows(); i++)
			(*facemap)[i] = i;
		tmpfacemap = *(facemap);
	}

	for(int i=0; i<numSubdivs; ++i)
	{
		Eigen::MatrixXi tempF = NF;
		Eigen::SparseMatrix<double> S;
		std::vector<int> faceTracing;
		midPoint(NV.rows(), tempF, S, NF, facemap ? &faceTracing : NULL);
		// This .eval is super important
		NV = (S*NV).eval();

		if (mat)
		{
			(*mat) = S * (*mat);
		}

		if (facemap)
		{
			std::vector<int> tmpfacemap1;
			tmpfacemap1.resize(NF.rows());
			for (int j = 0; j < NF.rows(); j++)
			{
				tmpfacemap1[j] = tmpfacemap[faceTracing[j]];
			}
			std::swap(tmpfacemap, tmpfacemap1);
		}
	}
	if(facemap)
		(*facemap) = tmpfacemap;
}