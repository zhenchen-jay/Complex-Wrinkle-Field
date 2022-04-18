#include "../../include/MeshLib/MeshUpsampling.h"

#include <igl/adjacency_list.h>
#include <igl/triangle_triangle_adjacency.h>
#include <igl/writeOBJ.h>
#include <igl/boundary_loop.h>
#include <deque>
#include <vector>

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
		f2 << VI(2), VI(5), VI(4);
		f3 << VI(3), VI(4), VI(5);

		NF.row((i * 4) + 0) = f0;
		NF.row((i * 4) + 1) = f1;
		NF.row((i * 4) + 2) = f2;
		NF.row((i * 4) + 3) = f3;

		if (faceTracing)
			for (int j = 0; j < 4; j++)
				(*faceTracing)[4 * i + j] = i;
	}
}


void meshUpSampling(Eigen::MatrixXd V, Eigen::MatrixXi F, Eigen::MatrixXd &NV, Eigen::MatrixXi &NF, int numSubdivs, Eigen::SparseMatrix<double> *mat, std::vector<int>* facemap, std::vector<std::pair<int, Eigen::Vector3d>> *bary)
{
	NV = V;
	NF = F;
	// midpoint subdivision
	std::vector<int> tmpfacemap;
	Eigen::SparseMatrix<double> tmpMat(V.rows(), V.rows());
	tmpMat.setIdentity();

	tmpfacemap.resize(F.rows());
	for (int i = 0; i < F.rows(); i++)
	    tmpfacemap[i] = i;


	for(int i=0; i<numSubdivs; ++i)
	{
		Eigen::MatrixXi tempF = NF;
		Eigen::SparseMatrix<double> S;
		std::vector<int> faceTracing;
		midPoint(NV.rows(), tempF, S, NF, &faceTracing);
		// This .eval is super important
		NV = (S*NV).eval();

		tmpMat = S * tmpMat;

		std::vector<int> tmpfacemap1;
		tmpfacemap1.resize(NF.rows());
		for (int j = 0; j < NF.rows(); j++)
		{
		    tmpfacemap1[j] = tmpfacemap[faceTracing[j]];
		}
		std::swap(tmpfacemap, tmpfacemap1);
	}
	if(facemap)
		(*facemap) = tmpfacemap;

	if(mat)
	    (*mat) = tmpMat;

	if(bary)
	{
	    bary->resize(NV.rows());
	    Eigen::VectorXi isVisited = Eigen::VectorXi::Zero(NV.rows());

	    std::vector<std::vector<std::pair<int, double>>> nonzeroTracing(NV.rows());

	    for (int k=0; k < tmpMat.outerSize(); ++k)
	    {
	        for (Eigen::SparseMatrix<double>::InnerIterator it(tmpMat,k); it; ++it)
	        {
	           nonzeroTracing[it.row()].push_back({it.col(), it.value()});
	        }
	    }

	    for(int i = 0; i < NF.rows(); i++)
	    {
	        for(int j = 0; j < 3; j++)
	        {
	            int vid = NF(i, j);
	            if(isVisited(vid))
                    continue;
	            std::vector<std::pair<int, double>> perFaceBary = nonzeroTracing[vid];
	            if(perFaceBary.size() >3)
	            {
	                std::cerr << "some error in the upsampling matrix." << std::endl;
	                exit(1);
	            }
	            int preFace = tmpfacemap[i];
	            Eigen::Vector3d ptBary = Eigen::Vector3d::Zero();

	            for(int k = 0; k < perFaceBary.size(); k++)
	            {
	                int preVid = perFaceBary[k].first;
	                for(int n = 0; n < 3; n++)
	                {
	                    if(F(preFace, n) == preVid)
	                    {
	                        ptBary(n) = perFaceBary[k].second;
	                    }
	                }
	            }
	            bary->at(vid) = std::pair<int, Eigen::Vector3d>(preFace, ptBary);

	            isVisited(vid) = 1;
	        }
	    }
	}
}


static int labelComponents(const Eigen::MatrixXi& F, Eigen::VectorXi& labels)
{
	MeshConnectivity mesh(F);
	int nfaces = F.rows();
	labels.resize(nfaces);
	std::vector<bool> visited(nfaces);
	int label = 0;
	for (int i = 0; i < nfaces; i++)
	{
		if (visited[i])
			continue;
		std::deque<int> tovisit;
		tovisit.push_back(i);
		while (!tovisit.empty())
		{
			int next = tovisit.front();
			tovisit.pop_front();
			if (visited[next])
				continue;
			visited[next] = true;
			labels[next] = label;
			for (int j = 0; j < 3; j++)
			{
				int e = mesh.faceEdge(next, j);
				int o = mesh.faceEdgeOrientation(next, j);
				int opp = mesh.edgeFace(e, 1 - o);
				if (opp != -1 && !visited[opp])
					tovisit.push_back(opp);
			}
		}
		label++;
	}

	return label;
}

static void findCorners(const Eigen::MatrixXd& V2D, const Eigen::MatrixXi& F2D, const Eigen::MatrixXd& V3D, const Eigen::MatrixXi& F3D, std::set<int>& corners)
{
	corners.clear();

	Eigen::VectorXi labels;
	int components = labelComponents(F2D, labels);
	std::cout << "2D mesh has " << components << " components." << std::endl;

	int nfaces = F3D.rows();
	int n2Dverts = V2D.rows();
	int n3Dverts = V3D.rows();
	MeshConnectivity mesh2D(F2D);
	MeshConnectivity mesh3D(F3D);

	std::vector<std::set<int> > vertlabels(n3Dverts);

	for (int i = 0; i < nfaces; i++)
	{
		int l = labels[i];
		for (int j = 0; j < 3; j++)
		{
			int v = mesh3D.faceVertex(i, j);
			vertlabels[v].insert(l);
		}
	}
	std::vector<std::vector<int> > L;
	igl::boundary_loop(F3D, L);

	for (auto& loop : L)
	{
		for (auto it : loop)
		{
			vertlabels[it].insert(-1);
		}
	}

	for (int i = 0; i < nfaces; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			int v2d = mesh2D.faceVertex(i, j);
			int v3d = mesh3D.faceVertex(i, j);
			if (vertlabels[v3d].size() > 2)
				corners.insert(v2d);
		}
	}
}


static void loopWithCorners(
		const int n_verts,
		const Eigen::MatrixXi& F,
		std::set<int> corners,
		Eigen::SparseMatrix<double>& S,
		Eigen::MatrixXi& NF)
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
		const std::vector<int>& localAdjList = adjacencyList[i];
		if (corners.count(i))
		{
			tripletList.emplace_back(i, i, 1.0);
		}
		else if (vertIsOnBdry(i) == 1)
		{
			//Boundary vertex
			tripletList.emplace_back(i, localAdjList.front(), 1. / 8.);
			tripletList.emplace_back(i, localAdjList.back(), 1. / 8.);
			tripletList.emplace_back(i, i, 3. / 4.);
		}
		else
		{
			const int n = localAdjList.size();
			const double dn = n;
			double beta;
			if (n == 3)
			{
				beta = 3. / 16.;
			}
			else
			{
				beta = 3. / 8. / dn;
				//double c0 = (3. / 8. + 1. / 4. * std::cos(2 * M_PI / dn));
				//beta = 1. / dn * (5. / 8. - c0 * c0);
			}
			for (int j = 0; j < n; ++j)
			{
				tripletList.emplace_back(i, localAdjList[j], beta);
			}
			tripletList.emplace_back(i, i, 1. - dn * beta);
		}
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
					tripletList.emplace_back(NI(i, j) + n_odd, F(i, j), 3. / 8.);
					tripletList.emplace_back(NI(i, j) + n_odd, F(i, (j + 1) % 3), 3. / 8.);
					tripletList.emplace_back(NI(i, j) + n_odd, F(i, (j + 2) % 3), 1. / 8.);
					tripletList.emplace_back(NI(i, j) + n_odd, F(FF(i, j), (FFi(i, j) + 2) % 3), 1. / 8.);
				}
			}
		}
	}
	S.resize(n_newverts, n_verts);
	S.setFromTriplets(tripletList.begin(), tripletList.end());

	// Build the new topology (Every face is replaced by four)
	NF.resize(F.rows() * 4, 3);
	for (int i = 0; i < F.rows(); ++i)
	{
		Eigen::VectorXi VI(6);
		VI << F(i, 0), F(i, 1), F(i, 2), NI(i, 0) + n_odd, NI(i, 1) + n_odd, NI(i, 2) + n_odd;

		Eigen::VectorXi f0(3), f1(3), f2(3), f3(3);
		f0 << VI(0), VI(3), VI(5);
		f1 << VI(1), VI(4), VI(3);
		f2 << VI(2), VI(5), VI(4);
		f3 << VI(3), VI(4), VI(5);

		NF.row((i * 4) + 0) = f0;
		NF.row((i * 4) + 1) = f1;
		NF.row((i * 4) + 2) = f2;
		NF.row((i * 4) + 3) = f3;
	}
}

 void loopUpsampling(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, Eigen::MatrixXd& NV, Eigen::MatrixXi& NF, int upsampledTimes, Eigen::SparseMatrix<double>* loopMat)
{
	NV = V;
	NF = F;

	std::set<int> fixedVids;
	/*std::vector<int> bnds;
	igl::boundary_loop(F, bnds);
	for (auto& it : bnds)
	{
		fixedVids.insert(it);
	}*/
	if (loopMat)
	{
		loopMat->resize(V.rows(), V.rows());
		loopMat->setIdentity();
	}


	for (int i = 0; i < upsampledTimes; i++)
	{
		Eigen::SparseMatrix<double> S;
		Eigen::MatrixXd tmpV = NV;
		Eigen::MatrixXi tmpF = NF;
		loopWithCorners(NV.rows(), tmpF, fixedVids, S, NF);
		
		NV = (S * NV).eval();
		if (loopMat)
		{
			*loopMat = S * (*loopMat);
		}
	}

}


