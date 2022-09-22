#include "../../include/OtherApproaches/TFWAlgorithm.h"
#include "../../include/MeshLib/MeshConnectivity.h"
#include "../../include/MeshLib/MeshGeometry.h"
#include <map>
#include <deque>
#include <iostream>
#include <CoMISo/Solver/ConstrainedSolver.hh>
#include <igl/per_vertex_normals.h>
#include <igl/principal_curvature.h>
#include <igl/boundary_loop.h>
#include <tbb/tbb.h>

namespace TFWAlg
{
    void firstFoundForms(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F, std::vector<Eigen::Matrix2d> &abars)
    {
        int nfaces = F.rows();
        for(int i = 0; i < nfaces; i++)
        {
            Eigen::Matrix2d I;
            Eigen::Matrix<double, 2, 3> dr;
            dr.row(0) = V.row(F(i, 1)) - V.row(F(i, 0));
            dr.row(1) = V.row(F(i, 2)) - V.row(F(i, 0));

            I = dr * dr.transpose();
            abars.push_back(I);
        }
    }

    void findCuts(const Eigen::MatrixXi &F, std::vector<std::vector<int> > &cuts)
    {
        cuts.clear();

        int nfaces = F.rows();

        if (nfaces == 0)
            return;

        std::map<std::pair<int, int>, std::vector<int> > edges;
        // build edges

        for (int i = 0; i < nfaces; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                int v0 = F(i, j);
                int v1 = F(i, (j + 1) % 3);
                std::pair<int, int> e;
                e.first = std::min(v0, v1);
                e.second = std::max(v0, v1);
                edges[e].push_back(i);
            }
        }

        int nedges = edges.size();
        Eigen::MatrixXi edgeVerts(nedges,2);
        Eigen::MatrixXi edgeFaces(nedges,2);
        Eigen::MatrixXi faceEdges(nfaces, 3);
        std::set<int> boundaryEdges;
        std::map<std::pair<int, int>, int> edgeidx;
        int idx = 0;
        for (auto it : edges)
        {
            edgeidx[it.first] = idx;
            edgeVerts(idx, 0) = it.first.first;
            edgeVerts(idx, 1) = it.first.second;
            edgeFaces(idx, 0) = it.second[0];
            if (it.second.size() > 1)
            {
                edgeFaces(idx, 1) = it.second[1];
            }
            else
            {
                edgeFaces(idx, 1) = -1;
                boundaryEdges.insert(idx);
            }
            idx++;
        }
        for (int i = 0; i < nfaces; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                int v0 = F(i, j);
                int v1 = F(i, (j + 1) % 3);
                std::pair<int, int> e;
                e.first = std::min(v0, v1);
                e.second = std::max(v0, v1);
                faceEdges(i, j) = edgeidx[e];
            }
        }

        bool *deleted = new bool[nfaces];
        for (int i = 0; i < nfaces; i++)
            deleted[i] = false;

        std::set<int> deletededges;

        // loop over faces
        for (int face = 0; face < nfaces; face++)
        {
            // stop at first undeleted face
            if (deleted[face])
                continue;
            deleted[face] = true;
            std::deque<int> processEdges;
            for (int i = 0; i < 3; i++)
            {
                int e = faceEdges(face, i);
                if (boundaryEdges.count(e))
                    continue;
                int ndeleted = 0;
                if (deleted[edgeFaces(e, 0)])
                    ndeleted++;
                if (deleted[edgeFaces(e, 1)])
                    ndeleted++;
                if (ndeleted == 1)
                    processEdges.push_back(e);
            }
            // delete all faces adjacent to edges with exactly one adjacent face
            while (!processEdges.empty())
            {
                int nexte = processEdges.front();
                processEdges.pop_front();
                int todelete = -1;
                if (!deleted[edgeFaces(nexte, 0)])
                    todelete = edgeFaces(nexte, 0);
                if (!deleted[edgeFaces(nexte, 1)])
                    todelete = edgeFaces(nexte, 1);
                if (todelete != -1)
                {
                    deletededges.insert(nexte);
                    deleted[todelete] = true;
                    for (int i = 0; i < 3; i++)
                    {
                        int e = faceEdges(todelete, i);
                        if (boundaryEdges.count(e))
                            continue;
                        int ndeleted = 0;
                        if (deleted[edgeFaces(e, 0)])
                            ndeleted++;
                        if (deleted[edgeFaces(e, 1)])
                            ndeleted++;
                        if (ndeleted == 1)
                            processEdges.push_back(e);
                    }
                }
            }
        }
        delete[] deleted;

        // accumulated non-deleted edges
        std::vector<int> leftedges;
        for (int i = 0; i < nedges; i++)
        {
            if (!deletededges.count(i))
                leftedges.push_back(i);
        }

        deletededges.clear();
        // prune spines
        std::map<int, std::vector<int> > spinevertedges;
        for (int i : leftedges)
        {
            spinevertedges[edgeVerts(i, 0)].push_back(i);
            spinevertedges[edgeVerts(i, 1)].push_back(i);
        }

        std::deque<int> vertsProcess;
        std::map<int, int> spinevertnbs;
        for (auto it : spinevertedges)
        {
            spinevertnbs[it.first] = it.second.size();
            if (it.second.size() == 1)
                vertsProcess.push_back(it.first);
        }
        while (!vertsProcess.empty())
        {
            int vert = vertsProcess.front();
            vertsProcess.pop_front();
            for (int e : spinevertedges[vert])
            {
                if (!deletededges.count(e))
                {
                    deletededges.insert(e);
                    for (int j = 0; j < 2; j++)
                    {
                        spinevertnbs[edgeVerts(e, j)]--;
                        if (spinevertnbs[edgeVerts(e, j)] == 1)
                        {
                            vertsProcess.push_back(edgeVerts(e, j));
                        }
                    }
                }
            }
        }
        std::vector<int> loopedges;
        for (int i : leftedges)
            if (!deletededges.count(i))
                loopedges.push_back(i);

        int nloopedges = loopedges.size();
        if (nloopedges == 0)
            return;

        std::map<int, std::vector<int> > loopvertedges;
        for (int e : loopedges)
        {
            loopvertedges[edgeVerts(e, 0)].push_back(e);
            loopvertedges[edgeVerts(e, 1)].push_back(e);
        }

        std::set<int> usededges;
        for (int e : loopedges)
        {
            // make a cycle or chain starting from this edge
            while (!usededges.count(e))
            {
                std::vector<int> cycleverts;
                std::vector<int> cycleedges;
                cycleverts.push_back(edgeVerts(e, 0));
                cycleverts.push_back(edgeVerts(e, 1));
                cycleedges.push_back(e);

                std::map<int, int> cycleidx;
                cycleidx[cycleverts[0]] = 0;
                cycleidx[cycleverts[1]] = 1;

                int curvert = edgeVerts(e, 1);
                int cure = e;
                bool foundcycle = false;
                while (curvert != -1 && !foundcycle)
                {
                    int nextvert = -1;
                    int nexte = -1;
                    for (int cande : loopvertedges[curvert])
                    {
                        if (!usededges.count(cande) && cande != cure)
                        {
                            int vidx = 0;
                            if (curvert == edgeVerts(cande, vidx))
                                vidx = 1;
                            nextvert = edgeVerts(cande, vidx);
                            nexte = cande;
                            break;
                        }
                    }
                    if (nextvert != -1)
                    {
                        auto it = cycleidx.find(nextvert);
                        if (it != cycleidx.end())
                        {
                            // we've hit outselves
                            std::vector<int> cut;
                            for (int i = it->second; i < cycleverts.size(); i++)
                            {
                                cut.push_back(cycleverts[i]);
                            }
                            cut.push_back(nextvert);
                            cuts.push_back(cut);
                            for (int i = it->second; i < cycleedges.size(); i++)
                            {
                                usededges.insert(cycleedges[i]);
                            }
                            usededges.insert(nexte);
                            foundcycle = true;
                        }
                        else
                        {
                            cycleidx[nextvert] = cycleverts.size();
                            cycleverts.push_back(nextvert);
                            cycleedges.push_back(nexte);
                        }
                    }
                    curvert = nextvert;
                    cure = nexte;
                }
                if (!foundcycle)
                {
                    // we've hit a dead end. reverse and try the other direction
                    std::reverse(cycleverts.begin(), cycleverts.end());
                    std::reverse(cycleedges.begin(), cycleedges.end());
                    cycleidx.clear();
                    for (int i = 0; i < cycleverts.size(); i++)
                    {
                        cycleidx[cycleverts[i]] = i;
                    }
                    curvert = cycleverts.back();
                    cure = cycleedges.back();
                    while (curvert != -1 && !foundcycle)
                    {
                        int nextvert = -1;
                        int nexte = -1;
                        for (int cande : loopvertedges[curvert])
                        {
                            if (!usededges.count(cande) && cande != cure)
                            {
                                int vidx = 0;
                                if (curvert == edgeVerts(cande, vidx))
                                    vidx = 1;
                                nextvert = edgeVerts(cande, vidx);
                                nexte = cande;
                                break;
                            }
                        }
                        if (nextvert != -1)
                        {
                            auto it = cycleidx.find(nextvert);
                            if (it != cycleidx.end())
                            {
                                // we've hit outselves
                                std::vector<int> cut;
                                for (int i = it->second; i < cycleverts.size(); i++)
                                {
                                    cut.push_back(cycleverts[i]);
                                }
                                cut.push_back(nextvert);
                                cuts.push_back(cut);
                                for (int i = it->second; i < cycleedges.size(); i++)
                                {
                                    usededges.insert(cycleedges[i]);
                                }
                                usededges.insert(nexte);
                                foundcycle = true;
                            }
                            else
                            {
                                cycleidx[nextvert] = cycleverts.size();
                                cycleverts.push_back(nextvert);
                                cycleedges.push_back(nexte);
                            }
                        }
                        curvert = nextvert;
                        cure = nexte;
                    }
                    if (!foundcycle)
                    {
                        // we've found a chain
                        std::vector<int> cut;
                        for (int i = 0; i < cycleverts.size(); i++)
                        {
                            cut.push_back(cycleverts[i]);
                        }
                        cuts.push_back(cut);
                        for (int i = 0; i < cycleedges.size(); i++)
                        {
                            usededges.insert(cycleedges[i]);
                        }
                    }
                }
            }
        }
    }

    void cutMesh(const Eigen::MatrixXi &F,
            // list of cuts, each of which is a list (in order) of vertex indices of one cut.
            // Cuts can be closed loops (in which case the last vertex index should equal the
            // first) or open (in which case the two endpoint vertices should be distinct).
            // Multiple cuts can cross but there may be strange behavior if cuts share endpoint
            // vertices, or are non-edge-disjoint.
                 const std::vector<std::vector<int> > &cuts,
            // new vertices and faces
            // **DO NOT ALIAS V OR F!**
                 Eigen::MatrixXi &newF
    )
    {
        int ncuts = (int)cuts.size();

        // junction vertices that lie on multiple cuts
        std::set<int> junctions;
        std::set<int> seenverts;
        for (int i = 0; i < ncuts; i++)
        {
            std::set<int> seenincut;
            for (int j = 0; j < cuts[i].size(); j++)
            {
                if (seenverts.count(cuts[i][j]))
                    junctions.insert(cuts[i][j]);
                seenincut.insert(cuts[i][j]);
            }
            for (int v : seenincut)
                seenverts.insert(v);
        }

        // "interior" cut vertices: vertices that are part of a cut but not a cut endpoint
        // or junction vertex
        std::vector<std::set<int> > cutints;
        cutints.resize(ncuts);
        for (int i = 0; i < ncuts; i++)
        {
            if (cuts[i].empty())
                continue;
            if (cuts[i].front() == cuts[i].back())
            {
                // closed loop
                for (int v : cuts[i])
                {
                    if(!junctions.count(v))
                        cutints[i].insert(v);
                }
            }
            else
            {
                // open cut
                for (int j = 1; j < cuts[i].size() - 1; j++)
                {
                    if(!junctions.count(cuts[i][j]))
                        cutints[i].insert(cuts[i][j]);
                }
            }
        }

        struct edge
        {
            std::pair<int, int> verts;
            edge(int v1, int v2)
            {
                verts.first = std::min(v1, v2);
                verts.second = std::max(v1, v2);
            }

            bool operator<(const edge &other) const
            {
                return verts < other.verts;
            }
        };

        // maps each edge to incident triangles
        std::map<edge, std::vector<int> > edgeTriangles;
        for (int i = 0; i < F.rows(); i++)
        {
            for (int j = 0; j < 3; j++)
            {
                edge e(F(i, j), F(i, (j + 1) % 3));
                edgeTriangles[e].push_back(i);
            }
        }

        // have we visited this face yet?
        bool *visited = new bool[F.rows()];

        // edges that form part of a cut
        std::set<edge> forbidden;
        for (int i = 0; i < ncuts; i++)
        {
            for (int j = 0; j < (int)cuts[i].size() - 1; j++)
            {
                // works for both open and closed curves
                edge e(cuts[i][j], cuts[i][j + 1]);
                forbidden.insert(e);
            }
        }

        // connected components of faces adjacent to the cuts
        std::vector<std::vector<std::vector<int> > > components;
        components.resize(ncuts);

        // for each cut
        for (int cut = 0; cut < ncuts; cut++)
        {
            for (int i = 0; i < (int)F.rows(); i++)
                visited[i] = false;

            // find a face we haven't visited yet
            for (int i = 0; i < F.rows(); i++)
            {
                if (visited[i]) continue;
                bool found = false;
                for (int j = 0; j < 3; j++)
                {
                    if (cutints[cut].count(F(i, j)))
                    {
                        found = true;
                    }
                }

                if (found)
                {
                    // run a BFS along the cut edges, accumulating one connected component
                    // cross only edges that contain a vertex in cutints[cut], but are not forbidden
                    std::deque<int> q;
                    std::vector<int> component;
                    q.push_back(i);
                    while (!q.empty())
                    {
                        int next = q.front();
                        q.pop_front();
                        if (visited[next])
                            continue;
                        visited[next] = true;
                        component.push_back(next);
                        for (int j = 0; j < 3; j++)
                        {
                            int v1 = F(next, j);
                            int v2 = F(next, (j + 1) % 3);
                            edge e(v1, v2);
                            if (cutints[cut].count(v1) == 0 && cutints[cut].count(v2) == 0)
                            {
                                continue;
                            }
                            if (forbidden.count(e))
                                continue;
                            for (int nb : edgeTriangles[e])
                            {
                                if (!visited[nb])
                                {
                                    q.push_back(nb);
                                }
                            }
                        } // end BFS
                    }
                    components[cut].push_back(component);
                } // end if found
            } // end loop over all faces
        } // end loop over cuts

        std::map<int, std::vector<std::vector<int> > > junctcomponents;

        // for each junction
        for (int junc : junctions)
        {
            for (int i = 0; i < (int)F.rows(); i++)
                visited[i] = false;

            // find a face we haven't visited yet
            for (int i = 0; i < F.rows(); i++)
            {
                if (visited[i]) continue;
                bool found = false;
                for (int j = 0; j < 3; j++)
                {
                    if (junc == F(i, j))
                    {
                        found = true;
                    }
                }

                if (found)
                {
                    // run a BFS along the cut edges, accumulating one connected component
                    // cross only edges that contain the junction, but are not forbidden
                    std::deque<int> q;
                    std::vector<int> component;
                    q.push_back(i);
                    while (!q.empty())
                    {
                        int next = q.front();
                        q.pop_front();
                        if (visited[next])
                            continue;
                        visited[next] = true;
                        component.push_back(next);
                        for (int j = 0; j < 3; j++)
                        {
                            int v1 = F(next, j);
                            int v2 = F(next, (j + 1) % 3);
                            edge e(v1, v2);
                            if (v1 != junc && v2 != junc)
                            {
                                continue;
                            }
                            if (forbidden.count(e))
                                continue;
                            for (int nb : edgeTriangles[e])
                            {
                                if (!visited[nb])
                                {
                                    q.push_back(nb);
                                }
                            }
                        } // end BFS
                    }
                    junctcomponents[junc].push_back(component);
                } // end if found
            } // end loop over all faces
        } // end loop over cuts

        int vertstoadd = 0;
        // create a copy of each vertex for each component of each cut
        for (int i = 0; i < ncuts; i++)
        {
            vertstoadd += components[i].size() * cutints[i].size();
        }
        // create a copy of each junction point for each component of each junction
        for (int v : junctions)
        {
            vertstoadd += junctcomponents[v].size();
        }

        // create new faces
        Eigen::MatrixXi augF = F;

        // duplicate vertices and reindex faces

        int idx = 0;
        for (int i = 0; i < F.rows(); i++)
            for (int j = 0; j < 3; j++)
                idx = std::max(idx, F(i, j));
        idx++;

        for (int cut = 0; cut < ncuts; cut++)
        {
            for (int i = 0; i < components[cut].size(); i++)
            {
                // duplicate vertices
                std::map<int, int> idxmap;
                for (int v : cutints[cut])
                {
                    idxmap[v] = idx;
                    idx++;
                }
                for (int f : components[cut][i])
                {
                    for (int j = 0; j < 3; j++)
                    {
                        int v = augF(f, j);
                        if (cutints[cut].count(v))
                            augF(f, j) = idxmap[v];
                    }
                }
            }
        }

        for (int junc : junctions)
        {
            for (int i = 0; i < junctcomponents[junc].size(); i++)
            {

                int newidx = idx;
                idx++;

                for (int f : junctcomponents[junc][i])
                {
                    for (int j = 0; j < 3; j++)
                    {
                        int v = augF(f, j);
                        if (v == junc)
                            augF(f, j) = newidx;
                    }
                }
            }
        }

        newF = augF;
        delete[] visited;
    }

    void reindex(Eigen::MatrixXi &F)
    {
        std::map<int, int> old2new;
        for(int i=0; i<F.rows(); i++)
        {
            for(int j=0; j<3; j++)
            {
                int vidx = F(i,j);
                if(old2new.find(vidx) == old2new.end())
                {
                    int newidx = old2new.size();
                    old2new[vidx] = newidx;
                }
            }
        }
        for(int i=0; i<F.rows(); i++)
        {
            for(int j=0; j<3; j++)
            {
                int vidx = F(i,j);
                int newvidx = old2new[vidx];
                F(i,j) = newvidx;
            }
        }
    }

    void ComisoWrapper(const Eigen::SparseMatrix<double> &constraints,
                       const Eigen::SparseMatrix<double> &A,
                       Eigen::VectorXd &result,
                       const Eigen::VectorXd &rhs,
                       const Eigen::VectorXi &toRound,
                       double reg)
    {
        int n = A.rows();
        assert(n == A.cols());
        assert(n + 1 == constraints.cols());
        std::cout << n << " " << rhs.size() << std::endl;
        assert(n == rhs.size());
        COMISO::ConstrainedSolver solver;

        gmm::col_matrix< gmm::wsvector< double > > Agmm(n,n);
        int nconstraints = constraints.rows();
        gmm::row_matrix< gmm::wsvector< double > > Cgmm(nconstraints, n+1); // constraints

        for (int k=0; k < A.outerSize(); ++k){
            for (Eigen::SparseMatrix<double>::InnerIterator it(A,k); it; ++it){
                int row = it.row();
                int col = it.col();
                Agmm(row, col) += it.value();
            }
        }
        for (int k=0; k < constraints.outerSize(); ++k){
            for (Eigen::SparseMatrix<double>::InnerIterator it(constraints,k); it; ++it){
                int row = it.row();
                int col = it.col();
                Cgmm(row, col) += it.value();
            }
        }

        std::vector<double> rhsv(n,0);
        for (int i = 0; i < n; i++)
            rhsv[i] = rhs[i];

        std::vector<double> X(n, 0);
        std::vector<int> toRoundv(toRound.size());
        for (int i = 0; i < toRound.size(); i++)
            toRoundv[i] = toRound[i];
        for (int i = 0; i < toRoundv.size(); i++)
        {
            assert(toRoundv[i] >= 0 && toRoundv[i] < n);
        }
        solver.solve(Cgmm, Agmm, X, rhsv, toRoundv, reg, false, false);
        result.resize(n);
        for (int i = 0; i < n; i++)
            result[i] = X[i];
    }


    void roundPhiFromEdgeOmega(const Eigen::MatrixXd& V,
                               const Eigen::MatrixXi& F,
                               const std::vector<Eigen::Matrix2d>& abars,
                               const Eigen::VectorXd& amp,
                               const Eigen::VectorXd& dPhi, // |E| vector of phi jump
                               Eigen::VectorXd& phi,
                               Eigen::MatrixXd& seamedV,
                               Eigen::MatrixXi& seamedF,
                               Eigen::VectorXd& seamedPhi,
                               Eigen::VectorXd& seamedAmp,
                               std::set<int> &problemFaces)
    {
        MeshConnectivity mesh(F);
        int nonzeroamps = 0;

        problemFaces.clear();

        for (int i = 0; i < F.rows(); i++)
        {
            double curl = 0;
            bool isZeroFace = true;
            for (int j = 0; j < 3; j++)
            {
                if (amp(F(i, j)) > 0)
                    isZeroFace = false;
                int eid = mesh.faceEdge(i, j);
                if(mesh.edgeVertex(eid, 0) == mesh.faceVertex(i, (j + 2) % 3))
                    curl += -dPhi(eid);
                else
                    curl += dPhi(eid);
            }
            if (isZeroFace || curl > M_PI)
                problemFaces.insert(i);
        }
        std::cout << "problem face size: " << problemFaces.size() << ", ratio: " << problemFaces.size() * 1.0 / F.rows() << std::endl;

        nonzeroamps = F.rows() - problemFaces.size();

        std::vector<int> nonzero2old(nonzeroamps);
        Eigen::MatrixXi nonzeroF(nonzeroamps, 3);
        Eigen::MatrixXi zeroF(F.rows() - nonzeroamps, 3);
        std::vector<int> zero2old(F.rows() - nonzeroamps);

        int nzidx = 0;
        int zidx = 0;
        for (int i = 0; i < F.rows(); i++)
        {
            if (problemFaces.find(i) == problemFaces.end())
            {
                nonzeroF.row(nzidx) = F.row(i);
                nonzero2old[nzidx] = i;
                nzidx++;
            }
            else
            {
                zeroF.row(zidx) = F.row(i);
                zero2old[zidx] = i;
                zidx++;
            }
        }

        MeshConnectivity origNonZeroMesh(nonzeroF);

        std::vector<int> bdryedgecount(V.rows());
        for (int i = 0; i < origNonZeroMesh.nEdges(); i++)
        {
            int face1 = origNonZeroMesh.edgeFace(i, 0);
            int face2 = origNonZeroMesh.edgeFace(i, 1);
            if (face1 == -1 || face2 == -1)
            {
                bdryedgecount[origNonZeroMesh.edgeVertex(i, 0)]++;
                bdryedgecount[origNonZeroMesh.edgeVertex(i, 1)]++;
            }
        }

        std::map<int, std::vector<std::pair<int, int> > > nonmanifoldVerts;
        for (int i = 0; i < nonzeroF.rows(); i++)
        {
            for (int j = 0; j < 3; j++)
            {
                if (bdryedgecount[nonzeroF(i, j)] > 2)
                {
                    nonmanifoldVerts[nonzeroF(i, j)].push_back({ i,j });
                }
            }
        }

        std::vector<std::vector<std::vector<std::pair<int, int> > > > nonmanifoldClusters;
        for (auto& it : nonmanifoldVerts)
        {
            const auto& cluster = it.second;
            nonmanifoldClusters.push_back(std::vector<std::vector<std::pair<int, int> > >());
            int nclusterfaces = cluster.size();
            std::map<int, int> face2id;
            for (int i = 0; i < nclusterfaces; i++)
            {
                face2id[cluster[i].first] = i;
            }
            std::vector<bool> visited(nclusterfaces);
            for (int i = 0; i < nclusterfaces; i++)
            {
                if (visited[i])
                    continue;
                std::deque<int> tovisit;
                tovisit.push_back(i);
                std::vector<std::pair<int, int> > newcluster;
                while (!tovisit.empty())
                {
                    int next = tovisit.front();
                    tovisit.pop_front();
                    if (visited[next])
                        continue;
                    visited[next] = true;
                    newcluster.push_back(cluster[next]);
                    for (int j = 0; j < 3; j++)
                    {
                        int edge = origNonZeroMesh.faceEdge(cluster[next].first, j);
                        int orient = origNonZeroMesh.faceEdgeOrientation(cluster[next].first, j);
                        int opp = origNonZeroMesh.edgeFace(edge, 1 - orient);
                        if (opp != -1)
                        {
                            auto it = face2id.find(opp);
                            if (it != face2id.end())
                            {
                                if (!visited[it->second])
                                {
                                    tovisit.push_back(it->second);
                                }
                            }
                        }
                    }
                }
                nonmanifoldClusters.back().push_back(newcluster);
            }
        }

        Eigen::MatrixXi origNonZeroF = nonzeroF;

        int freeidx = V.rows();
        for (auto& it : nonmanifoldClusters)
        {
            for (int i = 1; i < it.size(); i++)
            {
                for (auto vf : it[i])
                {
                    nonzeroF(vf.first, vf.second) = freeidx;
                }
                freeidx++;
            }
        }

        MeshConnectivity punctmesh(nonzeroF);

        std::vector<std::vector<int> > cuts;
        findCuts(nonzeroF, cuts);
        std::cout << "Used " << cuts.size() << " cuts" << std::endl;

        std::map<std::pair<int, int>, int> cutpairs;

        // convert cuts to edge indices
        for (int i = 0; i < cuts.size(); i++)
        {
            int len = cuts[i].size();
            for (int j = 0; j < len - 1; j++)
            {
                cutpairs[std::pair<int, int>(cuts[i][j], cuts[i][j + 1])] = i;
                //std::cout << cuts[i][j] << ", ";
            }
        }

        Eigen::MatrixXi newF;

        cutMesh(nonzeroF, cuts, newF);
        reindex(newF);

        int newverts = 0;
        for (int i = 0; i < newF.rows(); i++)
            for (int j = 0; j < 3; j++)
                newverts = std::max(newverts, newF(i, j) + 1);

        reindex(zeroF);
        int zeroverts = 0;
        for (int i = 0; i < zeroF.rows(); i++)
            for (int j = 0; j < 3; j++)
                zeroverts = std::max(zeroverts, zeroF(i, j) + 1);

        int intdofs = 0;

        std::vector<Eigen::Triplet<double> > Ccoeffs;

        // integer constraints on the two sides of the cut
        int row = 0;
        for (int i = 0; i < punctmesh.nEdges(); i++)
        {
            int f1 = punctmesh.edgeFace(i, 0);
            int f2 = punctmesh.edgeFace(i, 1);
            if (f1 == -1 || f2 == -1)
                continue;
            int v1 = punctmesh.edgeVertex(i, 0);
            int v2 = punctmesh.edgeVertex(i, 1);
            double sign = 1.0;
            auto it = cutpairs.find(std::pair<int, int>(v1, v2));
            if (it == cutpairs.end())
            {
                it = cutpairs.find(std::pair<int, int>(v2, v1));
                sign = -1.0;
            }
            if (it == cutpairs.end())
                continue;

            int newv1 = -1;
            int newv2 = -1;
            int neww1 = -1;
            int neww2 = -1;
            for (int j = 0; j < 3; j++)
            {
                if (nonzeroF(f1, j) == v1)
                    newv1 = newF(f1, j);
                if (nonzeroF(f2, j) == v1)
                    newv2 = newF(f2, j);
                if (nonzeroF(f1, j) == v2)
                    neww1 = newF(f1, j);
                if (nonzeroF(f2, j) == v2)
                    neww2 = newF(f2, j);
            }

            Ccoeffs.push_back(Eigen::Triplet<double>(row, newv1, sign));
            Ccoeffs.push_back(Eigen::Triplet<double>(row, newv2, -sign));
            Ccoeffs.push_back(Eigen::Triplet<double>(row, newverts + intdofs, 1.0));
            Ccoeffs.push_back(Eigen::Triplet<double>(row + 1, neww1, sign));
            Ccoeffs.push_back(Eigen::Triplet<double>(row + 1, neww2, -sign));
            Ccoeffs.push_back(Eigen::Triplet<double>(row + 1, newverts + intdofs, 1.0));
            row += 2;
            intdofs++;
        }

        // add integer jumps for non-manifold vertices that connect the same connected component to itself

        std::vector<int> parent(newverts);
        for (int i = 0; i < newverts; i++)
            parent[i] = i;

        for (int i = 0; i < newF.rows(); i++)
        {
            for (int j = 0; j < 3; j++)
            {
                int idx1 = newF(i, j);
                int idx2 = newF(i, (j + 1) % 3);
                while (parent[idx1] != idx1)
                    idx1 = parent[idx1];
                while (parent[idx2] != idx2)
                    idx2 = parent[idx2];
                if (idx1 != idx2)
                {
                    // union
                    parent[idx2] = idx1;
                }
            }
        }

        int redundantManiJumps = 0;
        int essentialManiJumps = 0;

        for (auto &it : nonmanifoldClusters)
        {
            int nclusters = it.size();
            assert(nclusters > 1);
            int basef = it[0][0].first;
            int basev = it[0][0].second;

            for (int j = 1; j < nclusters; j++)
            {
                int newv1 = newF(basef, basev);
                int newv2 = newF(it[j][0].first, it[j][0].second);

                int idx1 = newv1;
                int idx2 = newv2;
                while (parent[idx1] != idx1)
                    idx1 = parent[idx1];
                while (parent[idx2] != idx2)
                    idx2 = parent[idx2];
                if (idx1 == idx2)
                {
                    essentialManiJumps++;
                    Ccoeffs.push_back({ row, newv1, 1.0 });
                    Ccoeffs.push_back({ row, newv2, -1.0 });
                    Ccoeffs.push_back({ row, newverts + intdofs, 1.0 });
                    row++;
                    intdofs++;
                }
                else
                {
                    // integer jump not needed; force 0
                    redundantManiJumps++;
                    parent[idx2] = idx1;
                    Ccoeffs.push_back({ row, newv1, 1.0 });
                    Ccoeffs.push_back({ row, newv2, -1.0 });
                    row++;
                }
            }
        }
        Eigen::SparseMatrix<double> C(row, newverts + intdofs + 1);
        C.setFromTriplets(Ccoeffs.begin(), Ccoeffs.end());

        std::cout << "For non-manifold vertices, added " << essentialManiJumps << " integer jumps and " << redundantManiJumps << " zero-jumps" << std::endl;

        std::vector<Eigen::Triplet<double> > Acoeffs;
        int newfaces = newF.rows();
        //assert(newfaces == punctF.rows());

        for (int i = 0; i < newfaces; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                Acoeffs.push_back(Eigen::Triplet<double>(3 * i + j, newF(i, j), 1.0));
            }
        }
        Eigen::SparseMatrix<double> A(3 * newfaces, newverts + intdofs);
        A.setFromTriplets(Acoeffs.begin(), Acoeffs.end());

        std::vector<Eigen::Triplet<double> > L2Minvcoeffs;
        for (int i = 0; i < newfaces; i++)
        {
            double area = 0.5 * std::sqrt(abars[nonzero2old[i]].determinant());
            for (int j = 0; j < 3; j++)
            {
                //double avamp = 0.5 * (amp[F(nonzero2old[i], (j + 1) % 3)] + amp[F(nonzero2old[i], (j + 2) % 3)]);
                //L2Minvcoeffs.push_back(Eigen::Triplet<double>(3 * i + j, 3 * i + j, area * avamp));
                L2Minvcoeffs.push_back(Eigen::Triplet<double>(3 * i + j, 3 * i + j, area));
            }
        }
        Eigen::SparseMatrix<double> L2Minv(3 * newfaces, 3 * newfaces);
        L2Minv.setFromTriplets(L2Minvcoeffs.begin(), L2Minvcoeffs.end());



        std::vector<Eigen::Triplet<double> > Dcoeffs;
        for (int i = 0; i < newfaces; i++)
        {
            Dcoeffs.push_back(Eigen::Triplet<double>(3 * i, 3 * i + 2, 1.0));
            Dcoeffs.push_back(Eigen::Triplet<double>(3 * i, 3 * i + 1, -1.0));
            Dcoeffs.push_back(Eigen::Triplet<double>(3 * i + 1, 3 * i, 1.0));
            Dcoeffs.push_back(Eigen::Triplet<double>(3 * i + 1, 3 * i + 2, -1.0));
            Dcoeffs.push_back(Eigen::Triplet<double>(3 * i + 2, 3 * i + 1, 1.0));
            Dcoeffs.push_back(Eigen::Triplet<double>(3 * i + 2, 3 * i, -1.0));
        }
        Eigen::SparseMatrix<double> D(3 * newfaces, 3 * newfaces);
        D.setFromTriplets(Dcoeffs.begin(), Dcoeffs.end());

        Eigen::SparseMatrix<double> Mat = A.transpose() * D.transpose() * L2Minv * D * A;

        Eigen::VectorXd punctfield(3 * newfaces);
        for (int i = 0; i < newfaces; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                int edgeidx = mesh.faceEdge(nonzero2old[i], j);
                double sign = (mesh.faceEdgeOrientation(nonzero2old[i], j) == 0 ? 1.0 : -1.0);
                punctfield[3 * i + j] = dPhi[edgeidx] * sign / 2.0 / M_PI;
            }
        }
        Eigen::VectorXd rhs = A.transpose() * D.transpose() * L2Minv * punctfield;

        Eigen::VectorXd result;
        Eigen::VectorXi toRound(intdofs);
        for (int i = 0; i < intdofs; i++)
        {
            toRound[i] = newverts + i;
        }
        std::cout << intdofs << " integer variables" << std::endl;
        std::cout << "Comiso round" << std::endl;
        ComisoWrapper(C, Mat, result, rhs, toRound, 1e-6);

        std::cout << "Solver residual: " << (Mat * result - rhs).norm() << std::endl;
        std::cout << "Reconstruction residual: " << (D * A * result - punctfield).transpose() * L2Minv * (D * A * result - punctfield) << std::endl;
        std::cout << "Integer jumps: ";
        for (int i = 0; i < intdofs; i++)
            std::cout << result[newverts + i] << " ";
        std::cout << std::endl;

        // map phi back to the original mesh
        int oldverts = V.rows();
        phi.resize(oldverts);
        phi.setZero();
        for (int i = 0; i < newfaces; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                phi[origNonZeroF(i, j)] = 2.0 * M_PI * result[newF(i, j)];
            }
        }

        seamedV.resize(newverts + zeroverts, 3);

        // construct new position
        for (int i = 0; i < newfaces; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                seamedV.row(newF(i, j)) = V.row(origNonZeroF(i, j));
            }
        }

        for (int i = 0; i < zeroF.rows(); i++)
        {
            for (int j = 0; j < 3; j++)
            {
                seamedV.row(newverts + zeroF(i, j)) = V.row(F(zero2old[i], j));
            }
        }

        seamedF.resize(F.rows(), 3);
        for (int i = 0; i < newfaces; i++)
        {
            seamedF.row(nonzero2old[i]) = newF.row(i);
        }
        for (int i = 0; i < zeroF.rows(); i++)
        {
            for (int j = 0; j < 3; j++)
            {
                seamedF(zero2old[i], j) = newverts + zeroF(i, j);
            }
        }

        // construct new phi and amp with the seam
        seamedPhi.resize(newverts + zeroverts);
        seamedAmp.resize(newverts + zeroverts);
        seamedPhi.setZero();
        seamedAmp.setZero();
        std::set<int> freeDOFs;
        for (int i = 0; i < newfaces; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                seamedPhi(newF(i, j)) = 2.0 * M_PI * result(newF(i, j));
                seamedAmp(newF(i, j)) = amp(origNonZeroF(i, j));
            }
        }

        for (int i = 0; i < zeroF.rows(); i++)
        {
            for (int j = 0; j < 3; j++)
            {
                //seamedAmp(newverts + zeroF(i, j)) = amp(F(zero2old[i], j));
                seamedAmp(newverts + zeroF(i, j)) = 0;
            }
        }
    }


    void findSharpCorners(const Eigen::MatrixXd& uncutV, const Eigen::MatrixXi& uncutF, std::set<int>& cornerVerts)
    {
        int nverts = uncutV.rows();
        std::vector<double> anglesum(nverts);
        std::set<int> bdryverts;
        MeshConnectivity mesh(uncutF);
        for (int i = 0; i < uncutF.rows(); i++)
        {
            for (int j = 0; j < 3; j++)
            {
                int edge = mesh.faceEdge(i, j);
                int orient = mesh.faceEdgeOrientation(i, j);
                int opp = mesh.edgeFace(edge, 1 - orient);
                if (opp == -1)
                {
                    bdryverts.insert(mesh.edgeVertex(edge, 0));
                    bdryverts.insert(mesh.edgeVertex(edge, 1));
                }

                Eigen::Vector3d v0 = uncutV.row(uncutF(i, j)).transpose();
                Eigen::Vector3d v1 = uncutV.row(uncutF(i, (j + 1) % 3)).transpose();
                Eigen::Vector3d v2 = uncutV.row(uncutF(i, (j + 2) % 3)).transpose();

                Eigen::Vector3d e1 = v1 - v0;
                Eigen::Vector3d e2 = v2 - v0;
                double theta = std::atan2(e1.cross(e2).norm(), e1.dot(e2));
                anglesum[uncutF(i, j)] += theta;
            }
        }

        cornerVerts.clear();
        for (auto i : bdryverts)
        {
            double tol = M_PI / 4;
            if (std::fabs(anglesum[i] - M_PI) > tol)
                cornerVerts.insert(i);
        }
    }


    static void zeroProblemAmplitudes(const Eigen::MatrixXi& F, const std::set<int>& problemFaces, Eigen::VectorXd& soupAmps)
    {
        std::set<int> zeroverts;
        int nfaces = F.rows();
        for (int i = 0; i < nfaces; i++)
        {
            if (problemFaces.count(i) == 0)
                continue;
            for (int j = 0; j < 3; j++)
                zeroverts.insert(F(i, j));
        }

        for (int i = 0; i < nfaces; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                if (zeroverts.count(F(i, j)))
                {
                    soupAmps[3 * i + j] = 0;
                }
            }
        }
    }

    struct ClusterInfo
    {
        std::vector<std::vector<std::pair<int, int> > > clusters;
        std::vector<int> jump;
    };

    // computes the integer multiple of 2Pi jump between the values of cutFunction on edgeFace(i,0) and edgeFace(i,1)
    static void computeEdgeJumps(int uncutVerts, const Eigen::MatrixXi& uncutF, const std::set<int>& problemFaces,
                                 const Eigen::VectorXd &soupFunction, Eigen::VectorXi &jumps,
                                 std::map<int, ClusterInfo> &clusterjumps)
    {
        MeshConnectivity uncutMesh(uncutF);

        int nedges = uncutMesh.nEdges();
        jumps.resize(nedges);
        jumps.setZero();
        for (int i = 0; i < nedges; i++)
        {
            int face1 = uncutMesh.edgeFace(i, 0);
            int face2 = uncutMesh.edgeFace(i, 1);
            if (face1 == -1 || face2 == -1)
                continue;
            if (problemFaces.count(face1) || problemFaces.count(face2))
                continue;
            int vert[2];
            vert[0] = uncutMesh.edgeVertex(i, 0);
            vert[1] = uncutMesh.edgeVertex(i, 1);
            double vert1val[2];
            double vert2val[2];
            double jump[2];
            for (int k = 0; k < 2; k++)
            {
                for (int j = 0; j < 3; j++)
                {
                    if (uncutF(face1, j) == vert[k])
                        vert1val[k] = soupFunction[3 * face1 + j];
                    if (uncutF(face2, j) == vert[k])
                        vert2val[k] = soupFunction[3 * face2 + j];

                }
                jump[k] = (vert2val[k] - vert1val[k]) / 2.0 / M_PI;
            }

            int ijmp1 = std::round(jump[0]);
            int ijmp2 = std::round(jump[1]);
            int avjmp = (ijmp1 + ijmp2) / 2;
            jumps[i] = avjmp;
            if (ijmp1 != ijmp2 || std::fabs(avjmp - jumps[i]) > 1e-6)
            {
                std::cout << "Bad jump: " << i << " " << jump[0] << " " << jump[1] << " " << face1 << " " << face2 << " " << vert[0] << " " << vert1val[0] << " " << vert2val[0] << " " << vert[1] << " " << vert1val[1] << " " << vert2val[1] << " " << problemFaces.count(face1) << " " << problemFaces.count(face2) << std::endl;
            }
        }

        clusterjumps.clear();
        std::vector<std::vector<std::pair<int, int> > > vertexFaces(uncutVerts);
        for (int i = 0; i < uncutMesh.nFaces(); i++)
        {
            if (!problemFaces.count(i))
            {
                for (int j = 0; j < 3; j++)
                {
                    vertexFaces[uncutF(i, j)].push_back({ i, j});
                }
            }
        }

        for (int i = 0; i < uncutVerts; i++)
        {
            int ncfaces = vertexFaces[i].size();
            std::vector<bool> visited(ncfaces);
            std::map<int, int> face2id;
            for (int j = 0; j < ncfaces; j++)
            {
                face2id[vertexFaces[i][j].first] = j;
            }
            std::vector < std::vector<std::pair<int, int> > > clusters;
            for (int j = 0; j < ncfaces; j++)
            {
                if (visited[j])
                    continue;
                std::vector<std::pair<int, int> > newcluster;
                std::deque<int> tovisit;
                tovisit.push_back(j);
                while (!tovisit.empty())
                {
                    int next = tovisit.front();
                    tovisit.pop_front();
                    if (visited[next])
                        continue;
                    visited[next] = true;
                    newcluster.push_back(vertexFaces[i][next]);
                    for (int k = 0; k < 3; k++)
                    {
                        int edge = uncutMesh.faceEdge(vertexFaces[i][next].first, k);
                        int orient = uncutMesh.faceEdgeOrientation(vertexFaces[i][next].first, k);
                        int opp = uncutMesh.edgeFace(edge, 1 - orient);
                        if (opp != -1 && !problemFaces.count(opp))
                        {
                            auto it = face2id.find(opp);
                            if (it != face2id.end())
                            {
                                if (!visited[it->second])
                                {
                                    tovisit.push_back(it->second);
                                }
                            }
                        }
                    }
                }
                clusters.push_back(newcluster);
            }
            if (clusters.size() > 1)
            {
                ClusterInfo ci;
                ci.clusters = clusters;
                ci.jump.resize(clusters.size());
                ci.jump[0] = 0;
                for (int i = 1; i < clusters.size(); i++)
                {
                    double vf1val = soupFunction[3 * ci.clusters[0][0].first + ci.clusters[0][0].second];
                    double vf2val = soupFunction[3 * ci.clusters[i][0].first + ci.clusters[i][0].second];
                    double jump = (vf2val - vf1val) / 2.0 / M_PI;
                    ci.jump[i] = std::round(jump);
                    if (std::fabs(jump - ci.jump[i]) > 1e-6)
                    {
                        std::cout << "Bad cluster jump: " << i << std::endl;
                    }
                }
                clusterjumps[i] = ci;
            }
        }
    }

    static void neighborhoodSoupJumps(const MeshConnectivity& mesh,
                               const std::set<int>& problemFaces,
                               const std::vector<std::vector<int> >& vertEdges,
                               const Eigen::VectorXi& periodJumps,
                               int face, int vertidx,
                               std::map<int, int>& soupjumps,
                               const std::map<int, ClusterInfo> &clusterInfo
    )
    {
        soupjumps.clear();

        std::map<int, std::vector<std::pair<int, int> > > graph;
        for (auto it : vertEdges[mesh.faceVertex(face, vertidx)])
        {
            int face1 = mesh.edgeFace(it, 0);
            int face2 = mesh.edgeFace(it, 1);
            if (face1 == -1 || face2 == -1)
                continue;
            if (problemFaces.count(face1) || problemFaces.count(face2))
                continue;
            graph[face1].push_back({ face2, periodJumps[it] });
            graph[face2].push_back({ face1, -periodJumps[it] });
        }
        struct Visit
        {
            int face;
            int totaljump;
        };
        std::deque<Visit> q;
        q.push_back({ face, 0 });
        std::map<int, bool> visited;
        while (!q.empty())
        {
            Visit next = q.front();
            q.pop_front();
            if (visited[next.face])
                continue;
            visited[next.face] = true;
            soupjumps[next.face] = next.totaljump;
            for (auto nb : graph[next.face])
            {
                if (!visited[nb.first])
                {
                    q.push_back({ nb.first, next.totaljump + nb.second });
                }
            }
        }

        auto it = clusterInfo.find(mesh.faceVertex(face, vertidx));
        if (it != clusterInfo.end())
        {
            int curcluster = -1;
            int basejump = 0;
            int nclusters = it->second.clusters.size();
            for (int i = 0; i < nclusters; i++)
            {
                auto sjit = soupjumps.find(it->second.clusters[i][0].first);
                if (sjit != soupjumps.end())
                {
                    curcluster = i;
                    basejump = sjit->second - it->second.jump[i];
                }
            }

            for (int i = 0; i < nclusters; i++)
            {
                if (i == curcluster)
                    continue;

                int seedface = it->second.clusters[i][0].first;

                std::deque<Visit> q;
                q.push_back({ seedface, basejump + it->second.jump[i]});
                std::map<int, bool> visited;
                while (!q.empty())
                {
                    Visit next = q.front();
                    q.pop_front();
                    if (visited[next.face])
                        continue;
                    visited[next.face] = true;
                    soupjumps[next.face] = next.totaljump;
                    for (auto nb : graph[next.face])
                    {
                        if (!visited[nb.first])
                        {
                            q.push_back({ nb.first, next.totaljump + nb.second });
                        }
                    }
                }
            }
        }
    }

    static void loopUncut(
            int uncutVerts,
            const Eigen::MatrixXi& uncutF,
            const std::set<int> &cornerVerts,
            Eigen::SparseMatrix<double>& uncutS,
            Eigen::MatrixXi& newF
    )
    {
        MeshConnectivity mesh(uncutF);
        int nfaces = mesh.nFaces();
        int nedges = mesh.nEdges();

        std::vector<std::set<int> > vertexNeighbors(uncutVerts); // all neighbors
        std::vector<bool> boundaryVertex(uncutVerts); // on the boundary?
        std::vector<std::set<int> > boundaryNeighbors(uncutVerts); // neighbors on the boundary
        for (int i = 0; i < nfaces; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                vertexNeighbors[uncutF(i, j)].insert(uncutF(i, (j + 1) % 3));
                vertexNeighbors[uncutF(i, j)].insert(uncutF(i, (j + 2) % 3));
            }
        }

        for (int i = 0; i < nedges; i++)
        {
            int face1 = mesh.edgeFace(i, 0);
            int face2 = mesh.edgeFace(i, 1);
            int vert1 = mesh.edgeVertex(i, 0);
            int vert2 = mesh.edgeVertex(i, 1);
            if (face1 == -1 || face2 == -1)
            {
                boundaryVertex[vert1] = true;
                boundaryVertex[vert2] = true;
                boundaryNeighbors[vert1].insert(vert2);
                boundaryNeighbors[vert2].insert(vert1);
            }
        }

        // Step 2: make newF
        int newfaces = 4 * nfaces;
        int newverts = uncutVerts + nedges;

        newF.resize(newfaces, 3);

        for (int i = 0; i < nfaces; i++)
        {
            // the central triangle
            for (int j = 0; j < 3; j++)
            {
                newF(4 * i, j) = uncutVerts + mesh.faceEdge(i, j);
            }

            // the three corner triangles
            // vertex i, edge (i+2), edge (i+1)
            for (int j = 0; j < 3; j++)
            {
                newF(4 * i + j + 1, 0) = mesh.faceVertex(i, j);
                newF(4 * i + j + 1, 1) = uncutVerts + mesh.faceEdge(i, (j + 2) % 3);
                newF(4 * i + j + 1, 2) = uncutVerts + mesh.faceEdge(i, (j + 1) % 3);

            }
        }


        // Step 3: make the uncutS stencil
        std::vector<Eigen::Triplet<double> > uncutScoeffs;
        // the old vertices
        for (int i = 0; i < uncutVerts; i++)
        {
            if (boundaryVertex[i])
            {
                if (cornerVerts.count(i))
                {
                    uncutScoeffs.push_back({ i,i,1.0 });
                }
                else
                {
                    int valence = boundaryNeighbors[i].size();
                    assert(valence >= 2);
                    double beta = 1.0 / 4.0 / double(valence);
                    uncutScoeffs.push_back({ i,i,3.0 / 4.0 });
                    for (auto it : boundaryNeighbors[i])
                    {
                        uncutScoeffs.push_back({ i, it, beta });
                    }
                }
            }
            else
            {
                int valence = vertexNeighbors[i].size();
                assert(valence >= 3);
                double beta = 0;
                if (valence == 3)
                {
                    beta = 3.0 / 16.0;
                }
                else
                {
                    beta = 3.0 / 8.0 / double(valence);
                }

                uncutScoeffs.push_back({ i, i, 1.0 - double(valence) * beta });
                for (auto it : vertexNeighbors[i])
                {
                    uncutScoeffs.push_back({ i, it, beta });
                }
            }
        }
        // the new vertices
        for (int i = 0; i < nedges; i++)
        {
            int face1 = mesh.edgeFace(i, 0);
            int face2 = mesh.edgeFace(i, 1);
            int vert1 = mesh.edgeVertex(i, 0);
            int vert2 = mesh.edgeVertex(i, 1);
            if (face1 == -1 || face2 == -1)
            {
                uncutScoeffs.push_back({ uncutVerts + i, vert1, 0.5 });
                uncutScoeffs.push_back({ uncutVerts + i, vert2, 0.5 });
            }
            else
            {
                uncutScoeffs.push_back({ uncutVerts + i, vert1, 3.0/8.0 });
                uncutScoeffs.push_back({ uncutVerts + i, vert2, 3.0/8.0 });
                int oppvert1 = mesh.edgeOppositeVertex(i, 0);
                int oppvert2 = mesh.edgeOppositeVertex(i, 1);
                uncutScoeffs.push_back({ uncutVerts + i, oppvert1, 1.0 / 8.0 });
                uncutScoeffs.push_back({ uncutVerts + i, oppvert2, 1.0 / 8.0 });
            }
        }
        uncutS.resize(newverts, uncutVerts);
        uncutS.setFromTriplets(uncutScoeffs.begin(), uncutScoeffs.end());
    }


    // Applies Loop subdivision to a given mesh uncutF.
// Returns three stencils:
//  - uncutS: the subdivision stencil for uncutF (ordinary Loop subdivision)
//  - soupS: the subdivision stencil for a triangle soup with the same connectivity as uncutF (only makes sense if the function at coincident
//           vertices actually agree).
//  - periodicS, periodicb: affine stencil for a periodic function on the triangle soup whose period jumps on edges is given by periodJumps.
//                                  Extends the function along edges into the "problem faces" (and otherwise treats them as if they didn't exist)
    static void loopWithSeams(
            int uncutVerts,
            const Eigen::MatrixXi& uncutF,
            const Eigen::VectorXi& periodJumps,
            const std::map<int, ClusterInfo>& clusterJumps,
            const std::set<int>& problemFaces,
            const std::set<int>& cornerVerts,
            Eigen::SparseMatrix<double>& uncutS,
            Eigen::SparseMatrix<double>& soupS,
            Eigen::SparseMatrix<double>& periodicS,
            Eigen::VectorXd& periodicb,
            Eigen::MatrixXi& newF,
            std::set<int>& newProblemFaces)
    {
        MeshConnectivity mesh(uncutF);

        // Step 1: collect neighbors

        int nfaces = uncutF.rows();
        int nedges = mesh.nEdges();

        std::vector<std::set<int> > vertexNeighbors(uncutVerts); // all neighbors
        std::vector<bool> boundaryVertex(uncutVerts); // on the boundary?
        std::vector<std::set<int> > boundaryNeighbors(uncutVerts); // neighbors on the boundary
        std::vector<bool> problemVertex(uncutVerts); // touching a problem face?
        std::vector<bool> transitionVertex(uncutVerts); // on the edge between a regular and problem face?
        std::vector<bool> protectedEdges(nedges);
        /*
        for (int i = 0; i < uncutF.rows(); i++)
        {
            for (int j = 0; j < 3; j++)
            {
                if (uncutF(i, j) < 0 || uncutF(i, j) >= uncutVerts)
                    exit(0);
            }
        }
        */

        for (int i = 0; i < nfaces; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                vertexNeighbors[uncutF(i, j)].insert(uncutF(i, (j + 1) % 3));
                vertexNeighbors[uncutF(i, j)].insert(uncutF(i, (j + 2) % 3));
            }
        }

        for (int i = 0; i < nedges; i++)
        {
            int face1 = mesh.edgeFace(i, 0);
            int face2 = mesh.edgeFace(i, 1);
            int vert1 = mesh.edgeVertex(i, 0);
            int vert2 = mesh.edgeVertex(i, 1);
            if (face1 == -1 || face2 == -1)
            {
                boundaryVertex[vert1] = true;
                boundaryVertex[vert2] = true;
                boundaryNeighbors[vert1].insert(vert2);
                boundaryNeighbors[vert2].insert(vert1);
            }
            else
            {
                int problemfaces = 0;
                if (problemFaces.count(face1)) problemfaces++;
                if (problemFaces.count(face2)) problemfaces++;
                if (problemfaces == 1)
                {
                    transitionVertex[vert1] = true;
                    transitionVertex[vert2] = true;
                }
                if (problemfaces > 0)
                {
                    problemVertex[vert1] = true;
                    problemVertex[vert2] = true;
                }
            }
        }

        for (int i = 0; i < nedges; i++)
        {
            int face1 = mesh.edgeFace(i, 0);
            int face2 = mesh.edgeFace(i, 1);

            int vert1 = mesh.edgeVertex(i, 0);
            int vert2 = mesh.edgeVertex(i, 1);
            if (face1 != -1 && face2 != -1
                && problemFaces.count(face1) && problemFaces.count(face2)
                && transitionVertex[vert1] && transitionVertex[vert2])
            {
                // edge is chord through the problem region
                protectedEdges[i] = true;
            }
            if (((face1 == -1 && problemFaces.count(face2)) || (face2 == -1 && problemFaces.count(face1)))
                && transitionVertex[vert1] && transitionVertex[vert2])
            {
                protectedEdges[i] = true;
            }
        }

        loopUncut(uncutVerts, uncutF, cornerVerts, uncutS, newF);

        int newfaces = newF.rows();
        int newverts = uncutVerts + nedges;

        std::vector<int> soupNewToOld(3 * newfaces + newverts, -1);
        for (int i = 0; i < nfaces; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                soupNewToOld[3 * (4 * i + j + 1) + 0] = 3 * i + j;
            }
        }

        // Step 4: compute soup neighbor data
        std::vector<std::vector<std::pair<int, int> > > soupNeighbors(uncutVerts); // neighbors in the triangle soup
        std::vector<std::vector<std::pair<int, int> > > soupOrangeNeighbors(uncutVerts); // extrapolate into the problem region
        std::vector<std::vector<std::pair<int, int> > > soupBoundaryNeighbors(uncutVerts); // neighbors on the boundary or problem face in the soup
        std::vector<std::vector<std::pair<int, int> > > soupOrangeBoundaryNeighbors(uncutVerts); // extrapolate into the problem region
        std::map<int, std::pair<int, int> > nonProblemCopy;

        for (int i = 0; i < nfaces; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                if (!problemFaces.count(i))
                {
                    nonProblemCopy[uncutF(i, j)] = { i,j };
                }
                int edge = mesh.faceEdge(i, j);

                soupNeighbors[uncutF(i, (j + 1) % 3)].push_back({ i,(j + 2) % 3 });
                soupNeighbors[uncutF(i, (j + 2) % 3)].push_back({ i,(j + 1) % 3 });

                int face1 = mesh.edgeFace(edge, 0);
                int face2 = mesh.edgeFace(edge, 1);

                if ((face1 == -1 || !problemFaces.count(face1))
                    && (face2 == -1 || !problemFaces.count(face2)))
                {
                    soupOrangeNeighbors[uncutF(i, (j + 1) % 3)].push_back({ i,(j + 2) % 3 });
                    soupOrangeNeighbors[uncutF(i, (j + 2) % 3)].push_back({ i,(j + 1) % 3 });
                }
                else
                {
                    soupOrangeNeighbors[uncutF(i, (j + 1) % 3)].push_back({ i,(j + 1) % 3 });
                    soupOrangeNeighbors[uncutF(i, (j + 2) % 3)].push_back({ i,(j + 2) % 3 });
                }

                int orient = mesh.faceEdgeOrientation(i, j);
                int oppface = mesh.edgeFace(edge, 1 - orient);
                if (oppface == -1)
                {
                    soupBoundaryNeighbors[uncutF(i, (j + 1) % 3)].push_back({ i, (j + 2) % 3 });
                    soupBoundaryNeighbors[uncutF(i, (j + 2) % 3)].push_back({ i, (j + 1) % 3 });

                    if (!problemFaces.count(i))
                    {
                        soupOrangeBoundaryNeighbors[uncutF(i, (j + 1) % 3)].push_back({ i, (j + 2) % 3 });
                        soupOrangeBoundaryNeighbors[uncutF(i, (j + 2) % 3)].push_back({ i, (j + 1) % 3 });
                    }
                    else
                    {
                        soupOrangeBoundaryNeighbors[uncutF(i, (j + 1) % 3)].push_back({ i, (j + 1) % 3 });
                        soupOrangeBoundaryNeighbors[uncutF(i, (j + 2) % 3)].push_back({ i, (j + 2) % 3 });
                    }
                }
            }
        }

        // Step 5: make the soup stencil
        std::vector<Eigen::Triplet<double> > soupScoeffs;

        for(int faceitr = 0; faceitr < newfaces; faceitr++)
        {
            for (int vertitr = 0; vertitr < 3; vertitr++)
            {
                int vertid = newF(faceitr, vertitr);
                if (vertid < uncutVerts)
                {
                    if(boundaryVertex[vertid])
                    {
                        if (cornerVerts.count(vertid))
                        {
                            soupScoeffs.push_back({ 3 * faceitr + vertitr, soupNewToOld[3 * faceitr + vertitr], 1.0 });
                        }
                        else
                        {
                            int valence = soupBoundaryNeighbors[vertid].size();
                            assert(valence >= 2);
                            double beta = 1.0 / 4.0 / double(valence);
                            double alpha = 3.0 / 4.0;

                            soupScoeffs.push_back({ 3 * faceitr + vertitr, soupNewToOld[3 * faceitr + vertitr], alpha });

                            for (auto it : soupBoundaryNeighbors[vertid])
                            {
                                soupScoeffs.push_back({ 3 * faceitr + vertitr, 3 * it.first + it.second, beta });
                            }
                        }
                    }
                    else
                    {
                        int valence = soupNeighbors[vertid].size();
                        assert(valence >= 6);
                        double beta = 0;
                        if (valence == 6)
                        {
                            beta = 3.0 / 32.0;
                        }
                        else
                        {
                            beta = 3.0 / 8.0 / double(valence);
                        }
                        double alpha = (1.0 - double(valence) * beta);

                        soupScoeffs.push_back({ 3 * faceitr + vertitr, soupNewToOld[3 * faceitr + vertitr], alpha });

                        for (auto it : soupNeighbors[vertid])
                        {
                            soupScoeffs.push_back({ 3 * faceitr + vertitr, 3 * it.first + it.second, beta });
                        }
                    }
                }
                else
                {
                    // the new vertices
                    int edgeitr = vertid - uncutVerts;

                    int face1 = mesh.edgeFace(edgeitr, 0);
                    int face2 = mesh.edgeFace(edgeitr, 1);
                    int vert1 = mesh.edgeVertex(edgeitr, 0);
                    int vert2 = mesh.edgeVertex(edgeitr, 1);
                    if (face1 == -1 || face2 == -1 )
                    {
                        int okface = face1;
                        if (okface == -1)
                            okface = face2;
                        for (int i = 0; i < 3; i++)
                        {
                            if (uncutF(okface, i) == vert1 || uncutF(okface, i) == vert2)
                            {
                                soupScoeffs.push_back({ 3 * faceitr + vertitr, 3 * okface + i, 0.5 });
                            }
                        }
                    }
                    else
                    {
                        int oppvert1 = mesh.edgeOppositeVertex(edgeitr, 0);
                        int oppvert2 = mesh.edgeOppositeVertex(edgeitr, 1);
                        for (int i = 0; i < 3; i++)
                        {
                            if (uncutF(face1, i) == oppvert1)
                            {
                                soupScoeffs.push_back({ 3 * faceitr + vertitr, 3 * face1 + i, 1.0 / 8.0 });
                            }
                            else
                            {
                                soupScoeffs.push_back({ 3 * faceitr + vertitr, 3 * face1 + i, 3.0 / 16.0 });
                            }

                            if (uncutF(face2, i) == oppvert2)
                            {
                                soupScoeffs.push_back({ 3 * faceitr + vertitr, 3 * face2 + i, 1.0 / 8.0 });
                            }
                            else
                            {
                                soupScoeffs.push_back({ 3 * faceitr + vertitr, 3 * face2 + i, 3.0 / 16.0 });
                            }
                        }
                    }
                }
            }
        }
        soupS.resize(3 * newfaces, 3 * nfaces);
        soupS.setFromTriplets(soupScoeffs.begin(), soupScoeffs.end());

        // Step 6: convert the period jumps to (face, face) form
        std::map<std::pair<int, int>, int> faceFaceJumps;
        for (int i=0; i<nedges; i++)
        {
            int face0 = mesh.edgeFace(i, 0);
            int face1 = mesh.edgeFace(i, 1);
            if (face0 == -1 || face1 == -1)
                continue;
            faceFaceJumps[{face0, face1}] = periodJumps[i];
            faceFaceJumps[{face1, face0}] = -periodJumps[i];
        }

        // Step 8: compute new problem faces
        newProblemFaces.clear();
        for (auto it : problemFaces)
        {
            bool problemchild[3];
            for (int i = 0; i < 3; i++)
                problemchild[i] = false;

            for (int i = 0; i < 3; i++)
            {
                if (!transitionVertex[uncutF(it, i)])
                {
                    problemchild[i] = true;
                }
                if (protectedEdges[mesh.faceEdge(it, i)])
                {
                    problemchild[(i + 1) % 3] = true;
                    problemchild[(i + 2) % 3] = true;
                }
            }
            int problems = 0;
            for (int i = 0; i < 3; i++)
            {
                if (problemchild[i])
                {
                    problems++;
                    newProblemFaces.insert(4 * it + 1 + i);
                }
            }
            if (problems != 1)
            {
                newProblemFaces.insert(4 * it);
            }
        }

        // Step 9: compute vertex edge neighors
        std::vector<std::vector<int> > vertexEdges(uncutVerts);
        for (int i = 0; i < nedges; i++)
        {
            vertexEdges[mesh.edgeVertex(i, 0)].push_back(i);
            vertexEdges[mesh.edgeVertex(i, 1)].push_back(i);
        }

        // Step 11: build periodic stencil

        std::vector<Eigen::Triplet<double> > periodicScoeffs;
        periodicb.resize(3 * newfaces);
        periodicb.setZero();

        for(int faceitr = 0; faceitr < newfaces; faceitr++)
        {
            if (newProblemFaces.count(faceitr))
                continue;

            for (int vertitr = 0; vertitr < 3; vertitr++)
            {
                int vertid = newF(faceitr, vertitr);
                if (vertid < uncutVerts)
                {
                    // the old vertices

                    int oldfaceid = soupNewToOld[3 * faceitr + vertitr] / 3;
                    int oldvertidx = soupNewToOld[3 * faceitr + vertitr] % 3;

                    if (problemFaces.count(oldfaceid))
                    {
                        // old vertex on a previous problem-face
                        // use an arbitrary non-problem neighbor for the calculation
                        oldfaceid = nonProblemCopy[vertid].first;
                        oldvertidx = nonProblemCopy[vertid].second;
                        vertid = uncutF(oldfaceid, oldvertidx);
                    }

                    std::map<int, int> soupjumps;
                    neighborhoodSoupJumps(mesh, problemFaces, vertexEdges, periodJumps, oldfaceid, oldvertidx, soupjumps, clusterJumps);

                    if (boundaryVertex[vertid])
                    {
                        if (cornerVerts.count(vertid))
                        {
                            periodicScoeffs.push_back({ 3 * faceitr + vertitr, 3 * oldfaceid + oldvertidx, 1.0 });
                        }
                        else
                        {
                            int valence = soupOrangeBoundaryNeighbors[vertid].size();
                            assert(valence >= 2);
                            double beta = 1.0 / 4.0 / double(valence);
                            double alpha = 3.0 / 4.0;

                            periodicScoeffs.push_back({ 3 * faceitr + vertitr, 3 * oldfaceid + oldvertidx, alpha });

                            for (auto it : soupOrangeBoundaryNeighbors[vertid])
                            {
                                int nbface = it.first;
                                int nbvert = it.second;
                                if (problemFaces.count(nbface))
                                {
                                    nbface = nonProblemCopy[uncutF(nbface, nbvert)].first;
                                    nbvert = nonProblemCopy[uncutF(nbface, nbvert)].second;
                                }
                                periodicScoeffs.push_back({ 3 * faceitr + vertitr, 3 * nbface + nbvert, beta });
                                periodicb[3 * faceitr + vertitr] += -double(soupjumps[nbface]) * 2.0 * M_PI * beta;
                            }
                        }
                    }
                    else
                    {
                        int valence = soupOrangeNeighbors[vertid].size();
                        assert(valence >= 6);
                        double beta = 0;
                        if (valence == 6)
                        {
                            beta = 3.0 / 32.0;
                        }
                        else
                        {
                            beta = 3.0 / 8.0 / double(valence);
                        }
                        double alpha = (1.0 - double(valence) * beta);

                        periodicScoeffs.push_back({ 3 * faceitr + vertitr, 3*oldfaceid + oldvertidx, alpha });

                        for (auto it : soupOrangeNeighbors[vertid])
                        {
                            int nbface = it.first;
                            int nbvert = it.second;
                            if (problemFaces.count(nbface))
                            {
                                nbface = nonProblemCopy[uncutF(nbface, nbvert)].first;
                                nbvert = nonProblemCopy[uncutF(nbface, nbvert)].second;
                            }
                            periodicScoeffs.push_back({ 3 * faceitr + vertitr, 3 * nbface + nbvert, beta });
                            periodicb[3 * faceitr + vertitr] += -double(soupjumps[nbface]) * 2.0 * M_PI * beta;
                        }
                    }
                }
                else
                {
                    // the new vertices
                    int edgeitr = vertid - uncutVerts;
                    int oldfaceid = faceitr / 4;

                    if (problemFaces.count(oldfaceid))
                    {
                        // new vertex on a previously problem-face

                        int corner = faceitr % 4 - 1;
                        if (corner == -1)
                        {
                            // central triangle

                            int oppface = -1;
                            int oppedge = -1;
                            int oppvert = -1;
                            for (int i = 0; i < 3; i++)
                            {
                                int edge = mesh.faceEdge(oldfaceid, i);
                                int edgeorient = mesh.faceEdgeOrientation(oldfaceid, i);
                                int candopp = mesh.edgeFace(edge, 1 - edgeorient);
                                if (candopp != -1 && !problemFaces.count(candopp))
                                {
                                    oppface = candopp;
                                    oppedge = edge;
                                    oppvert = mesh.edgeOppositeVertex(edge, 1 - edgeorient);
                                }
                            }
                            assert(oppface != -1);
                            if (edgeitr == oppedge)
                            {
                                for (int i = 0; i < 3; i++)
                                {
                                    double alpha = 0;
                                    if (uncutF(oppface, i) == oppvert)
                                    {
                                        alpha = 1.0 / 8.0;
                                    }
                                    else
                                    {
                                        alpha = 7.0 / 16.0;
                                    }
                                    periodicScoeffs.push_back({ 3 * faceitr + vertitr, 3 * oppface + i, alpha });
                                }
                            }
                            else
                            {
                                for (int k = 0; k < 2; k++)
                                {
                                    int target = mesh.edgeVertex(edgeitr, k);
                                    for (int i = 0; i < 3; i++)
                                    {
                                        if (uncutF(oppface, i) == target)
                                        {
                                            periodicScoeffs.push_back({ 3 * faceitr + vertitr, 3 * oppface + i, 1.0 });
                                        }
                                    }
                                }
                            }
                        }
                        else
                        {
                            int vertid = uncutF(oldfaceid, corner);
                            int refface = nonProblemCopy[vertid].first;
                            int refvertidx = nonProblemCopy[vertid].second;
                            std::map<int, int> soupjumps;
                            neighborhoodSoupJumps(mesh, problemFaces, vertexEdges, periodJumps, refface, refvertidx, soupjumps, clusterJumps);

                            int eidx = 0;
                            int oppface = mesh.edgeFace(edgeitr, 0);
                            if (oppface == oldfaceid)
                            {
                                oppface = mesh.edgeFace(edgeitr, 1);
                                eidx = 1;
                            }
                            if (oppface == -1 || problemFaces.count(oppface))
                            {
                                periodicScoeffs.push_back({ 3 * faceitr + vertitr, 3 * refface + refvertidx, 1.0});
                            }
                            else
                            {
                                int opv = mesh.edgeOppositeVertex(edgeitr, eidx);
                                for (int i = 0; i < 3; i++)
                                {
                                    double alpha = 0;
                                    if (uncutF(oppface, i) == opv)
                                    {
                                        alpha = 1.0 / 8.0;
                                    }
                                    else
                                    {
                                        alpha = 7.0 / 16.0;
                                    }
                                    periodicScoeffs.push_back({ 3 * faceitr + vertitr, 3 * oppface + i, alpha });
                                    periodicb[3 * faceitr + vertitr] += -double(soupjumps[oppface]) * 2.0 * M_PI * alpha;
                                }
                            }
                        }
                    }
                    else
                    {
                        int face1 = mesh.edgeFace(edgeitr, 0);
                        int face2 = mesh.edgeFace(edgeitr, 1);
                        int vert1 = mesh.edgeVertex(edgeitr, 0);
                        int vert2 = mesh.edgeVertex(edgeitr, 1);

                        if (face1 == -1 || face2 == -1)
                        {
                            int okface = face1;
                            if (okface == -1)
                                okface = face2;
                            for (int i = 0; i < 3; i++)
                            {
                                if (uncutF(okface, i) == vert1 || uncutF(okface, i) == vert2)
                                {
                                    periodicScoeffs.push_back({ 3 * faceitr + vertitr, 3 * okface + i, 1.0 / 2.0 });
                                }
                            }
                        }
                        else
                        {
                            int face1jmp = 0;
                            int face2jmp = 0;
                            if (face1 == oldfaceid)
                            {
                                face2jmp = periodJumps[edgeitr];
                            }
                            else
                            {
                                face1jmp = -periodJumps[edgeitr];
                            }

                            int oppvert1 = mesh.edgeOppositeVertex(edgeitr, 0);
                            int oppvert2 = mesh.edgeOppositeVertex(edgeitr, 1);
                            double beta = 1.0 / 8.0;
                            int oppvalence = 0;
                            int edgevalence = 0;
                            if (!problemFaces.count(face1))
                            {
                                oppvalence++;
                                edgevalence += 2;
                            }
                            if (!problemFaces.count(face2))
                            {
                                oppvalence++;
                                edgevalence += 2;
                            }
                            double alpha = (1.0 - oppvalence * beta) / double(edgevalence);

                            for (int i = 0; i < 3; i++)
                            {
                                if (!problemFaces.count(face1))
                                {
                                    if (uncutF(face1, i) == oppvert1)
                                    {
                                        periodicScoeffs.push_back({ 3 * faceitr + vertitr, 3 * face1 + i, beta });
                                        periodicb[3 * faceitr + vertitr] += -double(face1jmp) * 2.0 * M_PI * beta;
                                    }
                                    else
                                    {
                                        periodicScoeffs.push_back({ 3 * faceitr + vertitr, 3 * face1 + i, alpha });
                                        periodicb[3 * faceitr + vertitr] += -double(face1jmp) * 2.0 * M_PI * alpha;
                                    }
                                }
                                if (!problemFaces.count(face2))
                                {
                                    if (uncutF(face2, i) == oppvert2)
                                    {
                                        periodicScoeffs.push_back({ 3 * faceitr + vertitr, 3 * face2 + i, beta });
                                        periodicb[3 * faceitr + vertitr] += -double(face2jmp) * 2.0 * M_PI * beta;
                                    }
                                    else
                                    {
                                        periodicScoeffs.push_back({ 3 * faceitr + vertitr, 3 * face2 + i, alpha });
                                        periodicb[3 * faceitr + vertitr] += -double(face2jmp) * 2.0 * M_PI * alpha;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        periodicS.resize(3 * newfaces, 3 * nfaces);
        periodicS.setFromTriplets(periodicScoeffs.begin(), periodicScoeffs.end());
    }

    void wrinkledMeshUpsamplingUncut(const Eigen::MatrixXd &uncutV, const Eigen::MatrixXi &uncutF,
                                     const Eigen::MatrixXd &restV, const Eigen::MatrixXi &restF,
                                     const Eigen::MatrixXd &cutV, const Eigen::MatrixXi &cutF,
                                     const Eigen::VectorXd &cutAmplitude, const Eigen::VectorXd &cutPhi,
                                     const std::set<int> &noPhiFaces,
                                     const std::set<int> &clampedVerts,
                                     Eigen::MatrixXd *wrinkledV, Eigen::MatrixXi *wrinkledF,
                                     Eigen::MatrixXd *upsampledTFTV, Eigen::MatrixXi *upsampledTFTF,
                                     Eigen::MatrixXd *soupPhiV, Eigen::MatrixXi *soupPhiF,
                                     Eigen::MatrixXd *soupProblemV, Eigen::MatrixXi *soupProblemF,
                                     Eigen::VectorXd *upsampledAmp, Eigen::VectorXd *soupPhi,
                                     int numSubdivs,
                                     bool isUseV2Term)
    {
        // fix the input amplitudes
        Eigen::VectorXd fixedCutAmplitude;
        fixedCutAmplitude = cutAmplitude;

        // find corner vertices
        std::set<int> cornerVerts;
        if (clampedVerts.size() == 0)
        {
            findSharpCorners(uncutV, uncutF, cornerVerts);
            std::cout << "Found " << cornerVerts.size() << " sharp corners" << std::endl;
        }
        else
        {
            cornerVerts = clampedVerts;
        }


        // turn amp and phi into a triangle soup
        Eigen::VectorXd uncutsoupamp(3 * uncutF.rows());
        Eigen::VectorXd uncutsoupphi(3 * uncutF.rows());
        for (int i = 0; i < uncutF.rows(); i++)
        {
            for (int j = 0; j < 3; j++)
            {
                uncutsoupamp[3 * i + j] = fixedCutAmplitude[cutF(i, j)];
                uncutsoupphi[3 * i + j] = cutPhi[cutF(i, j)];
            }
        }

        // Fix phi on problem faces
        std::vector<double> canonicalPhi(uncutV.rows());
        for (int i = 0; i < uncutF.rows(); i++)
        {
            if (!noPhiFaces.count(i))
            {
                for (int j = 0; j < 3; j++)
                {
                    canonicalPhi[uncutF(i, j)] = cutPhi[cutF(i, j)];
                }
            }
        }
        for (int i = 0; i < uncutF.rows(); i++)
        {
            if (noPhiFaces.count(i))
            {
                for (int j = 0; j < 3; j++)
                {
                    uncutsoupphi[3 * i + j] = canonicalPhi[uncutF(i, j)];
                }
            }
        }

        std::set<int> problemFaces;
        Eigen::VectorXd soupamp;
        Eigen::VectorXd soupphi;
        //splitIsolatedProblemFaces(uncutV, uncutF, noPhiFaces, uncutsoupamp, uncutsoupphi, NV, NF, problemFaces, soupamp, soupphi);
        Eigen::MatrixXd NV = uncutV;
        Eigen::MatrixXi NF = uncutF;

        problemFaces = noPhiFaces;
        soupamp = uncutsoupamp;
        soupphi = uncutsoupphi;

        Eigen::MatrixXd RV = restV;
        Eigen::MatrixXi RF = restF;

        std::set<int> restCornerVerts;
        for (int i = 0; i < NF.rows(); i++)
        {
            for (int j = 0; j < 3; j++)
            {
                if (cornerVerts.count(NF(i, j)))
                    restCornerVerts.insert(RF(i, j));
            }
        }

        zeroProblemAmplitudes(NF, problemFaces, soupamp);

        // period jumps
        Eigen::VectorXi periodJumps;
        std::map<int, ClusterInfo> clusterJumps;
        computeEdgeJumps(NV.rows(), NF, problemFaces, soupphi, periodJumps, clusterJumps);

        // Subdivide everything

        for(int i=0; i<numSubdivs; ++i)
        {
            Eigen::MatrixXi tempF = NF;
            Eigen::SparseMatrix<double> uncutS;
            Eigen::SparseMatrix<double> soupS;
            Eigen::SparseMatrix<double> periodicS;
            Eigen::SparseMatrix<double> restS;
            Eigen::VectorXd periodicb;
            std::set<int> tempProblemFaces = problemFaces;
            Eigen::MatrixXi tempRF = RF;
            loopWithSeams(NV.rows(), tempF, periodJumps, clusterJumps, tempProblemFaces, cornerVerts,
                          uncutS, soupS, periodicS, periodicb, NF, problemFaces);
            loopUncut(RV.rows(), tempRF, restCornerVerts, restS, RF);

            NV = (uncutS*NV).eval();
            soupamp = (soupS * soupamp).eval();
            soupphi = (periodicS * soupphi).eval() + periodicb;

            RV = (restS * RV).eval();

            zeroProblemAmplitudes(NF, problemFaces, soupamp);
            computeEdgeJumps(NV.rows(), NF, problemFaces, soupphi, periodJumps, clusterJumps);
        }

        Eigen::MatrixXd NN;
        igl::per_vertex_normals(NV, NF, NN);

        int uncutfaces = NF.rows();
        int uncutverts = NV.rows();
        std::vector<std::vector<double> > phivalues(uncutverts);
        std::vector<std::vector<double> > ampvalues(uncutverts);
        std::vector<std::vector<Eigen::Vector3d> > dphivalues(uncutverts);
        std::vector<std::vector<Eigen::Vector3d> > dphiperpvalues(uncutverts);
        MeshConnectivity finalRestMesh(RF);
        MeshConnectivity finalUncutMesh(NF);
        MeshGeometry restGeo(RV, finalRestMesh);

        Eigen::MatrixXd PD1, PD2;
        Eigen::VectorXd PV1, PV2;
        igl::principal_curvature(NV, NF, PD1, PD2, PV1, PV2);


        for (int i = 0; i < uncutfaces; i++)
        {
            if (problemFaces.count(i))
                continue;
            for (int j = 0; j < 3; j++)
            {
                phivalues[NF(i, j)].push_back(soupphi[3 * i + j]);
                ampvalues[NF(i, j)].push_back(soupamp[3 * i + j]);
            }
            Eigen::Matrix<double, 3, 2> dr;
            dr.col(0) = (NV.row(NF(i, 1)) - NV.row(NF(i, 0))).transpose();
            dr.col(1) = (NV.row(NF(i, 2)) - NV.row(NF(i, 0))).transpose();

            Eigen::Matrix2d a = dr.transpose() * dr;

            Eigen::Matrix<double, 3, 2> drbar;
            drbar.col(0) = (RV.row(RF(i, 1)) - RV.row(RF(i, 0))).transpose();
            drbar.col(1) = (RV.row(RF(i, 2)) - RV.row(RF(i, 0))).transpose();

            Eigen::Matrix2d abar = drbar.transpose() * drbar;

            for (int j = 0; j < 3; j++)
            {
                int vert = NF(i, j);
                Eigen::Matrix2d D;
                D << PV1[vert], 0,
                        0, PV2[vert];
                Eigen::Matrix<double,2,3> U;
                U.row(0) = PD1.row(vert);
                U.row(1) = PD2.row(vert);
            }

            double dphi0 = soupphi[3 * i + 1] - soupphi[3 * i + 0];
            double dphi1 = soupphi[3 * i + 2] - soupphi[3 * i + 0];
            Eigen::Vector2d dphi(dphi0, dphi1);
            Eigen::Vector3d extdphi = dr * a.inverse() * dphi;
            for(int j=0; j<3; j++)
                dphivalues[NF(i, j)].push_back(extdphi);

            Eigen::Vector2d u = abar.inverse() * dphi;
            Eigen::Vector2d uperp = restGeo.Js.block<2,2>(2 * i, 0) * u;

            double unormsq = u.transpose() * abar * u;


            Eigen::Vector3d extdphiperp = dr * a.inverse() * abar * uperp;
            if (unormsq < 1e-6)
                extdphiperp.setZero();


            for(int j=0; j<3; j++)
                dphiperpvalues[NF(i, j)].push_back(extdphiperp);
        }


        Eigen::VectorXd finalCosPhi(uncutverts);
        Eigen::VectorXd finalSin2Phi(uncutverts);
        Eigen::VectorXd finalSinPhi(uncutverts);
        Eigen::VectorXd finalAmp(uncutverts);
        Eigen::MatrixXd finalDphi(uncutverts, 3);
        Eigen::MatrixXd finalDphiperp(uncutverts, 3);

        double phivariance = 0;
        double ampvariance = 0;
        for (int i = 0; i < uncutverts; i++)
        {
            double meancosphi = 0;
            double meansin2phi = 0;
            double meansinphi = 0;
            for (auto it : phivalues[i])
            {
                meancosphi += std::cos(it);
                meansin2phi += std::sin(2*it);
                meansinphi += std::sin(it);
            }

            if (phivalues[i].size() > 0)
            {
                meancosphi /= double(phivalues[i].size());
                meansin2phi /= double(phivalues[i].size());
                meansinphi /= double(phivalues[i].size());
            }
            for (auto it : phivalues[i])
            {
                double variance = (std::cos(it) - meancosphi) * (std::cos(it) - meancosphi);
                phivariance += variance;
            }

            Eigen::Vector3d meandphi(0, 0, 0);
            Eigen::Vector3d meandphiperp(0, 0, 0);
            for (auto it : dphivalues[i])
            {
                meandphi += it;
            }
            if (dphivalues[i].size() > 0)
                meandphi /= dphivalues[i].size();
            for (auto it : dphiperpvalues[i])
            {
                meandphiperp += it;
            }
            if (dphivalues[i].size() > 0)
                meandphiperp /= dphiperpvalues[i].size();


            double meanamp = 0;
            for (auto it : ampvalues[i])
                meanamp += it;

            if (ampvalues[i].size() > 0)
                meanamp /= double(ampvalues[i].size());
            for (auto it : ampvalues[i])
                ampvariance += (it - meanamp) * (it - meanamp);

            finalCosPhi[i] = meancosphi;
            finalSin2Phi[i] = meansin2phi;
            finalSinPhi[i] = meansinphi;
            finalAmp[i] = meanamp;
            finalDphi.row(i) = meandphi.transpose();
            finalDphiperp.row(i) = meandphiperp.transpose();
        }
        std::cout << "Variances (should be zero): " << phivariance << " " << ampvariance << std::endl;

        if (upsampledTFTV)
            *upsampledTFTV = NV;
        if (upsampledTFTF)
            *upsampledTFTF = NF;

        int nonzerofaces = NF.rows() - problemFaces.size();
        if (soupPhiV)
            soupPhiV->resize(3 * nonzerofaces, 3);
        if (soupPhiF)
            soupPhiF->resize(nonzerofaces, 3);
        if (soupPhi)
            soupPhi->resize(3 * nonzerofaces);

        if (soupProblemV)
            soupProblemV->resize(3 * problemFaces.size(), 3);
        if (soupProblemF)
            soupProblemF->resize(problemFaces.size(), 3);

        int idx = 0;
        int pidx = 0;
        for (int i = 0; i < NF.rows(); i++)
        {
            if (problemFaces.count(i))
            {
                if (soupProblemV)
                {
                    for (int j = 0; j < 3; j++)
                        soupProblemV->row(3 * pidx + j) = NV.row(NF(i, j));
                }
                if (soupProblemF)
                {
                    for (int j = 0; j < 3; j++)
                    {
                        (*soupProblemF)(pidx, j) = 3 * pidx + j;
                    }
                }
                pidx++;
            }
            else
            {
                if (soupPhiV)
                {
                    for (int j = 0; j < 3; j++)
                    {
                        soupPhiV->row(3 * idx + j) = NV.row(NF(i, j));
                    }
                }
                if (soupPhiF)
                {
                    for (int j = 0; j < 3; j++)
                    {
                        (*soupPhiF)(idx, j) = 3 * idx + j;
                    }
                }
                if (soupPhi)
                {
                    for (int j = 0; j < 3; j++)
                    {
                        (*soupPhi)[3 * idx + j] = soupphi[3 * i + j];
                    }
                }
                idx++;
            }
        }

        for (int i = 0; i < uncutverts; i++)
        {
            NV.row(i) += finalAmp[i] * finalCosPhi[i] * NN.row(i);
            if(isUseV2Term)
            {
                NV.row(i) += finalAmp[i] * finalAmp[i] / 8.0 * finalDphi.row(i) * finalSin2Phi[i];
            }

        }

        if (wrinkledV)
            *wrinkledV = NV;
        if (wrinkledF)
            *wrinkledF = NF;
        if (upsampledAmp)
            *upsampledAmp = finalAmp;
    }

    void getTFWSurfacePerframe(const Eigen::MatrixXd& baseV, const Eigen::MatrixXi& baseF,
                               const Eigen::VectorXd& amp, const Eigen::VectorXd& omega,
                               Eigen::MatrixXd& wrinkledV, Eigen::MatrixXi& wrinkledF,
                               Eigen::MatrixXd& upsampledV, Eigen::MatrixXi& upsampledF,
                               Eigen::MatrixXd& soupPhiV, Eigen::MatrixXi& soupPhiF,
                               Eigen::MatrixXd& soupProblemV, Eigen::MatrixXi& soupProblemF,
                               Eigen::VectorXd& upsampledAmp, Eigen::VectorXd& soupPhi,
                               int numSubdivs, bool isUseV2Term, bool isFixedBnd
    )
    {
        std::vector<Eigen::Matrix2d> abars;
        firstFoundForms(baseV, baseF, abars);
        Eigen::MatrixXd seamedV;
        Eigen::MatrixXi seamedF;
        Eigen::VectorXd phi, seamedPhi, seamedAmp;
        std::set<int> problem_faces;
        roundPhiFromEdgeOmega(baseV, baseF, abars, amp, omega, phi, seamedV, seamedF, seamedAmp, seamedPhi, problem_faces);
        std::set<int> clampledVerts = {};
        if(isFixedBnd)
        {
            Eigen::VectorXi bnd;
            igl::boundary_loop(baseF, bnd);
            for(int i = 0; i < bnd.size(); i++)
                clampledVerts.insert(bnd[i]);
        }
        wrinkledMeshUpsamplingUncut(baseV, baseF, baseV, baseF, seamedV, seamedF, seamedAmp, seamedPhi, problem_faces, clampledVerts, &wrinkledV, &wrinkledF, &upsampledV, &upsampledF, &soupPhiV, &soupPhiF, &soupProblemV, &soupProblemF, &upsampledAmp, &soupPhi, numSubdivs, isUseV2Term);
    }

    void getTFWSurfaces(const Eigen::MatrixXd& baseV, const Eigen::MatrixXi& baseF,
                        const std::vector<Eigen::VectorXd>& ampList, const std::vector<Eigen::VectorXd>& omegaList,
                        std::vector<Eigen::MatrixXd>& wrinkledVList, std::vector<Eigen::MatrixXi>& wrinkledFList,
                        std::vector<Eigen::MatrixXd>& upsampledVList, std::vector<Eigen::MatrixXi>& upsampledFList,
                        std::vector<Eigen::MatrixXd>& soupPhiVList, std::vector<Eigen::MatrixXi>& soupPhiFList,
                        std::vector<Eigen::MatrixXd>& soupProblemVList, std::vector<Eigen::MatrixXi>& soupProblemFList,
                        std::vector<Eigen::VectorXd>& upsampledAmpList, std::vector<Eigen::VectorXd>& soupPhiList,
                        int numSubdivs, bool isUseV2Term, bool isFixedBnd
    )
    {
        int nframes = ampList.size();

        wrinkledVList.resize(nframes);
        wrinkledFList.resize(nframes);
        upsampledVList.resize(nframes);
        upsampledFList.resize(nframes);
        soupPhiVList.resize(nframes);
        soupPhiFList.resize(nframes);
        soupProblemVList.resize(nframes);
        soupProblemFList.resize(nframes);
        upsampledVList.resize(nframes);
        upsampledFList.resize(nframes);

        auto frameUpsampling = [&](const tbb::blocked_range<uint32_t>& range)
        {
            for (uint32_t i = range.begin(); i < range.end(); ++i)
            {
                getTFWSurfacePerframe(baseV, baseF, ampList[i], omegaList[i],
                                      wrinkledVList[i], wrinkledFList[i], upsampledVList[i], upsampledFList[i],
                                      soupPhiVList[i], soupPhiFList[i], soupProblemVList[i], soupProblemFList[i],
                                      upsampledAmpList[i], soupPhiList[i], numSubdivs, isUseV2Term, isFixedBnd
                                      );
            }
        };

        tbb::blocked_range<uint32_t> rangex(0u, (uint32_t)nframes);
        tbb::parallel_for(rangex, frameUpsampling);
    }

}