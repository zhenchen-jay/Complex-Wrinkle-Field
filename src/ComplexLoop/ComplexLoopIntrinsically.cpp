#include "../../include/CommonTools.h"
#include "../../include/ComplexLoop/ComplexLoopIntrinsically.h"
#include <iostream>
#include <cassert>
#include <memory>

std::complex<double> ComplexLoopIntrinsically::computeBarycentricZval(const Eigen::VectorXd& omega, const std::vector<std::complex<double>>& zvals, int fid, const Eigen::Vector3d& bary)
{
    Eigen::Matrix3d gradBary;
    Eigen::Vector3d P0 = _mesh.GetVertPos(_mesh.GetFaceVerts(fid)[0]);
    Eigen::Vector3d P1 = _mesh.GetVertPos(_mesh.GetFaceVerts(fid)[1]);
    Eigen::Vector3d P2 = _mesh.GetVertPos(_mesh.GetFaceVerts(fid)[2]);

    computeBaryGradient(P0, P1, P2, bary, gradBary);

    std::complex<double> z = 0;

    std::complex<double> tau = std::complex<double>(0, 1);

    for (int i = 0; i < 3; i++)
    {
        double alphai = bary(i);
        int vid = _mesh.GetFaceVerts(fid)[i];
        int eid0 = _mesh.GetFaceEdges(fid)[i];
        int eid1 = _mesh.GetFaceEdges(fid)[(i + 2) % 3];

        double w1 = omega(eid0);
        double w2 = omega(eid1);

        if (vid == _mesh.GetEdgeVerts(eid0)[1])
            w1 *= -1;
        if (vid == _mesh.GetEdgeVerts(eid1)[1])
            w2 *= -1;

        double deltaTheta = w1 * bary((i + 1) % 3) + w2 * bary((i + 2) % 3);
        std::complex<double> expip = std::complex<double>(std::cos(deltaTheta), std::sin(deltaTheta));

        z += alphai * zvals[vid] * expip;
    }
    return z;
}

std::complex<double> ComplexLoopIntrinsically::computeZvalAtPoint(const Eigen::VectorXd& omega, const std::vector<std::complex<double>>& zvals, int fid, const Eigen::Vector3d& p, int startviInFace)
{
    int vi = _mesh.GetFaceVerts(fid)[startviInFace];
    Eigen::Vector3d r0 = _mesh.GetVertPos(_mesh.GetFaceVerts(fid)[(startviInFace + 1) % 3]) - _mesh.GetVertPos(vi);
    Eigen::Vector3d r1 = _mesh.GetVertPos(_mesh.GetFaceVerts(fid)[(startviInFace + 2) % 3]) - _mesh.GetVertPos(vi);

    Eigen::Matrix2d Iinv, I;
    I << r0.dot(r0), r0.dot(r1), r1.dot(r0), r1.dot(r1);
    Iinv = I.inverse();

    if(I.determinant() < 1e-15)
    {
        return (zvals[vi] + zvals[_mesh.GetFaceVerts(fid)[(startviInFace + 1) % 3]] +
                        zvals[_mesh.GetFaceVerts(fid)[(startviInFace + 2) % 3]]) / 3.0;
    }
    Eigen::Vector2d rhs;
    rhs << (p - _mesh.GetVertPos(vi)).dot(r0),  (p - _mesh.GetVertPos(vi)).dot(r1);

    Eigen::Vector2d sol = Iinv * rhs;
    Eigen::Vector3d bary = Eigen::Vector3d::Zero();
    bary[startviInFace] = 1 - sol[0] - sol[1];
    bary[(startviInFace + 1) % 3] = sol[0];
    bary[(startviInFace + 2) % 3] = sol[1];

    return computeBarycentricZval(omega, zvals, fid, bary);
}

void ComplexLoopIntrinsically::updateLoopedZvals(const Eigen::VectorXd& omega, const std::vector<std::complex<double>>& zvals, std::vector<std::complex<double>>& upZvals)
{
    int V = _mesh.GetVertCount();
    int E = _mesh.GetEdgeCount();

    upZvals.resize(V + E);
    // Even verts
    for (int vi = 0; vi < V; ++vi)
    {
        if (_mesh.IsVertBoundary(vi))
        {
            if(_isFixBnd)
                upZvals[vi] = zvals[vi];
            else
            {
                // the new Loop vertex is 3/4 V0 + 1/8 V1 + 1 / 8 V2
                // We projected this vertex to the boundary edges, and compute the contribution and average back
                std::vector<int> boundary(2);
                boundary[0] = _mesh.GetVertEdges(vi).front();
                boundary[1] = _mesh.GetVertEdges(vi).back();
                std::vector<int> adjverts(2);



                for (int j = 0; j < boundary.size(); ++j)
                {
                    int edge = boundary[j];
                    int viInEdge = _mesh.GetVertIndexInEdge(edge, vi);
                    int vj = _mesh.GetEdgeVerts(edge)[(viInEdge + 1) % 2];
                    adjverts[j] = vj;
                }

                upZvals[vi] = 0;
                for(int j = 0; j < boundary.size(); ++j)
                {
                    double norm = (_mesh.GetVertPos(adjverts[j]) - _mesh.GetVertPos(vi)).norm();
                    if(norm < 1e-15)    // numerical error
                    {
                        upZvals[vi] += (zvals[vi] + zvals[adjverts[j]]) / 2.0;
                        continue;
                    }
                    double w = 1. / 8 * (norm +  (_mesh.GetVertPos(adjverts[(j + 1) % 2]) - _mesh.GetVertPos(vi)).dot(_mesh.GetVertPos(adjverts[j]) - _mesh.GetVertPos(vi)) / norm);
                    Eigen::Vector3d bary;
                    bary << 0, 0, 0;

                    int edge = boundary[j];
                    int face = _mesh.GetEdgeFaces(edge)[0];
                    int viInface = _mesh.GetVertIndexInFace(face, vi);
                    int vjInface = _mesh.GetVertIndexInFace(face, adjverts[j]);

                    bary[viInface] = 1 - w;
                    bary[vjInface] = w;
                    upZvals[vi] += computeBarycentricZval(omega, zvals, face, bary);
                }
                upZvals[vi] /= 2.0;
            }


        }
        else
        {
            const std::vector<int>& vEdges = _mesh.GetVertEdges(vi);
            int nNeiEdges = vEdges.size();

            // Fig5 left [Wang et al. 2006]
            Scalar alpha = 0.375;
            if (nNeiEdges == 3) alpha /= 2;
            else                    alpha /= nNeiEdges;

            // new vertex is (1 - n alpha) V0 + \sum alpha Vi
            Eigen::Vector3d p = _mesh.GetVertPos(vi) * (1 - nNeiEdges * alpha);
            for (int k = 0; k < nNeiEdges; ++k)
            {
                int edge = _mesh.GetVertEdges(vi)[k];
                int viInedge = _mesh.GetVertIndexInEdge(edge, vi);
                int vj = _mesh.GetEdgeVerts(edge)[1 - viInedge];

                p += alpha * _mesh.GetVertPos(vj);
            }

            // compute the projection on each tangent face
            upZvals[vi] = 0;
            for(int k = 0; k < nNeiEdges; ++k)
            {
                int face = _mesh.GetVertFaces(vi)[k];
                int viInface = _mesh.GetVertIndexInFace(face, vi);
                upZvals[vi] += computeZvalAtPoint(omega, zvals, face, p, viInface);
            }

            upZvals[vi] *= 1.0 / nNeiEdges;
        }
    }

    // Odd verts
    for (int edge = 0; edge < E; ++edge)
    {
        int row = edge + V;
        if (_mesh.IsEdgeBoundary(edge))
        {
            int face = _mesh.GetEdgeFaces(edge)[0];
            int eindexFace = _mesh.GetEdgeIndexInFace(face, edge);
            Eigen::Vector3d bary;
            bary.setConstant(0.5);
            bary((eindexFace + 2) % 3) = 0;

            upZvals[row] = computeBarycentricZval(omega, zvals, face, bary);
        }
        else
        {
            // new pos  is 3/8 (V0 + V1) + 1 / 8 (V2 + V3)
            Eigen::Vector3d p = 3. / 8. * (_mesh.GetVertPos(_mesh.GetEdgeVerts(edge)[0]) + _mesh.GetVertPos(_mesh.GetEdgeVerts(edge)[1]));
            std::vector<int> startId(2);

            for (int j = 0; j < 2; ++j)
            {
                int face = _mesh.GetEdgeFaces(edge)[j];
                int vjOpp = -1;
                for(int k = 0; k < 3; k++)
                {
                    if(_mesh.GetFaceVerts(face)[k] != _mesh.GetEdgeVerts(edge)[0] && _mesh.GetFaceVerts(face)[k] != _mesh.GetEdgeVerts(edge)[1])
                    {
                        startId[j] = k;
                        vjOpp = _mesh.GetFaceVerts(face)[k];
                    }

                }

                p += 1. / 8. * _mesh.GetVertPos(vjOpp);
            }

            upZvals[row] = 0;
            for(int j = 0; j < 2; j++)
            {
                int face = _mesh.GetEdgeFaces(edge)[j];
                upZvals[row] += computeZvalAtPoint(omega, zvals, face, p, startId[j]);
            }
            upZvals[row] /= 2.0;
        }
    }
}

void ComplexLoopIntrinsically::Subdivide(const Eigen::VectorXd& omega, const std::vector<std::complex<double>>& zvals, Eigen::VectorXd& omegaNew, std::vector<std::complex<double>>& upZvals, int level)
{

    int nverts = _mesh.GetVertCount();
    omegaNew = omega;
    upZvals = zvals;


    MatrixX X;
    _mesh.GetPos(X);

    Eigen::VectorXd amp(nverts);

    for (int i = 0; i < nverts; i++)
    {
        amp(i) = std::abs(zvals[i]);
    }

    for (int l = 0; l < level; ++l)
    {
        SparseMatrixX tmpS0, tmpS1;
        BuildS0(tmpS0);
        BuildS1(tmpS1);

        X = tmpS0 * X;
        amp = tmpS0 * amp;

        std::vector<std::complex<double>> upZvalsNew;
        //std::vector<Eigen::Vector3cd> upGradZvals;

        updateLoopedZvals(omegaNew, upZvals, upZvalsNew);

        //updateLoopedZvalsNew(meshNew, omegaNew, upZvals, upZvalsNew);
        omegaNew = tmpS1 * omegaNew;

        std::vector<Vector3> points;
        ConvertToVector3(X, points);

        std::vector< std::vector<int> > edgeToVert;
        GetSubdividedEdges(edgeToVert);

        std::vector< std::vector<int> > faceToVert;
        GetSubdividedFaces(faceToVert);

        _mesh.Populate(points, faceToVert, edgeToVert);

        upZvals.swap(upZvalsNew);
    }
}
