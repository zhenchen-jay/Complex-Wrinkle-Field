#include "../../include/CommonTools.h"
#include "../../include/ComplexLoop/ComplexLoopExtrinsically.h"
#include <iostream>
#include <cassert>
#include <memory>


std::complex<double> ComplexLoopZuenkoExtrinsically::interpZ(const std::vector<std::complex<double>>& zList, const std::vector<Eigen::Vector3d>& gradThetaList, std::vector<double>& coords, const std::vector<Eigen::Vector3d>& pList)
{
    Eigen::Vector3d P = Eigen::Vector3d::Zero();

    int n = pList.size();
    for (int i = 0; i < n; i++)
    {
        P += coords[i] * pList[i];
    }
    Eigen::Vector3d gradThetaAve = Eigen::Vector3d::Zero();
    for (int i = 0; i < n; i++)
    {
        gradThetaAve += coords[i] * gradThetaList[i];
    }

    std::complex<double> z = 0;

    std::complex<double> I = std::complex<double>(0, 1);

    std::vector<double> deltaThetaList(n);
    for (int i = 0; i < n; i++)
    {
        //deltaThetaList[i] = (P - pList[i]).dot(gradThetaList[i]);
        deltaThetaList[i] = (P - pList[i]).dot(gradThetaAve);
    }

    Eigen::VectorXcd dzdalpha;
    dzdalpha.setZero(n);

    for (int i = 0; i < n; i++)
    {
        z += coords[i] * zList[i] * std::complex<double>(std::cos(deltaThetaList[i]), std::sin(deltaThetaList[i]));
    }

    return z;
}

void ComplexLoopZuenkoExtrinsically::computeEdgeOmega2ExtrinsicalGradient(const Eigen::VectorXd& omega, Eigen::MatrixXd& gradTheta)
{
    int V = _mesh.GetVertCount();
    int E = _mesh.GetEdgeCount();
    int F = _mesh.GetFaceCount();
    gradTheta.setZero(V, 3);

    // loop over the faces
    for(int fid = 0; fid < F; fid++)
    {
        for(int vInF = 0; vInF < 3; vInF++)
        {
            int vid = _mesh.GetFaceVerts(fid)[vInF];
            int eid0 = _mesh.GetFaceEdges(fid)[vInF];
            int eid1 = _mesh.GetFaceEdges(fid)[(vInF + 2) % 3];
            Eigen::Vector3d r0 = _mesh.GetVertPos(_mesh.GetEdgeVerts(eid0)[1]) - _mesh.GetVertPos(_mesh.GetEdgeVerts(eid0)[0]);
            Eigen::Vector3d r1 = _mesh.GetVertPos(_mesh.GetEdgeVerts(eid1)[1]) - _mesh.GetVertPos(_mesh.GetEdgeVerts(eid1)[0]);

            Eigen::Matrix2d Iinv, I;
            I << r0.dot(r0), r0.dot(r1), r1.dot(r0), r1.dot(r1);
            Iinv = I.inverse();

            Eigen::Vector2d rhs;
            double w1 = omega(eid0);
            double w2 = omega(eid1);
            rhs << w1, w2;

            Eigen::Vector2d u = Iinv * rhs;
            gradTheta.row(vid) += (u[0] * r0 + u[1] * r1) / (_mesh.GetVertFaces(vid).size());       // simple average
        }
    }

}

void ComplexLoopZuenkoExtrinsically::updateLoopedZvals(const Eigen::VectorXd& omega, const std::vector<std::complex<double>>& zvals, std::vector<std::complex<double>>& upZvals)
{
    int V = _mesh.GetVertCount();
    int E = _mesh.GetEdgeCount();

    upZvals.resize(V + E);
    Eigen::MatrixXd gradTheta;
    computeEdgeOmega2ExtrinsicalGradient(omega, gradTheta);
    // Even verts
    for (int vi = 0; vi < V; ++vi)
    {
        if (_mesh.IsVertBoundary(vi))
        {
            if(_isFixBnd)
                upZvals[vi] = zvals[vi];
            else
            {
                std::vector<int> boundary(2);
                boundary[0] = _mesh.GetVertEdges(vi).front();
                boundary[1] = _mesh.GetVertEdges(vi).back();

                std::vector<std::complex<double>> zp(3);
                std::vector<Eigen::Vector3d> gradthetap(3);
                std::vector<double> coords = { 3. / 4, 1. / 8, 1./ 8 };
                std::vector<Eigen::Vector3d> pList(3);
                pList[0] = _mesh.GetVertPos(vi);
                gradthetap[0] = gradTheta.row(vi);
                zp[0] = zvals[vi];

                for (int j = 0; j < boundary.size(); ++j)
                {
                    int edge = boundary[j];
                    int face = _mesh.GetEdgeFaces(edge)[0];
                    int viInface = _mesh.GetVertIndexInFace(face, vi);

                    int viInEdge = _mesh.GetVertIndexInEdge(edge, vi);
                    int vj = _mesh.GetEdgeVerts(edge)[(viInEdge + 1) % 2];

                    int vjInface = _mesh.GetVertIndexInFace(face, vj);

                    pList[1 + j] = _mesh.GetVertPos(vj);
                    // grad from vi
                    gradthetap[1 + j] = gradTheta.row(vj);
                    zp[1 + j] = zvals[vj];
                }
                upZvals[vi] = interpZ(zp, gradthetap, coords, pList);
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

            std::vector<std::complex<double>> zp(1 + nNeiEdges);
            std::vector<Eigen::Vector3d> gradthetap(1 + nNeiEdges);
            std::vector<double> coords;
            coords.resize(1 + nNeiEdges, alpha);
            std::vector<Eigen::Vector3d> pList(1 + nNeiEdges);

            coords[0] = 1 - nNeiEdges * alpha;
            zp[0] = zvals[vi];
            gradthetap[0] = gradTheta.row(vi);
            pList[0] = _mesh.GetVertPos(vi);

            for (int k = 0; k < nNeiEdges; ++k)
            {
                int edge = _mesh.GetVertEdges(vi)[k];
                int viInedge = _mesh.GetVertIndexInEdge(edge, vi);
                int vj = _mesh.GetEdgeVerts(edge)[1 - viInedge];

                zp[1 + k] = zvals[vj];
                gradthetap[1 + k] = gradTheta.row(vj);
                pList[1 + k] = _mesh.GetVertPos(vj);
            }

            upZvals[vi] = interpZ(zp, gradthetap, coords, pList);
        }
    }

    // Odd verts
    for (int edge = 0; edge < E; ++edge)
    {
        int row = edge + V;
        if (_mesh.IsEdgeBoundary(edge))
        {
            std::vector<std::complex<double>> zp(2);
            std::vector<Eigen::Vector3d> gradthetap(2);
            std::vector<double> coords = { 1. / 2, 1. / 2 };
            std::vector<Eigen::Vector3d> pList(2);

            for(int j = 0; j < 2; j++)
            {
                int vj = _mesh.GetEdgeVerts(edge)[j];
                zp[j] = zvals[vj];
                gradthetap[j] = gradTheta.row(vj);
                pList[j] = _mesh.GetVertPos(vj);
            }
            upZvals[row] = interpZ(zp, gradthetap, coords, pList);
        }
        else
        {

            std::vector<std::complex<double>> zp(4);
            std::vector<Eigen::Vector3d> gradthetap(4);
            std::vector<double> coords = { 3. / 8, 3. / 8, 1. / 8, 1. / 8 };
            std::vector<Eigen::Vector3d> pList(4);

            for (int j = 0; j < 2; ++j)
            {
                int vj = _mesh.GetEdgeVerts(edge)[j];
                zp[j] = zvals[vj];
                gradthetap[j] = gradTheta.row(vj);
                pList[j] = _mesh.GetVertPos(vj);

                int face = _mesh.GetEdgeFaces(edge)[j];
                int vjOpp = -1;
                for(int k = 0; k < 3; k++)
                {
                    if(_mesh.GetFaceVerts(face)[k] != _mesh.GetEdgeVerts(edge)[0] && _mesh.GetFaceVerts(face)[k] != _mesh.GetEdgeVerts(edge)[1])
                        vjOpp = _mesh.GetFaceVerts(face)[k];
                }

                zp[2 + j] = zvals[vjOpp];
                gradthetap[2 + j] = gradTheta.row(vjOpp);
                pList[2 + j] = _mesh.GetVertPos(vjOpp);

            }

            upZvals[row] = interpZ(zp, gradthetap, coords, pList);
        }
    }
}

void ComplexLoopZuenkoExtrinsically::Subdivide(const Eigen::VectorXd& omega, const std::vector<std::complex<double>>& zvals, Eigen::VectorXd& omegaNew, std::vector<std::complex<double>>& upZvals, int level)
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
