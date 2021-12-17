#include <memory>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <igl/jet.h>
#include <igl/gaussian_curvature.h>
#include <igl/principal_curvature.h>
#include <igl/per_face_normals.h>
#include <igl/massmatrix.h>
#include <igl/readOBJ.h>
#include <igl/invert_diag.h>
#include <igl/hsv_to_rgb.h>
#include <igl/boundary_loop.h>
#include <igl/boundary_facets.h>
#include <igl/hsv_to_rgb.h>
#include <igl/adjacency_list.h>


#include "../../include/Visualization/PaintGeometry.h"



Eigen::MatrixXd PaintGeometry::paintPhi(const Eigen::VectorXd& phi, Eigen::VectorXd* brightness)      // brightness should between 0 and 1
{
    int nverts = phi.size();
    // std::cout << phi.minCoeff() << " " << phi.maxCoeff() << std::endl;
    Eigen::MatrixXd color(nverts, 3);
    if (isNormalize)
    {
        igl::jet(phi, true, color);
    }
    else
    {
        for (int i = 0; i < nverts; i++)
        {
            double r, g, b;
            double h = 360.0 * phi[i] / 2.0 / M_PI + 120;
            h = 360 + ((int)h % 360); // fix for libigl bug
            double s = 1.0;
            double v = 0.5;
            if(brightness)
            {
                double r = (*brightness)(i);
                v = r * r / (r * r + 1);
            }
//                v = (*brightness)(i);
            igl::hsv_to_rgb(h, s, v, r, g, b);
            color(i, 0) = r;
            color(i, 1) = g;
            color(i, 2) = b;
        }
    }

    return color;


}

Eigen::MatrixXd PaintGeometry::paintAmplitude(const Eigen::VectorXd& amplitude)
{
    int nverts = amplitude.size();
    Eigen::VectorXd trueAmp = amplitude;

    // std::cout << "amplitude: " << trueAmp.minCoeff() << " " << trueAmp.maxCoeff() << std::endl;

    Eigen::MatrixXd color(nverts, 3);
    igl::jet(trueAmp, isNormalize, color);

    return color;
}