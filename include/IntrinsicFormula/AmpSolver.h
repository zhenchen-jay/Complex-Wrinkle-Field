
#ifndef AMPSOLVER_H
#define AMPSOLVER_H

#include <Eigen/Core>
#include "../MeshLib/MeshConnectivity.h"

/*
 * Recovers amplitude given frequency, by solving
 * \min_{a, \theta} || d[a e^{i\theta}] ||^2   s.t.  d arg(theta) = w
 * Inputs:
 *  - omegas: size |E| x 2 matrix of frequencies. Row i are the two halfedge frequency values for edge i in mesh.
 * Output:
 *  - amplitudes: |V| vector of amplitudes. Not necessarily positive.
 */
void ampSolver(const Eigen::MatrixXd &V, const MeshConnectivity& mesh, const Eigen::MatrixXd& omegas, Eigen::VectorXd& amplitudes);

#endif