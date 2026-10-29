#pragma once

#include <vector>

// Stage 1: IPC in one dimension. A ball of mass m falls onto the ground; the state is the gap x between
// the ball's bottom and the ground (m) and the vertical velocity v (m/s). See notes/01_ipc1d.md.
namespace rcipc::ipc1d {

// Defaults are the book's example (section 3.8).
struct Params {
  double m = 1.0;      // mass, kg
  double g = 9.81;     // gravity towards the ground, m/s^2
  double h = 0.002;    // time step, s
  double dhat = 1e-3;  // barrier activation distance, m
  double kappa = 1e4;  // barrier stiffness, N/m
  double c = 0.9;      // CCD: one iteration closes at most this fraction of the gap
  double tol = 1e-6;   // Newton stops when |p| / h < tol, m/s
  int max_newton = 50;
};

// x~ = x0 + h v0 - h^2 g: where the ball would be after one implicit Euler step without the ground.
double predicted(const Params& P, double x0, double v0);

// The incremental potential E(x) = m / (2 h^2) (x - xt)^2 + kappa b(x) and its first two derivatives
// in x, where xt is the predicted position x~. Precondition: x > 0.
double energy(const Params& P, double x, double xt);
double gradient(const Params& P, double x, double xt);
double hessian(const Params& P, double x, double xt);

// CCD in one dimension, as conservative advancement: the largest alpha in [0, 1] such that the gap
// x + s p stays >= (1 - c) x for every s in [0, alpha]. That is 1 when p >= 0 or when the full step
// keeps the margin.
double ccd_max_step(double x, double p, double c);

// One Newton iteration that moved, for the tests and the demo's impact table.
struct Iterate {
  double x;          // gap at the start of the iteration
  double p;          // Newton direction
  double alpha_max;  // CCD bound
  double alpha;      // step accepted by the line search
};

struct StepResult {
  double x;    // gap at the end of the step
  double v;    // (x - x0) / h
  int newton;  // Newton iterations, counting the last one, which found |p| / h < tol
};

// One time step: Newton on E, the CCD bound, backtracking line search (notes/01_ipc1d.md, section 3).
// If trace is not null, every iteration that moved is appended to it.
StepResult step(const Params& P, double x0, double v0, std::vector<Iterate>* trace = nullptr);

}  // namespace rcipc::ipc1d
