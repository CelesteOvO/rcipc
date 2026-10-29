#include "ipc1d/ipc1d.h"

#include <cmath>
#include <stdexcept>

#include "rcipc/barrier.h"

namespace rcipc::ipc1d {

// Every TODO below is a few lines. Suggested order: predicted, energy, gradient, hessian
// (test_ipc1d's Energy* tests), ccd_max_step (Ccd*), then step (Step*, Drop*).

double predicted(const Params& P, double x0, double v0) {
  // TODO(stage 1): notes/01_ipc1d.md section 1.
  (void)P;
  (void)x0;
  (void)v0;
  throw std::logic_error("not implemented: ipc1d::predicted (stage 1)");
}

double energy(const Params& P, double x, double xt) {
  // TODO(stage 1): notes/01_ipc1d.md section 1.
  (void)P;
  (void)x;
  (void)xt;
  throw std::logic_error("not implemented: ipc1d::energy (stage 1)");
}

double gradient(const Params& P, double x, double xt) {
  // TODO(stage 1): dE/dx.
  (void)P;
  (void)x;
  (void)xt;
  throw std::logic_error("not implemented: ipc1d::gradient (stage 1)");
}

double hessian(const Params& P, double x, double xt) {
  // TODO(stage 1): d^2E/dx^2.
  (void)P;
  (void)x;
  (void)xt;
  throw std::logic_error("not implemented: ipc1d::hessian (stage 1)");
}

double ccd_max_step(double x, double p, double c) {
  // TODO(stage 1): notes/01_ipc1d.md section 3.2.
  (void)x;
  (void)p;
  (void)c;
  throw std::logic_error("not implemented: ipc1d::ccd_max_step (stage 1)");
}

StepResult step(const Params& P, double x0, double v0, std::vector<Iterate>* trace) {
  // TODO(stage 1): the loop in notes/01_ipc1d.md section 3.4.
  (void)P;
  (void)x0;
  (void)v0;
  (void)trace;
  throw std::logic_error("not implemented: ipc1d::step (stage 1)");
}

}  // namespace rcipc::ipc1d
