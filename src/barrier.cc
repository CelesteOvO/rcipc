#include "rcipc/barrier.h"

#include <cmath>
#include <stdexcept>

namespace rcipc {

BarrierValue barrier(double x, double xhat) {
  // TODO(stage 1): return {b, b', b''}; all three are 0 for x >= xhat.
  // Formulas and their derivation: notes/01_ipc1d.md section 2, book section 3.2 (p. 48).
  (void)x;
  (void)xhat;
  throw std::logic_error("not implemented: rcipc::barrier (stage 1)");
}

}  // namespace rcipc
