#pragma once

namespace rcipc {

// IPC's clamped log barrier and its first two derivatives at one argument.
struct BarrierValue {
  double b = 0;    // b(x)
  double db = 0;   // b'(x)
  double ddb = 0;  // b''(x)
};

// b(x) = -(x - xhat)^2 ln(x / xhat) for 0 < x < xhat, and 0 for x >= xhat (notes/01_ipc1d.md, section 2).
//
// Stage 1 calls it with a distance and the activation distance, barrier(d, dhat). From stage 4 on the
// argument is a squared distance, barrier(d * d, dhat * dhat), as in the old code; the function is the same.
// Precondition: x > 0 (the caller keeps every iterate intersection-free).
BarrierValue barrier(double x, double xhat);

}  // namespace rcipc
