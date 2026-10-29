// Stage 0: the build, Eigen and GoogleTest work.
#include <Eigen/Dense>
#include <gtest/gtest.h>

TEST(Smoke, EigenSolves) {
  Eigen::Matrix3d A;
  A << 4, 1, 0,
       1, 3, 1,
       0, 1, 2;
  Eigen::Vector3d b(1, 2, 3);
  Eigen::Vector3d x = A.ldlt().solve(b);
  EXPECT_LT((A * x - b).norm(), 1e-14);
}
