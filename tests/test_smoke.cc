// 阶段 0：冒烟测试。只检查编译、Eigen 和 GoogleTest 都能用：解一个 3x3 对称正定方程组。
#include <Eigen/Dense>
#include <gtest/gtest.h>

TEST(Smoke, EigenSolves) {
  Eigen::Matrix3d A;
  A << 4, 1, 0,
       1, 3, 1,
       0, 1, 2;
  Eigen::Vector3d b(1, 2, 3);
  Eigen::Vector3d x = A.ldlt().solve(b);  // LDL^T 分解，适合对称矩阵
  EXPECT_LT((A * x - b).norm(), 1e-14);
}
