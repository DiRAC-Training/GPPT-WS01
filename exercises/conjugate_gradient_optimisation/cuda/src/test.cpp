#include <cmath>
#include <iostream>
#include <vector>

#include "idx.hpp"
#include "solver.hpp"
#include "test.hpp"

bool nearly_eql(real x, real y, real eps = 1e-6) {
  if (std::fabs(x - y) < eps) {
    return true;
  } else {
    std::cout << x << " != " << y << "\n";
    return false;
  }
}

/// Test matvec with an identity matrix
bool test_matvec_identity() {
  const int n = 3;

  real A[n * n] = {0};
  for (int i = 0; i < n; ++i) {
    A[idx(i, i, n)] = 1.0;
  }
  real x[3] = {1.0, 2.0, 3.0};
  real y[3];

  matvec(y, A, x, n);

  // Because A is the identity matrix, y and x should be identical
  bool passed = true;
  for (int i = 0; i < n; ++i) {
    passed &= nearly_eql(x[i], y[i], 1e-6);
  }

  return passed;
}

bool test_matvec_simple() {
  const int n = 3;

  // Create a matrix with known values
  // clang-format off
  real A[9] = {
    -1, 4, 0,
    4, 3, -100,
    0, -100, 1
  };
  // clang-format on

  // x and y_soln are calculated solutions to y = Ax.
  real x[3] = {-1.0, 2.0, 0.0};
  real y_soln[3] = {9, 2, -200};

  // Calculate y with our matvec test
  real y[3];
  matvec(y, A, x, n);

  // Compare y to y_soln
  bool passed = true;
  for (int i = 0; i < n; ++i) {
    passed &= nearly_eql(y_soln[i], y[i], 1e-6);
  }

  return passed;
}

bool test_dot() {
  const int n = 1024;

  // Generate x and y and calculate their dot product in this loop
  auto x = std::vector<real>(n);
  auto y = std::vector<real>(n);
  real soln = 0.0;
  for (int i = 0; i < n; ++i) {
    const real p = real(i) / real(n);
    x[i] = p;
    y[i] = p;
    soln += p * p;
  }

  // Calculate dot with the function and test against above value
  const real *xp = x.data();
  const real *yp = y.data();
  real res = dot(xp, yp, n);

  return nearly_eql(res, soln, 1e-3);
}
