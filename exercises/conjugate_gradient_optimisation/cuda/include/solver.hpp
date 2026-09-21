#pragma once

#include "precision.hpp"

void matvec(real *y, const real *A, const real *x, const int n);
real dot(const real *a, const real *b, int n);
void axpby(real *y, const real *x, real alpha, real beta, int n);
int cg_solve(real *x, const real *A, const real *b, const int n, int max_iter);
