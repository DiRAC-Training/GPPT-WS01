#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <thrust/copy.h>
#include <thrust/device_vector.h>

using std::vector;

#define PRECISION_SINGLE

#ifdef PRECISION_SINGLE
typedef float real;
#else
typedef double real;
#endif

struct Vec2 {
  real x;
  real y;
};

__device__ __host__ real norm2(const Vec2 &v) { return v.x * v.x + v.y * v.y; }

__device__ __host__ Vec2 sub(const Vec2 &v1, const Vec2 &v2) { return {v1.x - v2.x, v1.y - v2.y}; }

#include "util.hpp" // This must be included *after* Vec2 definition

__device__ __host__ Vec2 calc_acc_pair(Vec2 pi, Vec2 pj, real mj, real eps = 0.0) {
  const Vec2 r = sub(pj, pi);
  const real d = norm2(r) + eps * eps;
#ifdef PRECISION_SINGLE
  const real inv_d = rsqrtf(d); // only when real = float and not double
#else
  const real inv_d = 1.0 / sqrt(d); // the fallback when real = double
#endif
  const real accs = mj * inv_d * inv_d * inv_d;
  return {r.x * accs, r.y * accs};
}

void calc_acc(vector<Vec2> &acc, const vector<Vec2> &pos,
              const vector<real> &mass, real eps = 0.0) {
  for (int i = 0; i < acc.size(); ++i) {
    Vec2 accl = {0.0, 0.0};
    const Vec2 pi = pos[i];

    for (int j = 0; j < acc.size(); ++j) {
      const Vec2 accll = calc_acc_pair(pi, pos[j], mass[j], eps);
      accl.x += accll.x;
      accl.y += accll.y;
    }

    acc[i] = accl;
  }
}

__global__ void calc_acc_k(Vec2 *acc, const Vec2 *pos, const real *mass, uint N,
                         real eps = 0.0) {
  const int gtid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gtid >= N)
    return;

  const Vec2 pi = pos[gtid];
  Vec2 accl = {0.0, 0.0};

  for (int j = 0; j < N; ++j) {
    const Vec2 accll = calc_acc_pair(pi, pos[j], mass[j], eps);
    accl.x += accll.x;
    accl.y += accll.y;
  }

  acc[gtid] = accl;
}

void advance_pos(vector<Vec2> &pos, vector<Vec2> &pos_prev,
                 const vector<Vec2> &acc, real dt) {
  for (int i = 0; i < pos.size(); ++i) {
    pos_prev[i].x = 2.0 * pos[i].x - pos_prev[i].x + acc[i].x * dt * dt;
    pos_prev[i].y = 2.0 * pos[i].y - pos_prev[i].y + acc[i].y * dt * dt;
  }
}

__global__ void advance_pos_k(Vec2 *pos_prev, const Vec2 *pos, const Vec2 *acc,
                            uint N, real dt) {
  const int gtid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gtid >= N)
    return;
  pos_prev[gtid].x =
      2.0 * pos[gtid].x - pos_prev[gtid].x + acc[gtid].x * dt * dt;
  pos_prev[gtid].y =
      2.0 * pos[gtid].y - pos_prev[gtid].y + acc[gtid].y * dt * dt;
}

bool all_tests_pass();

int main(int argc, char *argv[]) {

  // Parameters
  const int seed = get_argval<int>(argv, argv + argc, "--seed", 42);
  const uint N_PARTICLES = get_argval<uint>(argv, argv + argc, "-n", 256);
  const real dt = get_argval<real>(argv, argv + argc, "--dt", 0.01);
  const real total_time =
      get_argval<real>(argv, argv + argc, "--total_time", 0.1);
  const bool dump_data = get_arg(argv, argv + argc, "--dump");
  const bool quiet = get_arg(argv, argv + argc, "--quiet");
  const bool disable_unit_tests =
      get_arg(argv, argv + argc, "--disable_unit_tests");
  const bool only_unit_tests =
      get_arg(argv, argv + argc, "--only_unit_tests");
  const real time_between_dumps = total_time/10;
  const real time_between_reports = total_time/100;

  const int block_size = 256;
  const int n_blocks = (N_PARTICLES + block_size - 1) / block_size;

  // Check tests
  bool tests_passed = true;
  if (!disable_unit_tests) {
    tests_passed = all_tests_pass();
    if (!tests_passed)
      return -1;

    if(tests_passed && only_unit_tests) {
      std::cout << "All tests passed!\n";
      return 0;
    }
  }

  // This prevents numerical errors when two particles are very close
  const real epsilon = 1.1 * std::pow(real(N_PARTICLES), -0.48);

  // Main variables
  vector<Vec2> pos(N_PARTICLES);
  vector<Vec2> acc(N_PARTICLES);
  vector<Vec2> pos_prev(N_PARTICLES);
  vector<real> mass(N_PARTICLES);

  // Setup initial conditions
  {
    vector<Vec2> vel(N_PARTICLES);
    vector<real> r(N_PARTICLES);
    vector<real> theta(N_PARTICLES);

    fill_rand_vec(r.data(), r.size(), 15, 20.0, seed);
    fill_rand_vec(theta.data(), theta.size(), 0, 2.0 * M_PI, seed + 1);
    fill_rand_vec(mass.data(), mass.size(), 1.0 / 6000000, 1.0 / 1000,
                  seed + 2);

    calc_stable_orbit(pos, vel, r, theta);

    pos[0] = {0, 0};
    vel[0] = {0, 0};
    mass[0] = 1.0;

    calc_acc(acc, pos, mass, epsilon);

    for (int i = 0; i < pos.size(); ++i) {
      pos_prev[i].x = pos[i].x - vel[i].x * dt - 0.5 * acc[i].x * dt * dt;
      pos_prev[i].y = pos[i].y - vel[i].y * dt - 0.5 * acc[i].y * dt * dt;
    }
  }

  thrust::device_vector<Vec2> pos_d(N_PARTICLES);
  thrust::device_vector<Vec2> acc_d(N_PARTICLES);
  thrust::device_vector<Vec2> pos_prev_d(N_PARTICLES);
  thrust::device_vector<real> mass_d(N_PARTICLES);

  thrust::copy(pos.begin(), pos.end(), pos_d.begin());
  thrust::copy(mass.begin(), mass.end(), mass_d.begin());
  thrust::copy(pos_prev.begin(), pos_prev.end(), pos_prev_d.begin());
  thrust::copy(acc.begin(), acc.end(), acc_d.begin());

  // Timers & counters
  real t = 0; // simulation time
  int loop_counter = 0;
  int dump_counter = 0;
  real time_per_loop = 0.0;
  real total_elapsed_us = 0.0;
  real time_to_next_dump = 0.0;
  real time_to_next_report = 0.0;

  Timer<std::chrono::microseconds> timer;

  while (t < total_time) {
    calc_acc_k<<<n_blocks, block_size>>>(acc_d.data().get(), pos_d.data().get(),
                                       mass_d.data().get(), N_PARTICLES,
                                       epsilon);
    advance_pos_k<<<n_blocks, block_size>>>(pos_prev_d.data().get(),
                                          pos_d.data().get(),
                                          acc_d.data().get(), N_PARTICLES, dt);
    pos_d.swap(pos_prev_d);

    cudaDeviceSynchronize();

    time_per_loop = timer.lap();
    total_elapsed_us += time_per_loop;

    t += dt;
    loop_counter += 1;

    if (t > time_to_next_dump and dump_data) {
      time_to_next_dump += time_between_dumps;
      thrust::copy(pos_d.begin(), pos_d.end(), pos.begin());
      dump_to_file(format_fname(dump_counter), pos);
      dump_counter += 1;
    }

    if (t > time_to_next_report and !quiet) {
      time_to_next_report += time_between_reports;

      const std::string ESC = "\x1b";
      const std::string CLEAR_SCREEN = ESC + "[2J";
      const std::string JUMP_HOME = ESC + "[H";
      std::cout << CLEAR_SCREEN << JUMP_HOME;

      if (tests_passed and !disable_unit_tests) {
        std::cout << "ALL TESTS PASSED\n";
      }
      std::cout << "N_PARTICLES: " << N_PARTICLES << "\n";
      std::cout << "Loop time: " << time_per_loop << " us\n";
      const real pc_remaining = t / total_time * 100;
      printf("Sim time remaining: %.2f (%.2f \\%)\n", total_time - t,
             pc_remaining);
      const int n_loops_remaining = (total_time - t) / dt;
      const real realtime_remaining = n_loops_remaining * time_per_loop;
      printf("Real time remaining: %.2f s\n", realtime_remaining / 1e6);
    }
  }

  thrust::copy(pos_d.begin(), pos_d.end(), pos.begin());
  dump_to_file("final.csv", pos);
  std::cout << "N_PARTICLES: " << N_PARTICLES << "\n";
  std::cout << "Steps: " << loop_counter << "\n";
  std::cout << "Total: " << total_elapsed_us << " us\n";
  std::cout << "Mean per step: " << total_elapsed_us / loop_counter << " us\n";
}

// TESTS

bool assert_nearly_eql(real x, real y, real eps = 1e-6) {
  if (std::fabs(x - y) < eps) {
    return true;
  } else {
    std::cout << x << " != " << y << "\n";
    return false;
  }
}

bool assert_nearly_eql(const Vec2 &v1, const Vec2 &v2, real eps = 1e-6) {
  return assert_nearly_eql(v1.x, v2.x, eps) &&
         assert_nearly_eql(v1.y, v2.y, eps);
}

bool test_calc_acc_x() {
  thrust::device_vector<real> mass({2.0, 0.5});
  thrust::device_vector<Vec2> pos({{0.0, 0.0}, {1.0, 0.0}});
  thrust::device_vector<Vec2> acc(pos.size());

  const real epsilon = 1.1 * std::pow(real(pos.size()), -0.48);
  const real e2 = std::pow(epsilon, 2);
  const real inv_d3 = std::pow(1 + e2, -1.5);

  calc_acc_k<<<1, 32>>>(acc.data().get(), pos.data().get(), mass.data().get(),
                      pos.size(), epsilon);
  cudaDeviceSynchronize();

  if (!assert_nearly_eql(acc[0], {mass[1] * inv_d3, 0}, 1e-5)) {
    return false;
  }

  if (!assert_nearly_eql(acc[1], {-mass[0] * inv_d3, 0}, 1e-5)) {
    return false;
  }

  return true;
}

bool test_calc_acc_y() {
  thrust::device_vector<real> mass({2.0, 0.5});
  thrust::device_vector<Vec2> pos({{0.0, 0.0}, {0.0, 1.0}});
  thrust::device_vector<Vec2> acc(pos.size());

  const real epsilon = 1.1 * std::pow(real(pos.size()), -0.48);
  const real e2 = std::pow(epsilon, 2);
  const real inv_d3 = std::pow(1 + e2, -1.5);

  calc_acc_k<<<1, 32>>>(acc.data().get(), pos.data().get(), mass.data().get(),
                      pos.size(), epsilon);
  cudaDeviceSynchronize();

  if (!assert_nearly_eql(acc[0], {0, mass[1] * inv_d3}, 1e-5)) {
    return false;
  }

  if (!assert_nearly_eql(acc[1], {0, -mass[0] * inv_d3}, 1e-5)) {
    return false;
  }

  return true;
}

bool test_advance_pos() {
  const real dt = 0.5;

  thrust::device_vector<Vec2> pos({{1.0, 2.0}});
  thrust::device_vector<Vec2> pos_prev({{0.5, 3.0}});
  thrust::device_vector<Vec2> acc({{0.5, -1.0}});

  advance_pos_k<<<1, 32>>>(pos_prev.data().get(), pos.data().get(),
                        acc.data().get(), pos.size(), dt);
  cudaDeviceSynchronize();
  pos.swap(pos_prev);

  return assert_nearly_eql(
      pos[0], {2.0 - 0.5 + 0.5 * dt * dt, 4.0 - 3.0 + (-1.0) * dt * dt});
}

bool all_tests_pass() {
  bool all_tests_passed = true;
  if (!test_advance_pos()) {
    std::cout << "test_advance_pos failed!\n";
    all_tests_passed = false;
  }

  if (!test_calc_acc_x()) {
    std::cout << "test_calc_acc_x failed!\n";
    all_tests_passed = false;
  }

  if (!test_calc_acc_y()) {
    std::cout << "test_calc_acc_y failed!\n";
    all_tests_passed = false;
  }

  return all_tests_passed;
}
