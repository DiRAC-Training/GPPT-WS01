#pragma once

// Special characters for manipulating terminal
const std::string ESC = "\x1b";
const std::string CLEAR_SCREEN = ESC + "[2J";
const std::string JUMP_HOME = ESC + "[H";

// Easy argument handling
#include <sstream>
#include <algorithm>

template <typename T>
T get_argval(char **begin, char **end, const std::string &arg,
             const T default_val) {
  T argval = default_val;
  char **itr = std::find(begin, end, arg);
  if (itr != end && ++itr != end) {
    std::istringstream inbuf(*itr);
    inbuf >> argval;
  }
  return argval;
}

bool get_arg(char **begin, char **end, const std::string &arg) {
  char **itr = std::find(begin, end, arg);
  if (itr != end) {
    return true;
  }
  return false;
}

/*
Use:

int main(int argc, char* argv[]) {
    const int iter_max = get_argval<int>(argv, argv + argc, "-niter", 1000);
    const int nx = get_argval<int>(argv, argv + argc, "-nx", 16384);
    const bool csv = get_arg(argv, argv + argc, "-csv");
*/

// Timer
#include <chrono>

template <typename TimeUnit> class Timer {
private:
  // Type aliases to make accessing nested type easier
  using Clock = std::chrono::steady_clock;
  std::chrono::time_point<Clock> m_beg;

public:
  Timer() : m_beg{Clock::now()} {};
  void reset() { m_beg = Clock::now(); }
  double elapsed() const {
    return std::chrono::duration_cast<TimeUnit>(Clock::now() - m_beg).count();
  }
  double lap() {
    auto now = Clock::now();
    auto elapsed = std::chrono::duration_cast<TimeUnit>(now - m_beg).count();
    m_beg = now;
    return elapsed;
  }
};

// File IO
/// Output list of positions as CSV file
void dump_to_file(const std::string &fname, const vector<Vec2> &pos) {
  std::ofstream out(fname);

  if (out.is_open()) {
    for (int i = 0; i < pos.size(); ++i) {
      out << pos[i].x << "," << pos[i].y << "\n";
    }
    out.close();
  } else
    std::cout << "Unable to open file";
}

/// Format integer into CSV filename
std::string format_fname(int count) {
  char buffer[16];
  sprintf(buffer, "%04d.csv", count);
  return std::string(buffer);
}

// Random
/// Fill input array `x` with `size` real numbers
void fill_rand_vec(real *x, uint size, real min, real max, int seed = 42) {
  srand(seed);

  // rand() carries global state and is not thread safe, so this cannot be
  // parallelised as written
  for (int i = 0; i < size; ++i) {
    x[i] = min + (static_cast<real>(rand()) /
                  (static_cast<real>(RAND_MAX / (max - min))));
  }
}

/// Calculate the position and velocity that would give a stable orbit around a
/// huge mass at position (0,0)
void calc_stable_orbit(vector<Vec2> &pos, vector<Vec2> &vel,
                       const vector<real> &r, const vector<real> &theta) {
  for (int i = 0; i < r.size(); ++i) {
    real v_mag = 1. / std::sqrt(r[i]);

    pos[i].x = r[i] * std::sin(theta[i]);
    pos[i].y = r[i] * std::cos(theta[i]);

    vel[i].x = -v_mag * std::cos(theta[i]);
    vel[i].y = v_mag * std::sin(theta[i]);
  }
}
