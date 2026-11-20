#include "maze_planning/clearance.hpp"

#include <cmath>
#include <limits>

namespace maze_planning
{

namespace
{

constexpr double kInf = std::numeric_limits<double>::infinity();

// 1-D squared distance transform of f (Felzenszwalb & Huttenlocher 2012): d[q] = min_p (q - p)^2 + f[p]
void dt_1d(const std::vector<double> & f, std::vector<double> & d)
{
  const int n = static_cast<int>(f.size());
  std::vector<int> v(n);         // parabola locations of the lower envelope
  std::vector<double> z(n + 1);  // boundaries between them
  int k = -1;
  for (int q = 0; q < n; ++q) {
    if (std::isinf(f[q])) {
      continue;
    }
    while (k >= 0) {
      const double s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2.0 * (q - v[k]));
      if (s > z[k]) {
        break;
      }
      --k;
    }
    ++k;
    v[k] = q;
    z[k] = k == 0 ? -kInf : ((f[q] + q * q) - (f[v[k - 1]] + v[k - 1] * v[k - 1])) / (2.0 * (q - v[k - 1]));
    z[k + 1] = kInf;
  }
  if (k < 0) {
    d.assign(n, kInf);
    return;
  }
  int j = 0;
  for (int q = 0; q < n; ++q) {
    while (z[j + 1] < q) {
      ++j;
    }
    d[q] = (q - v[j]) * (q - v[j]) + f[v[j]];
  }
}

}  // namespace

std::vector<double> clearance_map(const Grid & grid)
{
  validate(grid);
  const int w = grid.width, h = grid.height;
  // squared distance in cells to the nearest obstacle cell center, first along columns, then rows
  std::vector<double> sq(static_cast<size_t>(w * h));
  std::vector<double> f(h), d(h);
  for (int c = 0; c < w; ++c) {
    for (int r = 0; r < h; ++r) {
      f[r] = is_obstacle(grid.data[r * w + c]) ? 0.0 : kInf;
    }
    dt_1d(f, d);
    for (int r = 0; r < h; ++r) {
      sq[r * w + c] = d[r];
    }
  }
  f.resize(w);
  d.resize(w);
  for (int r = 0; r < h; ++r) {
    for (int c = 0; c < w; ++c) {
      f[c] = sq[r * w + c];
    }
    dt_1d(f, d);
    for (int c = 0; c < w; ++c) {
      sq[r * w + c] = d[c];
    }
  }
  // to m, and from the obstacle cell's center to its edge
  std::vector<double> out(sq.size());
  for (size_t i = 0; i < sq.size(); ++i) {
    out[i] = sq[i] == 0.0 ? 0.0 : std::sqrt(sq[i]) * grid.resolution - grid.resolution / 2;
  }
  return out;
}

}  // namespace maze_planning
