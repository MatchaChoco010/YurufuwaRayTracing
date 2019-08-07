#define _USE_MATH_DEFINES
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <omp.h>
#include <optional>
#include <random>
#include <tuple>
#include <vector>

// Vector ----

struct V {
  double x;
  double y;
  double z;

  V(const double v = 0) : V(v, v, v) {}
  V(const double x, const double y, const double z) : x(x), y(y), z(z) {}
};

V operator+(const V a, const V b) { return V(a.x + b.x, a.y + b.y, a.z + b.z); }
V operator-(const V a, const V b) { return V(a.x - b.x, a.y - b.y, a.z - b.z); }
V operator*(const V a, const V b) { return V(a.x * b.x, a.y * b.y, a.z * b.z); }
V operator/(const V a, const V b) { return V(a.x / b.x, a.y / b.y, a.z / b.z); }
V operator-(const V v) { return V(-v.x, -v.y, -v.z); }

double dot(const V a, const V b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
V cross(const V a, const V b) {
  return V(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
V normalize(const V v) { return v / std::sqrt(dot(v, v)); }
std::tuple<V, V> tangentSpace(const V &n) {
  const double s = std::copysign(1, n.z);
  const double a = -1 / (s + n.z);
  const double b = n.x * n.y * a;
  return {V(1 + s * n.x * n.x * a, s * b, -s * n.x),
          V(b, s + n.y * n.y * a, -n.y)};
}

// tonemap ----
// 0-1の値にgamma補正をかけて0-255に変換する

int tonemap(const double v) {
  return std::min(std::max(int(std::pow(v, 1 / 2.2) * 255), 0), 255);
}

// Random ----
// 0-1のランダムなdoubleを返す

struct Random {
  std::mt19937 engine;
  std::uniform_real_distribution<double> dist;
  Random(){};
  Random(int seed) {
    engine.seed(seed);
    dist.reset();
  }
  double next() { return dist(engine); }
};

// Ray ----

struct Ray {
  V origin;
  V direction;
};

// Hit ----

struct Sphere;
struct Hit {
  double t;
  V position;
  V normal;
  const Sphere *sphere;
};

// Sphere ----

struct Sphere {
  V position;
  double r;
  V reflectance;
  V illuminance;

  std::optional<Hit> intersect(const Ray &ray, const double tmin,
                               const double tmax) const {
    // |x-p| = r
    // x = o + td
    // tについての2次方程式の解の公式から衝突判定を計算する

    const V op = position - ray.origin;
    const double b = -dot(op, ray.direction);
    const double det = b * b - dot(op, op) + r * r;

    if (det < 0) {
      return std::nullopt;
    }

    const double t1 = -b - std::sqrt(det);
    if (tmin < t1 && t1 < tmax) {
      return Hit{t1, {}, {}, this};
    }

    const double t2 = -b + std::sqrt(det);
    if (tmin < t2 && t2 < tmax) {
      return Hit{t2, {}, {}, this};
    }

    return std::nullopt;
  }
};

// Scene ----

struct Scene {
  std::vector<Sphere> spheres{
      // 1
      // {V(-0.5, 0, 0), 1, V(1, 0, 0)},
      // {V(0.5, 0, 0), 1, V(0, 1, 0)},

      // 2
      {V(1e5 + 1, 40.8, 81.6), 1e5, V(0.75, 0.25, 0.25)},
      {V(-1e5 + 99, 40.8, 81.6), 1e5, V(0.25, 0.25, 0.75)},
      {V(50, 40.8, 1e5), 1e5, V(0.75)},
      {V(50, 1e5, 81.6), 1e5, V(0.75)},
      {V(50, -1e5 + 81.6, 81.6), 1e5, V(0.75)},
      {V(27, 16.5, 47), 16.5, V(0.999)},
      {V(73, 16.5, 78), 16.5, V(0.999)},
      {V(50, 681.6 - 0.27, 81.6), 600, V(), V(12)},
  };

  std::optional<Hit> intersect(const Ray &ray, const double tmin,
                               double tmax) const {
    std::optional<Hit> minHit;

    for (const auto &sphere : spheres) {
      const auto hit = sphere.intersect(ray, tmin, tmax);
      if (!hit) {
        continue;
      }
      minHit = hit;
      tmax = minHit->t;
    }

    if (minHit) {
      const auto *s = minHit->sphere;
      minHit->position = ray.origin + ray.direction * minHit->t;
      minHit->normal = (minHit->position - s->position) / s->r;
    }

    return minHit;
  }
};

// main ----

int main() {
  // Image size
  const int w = 1200;
  const int h = 800;

  // Sampes per pixel
  const int spp = 1000;

  // Camera parameters
  // 1
  // const V eye(5, 5, 5);
  // const V center(0, 0, 0);
  // const V up(0, 1, 0);
  // const double fov = 30 * M_PI / 180;
  // 2
  const V eye(50, 52, 295.6);
  const V center = eye + V(0, -0.042612, -1);
  const V up(0, 1, 0);
  const double fov = 30 * M_PI / 180;

  const double aspect = static_cast<double>(w) / h;

  // Basis vectors for camera coordinates
  const auto wE = normalize(eye - center);
  const auto uE = normalize(cross(up, wE));
  const auto vE = cross(wE, uE);

  // Rendering
  Scene scene;
  std::vector<V> I(w * h);
#pragma omp parallel for schedule(dynamic, 1)
  for (int i = 0; i < w * h; i++) {

    thread_local Random rng(42 + omp_get_thread_num());

    for (int j = 0; j < spp; j++) {

      // Initial Ray ----
      const int x = i % w;
      const int y = h - i / w;
      Ray ray;
      ray.origin = eye;
      ray.direction = [&]() {
        const double tf = std::tan(fov * 0.5);
        const double rpx = 2.0 * (x + rng.next()) / w - 1.0;
        const double rpy = 2.0 * (y + rng.next()) / h - 1.0;
        const V w = normalize(V(aspect * tf * rpx, tf * rpy, -1));
        return uE * w.x + vE * w.y + wE * w.z;
      }();

      // Ray tracing loop
      V L(0), throughput(1);
      for (int depth = 0; depth < 10; depth++) {
        // Intersection
        const auto hit = scene.intersect(ray, 1e-4, 1e+10);

        if (!hit) {
          break;
        }

        // Add contribution
        L = L + throughput * hit->sphere->illuminance;
        // Update next direction
        ray.origin = hit->position;
        ray.direction = [&]() {
          // Sample direction in local coordinates
          const auto n =
              dot(hit->normal, -ray.direction) > 0 ? hit->normal : -hit->normal;
          const auto &[u, v] = tangentSpace(n);
          const auto d = [&]() {
            const double r = sqrt(rng.next());
            const double t = 2 * M_PI * rng.next();
            double x = r * cos(t);
            double y = r * sin(t);
            return V(x, y, std::sqrt(std::max(0.0, 1 - x * x - y * y)));
          }();
          // Convert to world coordinates
          return u * d.x + v * d.y + n * d.z;
        }();

        // Update throughput
        throughput = throughput * hit->sphere->reflectance;
        if (std::max({throughput.x, throughput.y, throughput.z}) == 0) {
          break;
        }
      }
      I[i] = I[i] + L / spp;
    }
  }

  // Write result.ppm ----
  std::ofstream ofs("result.ppm");
  ofs << "P3\n" << w << " " << h << "\n255\n";
  for (const auto &i : I) {
    ofs << tonemap(i.x) << " " << tonemap(i.y) << " " << tonemap(i.z) << "\n";
  }

  return 0;
}
