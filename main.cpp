#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
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

// tonemap ----
// 0-1の値にgamma補正をかけて0-255に変換する

int tonemap(const double v) {
  return std::min(std::max(int(std::pow(v, 1 / 2.2) * 255), 0), 255);
}

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

  std::optional<Hit> intersect(const Ray &ray, const double tmin,
                               const double tmax) const {
    // |x-p| = r
    // x = o + td
    // tについての2次方程式の解の公式を作る

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
  std::vector<Sphere> spheres{{V(), 1}};

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
  const int w = 1200;
  const int h = 800;

  Scene scene;

  std::ofstream ofs("result.ppm");
  ofs << "P3\n" << w << " " << h << "\n255\n";
  for (int i = 0; i < w * h; i++) {
    const int x = i % w;
    const int y = i / w;
    Ray ray;
    ray.origin = V(2.0 * (static_cast<double>(x) / w) - 1.0,
                   2.0 * (static_cast<double>(y) / h) - 1.0, 5);
    ray.direction = V(0, 0, -1);

    const auto hit = scene.intersect(ray, 0, 1e+10);
    if (hit) {
      const auto n = hit->normal;
      ofs << tonemap(n.x) << " " << tonemap(n.y) << " " << tonemap(n.z) << "\n";
    } else {
      ofs << "0 0 0\n";
    }
  }
  return 0;
}
