#include "renderable.h"
#include "primitives.h"
#include "math.h"
#include <stdio.h>

Plane::Plane(Vector3f point_, Vector3f norm_, Color color_) :
    Renderable(color_), 
    point(point_), norm(norm_.normalize()) {};

IntersectData Plane::intersects(Ray r) {
    IntersectData out;
    float d = this->norm.dot(r.direction);
    //printf("<%f %f %f> * <%f %f %f> = <%f>", norm.x, norm.y, norm.z, r.direction.x, r.direction.y, r.direction.z, d);
    out.normal = norm;
    if(d != 0) {
        out.t = (this->point - r.origin).dot(this->norm) / d;
        out.color = this->color;
    } else {
        out.t = nan("");
    }

    return out;
}

Sphere::Sphere(Vector3f center_, float radius_, Color color_) :
    Renderable(color_), 
    center(center_), radius(radius_) {};

// Matg based off of https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-sphere-intersection
IntersectData Sphere::intersects(Ray r) {

    IntersectData out;
    out.color = this->color;
    Vector3f L = this->center - r.origin;
    float t_ca = L.dot(r.direction);
    if(t_ca < 0) {
        out.t = nan("");
    } else {
        float d = sqrt(L.dot(L) - t_ca * t_ca);
        if(d < 0) {
            out.t = nan("");
        } else {
            float t_hc = sqrt(this->radius * this->radius - d * d);
            float t0 = t_ca - t_hc;
            float t1 = t_ca + t_hc;

            if(t0 > 0) { // Both intersect, t0 is closer so use it
                out.t = t0;
            } else if(t1 > 0) { // Front of sphere behind camera, but back infront
                out.t = t1;
            } else { // Sphere behind camera
                out.t = nan("");
            }

            if(out.t > 0) {
                out.normal = (r.getPoint(out.t) - this->center).normalize();
            }
        }
    }

    return out;
}
