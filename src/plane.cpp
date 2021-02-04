#include "renderable.h"
#include "plane.h"
#include "math.h"
#include <stdio.h>

Plane::Plane(Vector3f point_, Vector3f norm_, Color color_) :
    point(point_), norm(norm_.normalize()), color(color_) {};

IntersectData Plane::intersects(Ray r) {
    IntersectData out;
    float d = this->norm.dot(r.direction);
    //printf("<%f %f %f> * <%f %f %f> = <%f>", norm.x, norm.y, norm.z, r.direction.x, r.direction.y, r.direction.z, d);
    if(d != 0) {
        out.t = (this->point - r.origin).dot(this->norm) / d;
        out.color = this->color;
    } else {
        out.t = nan("");
    }

    return out;
}


