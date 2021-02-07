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

Triangle::Triangle(Vector3f A_, Vector3f B_, Vector3f C_, Color color_) :
    Renderable(color_), A(A_), B(B_), C(C_) {

        // Precompute normal
        Vector3f v1 = A - B;
        Vector3f v2 = C - B;

        normal = v1.cross(v2).normalize();
};

// https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/ray-triangle-intersection-geometric-solution
IntersectData Triangle::intersects(Ray r) {

    float d = normal.dot(r.direction);
    IntersectData out;
    out.t = nan("");
    out.normal = normal;
    out.color = color;
    if(d == 0) { // Paralell case
        return out;
    } else {
        //float t = (normal.dot(r.origin) + normal.dot(A)) / d;
        //out.t = (this->point - r.origin).dot(this->norm) / d;
        float t = (A - r.origin).dot(normal) / d;
        if(t < 0) { // Ray behind case
            return out;
        } else {

            Vector3f P = r.getPoint(t);

            Vector3f edge0 = B - A;
            Vector3f edge1 = C - B;
            Vector3f edge2 = A - C;
            Vector3f C0 = P - A;
            Vector3f C1 = P - B;
            Vector3f C2 = P - C;

            if (normal.dot(edge0.cross(C0)) <= 0 &&
                normal.dot(edge1.cross(C1)) <= 0 &&
                normal.dot(edge2.cross(C2)) <= 0) {
                out.t = t;
                return out;
            } else {
                return out;
            }
        }

    }
}

Prism::Prism(Vector3f center_, Vector3f up_, Vector3f right_,
        Vector3f dimensions_, Color color) 
: center(center_), up(up_.normalize()), right(right_.normalize()), dimensions(dimensions_) {

    float half_width = dimensions.x / 2.0;
    float half_height = dimensions.y / 2.0;
    float half_depth = dimensions.z / 2.0;

    Vector3f back = up.cross(right).normalize();

    Vector3f back_top_right  = center + back * half_depth + up * half_height + right * half_width;
    Vector3f back_top_left   = center + back * half_depth + up * half_height - right * half_width;
    Vector3f back_bot_right  = center + back * half_depth - up * half_height + right * half_width;
    Vector3f back_bot_left   = center + back * half_depth - up * half_height - right * half_width;

    Vector3f front_top_right = center - back * half_depth + up * half_height + right * half_width;
    Vector3f front_top_left  = center - back * half_depth + up * half_height - right * half_width;
    Vector3f front_bot_right = center - back * half_depth - up * half_height + right * half_width;
    Vector3f front_bot_left  = center - back * half_depth - up * half_height - right * half_width;

    // Top
    triangles[0] = Triangle(back_top_right, back_top_left, front_top_left, color);
    triangles[1] = Triangle(front_top_right, back_top_right, front_top_left, color);
    // Bot
    triangles[2] = Triangle(back_bot_right, back_bot_left, front_bot_left, color);
    triangles[3] = Triangle(front_bot_right, back_bot_right, front_bot_left, color);
    // Right
    triangles[4] = Triangle(front_top_right, back_top_right, front_bot_right, color);
    triangles[5] = Triangle(back_bot_right, back_top_right, front_bot_right, color);
    // Left
    triangles[6] = Triangle(front_top_left, back_top_left, front_bot_left, color);
    triangles[7] = Triangle(back_bot_left, back_top_left, front_bot_left, color);
    // Front
    triangles[8] = Triangle(front_top_left, front_top_right, front_bot_left, color);
    triangles[9] = Triangle(front_bot_right, front_top_right, front_bot_left, color);
    // Back
    triangles[10] = Triangle(back_top_left, back_top_right, back_bot_left, color);
    triangles[11] = Triangle(back_bot_right, back_top_right, back_bot_left, color);

    // Add triangles to mesh
    for(int i = 0; i < 12; i++) {
        objects.push_back(&(triangles[i]));
    }


}
