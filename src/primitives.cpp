#include "renderable.h"
#include "primitives.h"
#include "math.h"
#include "vector.h"
#include <stdio.h>

// Plane constructor. Normalizes the normal
Plane::Plane(Vector3f point_, Vector3f norm_, Material material_) :
    Renderable(material_), 
    point(point_), norm(norm_.normalize()) {
};

// Plane ray intersection math
IntersectData Plane::intersects(Ray r) {
    IntersectData out;
    float d = this->norm.dot(r.direction);
    out.normal = norm;
    if(d != 0) {
        out.t = (this->point - r.origin).dot(this->norm) / d;
        out.material = this->material;
    } else {
        out.t = nan("");
    }

    return out;
}

// Creates bounding box. Planes do not have one so returns null
AABB* Plane::buildBoundingBox() {return NULL; }

// Sphere constructor
Sphere::Sphere(Vector3f center_, float radius_, Material material_) :
    Renderable(material_), 
    center(center_), radius(radius_) {

    this->bounding_box = this->buildBoundingBox();
};

// Buildings the bounding box for the sphere
AABB* Sphere::buildBoundingBox() {
    return new AABB{
        center,
        Vector3f(radius * 2, radius * 2, radius * 2),
        material
    };

}

// Math based off of https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-sphere-intersection
// Tests if ray intersects sphere, and calculates normal if so
IntersectData Sphere::intersects(Ray r) {

    IntersectData out;
    out.material = this->material;
    Vector3f L = this->center - r.origin;

    float dir_len = r.direction.length();
    r.direction.normalize();
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

            if((t0 > 0) && (t1 > 0)) {
                out.t =  t0 < t1 ? t0 : t1;
            } else if(t0 > 0) {
                out.t = t0;
            } else if(t1 > 0) {
                out.t = t1;
            } else {
                out.t = nan("");
            }

            out.t /= dir_len;

            out.normal = (r.getPoint(out.t) - this->center).normalize();
        }
    }

    return out;
}

// Constructor for a triangle that has per vertex normals. Still
// calculates the face normal
Triangle::Triangle(Vector3f A_, Vector3f B_, Vector3f C_, 
        Vector3f A_normal_, Vector3f B_normal_, Vector3f C_normal_, Material material_) :
    Renderable(material_), A(A_), B(B_), C(C_), 
    A_normal(A_normal_), B_normal(B_normal_), C_normal(C_normal_){

    Vector3f v1 = A - B;
    Vector3f v2 = C - B;
    normal = v1.cross(v2);


}


// Triangle constructor, which computes the triangles normal. Vertex
// normals are the same as the face normals
Triangle::Triangle(Vector3f A_, Vector3f B_, Vector3f C_, Material material_) :
    Renderable(material_), A(A_), B(B_), C(C_) {

        // Precompute normal
        Vector3f v1 = A - B;
        Vector3f v2 = C - B;

        normal = v1.cross(v2);
        A_normal = normal;
        B_normal = normal;
        C_normal = normal;

        this->bounding_box = buildBoundingBox();
};

// Builds the AABB for this triangle
AABB* Triangle::buildBoundingBox() {

        Vector2f spreads[3];
        spreads[0].x = spreads[0].y = A.x;
        spreads[1].x = spreads[1].y = A.y;
        spreads[2].x = spreads[2].y = A.z;

        spreads[0].x = std::min(spreads[0].x, B.x);
        spreads[0].y = std::max(spreads[0].y, B.x);
        spreads[1].x = std::min(spreads[1].x, B.y);
        spreads[1].y = std::max(spreads[1].y, B.y);
        spreads[2].x = std::min(spreads[2].x, B.z);
        spreads[2].y = std::max(spreads[2].y, B.z);

        spreads[0].x = std::min(spreads[0].x, C.x);
        spreads[0].y = std::max(spreads[0].y, C.x);
        spreads[1].x = std::min(spreads[1].x, C.y);
        spreads[1].y = std::max(spreads[1].y, C.y);
        spreads[2].x = std::min(spreads[2].x, C.z);
        spreads[2].y = std::max(spreads[2].y, C.z);

        return  new AABB(
            Vector3f(
                (spreads[0].y + spreads[0].x) / 2.0,
                (spreads[1].y + spreads[1].x) / 2.0,
                (spreads[2].y + spreads[2].x) / 2.0
            ),
            Vector3f(
                (spreads[0].y - spreads[0].x),
                (spreads[1].y - spreads[1].x),
                (spreads[2].y - spreads[2].x)
            ),
            material
        );

}

// https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/ray-triangle-intersection-geometric-solution
// https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/barycentric-coordinates
IntersectData Triangle::intersects(Ray r) {

    float d = normal.dot(r.direction);
    IntersectData out;
    out.t = nan("");
    out.normal = normal;
    out.normal.normalize();
    out.material = material;
    if(d == 0) { // Paralell case
        return out;
    } else {
        float t = (A - r.origin).dot(normal) / d;
        if(t < 0) { // Ray behind case
            return out;
        } else {

            // Compute barycentric
            Vector3f P = r.getPoint(t);

            Vector3f edge0 = B - A;
            Vector3f edge1 = C - B;
            Vector3f edge2 = A - C;

            Vector3f VP0 = P - A;
            Vector3f VP1 = P - B;
            Vector3f VP2 = P - C;

            Vector3f C0 = edge0.cross(VP0);
            Vector3f C1 = edge1.cross(VP1);
            Vector3f C2 = edge2.cross(VP2);

            float area = normal.length() / 2;
            float u = (C0.length() / 2) / area;
            float v = (C1.length() / 2) / area;
            float w = (C2.length() / 2) / area;


            if (normal.dot(C0) <= 0 &&
                normal.dot(C1) <= 0 &&
                normal.dot(C2) <= 0) {
                out.t = t;
                out.normal = A_normal * v + B_normal * w + C_normal * u;
                out.normal.normalize();
                return out;
            } else {
                return out;
            }
        }

    }
}

// Prism constructor. Breaks up the prisim into 12 triangles and
// pushes them to a mesh
Prism::Prism(Vector3f center_, Vector3f up_, Vector3f right_,
        Vector3f dimensions_, Material material) 
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
    triangles[0] = new Triangle(back_top_right, back_top_left, front_top_left, material);
    triangles[1] = new Triangle(front_top_right, back_top_right, front_top_left, material);
    // Bot
    triangles[2] = new Triangle(back_bot_right, back_bot_left, front_bot_left, material);
    triangles[3] = new Triangle(front_bot_right, back_bot_right, front_bot_left, material);
    // Right
    triangles[4] = new Triangle(front_top_right, back_top_right, front_bot_right, material);
    triangles[5] = new Triangle(back_bot_right, back_top_right, front_bot_right, material);
    // Left
    triangles[6] = new Triangle(front_top_left, back_top_left, front_bot_left, material);
    triangles[7] = new Triangle(back_bot_left, back_top_left, front_bot_left, material);
    // Front
    triangles[8] = new Triangle(front_top_left, front_top_right, front_bot_left, material);
    triangles[9] = new Triangle(front_bot_right, front_top_right, front_bot_left, material);
    // Back
    triangles[10] = new Triangle(back_top_left, back_top_right, back_bot_left, material);
    triangles[11] = new Triangle(back_bot_right, back_top_right, back_bot_left, material);

    // Add triangles to mesh
    for(int i = 0; i < 12; i++) {
        objects.push_back(triangles[i]);
    }

    this->bounding_box = this->buildBoundingBox();

}

Prism::~Prism() {
    for(Renderable* r : objects)
        delete r;
}
