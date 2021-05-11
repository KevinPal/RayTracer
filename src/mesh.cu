
#include "mesh.h"
#include "renderable.h"
#include "vector.h"
#include "primitives.h"

#include <stdio.h>
#include <math.h>
#include <string>
#include <cstring>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cassert>

// Default constructor for a mesh with an empty AABB
Mesh::Mesh(Material* material_)
 : Renderable(material_) {

    this->bounding_box = new AABB();
}

// Creates a mesh from already existing renderables, where start is an iterator 
// pointing to the start of elements to copy and end is the ending iterator to copy
Mesh::Mesh(Material* material_, std::vector<Renderable*>::iterator start, std::vector<Renderable*>::iterator end)
    : Renderable(material_) { 
    
    this->bounding_box = new AABB();
    this->objects.assign(start, end);
}
// Mesh intersection loops across all objects
// in the mesh, and checks if the ray intersects 
// it. We keep track of the closest intersection,
// and return it
IntersectData Mesh::intersects(Ray ray) {
    bool hit = false;
    IntersectData min_hit;
    min_hit.t = -1;
    //printf("Running general intersection on size: %d\n", objects.size() + large_objects.size());
    for(Renderable* r : objects) {

        IntersectData data = r->intersects(ray);
        if((data.t >= 0) && (!hit || (data.t < min_hit.t))) {
            hit = true;
            min_hit = IntersectData(data);
            min_hit.object = r;
        }
    }

    for(Renderable* r : large_objects) {

        IntersectData data = r->intersects(ray);
        if((data.t >= 0) && (!hit || (data.t < min_hit.t))) {
            hit = true;
            min_hit = IntersectData(data);
            min_hit.object = r;
        }
    }

    return min_hit;
}
// Same as intersects, except only for the large objects
// in this mesh. This is done sepereatly because large objects, such
// as planes, do not have a valid AABB and must always be checked
IntersectData Mesh::intersects_large(Ray ray) {
    bool hit = false;
    IntersectData min_hit;
    min_hit.t = -1;

    //printf("Running large intersection on size: %d\n", large_objects.size());
    for(Renderable* r : large_objects) {

        IntersectData data = r->intersects(ray);
        if((data.t >= 0) && (!hit || (data.t < min_hit.t))) {
            hit = true;
            min_hit = IntersectData(data);
            min_hit.object = r;
        }
    }

    return min_hit;

}

// Pushes an object to this mesh. Attempts to build its bounding box
// if it does not have one. If it does, the meshes bounding box
// expands to include it, otherwise it is held with the large objects
void Mesh::addObject(Renderable* obj) {

    // If it has no bounding box but it is a mesh, build the box
    if(!obj->bounding_box) {
        obj->bounding_box = obj->buildBoundingBox();
    }

    if(obj->bounding_box) {
        objects.push_back(obj);
        this->bounding_box->merge(obj->bounding_box);
    } else {
        large_objects.push_back(obj);
    }
}

// Builds a bounding box for this mesh by unioning all
// the bounding boxes of the elements in this mesh
AABB* Mesh::buildBoundingBox(void) {

    for(Renderable* r : this->objects) {
        if(r->bounding_box) {
            this->bounding_box->merge(r->bounding_box);
        }
    }

    return this->bounding_box;
}

// Builds a mesh from an obj file. Only supports 'v' and 'f'
// statements. Vertex normals are averaged for each face
// connected to the vertex, and are area weighted. Adds
// the new triangles to the mesh
void Mesh::fromOBJ(std::string path) {
    std::vector<Vector3f> verticies;
    std::vector<Vector3f> faces;

    std::ifstream infile(path, std::ios_base::in);
    assert(infile.is_open());
    std::string line;

    char c;
    float f1, f2, f3;

    // Read data from file into faces and verticies arrays
    while(std::getline(infile, line)) {
        std::istringstream ss(line);

        if(!(ss >> c >> f1 >> f2 >> f3)) {
            assert(false);
        }  else {
            if(c == 'v') {
                Vector3f data(f1, f2, f3 * -1);
                verticies.push_back(data);
            } else if(c == 'f') {
                Vector3f data(f1, f2, f3);
                faces.push_back(data);
            }
        }
    }

    printf("Mesh has %d verticies with %d faces\n", verticies.size(), faces.size());

    Vector3f face_norms[faces.size()];
    float face_norm_size[faces.size()];

    // Calculate per face normals
    for(int i = 0; i < faces.size(); i++) {

        Vector3f& v0 = verticies[faces[i][0] - 1];
        Vector3f& v1 = verticies[faces[i][1] - 1];
        Vector3f& v2 = verticies[faces[i][2] - 1];

        face_norms[i] = (v1 - v0).cross(v2 - v0);
        face_norm_size[i] = face_norms[i].length();
    }

    // Sum up normals per vertex
    Vector3f vertex_norms[verticies.size()];

    float vertex_norm_size[verticies.size()];
    memset(vertex_norm_size, 0, verticies.size() * sizeof(float));

    // Sum up normals per vertex and face areas
    for(int i = 0; i < faces.size(); i++) {
        for(int j = 0; j < 3; j++) {
            int vert_index = faces[i][j] - 1;
            vertex_norms[vert_index] = vertex_norms[vert_index] + face_norms[i] * face_norm_size[i];
            vertex_norm_size[vert_index] += face_norm_size[i];
        }
    }

    // Weight each normal according to area
    for(int i = 0; i < verticies.size(); i++) {
        vertex_norms[i] = vertex_norms[i] / vertex_norm_size[i];
    }

    // Make triangles and push into mesh
    for(int i = 0; i < faces.size(); i++) {

        int v1 = (int) faces[i][0] - 1;
        int v2 = (int) faces[i][1] - 1;
        int v3 = (int) faces[i][2] - 1;

        Triangle* t = new Triangle(
            verticies[v1],
            verticies[v2],
            verticies[v3],
            vertex_norms[v1],
            vertex_norms[v2],
            vertex_norms[v3],
            this->material
        );
        addObject(t);
    }

}
