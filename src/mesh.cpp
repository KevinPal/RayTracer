
#include "mesh.h"
#include "renderable.h"
#include "vector.h"
#include <stdio.h>
#include <math.h>

Mesh::Mesh()
 : Renderable(Color()) {

    this->bounding_box = new AABB();
}

Mesh::Mesh(std::vector<Renderable*>::iterator start, std::vector<Renderable*>::iterator end)
    : Renderable(Color()) { 
    
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
    min_hit.t = nan("");
    for(Renderable* r : objects) {

        IntersectData data = r->intersects(ray);
        if((data.t >= 0) && (!hit || (data.t < min_hit.t))) {
            hit = true;
            min_hit = IntersectData(data);
        }
    }

    for(Renderable* r : large_objects) {

        IntersectData data = r->intersects(ray);
        if((data.t >= 0) && (!hit || (data.t < min_hit.t))) {
            hit = true;
            min_hit = IntersectData(data);
        }
    }

    return min_hit;
}
// Same as intersects, except only for the large objects
// in this mesh
IntersectData Mesh::intersects_large(Ray ray) {
    bool hit = false;
    IntersectData min_hit;
    min_hit.t = nan("");

    for(Renderable* r : large_objects) {

        IntersectData data = r->intersects(ray);
        if((data.t >= 0) && (!hit || (data.t < min_hit.t))) {
            hit = true;
            min_hit = IntersectData(data);
        }
    }

    return min_hit;

}

template<typename Base, typename T>
inline bool instanceof(const T*) {
   return std::is_base_of<Base, T>::value;
}

// Pushes an object to this mesh
void Mesh::addObject(Renderable* obj) {

    // If it has no bounding box but it is a mesh, build the box
    if(!obj->bounding_box && instanceof<Mesh>(obj)) {
        ((Mesh*) obj)->buildBoundingBox();
    }

    if(obj->bounding_box) {
        objects.push_back(obj);
        this->bounding_box->merge(obj->bounding_box);
    } else {
        large_objects.push_back(obj);
    }
}

void Mesh::buildBoundingBox(void) {

    for(Renderable* r : this->objects) {
        if(r->bounding_box) {
            this->bounding_box->merge(r->bounding_box);
        }
    }
}


