
#include "mesh.h"
#include "renderable.h"
#include "vector.h"
#include <math.h>

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

    return min_hit;
}

// Pushes an object to this mesh
void Mesh::addObject(Renderable* obj) {
    objects.push_back(obj);
}


