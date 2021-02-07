
#include "mesh.h"
#include "renderable.h"
#include "vector.h"
#include <math.h>

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

void Mesh::addObject(Renderable* obj) {
    objects.push_back(obj);
}


