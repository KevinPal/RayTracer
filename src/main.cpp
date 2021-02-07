#include <stdio.h>
#include "display.h"

#include "primitives.h"
#include "mesh.h"
#include "camera.h"
#include <vector>
#include <math.h>
#include <cassert>

int main (int argc, char **argv) {

    Display* display = Display::getInstance();

    display->init(1000, 1000);

    unsigned char* buf = display->getBuffer();

    Plane p(Vector3f(0, 0, 0), Vector3f(-1, 0, 0), Color(0, 0, 255));
    Plane p2(Vector3f(0, 0, 0), Vector3f(0, 1, 0), Color(255, 0, 0));
    Sphere s(Vector3f(0, 5, 5), 5, Color(255, 0, 255));
    Triangle t(
            Vector3f(0, 0, 10),
            Vector3f(0, 15, 10),
            Vector3f(15, 0, 10),
            Color(255, 255, 0));

    Prism r(
            Vector3f(5, 0, 2.5),
            Vector3f(0, 1, 0),
            Vector3f(1, 0, 1),
            Vector3f(5, 5, 5),
            Color(0, 255, 255));

    Mesh scene;
    scene.addObject(&p);
    scene.addObject(&p2);
    scene.addObject(&s);
    scene.addObject(&t);
    scene.addObject(&r);

    IntersectData data;
    Color bg(100, 100, 100);

    PerspectiveCamera cam = PerspectiveCamera(
        Vector3f(5, 5, -5),
        Vector3f(0, 1.0, 0),
        Vector3f(1.0, 0, 0),
        Vector2f(25.0, 25.0),
        Vector2f(display->getWidth(), display->getHeight()),
        Vector3f(5, 5, -25)
    );

    Vector3f light(5, 5, -10);

    for (auto it = cam.begin(), end = cam.end(); it != end; ++it) {
        Ray ray = *it;
        Vector2f screen_coord = it.getScreenCord();


        bool hit = false;
        IntersectData min_hit = scene.intersects(ray);
        hit = (min_hit.t >= 0);
        /*
        for(Renderable* r : objects) {

            data = r->intersects(ray);
            if((data.t >= 0) && (!hit || (data.t < min_hit.t))) {
                hit = true;
                min_hit = IntersectData(data);
            }
            
        }
        */

       
        // Do lighting
        if(hit) {
            Vector3f hit_pos = ray.getPoint(min_hit.t);
            float list_dist = (light - hit_pos).length();
            Vector3f light_ray = (light - hit_pos).normalize();
            float light_factor =  abs(min_hit.normal.dot(light_ray));

            min_hit.color = min_hit.color.color / 2 + (Vector3f(255, 255, 255) * CLAMP(light_factor / (list_dist) * 7 , 0, 1));
            min_hit.color.clamp();
        }
   

        (hit ? min_hit.color : bg).writeToBuff(&(buf[(int)(4 * (screen_coord.y * display->getWidth() + screen_coord.x))]));
    }

    display->run();
    display->destory();

  return 0;
}
