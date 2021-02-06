#include <stdio.h>
#include "display.h"

#include "primitives.h"
#include "camera.h"
#include <vector>
#include <math.h>
#include <cassert>

int main (int argc, char **argv) {

    Display* display = Display::getInstance();

    display->init(500, 500);

    unsigned char* buf = display->getBuffer();

    Plane p(Vector3f(0, 0, 0), Vector3f(-1, 0, 0), Color(0, 0, 255));
    Plane p2(Vector3f(0, 0, 0), Vector3f(0, 1, 0), Color(255, 0, 0));
    Sphere s(Vector3f(0, 0, 7), 10, Color(255, 0, 255));

    std::vector<Renderable*> objects;
    objects.push_back(&p);
    objects.push_back(&p2);
    objects.push_back(&s);

    IntersectData data;
    Color bg;

    PerspectiveCamera cam = PerspectiveCamera(
        Vector3f(5, 5, -5),
        Vector3f(0, 1.0, 0),
        Vector3f(1.0, 0, 0),
        Vector2f(25.0, 25.0),
        Vector2f(display->getWidth(), display->getHeight()),
        Vector3f(5, 5, -10)
    );

    Vector3f light(5, 5, -10);

    for (auto it = cam.begin(), end = cam.end(); it != end; ++it) {
        Ray ray = *it;
        Vector2f screen_coord = it.getScreenCord();


        bool hit = false;
        IntersectData min_hit;

        for(Renderable* r : objects) {


            data = r->intersects(ray);
            /*
            if(screen_coord == Vector2f(251, 250)) {
                printf("%f\n", data.t);
                ray.origin.print();
                printf("\n");
                ray.direction.print();
                printf("%f \n" ,ray.direction.length());
            }
            */
            if((data.t >= 0) && (!hit || (data.t < min_hit.t))) {
                hit = true;
                min_hit = IntersectData(data);
            }
            
        }

    
       
        if(hit) {
            Vector3f hit_pos = ray.getPoint(min_hit.t);
            float list_dist = (light - hit_pos).length();
            Vector3f light_ray = (light - hit_pos).normalize();
            float light_factor =  abs(min_hit.normal.dot(light_ray));

            if(screen_coord == Vector2f(0, 250)) {
                printf("YEETING\n");
                min_hit.color.color.print();
                printf("%f\n", data.t);
                printf("\n");
            }
            min_hit.color = min_hit.color.color / 2 + (Vector3f(255, 255, 255) * CLAMP(light_factor / (list_dist) * 7 , 0, 1));
            min_hit.color.clamp();
        }
        
   

        (hit ? min_hit.color : bg).writeToBuff(&(buf[(int)(4 * (screen_coord.y * display->getWidth() + screen_coord.x))]));
    }

    /*
    for(int x = 0; x < display->getWidth(); x++) {
        for(int y = 0; y < display->getHeight(); y++) {

            Ray r(Vector3f(x - display->getWidth()/2, (y - display->getHeight()/2), 0), Vector3f(0, 0, 1));

        }
    }
    */


    display->run();
    display->destory();

  return 0;
}
