#include <stdio.h>

#include "primitives.h"
#include "mesh.h"
#include "camera.h"
#include <vector>
#include <math.h>
#include <cassert>
#include "display.h"
#include <math.h>

#include <stack>

int main (int argc, char **argv) {


    Display* display = Display::getInstance();

    display->init(500, 500);

    unsigned char* buf = display->getBuffer();

    Plane p(Vector3f(0, 0, 15), Vector3f(0, 0, -1), Color(0, 0, 255, 1, 0));
    Plane p2(Vector3f(0, 0, 0), Vector3f(0, 1, 0), Color(255, 0, 0, 1, .4));
    Sphere s(Vector3f(0, 5, 5), 5, Color(255, 0, 255, .5, 0));
    Triangle t(
            Vector3f(0, 0, 10),
            Vector3f(15, 0, 5),
            Vector3f(0, 15, 10),
            Color(255, 255, 0, 1));

    Prism r(
            Vector3f(5, 0, -5),
            Vector3f(0, 1, 0),
            Vector3f(1, 0, 1),
            Vector3f(5, 10, 5),
            Color(0, 255, 255, .5));

    Prism r2(
            Vector3f(-7, 1, -6),
            Vector3f(0, 1, 0),
            Vector3f(0, 1, 2),
            Vector3f(5, 5, 5),
            Color(100, 100, 100, .8, 0));

    Mesh scene;
    scene.addObject(&p);
    scene.addObject(&p2);
    scene.addObject(&s);
    scene.addObject(&t);
    scene.addObject(&r);
    scene.addObject(&r2);

    Vector3f light(10, 10, -15);
    float l_int = 10.0;

    IntersectData data;
    Color bg(100, 100, 100);

    PerspectiveCamera cam = PerspectiveCamera(
        Vector3f(0, 5, -5),
        Vector3f(0, 1.0, 0),
        Vector3f(1.0, 0, 0),
        Vector2f(25.0, 25.0),
        Vector2f(display->getWidth(), display->getHeight()),
        Vector3f(0, 5, -25)
    );
    
    /*
    OrthoCamera cam = OrthoCamera(
        Vector3f(0, 5, -25),
        Vector3f(0, 1.0, .1),
        Vector3f(1.0, 0, 0),
        Vector2f(50, 50),
        Vector2f(display->getWidth(), display->getHeight())
    );
    */


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

       
        if(true) {
            // Do lighting
            if(hit) {
                Vector3f hit_pos = ray.getPoint(min_hit.t);
                float spec = min_hit.color.specular;

                std::stack<Color> alpha_stack;

                Ray alpha_ray;
                alpha_ray.direction = ray.direction;
                alpha_ray.origin = hit_pos;

                alpha_stack.push(min_hit.color);

                while(alpha_stack.top().alpha != 1) {
                    alpha_ray.origin = alpha_ray.getPoint(1e-4);
                    IntersectData alpha_data = scene.intersects(alpha_ray);
                    if(alpha_data.t != nan("") && (alpha_data.t >= 0)) {
                        alpha_stack.push(alpha_data.color);
                        alpha_ray.origin = alpha_ray.getPoint(alpha_data.t);
                    } else {
                        break;
                    }
                }

                min_hit.color = alpha_stack.top();
                alpha_stack.pop();
                while(!alpha_stack.empty()) {
                    Color mix_color = alpha_stack.top();

                    min_hit.color.color = (mix_color.color * (mix_color.alpha)) + (min_hit.color.color * (1 - mix_color.alpha));
                    alpha_stack.pop();
                }

                if(spec > 0) {
                    Ray reflection_ray;
                    reflection_ray.origin = hit_pos;
                    //reflection_ray.direction = min_hit.normal;
                    reflection_ray.direction = (min_hit.normal * 2 * ray.direction.dot(min_hit.normal) - ray.direction) * -1;

                    reflection_ray.origin = reflection_ray.getPoint(1e-4);
                    IntersectData reflection_data = scene.intersects(reflection_ray);

                    if(reflection_data.t != nan("") && (reflection_data.t >= 0)) {
                        min_hit.color.color = (min_hit.color.color * (1-spec)) + (reflection_data.color.color * spec);
                    }
                }


                // Do shadow
                Ray shadow_ray;
                float dist;
                shadow_ray.fromPoints(hit_pos, light);
                shadow_ray.origin = shadow_ray.getPoint(1e-4);
                IntersectData shadow_data = scene.intersects(shadow_ray);

                if(shadow_data.t != nan("") && (shadow_data.t >= 0) && (shadow_data.t < 1)) {
                    min_hit.color = min_hit.color.color / 2;
                } else {
                    float list_dist = (light - hit_pos).length();
                    Vector3f light_ray = (light - hit_pos).normalize();
                    float light_factor =  sqrt(abs(min_hit.normal.dot(light_ray)));
                    min_hit.color = min_hit.color.color / 2 + (Vector3f(255, 255, 255) * CLAMP(light_factor / (list_dist) * l_int , 0, 1));
                }

                min_hit.color.clamp();


            }
        }
   

        (hit ? min_hit.color : bg).writeToBuff(&(buf[(int)(4 * (screen_coord.y * display->getWidth() + screen_coord.x))]));
    }

    display->run();
    display->destory();

  return 0;
}
