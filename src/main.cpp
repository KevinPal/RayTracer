#include <stdio.h>

#include "primitives.h"
#include "mesh.h"
#include "camera.h"
#include "material.h"
#include "display.h"
#include "renderer.h"
#include "antialias.h"
#include "BVH.h"

#include <vector>
#include <math.h>
#include <cassert>
#include <math.h>
#include <stack>
#include <ctime>
#include <chrono>
#include <thread>

#define DO_ANTI_ALIASING 1
#define ANTI_ALIASING_NUM 16

#define BVH_LEAF_SIZE 2
#define CAMERA_NUM 1
#define REC_DEPTH 2

#define RAND ((float) (rand() / (float) RAND_MAX))

#define SCREEN_WIDTH 1000
#define SCREEN_HEIGHT 1000

using namespace std::chrono;

int main (int argc, char **argv) {

    // Setup the display and get the buffer
    Display* display = Display::getInstance();
    display->init(SCREEN_WIDTH, SCREEN_HEIGHT);
    unsigned char* buf = display->getBuffer();

    std::thread display_thread(
        [&]() {
        display->run();
      });

    Material* mat_blue = new DiffuseMaterial(
        Color(0.0, 0.0, 1.0)
    );

    Material* mat_red = new DiffuseMaterial(
        Color(1.0, 0.0, 0.0)
    );

    Material* mat_pink = new MirrorMaterial(
        Color(1.0, 0.0, 1.0)
    );

    Material* mat_cyan = new MirrorMaterial(
        Color(1.0, 1.0, 1.0)
    );
    
    Material* mat_yellow = new MirrorMaterial(
        Color(0.0, 1.0, 1.0)
    );

    Material* mat_grey = new MirrorMaterial(
        Color(0.5, 0.5, 0.5)
    );

    Material* mat_light = new EmissiveMaterial(
        Color(250.0 / 255.0, 1, 244.0 / 255.0)
    );

    Material* mat_forest_green = new DiffuseMaterial(
        Color(0.133, 0.745, 0.133)
    );

    Material* mat_trans = new RefractiveMaterial(
        1.52,
        0.9
    );


    BVHNode scene(mat_pink, BVH_LEAF_SIZE);
    BVHNode lighting(mat_pink, BVH_LEAF_SIZE);


    Plane p(
         Vector3f(0, 0, 15), // was 0 0 1 for dragon
         Vector3f(0, 0, 1),
         mat_red);

    Plane p2(
         Vector3f(0, -0.5, 0),
         Vector3f(0, 1, 0),
         mat_blue);

    Sphere s( 
         Vector3f(0, 5, 5), 
         5,
         mat_pink);

    Sphere s2(
         Vector3f(15, 5, 0), 
         5,
         mat_yellow);

    Sphere s3(
         Vector3f(15, 5, -5), 
         4,
         mat_trans);
    s3.invert = true;

    Triangle t(
            Vector3f(-15, 10, 10),
            Vector3f(0, 20, 0),
            Vector3f(15, 10, 10),
            mat_light);

    /*
    Prism r(
            Vector3f(5, 0, -5),
            Vector3f(0, 1, 0),
            Vector3f(1, 0, 1),
            Vector3f(5, 10, 5),
            mat_yellow);

    Prism r2(
            Vector3f(-7, 1, -6),
            Vector3f(0, 1, 0),
            Vector3f(0, 1, 2),
            Vector3f(5, 5, 5),
            mat_grey);
            */

    //scene.addObject(&s);
    //scene.addObject(&s2);
    //scene.addObject(&s3);
    scene.addObject(&t);
    //scene.addObject(&r);
    //scene.addObject(&r2);

    lighting.addObject(&t);

    /*
    BVHNode m(5);
    m.material.alpha = 0;
    m.fromOBJ("./res/dragon.obj");
    m.partition();
    */


    //scene.addObject(&p);
    //scene.addObject(&p2);
    //6638.986000
    //scene.addObject(&p);
    //scene.addObject(&p2);
   // scene.addObject(&m);

    //scene.addObject(m.bounding_box);


    scene.partition();

    IntersectData data;
    Color bg(100, 100, 100);


    // Specify the camera. Currently hard coding
    // until we get something more fancy

    Camera* cam;
    int cam_num = CAMERA_NUM;
    
    switch(cam_num) {
        case 1:
            cam = new PerspectiveCamera(
                Vector3f(3, 5, -10),
                Vector3f(0, 1.0, 0),
                Vector3f(1.0, 0, 0),
                Vector2f(25.0, 25.0),
                Vector2f(display->getWidth(), display->getHeight()),
                Vector3f(3, 5, -30)
            );
            break;
        case 2:
            // Other angle
            cam = new PerspectiveCamera(
                Vector3f(10, 5, -5),
                Vector3f(0, 1.0, 0),
                Vector3f(1.0, 0, 1),
                Vector2f(25.0, 25.0),
                Vector2f(display->getWidth(), display->getHeight()),
                Vector3f(30, 5, -25)
            );
            break;
        case 3:
            cam = new OrthoCamera(
                Vector3f(0, 5, -25),
                Vector3f(0, 1.0, 0),
                Vector3f(1.0, 0, 0),
                Vector2f(25, 25),
                Vector2f(display->getWidth(), display->getHeight())
            );
            break;
        case 4: // Dragon
            cam = new PerspectiveCamera(
                Vector3f(1, 0, -1),
                Vector3f(0, 1.0, 0),
                Vector3f(1.0, 0, 1),
                Vector2f(2, 2),
                Vector2f(display->getWidth(), display->getHeight()),
                Vector3f(4, 0, -4)
            );
            break;
        case 5:
            cam = new PerspectiveCamera(
                Vector3f(1, 0, -2),
                Vector3f(0, 1.0, 0),
                Vector3f(1.0, 0, 1),
                Vector2f(2, 2),
                Vector2f(display->getWidth(), display->getHeight()),
                Vector3f(5, 0, -5)
            );
            break;
    }

    auto start = duration_cast< milliseconds >(
        system_clock::now().time_since_epoch()
    ).count();

    // Loop across all the rays generated by the camera
    /*
    int i = 0;
    long rays = 0;
    for (auto it = cam->begin(), end = cam->end(); it != end; ++it) {
        Ray ray = *it;
        Vector2f screen_coord = it.getScreenCord();
        Vector3f color = Vector3f(0, 0, 0);

        AntiAliaser* anti_aliaser;
        int count = 0;

        // If we have an anti aliaser, loop across all the rays from it, and do an intersection test
        // Otherwise just do it for the main ray
        //
        if(DO_ANTI_ALIASING) {
            for(anti_aliaser = new GridAntiAliaser(&it, ANTI_ALIASING_NUM); !anti_aliaser->isDone(); ++(*anti_aliaser)) {
                Color hit_color = renderRay(**anti_aliaser, &scene, &lighting, REC_DEPTH, &rays);
                count += 1;
                color = color + hit_color;
            }
        } else  {
            Color hit_color = renderRay(ray, &scene, &lighting, REC_DEPTH, &rays);
            count += 1;
            color = color + hit_color;
        }

        // Average the colors and write to the buffer
        color = color / count;
        Color(color).writeToBuff(&(buf[(int)(4 * (screen_coord.y * display->getWidth() + screen_coord.x))]));

        // Print progress
        i += 1;
        if((i * 100 / (display->getWidth() * display->getHeight())) % 5 == 0)
            printf("%.3f\%\n",  i * 100.0 / (display->getWidth() * display->getHeight()));
    }
    */

    Ray* hostRayBuffer = (Ray*) malloc(sizeof(Ray) * display->getWidth() * display->getHeight() * 2);

    float theta = 0;

    for (auto it = cam->begin(), end = cam->end(); it != end; ++it) {
        Vector2f screen_coord = it.getScreenCord();
        if(screen_coord.x < display->getWidth() && screen_coord.y < display->getHeight()) {
            hostRayBuffer[(int)(screen_coord.x + screen_coord.y * display->getWidth())] = *it;
        }
    }

    auto last = duration_cast< milliseconds >(
        system_clock::now().time_since_epoch()
    ).count();
    auto curr = duration_cast< milliseconds >(
        system_clock::now().time_since_epoch()
    ).count();

    int frames = 0;

    int frame_time = 0;

    while(true) {

        frames += 1;

        curr = duration_cast< milliseconds >(
            system_clock::now().time_since_epoch()
        ).count();

        theta += (curr - last) / 1000.0 * 3;

        frame_time += (curr - last);

        if(frame_time > 1000) {
            frame_time -= 1000;
            printf("FPS: %d\n", frames);
            frames = 0;
        }

        s.center[0] = cos(theta) * 4;
        s.center[1] = sin(theta) * 7 + 5;

        renderRays(hostRayBuffer, buf, display->getWidth(), display->getHeight(), &s);


        display->redraw();

        last = curr;

    }

    free(hostRayBuffer);


    auto end = duration_cast< milliseconds >(
        system_clock::now().time_since_epoch()
    ).count();
    printf("Took %f\n", (end - start) / 1000.0);
    //printf("Used %ld rays\n", rays);

    // Write to a PNG for handing
    //display->writeToPNG("images/mp3/reflect2.png");

    // Display the image
    //display->run();
    //display_thread.join();
    display->destory();

  return 0;
}
