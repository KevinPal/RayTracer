#include <stdio.h>

#include "primitives.h"
#include "mesh.h"
#include "camera.h"
#include "material.h"
#include "display.h"
#include "renderer.h"
#include "antialias.h"
#include "BVH.h"
#include "matrix.h"

#include <vector>
#include <math.h>
#include <cassert>
#include <math.h>
#include <stack>
#include <ctime>
#include <chrono>
#include <thread>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstring>

#define DO_ANTI_ALIASING 1
#define ANTI_ALIASING_NUM 16

#define BVH_LEAF_SIZE 2
#define CAMERA_NUM 4
#define REC_DEPTH 2

#define RAND ((float) (rand() / (float) RAND_MAX))

#define SCREEN_WIDTH 500
#define SCREEN_HEIGHT 500

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      printf("%s%s\n", "CUDA error: ", cudaGetErrorString(err));              \
      printf("%s%s\n", "Failed to run stmt ", #stmt);                         \
    }                                                                     \
  } while (0)

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

    // Parse in input texture
    std::ifstream infile("./res/texture.ppm", std::ios_base::in);
    assert(infile.is_open());
    std::string line;

    int line_num = 0;

    char* hostTextureMem = NULL;
    int textureWidth;
    int textureHeight;
    int textureIdx = 0;

    // assuming scale is 255
    bool read_scale = false;

    while(std::getline(infile, line)) {
        line_num++;

        if(line_num == 1) {
            assert(line == "P3");
        } else if(line.at(0) == '#') {
            std::cout << "Texture comment: " << line << std::endl;
        } else if(hostTextureMem == NULL) {
            std::istringstream ss(line);

            if(!(ss >> textureWidth >> textureHeight)) {
                std::cout << "Error reading texture dims. Line: " << line << std::endl;
            } else {
                hostTextureMem = (char*) malloc(textureWidth * textureHeight * 3 * sizeof(char));
                printf("Texture size %d x %d\n", textureWidth, textureHeight);
            }
        } else {
            if(!read_scale) {
                read_scale = true;
                continue;
            }
            //std::cout << textureIdx << " " << line << std::endl;
            hostTextureMem[textureIdx++] = std::stoi(line);
        }
    }

    printf("Texutre[0:3]: %d %d %d\n", hostTextureMem[0], hostTextureMem[1], hostTextureMem[2]);


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

    int t_size = 50000;
    int t_y = -2;
    int t_z = 3;
    int t_x = 0;

    Triangle t(
            Vector3f(-t_size + t_x, t_y, t_size + t_z),
            Vector3f(-t_size + t_x, t_y, -t_size + t_z),
            Vector3f(t_size + t_x, t_y, -t_size + t_z),
            mat_light);

    Triangle t2(
            Vector3f(-t_size + t_x, t_y, t_size + t_z),
            Vector3f(t_size + t_x, t_y, -t_size + t_z),
            Vector3f(t_size + t_x, t_y, t_size + t_z),
            mat_light);


    lighting.addObject(&t);

    BVHNode m(mat_forest_green, 3);
    m.fromOBJ("./res/cube.obj");
    m.addObject(&t);
    m.addObject(&t2);

    m.partition();


    //scene.addObject(&p);
    //scene.addObject(&p2);
    //6638.986000
    //scene.addObject(&p);
    //scene.addObject(&p2);
    //scene.addObject(&m);

    //scene.addObject(m.bounding_box);
    //
    //Vector3f(0, -0.5, 0)
    //Vector3f(0, 1, 0)


    scene.partition();

    IntersectData data;

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
                Vector3f(2, 0, -2),
                Vector3f(0, 1.0, 0),
                Vector3f(1.0, 0, 1),
                Vector2f(2, 2),
                Vector2f(display->getWidth(), display->getHeight()),
                Vector3f(5, 0, -5)
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

    device_data_t device_data;

    renderRaysInit(hostRayBuffer, &m, &device_data, display->getWidth(), display->getHeight(), hostTextureMem, textureWidth, textureHeight);


    while(true) {

        float transform_raw[16] = {
            cos(theta), 0, sin(theta), 0,
            0,          1,          0, 0,
           -sin(theta), 0, cos(theta), 0,
            0,          0,          0, 1
        };

        Matrix44 transform(transform_raw);
        Matrix44 inv_transform = transform.invert();
        Matrix44 inv_t_transform = inv_transform.transpose();
        
        wbCheck(cudaMemcpy(device_data.deviceTransform, &inv_transform, sizeof(Matrix44), cudaMemcpyHostToDevice));
        wbCheck(cudaMemcpy(device_data.deviceInvTransform, &inv_t_transform, sizeof(Matrix44), cudaMemcpyHostToDevice));

        frames += 1;

        curr = duration_cast< milliseconds >(
            system_clock::now().time_since_epoch()
        ).count();

        theta += (curr - last) / 1000.0;

        frame_time += (curr - last);

        if(frame_time > 1000) {
            printf("FPS: %d (%d ms)\n", frames, frame_time / frames);
            frame_time = 0;
            frames = 0;
        }

        renderRays(&device_data, buf, display->getWidth(), display->getHeight(), textureWidth, textureHeight);

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
