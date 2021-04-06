#include "renderer.h"
#include "renderable.h"
#include "mesh.h"
#include "ray.h"

#include <vector>
#include <math.h>
#include <cassert>
#include <math.h>
#include <cmath>
#include <stack>

#define CLAMP(X, A, B) ((X) < (A) ? (A) : ((X) > (B) ? (B) : (X)))

#define DO_ALPHA 1
#define DO_SPEC 1


Color renderRay(Ray ray, Mesh* scene, Mesh* lights, int depth) {

    // We hard code a single light position for now
    Vector3f light(10, 10, -20);
    float l_int = 10.0;

    // Test if the ray hits anything in the scene
    bool hit = false;
    IntersectData min_hit = scene->intersects(ray);
    Vector3f hit_pos = ray.getPoint(min_hit.t);
    hit = (min_hit.t >= 0);

    if(!hit)
        return Color(0.0, 0.0, 0.0);
    if(depth <= 1) {
        return min_hit.material->emission;
    } else {
        Color base = min_hit.material->emission;
        Color rec_color = Color(0, 0, 0);

        float weighting = 0;
        int samples = depth;

        int light_samples = depth;

        for(int i = 0; i < samples; i++) {

            Vector3f hit_norm = min_hit.normal;
            float angleCos = hit_norm.angleCos(ray.direction);
            if(0 < angleCos && angleCos < 1) {
                hit_norm = hit_norm * -1.0;
            }

            //Vector3f out_dir = hit_norm + Vector3f::randomSphere().norm();
            Vector3f out_dir = min_hit.material->scatterRay(ray.direction, hit_norm);

            if(out_dir.isClose(Vector3f()))
                out_dir = min_hit.normal;

            Ray out_ray = Ray(hit_pos, out_dir);
            out_ray.origin = out_ray.getPoint(1e-4);
            //Ray out_ray = Ray(hit_pos, min_hit.normal);


            Color base_rec_color = renderRay(out_ray, scene, lights, (int) depth / 10);
            float brdf = min_hit.material->BRDF(ray, out_ray, min_hit.normal);

            rec_color = rec_color +  base_rec_color * brdf;
            weighting += brdf;
        }

        for(Renderable* light_obj : lights->objects) {
            AABB* aabb = light_obj->bounding_box;
            if(aabb) {
                for(int i = 0; i < depth; i++) {
                    Ray out_ray;
                    Vector3f end_pos = Vector3f::randomVect(aabb->mins[0], aabb->maxs[0],
                                                                 aabb->mins[1], aabb->maxs[1],
                                                                 aabb->mins[2], aabb->maxs[2]);
                    out_ray.fromPoints(hit_pos, end_pos);
                    out_ray.direction.normalize();
                    out_ray.origin = out_ray.getPoint(1e-4);

                    IntersectData light_intersect = scene->intersects(out_ray);
                    if(light_intersect.object == light_obj && light_intersect.t >= 0) {
                        Color light_color = light_obj->material->emission;
                        float brdf = min_hit.material->BRDF(ray, out_ray, min_hit.normal);

                        rec_color = rec_color + light_color * brdf;
                        weighting += brdf;
                    }
                }
            }
        }

        rec_color  = rec_color / weighting;
        rec_color = rec_color * min_hit.material->albedo + base;
        rec_color = rec_color + min_hit.material->albedo * 0.1;
        return rec_color;
    }

}

/*
// The bulk of the rendering code is here for now, but will
// get moved as the lightning logic gets cleaned up
IntersectData renderRay(Ray ray, Mesh* scene, int depth) {

    // We hard code a single light position for now
    Vector3f light(10, 10, -20);
    float l_int = 10.0;

    // Test if the ray hits anything in the scene
    bool hit = false;
    IntersectData min_hit = scene->intersects(ray);
    hit = (min_hit.t >= 0);
   
    if(hit) {
        Vector3f hit_pos = ray.getPoint(min_hit.t);
        float spec = min_hit.material.specular;

        // 
        // transperency support. Everytime the ray hits something,
        // we move it slightly forward and send the ray out again, until
        // it hits something opaque or nothing at all. We then blend
        // these colors backwards according to their alphas to get the final
        // color
        //
        if(DO_ALPHA) {
            std::stack<Material> alpha_stack;
            Ray alpha_ray;
            alpha_ray.direction = ray.direction;
            alpha_ray.origin = hit_pos;

            alpha_stack.push(min_hit.material);

            // Repededly move the ray forward and retest
            while(alpha_stack.top().alpha != 1) {
                alpha_ray.origin = alpha_ray.getPoint(1e-4);
                //IntersectData alpha_data = scene->intersects(alpha_ray);
                IntersectData alpha_data = renderRay(alpha_ray, scene, depth - 1);

                if(alpha_data.t != nan("") && (alpha_data.t >= 0)) {
                    alpha_stack.push(alpha_data.material);
                    alpha_ray.origin = alpha_ray.getPoint(alpha_data.t);
                } else {
                    break;
                }
            }

            // Process the colors in revere and blend the colors together
            min_hit.material = alpha_stack.top();
            alpha_stack.pop();
            while(!alpha_stack.empty()) {
                Material mix_color = alpha_stack.top();

                min_hit.material.color = (mix_color.color * (mix_color.alpha)) + (min_hit.material.color * (1 - mix_color.alpha));
                alpha_stack.pop();
            }
        }
        

        //
        // Recursive reflection support. Light gets bounced across the normal if the material is reflective,
        // and gets recursively processed
        //
        if(DO_SPEC) {
            if((spec > 0) && (depth > 0)) {
                Ray reflection_ray;
                reflection_ray.origin = hit_pos;
                reflection_ray.direction = (min_hit.normal * 2 * ray.direction.dot(min_hit.normal) - ray.direction) * -1;

                reflection_ray.origin = reflection_ray.getPoint(1e-4);
                //IntersectData reflection_data = scene->intersects(reflection_ray);
                IntersectData reflection_data = renderRay(reflection_ray, scene, depth-1);

                if(reflection_data.t != nan("") && (reflection_data.t >= 0)) {
                    min_hit.material.color = (min_hit.material.color * (1-spec)) + (reflection_data.material.color * spec);
                }
            }
        }

        // 
        // Shadow and shading. We check if the shadow ray has a direct path to
        // the light. If so, we shadw acording to phong model
        // Otherwise, we blacken the color to represent some ambient light
        //

        Ray shadow_ray;
        float dist;
        shadow_ray.fromPoints(hit_pos, light);
        shadow_ray.origin = shadow_ray.getPoint(1e-4);
        IntersectData shadow_data = scene->intersects(shadow_ray);

        if(shadow_data.t != nan("") && (shadow_data.t >= 0) && (shadow_data.t < 1)) {
            min_hit.material.color = min_hit.material.color / 2;
        } else {
            float list_dist = (light - hit_pos).length();
            Vector3f light_ray = (light - hit_pos).normalize();
            float light_factor =  sqrt(abs(min_hit.normal.dot(light_ray)));

            Vector3f light_bounce = min_hit.normal * light_ray.dot(min_hit.normal) * 2 - light_ray;

            Vector3f cAmbient = min_hit.material.color;
            Vector3f cDiffuse = Vector3f(1, 1, 1);
            Vector3f cSpecular = Vector3f(1, 1, 1);

            float kAmbient = 0.5;
            float kDiffuse = CLAMP(light_factor / (list_dist) * l_int , 0, 1);
            float kSpecular = pow(light_bounce.dot(ray.direction), 100);

            min_hit.material.color = cAmbient * kAmbient + cDiffuse * kDiffuse + cSpecular * kSpecular;
        }

        min_hit.material.color.clamp();
    }

    return min_hit;
}
*/
