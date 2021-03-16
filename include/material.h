#ifndef MATERIAL_H
#define MATERIAL_H

#include "color.h"


/* Represents a material. This class is not fully acurate yet,
 * but works for now. Specifies a color, its transperency,
 * and its reflectiveness
 *
 */
class Material {
    public:
        Color color;
        float alpha;
        float diffuse; // unused
        float specular;

        // Basic fully black color
        Material() :
            color(Color()), alpha(1), diffuse(1), specular(0) {}

        // Opaque, nonreflective color
        Material(Color c) :
            color(c), alpha(1), diffuse(1), specular(0) {};

        // Generic color with specified color, transperncy, and reflectiveness
        Material(Color c, float alpha_, float diffuse_, float specular_) :
            color(c), alpha(alpha_), diffuse(diffuse_), specular(specular_) {};

        // Copy constructor
        Material(const Material& other) :
            color(other.color), alpha(other.alpha), diffuse(other.diffuse), specular(other.specular) {};

};

#endif
