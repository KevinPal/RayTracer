#ifndef MATERIAL_H
#define MATERIAL_H

#include "color.h"

class Material {
    public:
        Color color;
        float alpha;
        float diffuse;
        float specular;

        Material() :
            color(Color()), alpha(1), diffuse(1), specular(1) {}

        Material(Color c) :
            color(c), alpha(1), diffuse(1), specular(1) {};

        Material(Color c, float alpha_, float diffuse_, float specular_) :
            color(c), alpha(alpha_), diffuse(diffuse_), specular(specular_) {};

        Material(const Material& other) :
            color(other.color), alpha(other.alpha), diffuse(other.diffuse), specular(other.specular) {};

};

#endif
