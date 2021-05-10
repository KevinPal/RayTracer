#ifndef VECTOR_H
#define VECTOR_H

#include <cuda.h>
#include <cuda_runtime_api.h>
#define HOSTDEVICE __host__ __device__

class Vector2f;
class Vector3f;

/*
 * Represents a 2 dimentional vector, and operators
 * on them.
 */
class Vector2f {
    public:
        float x;
        float y;

        // Constructors. Any unspecified component will default to 
        // 0
        Vector2f(const Vector2f& other);
        Vector2f(const Vector3f& other);
        Vector2f(float x_, float y_);
        Vector2f();

        // Gets the length of this vector
        float length();

        // Normalizes this vector
        Vector2f normalize();

        // Vector Vector operations
        Vector2f operator-(const Vector2f& other) const;
        Vector2f operator+(const Vector2f& other) const;
        Vector2f operator/(const Vector2f& other) const;
        Vector2f operator*(const Vector2f& other) const;

        // Component wise operations
        Vector2f operator/(const float& other) const;
        Vector2f operator*(const float& other) const;

        // Comparison operations
        bool operator==(const Vector2f& other) const;
        bool operator!=(const Vector2f& other) const;

        // Prints this vector
        void print(void) const;
};

/*
 * Represents a 3 dimentional vector, and operators
 * on them
 */
class Vector3f {

    public:
        float x;
        float y;
        float z;

        // Constructors. Any unspecified component will defaul to 0
        HOSTDEVICE Vector3f();
        HOSTDEVICE Vector3f(const Vector3f& other);
        HOSTDEVICE Vector3f(const Vector2f& other);
        HOSTDEVICE Vector3f(float x_, float y_);
        HOSTDEVICE Vector3f(float x_, float y_, float z_);

        // Gets the length or squared length of this vector
        HOSTDEVICE float length() const;
        HOSTDEVICE float length2() const;
        // Normalizes this vector
        HOSTDEVICE Vector3f normalize();
        // Returns the normal of this vector. This vector is unchanged
        HOSTDEVICE Vector3f norm();
        HOSTDEVICE Vector3f abs() const;

        // Vector vector operations
        HOSTDEVICE Vector3f operator-(const Vector3f& other) const;
        HOSTDEVICE Vector3f operator+(const Vector3f& other) const;
        HOSTDEVICE Vector3f operator/(const Vector3f& other) const;
        HOSTDEVICE Vector3f operator*(const Vector3f& other) const;

        // component wise operations
        HOSTDEVICE Vector3f operator/(const float& other) const;
        HOSTDEVICE Vector3f operator*(const float& other) const;

        // Dot and cross product
        HOSTDEVICE float dot(const Vector3f& other) const;
        HOSTDEVICE Vector3f cross(const Vector3f& other) const;

        // Equaility operations
        HOSTDEVICE bool operator==(const Vector3f& other) const;
        HOSTDEVICE bool operator!=(const Vector3f& other) const;

        // Array access operators
        HOSTDEVICE float& operator[](int i);
        HOSTDEVICE float operator[](int i) const;

        // Determines if two vectors are "close" by some epsilon
        HOSTDEVICE bool isClose(const Vector3f& other) const;
        HOSTDEVICE bool isClose(const Vector3f& other, float epsilon) const;

        HOSTDEVICE float angleCos(const Vector3f& other) const;

        // Prints the vector to console
        HOSTDEVICE void print(void) const;

        // Random vector generation
        static Vector3f randomVect(float minX, float maxX, float minY, float maxY, float minZ, float maxZ);
        static Vector3f randomVect(float min, float max);
        static Vector3f randomSphere();
};

#endif
