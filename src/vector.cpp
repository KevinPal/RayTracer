#include "vector.h"
#include <math.h>
#include <stdio.h>

// Vector 2f

// Copy constructor for vec2 -> vec2
Vector2f::Vector2f(const Vector2f& other) :
    x(other.x), y(other.y) {}

// Copy constructor for vec3 -> vec2. This drops the z component
Vector2f::Vector2f(const Vector3f& other) :
    x(other.x), y(other.y) {}

Vector2f::Vector2f(float x_, float y_) : x(x_),y(y_) {};

Vector2f::Vector2f() : Vector2f(0, 0) {}

float Vector2f::length() {
    return sqrt(x * x + y * y);
}

Vector2f Vector2f::normalize() {
    float len = this->length();
    this->x = x / len;
    this->y = y / len;
    return *this;
}

/*
Vector2f Vector2f::subtract(const Vector2f& v1, const Vector2f& v2) {
    return Vector2f(v1.x - v2.x, v1.y - v2.y);
}

Vector2f Vector2f::add(const Vector2f& v1, const Vector2f& v2) {
    return Vector2f(v1.x + v2.x, v1.y + v2.y);
}
*/

Vector2f Vector2f::operator-(const Vector2f& other) const {
    return Vector2f(this->x - other.x, this->y - other.y);
}

Vector2f Vector2f::operator+(const Vector2f& other) const {
    return Vector2f(this->x + other.x, this->y + other.y);
}

Vector2f Vector2f::operator/(const float& other) const {
    return Vector2f(this->x / other , this->y / other);
}

Vector2f Vector2f::operator/(const Vector2f& other) const {
    return Vector3f(this->x / other.x, this->y / other.y);
}

Vector2f Vector2f::operator*(const Vector2f& other) const {
    return Vector3f(this->x * other.x, this->y * other.y);
}

Vector2f Vector2f::operator*(const float& other) const {
    return Vector2f(this->x * other , this->y * other);
}

bool Vector2f::operator==(const Vector2f& other) const {
    return (this->x == other.x) && (this->y == other.y);
}

bool Vector2f::operator!=(const Vector2f& other) const {
    return !(*this == other);
}

void Vector2f::print(void) const {
    printf("<%f %f>", x, y);
}

// ------ Vector 3f -----

// Copy constructor for vec3 -> vec3
Vector3f::Vector3f(const Vector3f& other) : 
    x(other.x), y(other.y), z(other.z) {}

// Copy constructor for vec2 -> vec3. Sets z to 0
Vector3f::Vector3f(const Vector2f& other) : 
    x(other.x), y(other.y), z(0) {}

Vector3f::Vector3f() : Vector3f(0, 0, 0) {}

// Constructor that defaults z to 0
Vector3f::Vector3f(float x_, float y_):
    Vector3f(x_, y_, 0) {}

Vector3f::Vector3f(float x_, float y_, float z_) :
    x(x_), y(y_), z(z_) {}

float Vector3f::length() const {
    return sqrt(x * x + y * y + z * z);
}

Vector3f Vector3f::normalize() {
    float len = this->length();
    this->x = x / len;
    this->y = y / len;
    this->z = z / len;
    return *this;
}

Vector3f Vector3f::norm() {
    float len = this->length();
    return Vector3f(
        x / len,
        y / len,
        z / len
    );
}

Vector3f Vector3f::operator-(const Vector3f& other) const {
    return Vector3f(this->x - other.x, this->y - other.y, this->z - other.z);
}

Vector3f Vector3f::operator+(const Vector3f& other) const {
    return Vector3f(this->x + other.x, this->y + other.y, this->z + other.z);
}

Vector3f Vector3f::operator/(const float& other) const {
    return Vector3f(this->x / other , this->y / other, this->z / other);
}

Vector3f Vector3f::operator/(const Vector3f& other) const {
    return Vector3f(this->x / other.x, this->y / other.y, this->z / other.z);
}

Vector3f Vector3f::operator*(const Vector3f& other) const {
    return Vector3f(this->x * other.x, this->y * other.y, this->z * other.z);
}

Vector3f Vector3f::operator*(const float& other) const {
    return Vector3f(this->x * other , this->y * other, this->z * other);
}

float Vector3f::dot(const Vector3f& other) const{
    return this->x * other.x + this->y * other.y + this->z * other.z;
}

Vector3f Vector3f::cross(const Vector3f& other) const{
    return Vector3f(
            this->y * other.z - this->z * other.y,
            -(this->x * other.z - this->z * other.x),
            this->x * other.y - this->y * other.x
        );
}

bool Vector3f::operator==(const Vector3f& other) const {
    return (this->x == other.x) && (this->y == other.y) && (this->z == other.z);
}

bool Vector3f::operator!=(const Vector3f& other) const {
    return !(*this == other);
}

bool Vector3f::isClose(const Vector3f& other) const {
    return this->isClose(other, .000001);
}

bool Vector3f::isClose(const Vector3f& other, float epsilon) const {
    return (abs(x - other.x) <= epsilon) && 
                (abs(y - other.y) <= epsilon) && 
                (abs(z - other.z) <= epsilon);
}

void Vector3f::print(void) const {
    printf("<%f %f %f>\n", x, y, z);
}
