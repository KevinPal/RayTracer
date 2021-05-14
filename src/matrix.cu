
#include "matrix.h"
#include "vector.h"
#include <string.h>
#include <stdio.h>
#include <cassert>

Matrix44::Matrix44(float data[16]) {

    memcpy(this->data, data, 16 * sizeof(float));

}

Vector3f Matrix44::transform(Vector3f v) {
    float x = data[0] * v.x + data[1] * v.y + data[2] * v.z + data[3];
    float y = data[4] * v.x + data[5] * v.y + data[6] * v.z + data[7];
    float z = data[8] * v.x + data[9] * v.y + data[10] * v.z + data[11];

    return Vector3f(x, y, z);
}

// From https://stackoverflow.com/a/7596981
Matrix44 Matrix44::invert() {

    float i00 = data[0];
    float i01 = data[1];
    float i02 = data[2];
    float i03 = data[3];

    float i10 = data[4];
    float i11 = data[5];
    float i12 = data[6];
    float i13 = data[7];

    float i20 = data[8];
    float i21 = data[9];
    float i22 = data[10];
    float i23 = data[11];

    float i30 = data[12];
    float i31 = data[13];
    float i32 = data[14];
    float i33 = data[15];

    float m[16];

    float s0 = i00 * i11 - i10 * i01;
    float s1 = i00 * i12 - i10 * i02;
    float s2 = i00 * i13 - i10 * i03;
    float s3 = i01 * i12 - i11 * i02;
    float s4 = i01 * i13 - i11 * i03;
    float s5 = i02 * i13 - i12 * i03;

    float c5 = i22 * i33 - i32 * i23;
    float c4 = i21 * i33 - i31 * i23;
    float c3 = i21 * i32 - i31 * i22;
    float c2 = i20 * i33 - i30 * i23;
    float c1 = i20 * i32 - i30 * i22;
    float c0 = i20 * i31 - i30 * i21;

    // Should check for 0 determinant

    float det_denom = (s0 * c5 - s1 * c4 + s2 * c3 + s3 * c2 - s4 * c1 + s5 * c0);
    assert(det_denom != 0);
    float invdet = 1 / det_denom;

    m[0] = (i11 * c5 - i12 * c4 + i13 * c3) * invdet;
    m[1] = (-i01 * c5 + i02 * c4 - i03 * c3) * invdet;
    m[2] = (i31 * s5 - i32 * s4 + i33 * s3) * invdet;
    m[3] = (-i21 * s5 + i22 * s4 - i23 * s3) * invdet;

    m[4] = (-i10 * c5 + i12 * c2 - i13 * c1) * invdet;
    m[5] = (i00 * c5 - i02 * c2 + i03 * c1) * invdet;
    m[6] = (-i30 * s5 + i32 * s2 - i33 * s1) * invdet;
    m[7] = (i20 * s5 - i22 * s2 + i23 * s1) * invdet;

    m[8] = (i10 * c4 - i11 * c2 + i13 * c0) * invdet;
    m[9] = (-i00 * c4 + i01 * c2 - i03 * c0) * invdet;
    m[10] = (i30 * s4 - i31 * s2 + i33 * s0) * invdet;
    m[11] = (-i20 * s4 + i21 * s2 - i23 * s0) * invdet;

    m[12] = (-i10 * c3 + i11 * c1 - i12 * c0) * invdet;
    m[13] = (i00 * c3 - i01 * c1 + i02 * c0) * invdet;
    m[14] = (-i30 * s3 + i31 * s1 - i32 * s0) * invdet;
    m[15] = (i20 * s3 - i21 * s1 + i22 * s0) * invdet;

    return Matrix44(m);
}

Matrix44 Matrix44::transpose() {
    float out[16];
    for(int i = 0; i < 4; i++) {
        for(int j = 0; j < 4; j++) {
            out[i * 4 + j] = data[j * 4 + i];
        }
    }
    return Matrix44(out);
}

void Matrix44::print() {

    for(int i = 0; i < 4; i++) {
        for(int j = 0; j < 4; j++) {
            printf("%.02f ", data[i*4 + j]);
        }
        printf("\n");
    }
    printf("\n");
}
