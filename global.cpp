#include <stdlib.h>
#include "global.h"

const GLfloat cubeVertices[] = {  1, 1, 1,  -1, 1, 1,  -1,-1, 1,   1,-1, 1,
							1, 1, 1,   1,-1, 1,   1,-1,-1,   1, 1,-1,
							1, 1, 1,   1, 1,-1,  -1, 1,-1,  -1, 1, 1,
							-1, 1, 1,  -1, 1,-1,  -1,-1,-1,  -1,-1, 1,
							-1,-1,-1,   1,-1,-1,   1,-1, 1,  -1,-1, 1,
							1,-1,-1,  -1,-1,-1,  -1, 1,-1,   1, 1,-1  };

const GLuint cubeIndices[]  = { 0, 1, 2,   2, 3, 0,
						   4, 5, 6,   6, 7, 4,
						   8, 9,10,  10,11, 8,
						   12,13,14,  14,15,12,
						   16,17,18,  18,19,16,
						   20,21,22,  22,23,20 };


const int cubeVerticesSize = sizeof(cubeVertices);
const int cubeIndicesSize = sizeof(cubeIndices);

float getRandomFloat(float minimum, float maximum) {
	return minimum + (float)rand()/((float)RAND_MAX/(maximum-minimum));
}
