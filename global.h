#include <GL/glew.h>

#define ACTOR_SIZE 5

// All devices (even old ones) can handle this block size,
// 512 is the max on the oldest devices.
#define MAX_BLOCK_SIZE 512

#define SUCEPTABLE 1
#define INFECTED 2
#define DEAD 3
#define RECOVERED 4

#define ROTATION_MATRIX_LENGTH 16

#define INFECTION_RADIUS_RED 0.2f
#define INFECTION_RADIUS_GREEN 0.5f
#define INFECTION_RADIUS_BLUE 0.3f
#define INFECTION_RADIUS_ALPHA 0.2f

#define MAX_MOVE_SPEED 1.0f
#define ROTATION_SPEED_MAX 0.7f

extern const GLfloat cubeVertices[];
extern const GLuint  cubeIndices[];

extern const int cubeVerticesSize;
extern const int cubeIndicesSize;

float getRandomFloat(float minimum, float maximum);