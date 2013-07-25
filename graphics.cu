#include "graphics_h.cu"
#include "global.h"

#include <cuda_gl_interop.h>
#include <helper_cuda.h>
#include <curand.h>
#include <curand_kernel.h>

#include <string.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <GL/glew.h>
#include <GL/freeglut.h>

GLfloat * Entity::defaultSphereVertices(0);
GLuint * Entity::defaultSphereIndices(0);
int Entity::defaultSphereVerticeSize(0);
int Entity::defaultSphereIndiceSize(0);

Entity::Entity(const GLfloat * baseVertices, int baseVerticesBytes, const GLuint * baseIndices, int baseIndicesBytes, int numEntities, GLenum drawMode, bool cleanupBase) :
	numEntities(numEntities),
	baseVerticesLength(baseVerticesBytes / sizeof(GLfloat)),
	baseVerticesBytes(baseVerticesBytes),
	baseVerticesNum(baseVerticesLength / 3),
	baseIndicesLength(baseIndicesBytes / sizeof(GLuint)),
	baseColoursLength(baseVerticesNum * Entity::COLOUR_ELEMENTS),
	verticesLength(baseVerticesLength * numEntities),
	indicesLength(baseIndicesLength * numEntities),
	coloursLength(baseColoursLength * numEntities),
	verticesBytes(verticesLength * sizeof(GLfloat)),
	indicesBytes(indicesLength * sizeof(GLuint)),
	coloursBytes(coloursLength * sizeof(GLfloat)),
	cleanupBase(cleanupBase),
	drawMode(drawMode),
	centersLength(numEntities * 3),
	centersBytes(centersLength * sizeof(GLfloat)),
	initialisedForCuda(false),
	bufferVerticeData(false),
	bufferColourData(false),
	hostBaseVertices(baseVertices),
	baseIndices(baseIndices),
	mappedCudaResources(false),
	deviceCenters(0),
	deviceBaseVertices(0),
	verticesNum(verticesLength / 3)
{
	this->indices = new GLuint[this->indicesLength];
	this->hostVertices = new GLfloat[this->verticesLength];
	this->hostColours = new GLfloat[this->coloursLength];
	this->hostCenters = new GLfloat[this->centersLength];

	this->vertices = this->hostVertices;
	this->colours = this->hostColours;
	this->centers = this->hostCenters;
	this->baseVertices = this->hostBaseVertices;
	
	// Initialise indices (these never change).
	for(int n = 0;n<this->numEntities;n++) {
		int startIndexIndicesArray = n * this->baseIndicesLength;
		int offsetIndice =			 n * this->baseVerticesNum;

		for(int i = 0; i<this->baseIndicesLength; i++) {
			indices[i + startIndexIndicesArray] = baseIndices[i] + offsetIndice;
		}
	}

	// Assume center starts at 0,0,0 - this is decided by baseVertices.
	for(int n = 0;n<this->centersLength;n++) {
		this->centers[n] = 0;
	}

	GLuint buffers[2]; 
	glGenBuffers(2, buffers);

	verticeBufferId = buffers[0];
	colourBufferId = buffers[1];

	glBindBuffer( GL_ARRAY_BUFFER, this->verticeBufferId );
	glBufferData( GL_ARRAY_BUFFER, this->verticesBytes, 0, GL_DYNAMIC_DRAW );
	glBindBuffer( GL_ARRAY_BUFFER, 0);

	glBindBuffer( GL_ARRAY_BUFFER, this->colourBufferId );
	glBufferData( GL_ARRAY_BUFFER, this->coloursBytes, 0, GL_DYNAMIC_DRAW );
	glBindBuffer( GL_ARRAY_BUFFER, 0);
}

Entity::~Entity() {
	if(mappedCudaResources) {
		cudaGraphicsUnregisterResource(resources[0]);
		cudaGraphicsUnregisterResource(resources[1]);
	}

	if(initialisedForCuda) {
		cudaFree(centers);
		cudaFree((void*)baseVertices);
	}

	delete[] indices;
	delete[] hostColours;
	delete[] hostVertices;
	delete[] hostCenters;
	
	if(cleanupBase) {
		delete[] hostBaseVertices;
		delete[] baseIndices;
	}

	GLuint buffers[2];
	buffers[0] = verticeBufferId;
	buffers[1] = colourBufferId;

	glDeleteBuffers(2,buffers);
}

void Entity::setPosition(int entity, GLfloat x, GLfloat y, GLfloat z, GLfloat sizeMultiplierX, GLfloat sizeMultiplierY, GLfloat sizeMultiplierZ) {
	setPositionRaw(entity, this->baseVertices, this->baseVerticesLength, this->vertices, this->centers, x, y, z, sizeMultiplierX, sizeMultiplierY, sizeMultiplierZ);
	bufferVerticeData = true;
}

void Entity::setColour(int entity, GLfloat red, GLfloat green, GLfloat blue, GLfloat alpha) {
	setColourRaw(entity, this->colours, this->baseColoursLength, red, green, blue, alpha);
	bufferColourData = true;
}

void Entity::draw(GLfloat cameraDistance, GLfloat cameraAngleX, GLfloat cameraAngleY) {
	glBindBuffer(GL_ARRAY_BUFFER, this->colourBufferId);
	if(this->bufferColourData) {
		glBufferSubData(GL_ARRAY_BUFFER, 0, this->coloursBytes, this->colours);
		this->bufferColourData = false;
	}
	glColorPointer(4, GL_FLOAT, 0, 0);
	glEnableClientState(GL_COLOR_ARRAY);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindBuffer(GL_ARRAY_BUFFER, this->verticeBufferId);
	if(this->bufferVerticeData) {
		glBufferSubData(GL_ARRAY_BUFFER, 0, this->verticesBytes, this->vertices);
		this->bufferVerticeData = false;
	}
	glVertexPointer(3, GL_FLOAT, 0, 0);
	glEnableClientState(GL_VERTEX_ARRAY);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glPushMatrix();

	// Position/rotate camera (based on user input received earlier).
	glTranslatef(0, 0, -cameraDistance);
	glRotatef(cameraAngleX, 1, 0, 0);   // pitch
	glRotatef(cameraAngleY, 0, 1, 0);   // heading

	glPushMatrix();

	// Draw cubes.
	glDrawElements(this->drawMode, this->indicesLength, GL_UNSIGNED_INT, this->indices);

	// Reset matrix and states.
	glDisableClientState(GL_COLOR_ARRAY);
	glDisableClientState(GL_VERTEX_ARRAY);

	glPopMatrix();
	
	glPopMatrix();
}

// Code modified from stack overflow from a post by datenwolf
// Link: http://stackoverflow.com/questions/7957254/connecting-sphere-vertices-opengl
//
// There was a bug in the code from the above link, explained here:
// Link: http://stackoverflow.com/questions/14080932/implementing-opengl-sphere-example-code 
#define M_PI 3.14159265358979323846
#define M_PI_2 1.57079632679489661923
void Entity::buildSphereRaw(GLfloat ** verticesOutput, int * verticeLengthOutput, GLuint ** indicesOutput, int * indiceLengthOutput, float radius, unsigned int rings, unsigned int sectors) {
	GLfloat * vertices;
	GLuint * indices;

	int verticeLength, indiceLength;

    float const R = 1./(float)(rings-1);
    float const S = 1./(float)(sectors-1);
    int r, s;

	verticeLength = rings * sectors * 3;
    vertices = new GLfloat[verticeLength];

	int index = 0;
    for(r = 0; r < rings; r++) {
		for(s = 0; s < sectors; s++) {
            float const y = sin( -M_PI_2 + M_PI * r * R );
            float const x = cos(2*M_PI * s * S) * sin( M_PI * r * R );
            float const z = sin(2*M_PI * s * S) * sin( M_PI * r * R );

            vertices[index]     = x * radius;
            vertices[index + 1] = y * radius;
            vertices[index + 2] = z * radius;
			index += 3;
		}
    }

	index = 0;
	indiceLength = rings * sectors * 4;
    indices = new GLuint[indiceLength];
    for(r = 0; r < rings-1; r++) {
		for(s = 0; s < sectors-1; s++) {
            indices[index    ] = r * sectors + s;
			indices[index + 1] = r * sectors + (s+1);
            indices[index + 2] = (r+1) * sectors + (s+1);
			indices[index + 3] = (r+1) * sectors + s;
			index += 4;
		}
	}

	*verticesOutput = vertices;
	*verticeLengthOutput = verticeLength;
	*indicesOutput = indices;
	*indiceLengthOutput = indiceLength;
}


Entity * Entity::buildSphere(float radius, unsigned int rings, unsigned int sectors, int numEntities)
{
	GLfloat * vertices;
	GLuint * indices;
	int verticeLength, indiceLength;
	Entity::buildSphereRaw(&vertices, &verticeLength, &indices, &indiceLength, radius, rings, sectors);
	return new Entity(vertices, verticeLength * sizeof(GLfloat), indices, indiceLength * sizeof(GLuint), numEntities, GL_QUADS, true);
}

Entity * Entity::buildDefaultSphere(int numEntities) {
	if(Entity::defaultSphereVertices == 0) {
		Entity::buildSphereRaw(&Entity::defaultSphereVertices,
							   &Entity::defaultSphereVerticeSize,
							   &Entity::defaultSphereIndices,
							   &Entity::defaultSphereIndiceSize,
							   1,
							   12,
							   24);

		Entity::defaultSphereIndiceSize *= sizeof(GLuint);
		Entity::defaultSphereVerticeSize *= sizeof(GLfloat);
	}

	return new Entity(Entity::defaultSphereVertices, Entity::defaultSphereVerticeSize, Entity::defaultSphereIndices, Entity::defaultSphereIndiceSize, numEntities, GL_QUADS, false);
}

Entity * Entity::buildCube(int numEntities) {
	return new Entity(cubeVertices, cubeVerticesSize, cubeIndices, cubeIndicesSize, numEntities, GL_TRIANGLES, false);
}


__device__ __host__ void Entity::clearRotationMatrix(GLfloat * rotationMatrix) {
	rotationMatrix[0] = 1.0f;
	rotationMatrix[1] = 0;
	rotationMatrix[2] = 0;
	rotationMatrix[3] = 0;
	rotationMatrix[4] = 0;
	rotationMatrix[5] = 1.0f;
	rotationMatrix[6] = 0;
	rotationMatrix[7] = 0;
	rotationMatrix[8] = 0;
	rotationMatrix[9] = 0;
	rotationMatrix[10] = 1.0f;
	rotationMatrix[11] = 0;
	rotationMatrix[12] = 0;
	rotationMatrix[13] = 0;
	rotationMatrix[14] = 0;
	rotationMatrix[15] = 1.0f;
}

void Entity::getRotationMatrix(GLfloat angleX, GLfloat angleY, GLfloat angleZ, GLfloat moveX, GLfloat moveY, GLfloat moveZ, GLfloat * rotationMatrix) {
	const int n = 4;

	for(int i=0;i<ROTATION_MATRIX_LENGTH;i++) {
		rotationMatrix[i] = 0;
	}
	rotationMatrix[0 + (0 * n)] = cos(angleY)*cos(angleZ);
	rotationMatrix[0 + (1 * n)] = cos(angleY)*sin(angleZ);
	rotationMatrix[0 + (2 * n)] = -sin(angleY);

	rotationMatrix[1 + (0 * n)] = cos(angleZ)*sin(angleX)*sin(angleY)-cos(angleX)*sin(angleZ);
	rotationMatrix[1 + (1 * n)] = cos(angleX)*cos(angleZ)+sin(angleX)*sin(angleY)*sin(angleZ);
	rotationMatrix[1 + (2 * n)] = cos(angleY)*sin(angleX);

	rotationMatrix[2 + (0 * n)] = cos(angleX)*cos(angleZ)*sin(angleY)+sin(angleX)*sin(angleZ);
	rotationMatrix[2 + (1 * n)] = -cos(angleZ)*sin(angleX)+cos(angleX)*sin(angleY)*sin(angleZ);
	rotationMatrix[2 + (2 * n)] = cos(angleX)*cos(angleY);

	rotationMatrix[3 + (3 * n)] = 1;

	rotationMatrix[3 + (0 * n)] = moveX;
	rotationMatrix[3 + (1 * n)] = moveY;
	rotationMatrix[3 + (2 * n)] = moveZ;
}


__device__ __host__ void Entity::transformVector(const GLfloat * vectorInput, GLfloat * vectorOutput, const GLfloat * matrix) {
	const int n = 4;

	GLfloat origX = vectorInput[0];
	GLfloat origY = vectorInput[1];
	GLfloat origZ = vectorInput[2];
	//GLfloat origW = 1.0f;

	vectorOutput[0] = matrix[0 + (0 * n)] * origX + matrix[1 + (0 * n)] * origY + matrix[2 + (0 * n)] * origZ + matrix[3 + (0 * n)];
	vectorOutput[1] = matrix[0 + (1 * n)] * origX + matrix[1 + (1 * n)] * origY + matrix[2 + (1 * n)] * origZ + matrix[3 + (1 * n)];
	vectorOutput[2] = matrix[0 + (2 * n)] * origX + matrix[1 + (2 * n)] * origY + matrix[2 + (2 * n)] * origZ + matrix[3 + (2 * n)];	
}

__device__ __host__ void Entity::transformVectorNoRotate(const GLfloat * vectorInput, GLfloat * vectorOutput, const GLfloat * matrix) {
	const int n = 4;

	GLfloat origX = vectorInput[0];
	GLfloat origY = vectorInput[1];
	GLfloat origZ = vectorInput[2];
	//GLfloat origW = 1.0f;

	vectorOutput[0] = origX + matrix[3 + (0 * n)];
	vectorOutput[1] = origY + matrix[3 + (1 * n)];
	vectorOutput[2] = origZ + matrix[3 + (2 * n)];	
}


void Entity::transform(int entity, const GLfloat * rotationMatrix) {
	transformRaw(entity, this->centers, this->vertices, this->baseVerticesLength, rotationMatrix);
	bufferVerticeData = true;
}

GLfloat Entity::getDistance(int entity1, int entity2, bool precise) {
	return getDistanceRaw(entity1, entity2, this->centers, precise);
}

void cutilSafeCall(int result) {
	if(result != cudaSuccess) {
		printf("FAILURE\n");
		exit(0);
	}
}

void Entity::initialiseForCuda() {
	if(!mappedCudaResources) {
		cudaGraphicsGLRegisterBuffer(&resources[0], verticeBufferId, cudaGraphicsMapFlagsNone);
		cudaGraphicsGLRegisterBuffer(&resources[1], colourBufferId,  cudaGraphicsMapFlagsNone);
		mappedCudaResources = true;
	}

	if(!initialisedForCuda) {
		if(deviceCenters == 0) {
			cutilSafeCall(cudaMalloc((void**) &deviceCenters, this->centersBytes));
		}
		cutilSafeCall(cudaMemcpy(deviceCenters, hostCenters, this->centersBytes, cudaMemcpyHostToDevice));
		centers = deviceCenters;
	
		if(deviceBaseVertices == 0) {	
			cutilSafeCall(cudaMalloc((void**) &deviceBaseVertices, this->baseVerticesBytes));
		}
		cutilSafeCall(cudaMemcpy((void*)deviceBaseVertices, hostBaseVertices, this->baseVerticesBytes, cudaMemcpyHostToDevice));
		baseVertices = deviceBaseVertices;

		// These are initialised every frame.
		vertices = 0;
		colours = 0;

		initialisedForCuda = true;
	}
}

void Entity::uninitialiseForCuda(bool full) {
	if(initialisedForCuda) {
		if(full && mappedCudaResources) {
			cudaGraphicsUnregisterResource(resources[0]);
			cudaGraphicsUnregisterResource(resources[1]);
			mappedCudaResources = false;
		}

		// These are set by the constructor.
		centers = hostCenters;
		colours = hostColours;
		vertices = hostVertices;
		baseVertices = hostBaseVertices;

		initialisedForCuda = false;
	}
}

/* After calling this method CUDA takes control of the entity vertices and colours. Any attempt to modify these from
   the host will fail. */
void Entity::prepareCudaKernel() {
	if(this->bufferColourData) {
		glBindBuffer(GL_ARRAY_BUFFER, this->colourBufferId);
		glBufferSubData(GL_ARRAY_BUFFER, 0, this->coloursBytes, this->colours);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		this->bufferColourData = false;
	}

	if(this->bufferVerticeData) {
		glBindBuffer(GL_ARRAY_BUFFER, this->verticeBufferId);
		glBufferSubData(GL_ARRAY_BUFFER, 0, this->verticesBytes, this->vertices);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		this->bufferVerticeData = false;
	}
	
	initialiseForCuda();

	GLfloat * d_gl_actorVertices = NULL;
	GLfloat * d_gl_actorColours = NULL;
	size_t size;

	cudaStreamCreate(&cudaStream);

	cutilSafeCall(cudaGraphicsMapResources(2, resources, cudaStream)); 

	cutilSafeCall(cudaGraphicsResourceGetMappedPointer((void**)&d_gl_actorVertices,&size,resources[0]));
	cutilSafeCall(cudaGraphicsResourceGetMappedPointer((void**)&d_gl_actorColours,&size,resources[1]));

	vertices = d_gl_actorVertices;
	colours = d_gl_actorColours;
}

void Entity::unprepareCudaKernel() {
	cudaGraphicsUnmapResources(2, resources, cudaStream);
	cudaStreamDestroy(cudaStream);
}

__device__ __host__ void transformRaw(int entity, GLfloat * centers, GLfloat * vertices, const int baseVerticesLength, const GLfloat * rotationMatrix) {
	int baseIndexCenter = entity * 3;
	int baseIndexVertices = entity * baseVerticesLength;

	for(int i = 0; i<baseVerticesLength; i+=3) {
		vertices[baseIndexVertices + i + 0] -= centers[baseIndexCenter];
		vertices[baseIndexVertices + i + 1] -= centers[baseIndexCenter + 1];
		vertices[baseIndexVertices + i + 2] -= centers[baseIndexCenter + 2];

		Entity::transformVector(vertices + baseIndexVertices + i, vertices + baseIndexVertices + i, rotationMatrix);
				
 		vertices[baseIndexVertices + i + 0] += centers[baseIndexCenter];
		vertices[baseIndexVertices + i + 1] += centers[baseIndexCenter + 1];
		vertices[baseIndexVertices + i + 2] += centers[baseIndexCenter + 2];
	}

	Entity::transformVectorNoRotate(centers + baseIndexCenter, centers + baseIndexCenter, rotationMatrix);
}

__device__ __host__ void setPositionRaw(int entity, const GLfloat * baseVertices, const int baseVerticesLength, GLfloat * vertices, GLfloat * centers, GLfloat x, GLfloat y, GLfloat z, GLfloat sizeMultiplierX, GLfloat sizeMultiplierY, GLfloat sizeMultiplierZ) {
	const int baseIndex = entity * baseVerticesLength;
	for(int n = baseIndex, i = 0; n < baseIndex + baseVerticesLength; n+=3, i+=3) {
		vertices[n    ] = baseVertices[i    ] * sizeMultiplierX + x;
		vertices[n + 1] = baseVertices[i + 1] * sizeMultiplierY + y;
		vertices[n + 2] = baseVertices[i + 2] * sizeMultiplierZ + z;
	}

	const int baseIndexCenters = entity * 3;
	centers[baseIndexCenters]     = x;
	centers[baseIndexCenters + 1] = y;
	centers[baseIndexCenters + 2] = z;
}

__device__ __host__ void setColourRaw(int entity, GLfloat * colours, int baseColoursLength, GLfloat red, GLfloat green, GLfloat blue, GLfloat alpha) {
	int startIndex = entity * baseColoursLength;
	for(int n = 0; n<baseColoursLength; n+=4) {
		colours[n + startIndex	  ]	= red;
		colours[n + startIndex + 1]	= green;
		colours[n + startIndex + 2]	= blue;
		colours[n + startIndex + 3]	= alpha;
	}
}

__device__ __host__ GLfloat getDistanceRaw(int entity1, int entity2, const GLfloat * centers, bool precise) {
	int centersIndex1 = entity1 * 3;
	int centersIndex2 = entity2 * 3;

	GLfloat xDif = (centers[centersIndex1    ] - centers[centersIndex2    ]);
	GLfloat yDif = (centers[centersIndex1 + 1] - centers[centersIndex2 + 1]);
	GLfloat zDif = (centers[centersIndex1 + 2] - centers[centersIndex2 + 2]);

	GLfloat result = xDif * xDif + yDif * yDif + zDif * zDif;

	// We can save some computation time by not sqrting, we dont need to do this if we
	// are just using the value for comparison.
	if(precise) {
		return std::sqrt(result);
	}

	return result;
}

__device__ __host__ GLfloat getDistanceRawEx(int entity1, GLfloat x2, GLfloat y2, GLfloat z2, const GLfloat * centers, bool precise) {
	int centersIndex1 = entity1 * 3;

	GLfloat xDif = (centers[centersIndex1    ] - x2);
	GLfloat yDif = (centers[centersIndex1 + 1] - y2);
	GLfloat zDif = (centers[centersIndex1 + 2] - z2);

	GLfloat result = xDif * xDif + yDif * yDif + zDif * zDif;

	// We can save some computation time by not sqrting, we dont need to do this if we
	// are just using the value for comparison.
	if(precise) {
		return std::sqrt(result);
	}

	return result;
}

__global__ void applyRotationMatrix(GLfloat * vertices, GLfloat * centers, const GLfloat * baseVertices, const int baseVerticesLength, const GLfloat * rotationMatrices, const int numEntities, const int halfMapSize, const GLfloat sizeMultiplierX, const GLfloat sizeMultiplierY, const GLfloat sizeMultiplierZ) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);

	if(index < numEntities) {
		transformRaw(index, centers, vertices, baseVerticesLength, rotationMatrices + (index * ROTATION_MATRIX_LENGTH));

		int centersIndex = index * 3;
		GLfloat centersX = centers[centersIndex], centersY = centers[centersIndex + 1], centersZ = centers[centersIndex + 2];

		int lowerBound = -halfMapSize, upperBound = halfMapSize;
		if(centersX < lowerBound) {
			setPositionRaw(index, baseVertices, baseVerticesLength, vertices, centers, upperBound, centersY, centersZ, sizeMultiplierX, sizeMultiplierY, sizeMultiplierZ);
		} else if(centersX > upperBound) {
			setPositionRaw(index, baseVertices, baseVerticesLength, vertices, centers, lowerBound, centersY, centersZ, sizeMultiplierX, sizeMultiplierY, sizeMultiplierZ);
		}

		if(centersY < lowerBound) {
			setPositionRaw(index, baseVertices, baseVerticesLength, vertices, centers, centersX, upperBound, centersZ, sizeMultiplierX, sizeMultiplierY, sizeMultiplierZ);
		} else if(centersY > upperBound) {
			setPositionRaw(index, baseVertices, baseVerticesLength, vertices, centers, centersX, lowerBound, centersZ, sizeMultiplierX, sizeMultiplierY, sizeMultiplierZ);
		}

		if(centersZ < lowerBound) {
			setPositionRaw(index, baseVertices, baseVerticesLength, vertices, centers, centersX, centersY, upperBound, sizeMultiplierX, sizeMultiplierY, sizeMultiplierZ);
		} else if(centersZ > upperBound) {
			setPositionRaw(index, baseVertices, baseVerticesLength, vertices, centers, centersX, centersY, lowerBound, sizeMultiplierX, sizeMultiplierY, sizeMultiplierZ);
		}
	}
}

void gpuApplyRotationMatrix(int gridSize, int blockSize, Entity * entity, const GLfloat * rotationMatrices, const int halfMapSize, const int sizeMultiplierX, const int sizeMultiplierY, const int sizeMultiplierZ) {
	applyRotationMatrix<<<gridSize, blockSize>>>(entity->vertices, entity->centers, entity->baseVertices, entity->baseVerticesLength, rotationMatrices, entity->numEntities, halfMapSize, sizeMultiplierX, sizeMultiplierY, sizeMultiplierZ);
	getLastCudaError("Kernel execution failed");
	cutilSafeCall(cudaThreadSynchronize());
}


__global__ void matchPosition(GLfloat * vertices, GLfloat * centers, const GLfloat * baseVertices, const int baseVerticesLength, const GLfloat * sourceCenters, const int numEntities, const GLfloat sizeMultiplierX, const GLfloat sizeMultiplierY, const GLfloat sizeMultiplierZ) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);

	if(index < numEntities) {
		int centersIndex = index * 3;
		GLfloat centersX = sourceCenters[centersIndex], centersY = sourceCenters[centersIndex + 1], centersZ = sourceCenters[centersIndex + 2];

		setPositionRaw(index, baseVertices, baseVerticesLength, vertices, centers, centersX, centersY, centersZ, sizeMultiplierX, sizeMultiplierY, sizeMultiplierZ);
	}
}

void gpuMatchPosition(int gridSize, int blockSize, Entity * entity, Entity * sourceEntity, const int sizeMultiplierX, const int sizeMultiplierY, const int sizeMultiplierZ) {
	matchPosition<<<gridSize, blockSize>>>(entity->vertices, entity->centers, entity->baseVertices, entity->baseVerticesLength, sourceEntity->centers, entity->numEntities, sizeMultiplierX, sizeMultiplierY, sizeMultiplierZ);
	getLastCudaError("Kernel execution failed");
	cutilSafeCall(cudaThreadSynchronize());
}

curandState * cudaRandomStates = 0;
int lastCudaRandomStatesBytes = -1;
__global__ void prepareRandomNumberGeneratorDevice ( curandState * state, unsigned long seed ) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	curand_init ( seed, index, 0, &state[index] );
}

void prepareRandomNumberGeneratorHost(int gridSize, int blockSize) { 
	int newBytes = gridSize * blockSize * sizeof(curandState);

	if(newBytes != lastCudaRandomStatesBytes) {
		cutilSafeCall(cudaFree(cudaRandomStates));
		cutilSafeCall(cudaMalloc(&cudaRandomStates, newBytes));
		lastCudaRandomStatesBytes = newBytes;

		prepareRandomNumberGeneratorDevice <<< gridSize, blockSize >>> ( cudaRandomStates,unsigned(time(NULL)) );
		getLastCudaError("Kernel execution failed");
		cutilSafeCall(cudaThreadSynchronize());
	}
}


__device__ float d_getRandomFloat( curandState* globalState, int index ) 
{
    //int ind = threadIdx.x;
    curandState localState = globalState[index];
    float randomFloat = curand_uniform( &localState );
    globalState[index] = localState;
	return randomFloat;
}

__host__ __device__ void setActorSuceptable(int entity, int baseColoursLength, GLfloat * colours, int * states) {
	states[entity] = SUCEPTABLE;
	setColourRaw(entity,colours,baseColoursLength,0,1.0f,0,1.0f);
}

__host__ __device__ void setActorInfected(int entity, int baseColoursLength, GLfloat * colours, int * states) {
	states[entity] = INFECTED;
	setColourRaw(entity,colours,baseColoursLength,1.0f,0,0,1.0f);
}

__host__ __device__ void setActorRecovered(int entity, int baseColoursLength, GLfloat * colours, int * states) {
	states[entity] = RECOVERED;
	setColourRaw(entity,colours,baseColoursLength,0,0,1.0f,1.0f);
}

__host__ __device__ void setActorDead(int entity, int baseColoursLength, GLfloat * colours, int * states) {
	states[entity] = DEAD;
	setColourRaw(entity,colours,baseColoursLength,1.0f,1.0f,1.0f,0);
}

__global__ void transitionStates(curandState* globalState, GLfloat * colours, int baseColoursLength, int * statesInput, int * statesOutput, const GLfloat * centers, GLfloat * rotationMatrices, GLfloat * infectionRadiusConditionalColours, int infectionRadiusConditionalBaseColoursLength, GLfloat * infectionRadiusColours, int infectionRadiusBaseColoursLength, GLfloat infectionRadiusValue, float infectionProbability, float deathProbability, float recoveryProbability) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);

	// experimental optimized distance finding code.
	int sharedMemoryIndex = threadIdx.x;
	int globalMemoryIndex = threadIdx.x;

	int sharedMemoryIndexMult = sharedMemoryIndex * 3;
	int globalMemoryIndexMult = globalMemoryIndex * 3;
	int indexMult = index * 3;

	int globalMemoryBaseIndex = 0;

	//int closestIndex = 0;
	GLfloat closestDistance = -1;

	// Load into registers.
	GLfloat centerX = centers[indexMult], centerY = centers[indexMult + 1], centerZ = centers[indexMult + 2];
	int state = statesInput[index];

	__shared__ GLfloat centerSearchSpace[MAX_BLOCK_SIZE * 3];
	__shared__ int	   stateSearchSpace[MAX_BLOCK_SIZE];

	int numActors = gridDim.x * blockDim.x;

	while(globalMemoryBaseIndex < numActors) {
		// Load next set of centers from global memory.
		centerSearchSpace[sharedMemoryIndexMult] =     centers[globalMemoryIndexMult];
		centerSearchSpace[sharedMemoryIndexMult + 1] = centers[globalMemoryIndexMult + 1];
		centerSearchSpace[sharedMemoryIndexMult + 2] = centers[globalMemoryIndexMult + 2];

		stateSearchSpace[sharedMemoryIndex] = statesInput[globalMemoryIndex];
		__syncthreads();

		// Find smallest distance in current search space.
		#pragma unroll
		for(int n = 0;n<blockDim.x;n++) {
			if(n + globalMemoryBaseIndex != index) {
				GLfloat currentDistance = getDistanceRawEx(n, centerX, centerY, centerZ, centerSearchSpace, true);
				if(stateSearchSpace[n] == INFECTED && (closestDistance < 0 || currentDistance < closestDistance)) {
					closestDistance = currentDistance;
					//closestIndex = n;
				}
			}
		}
		__syncthreads();

		globalMemoryBaseIndex += blockDim.x;
		globalMemoryIndex += blockDim.x;
		globalMemoryIndexMult += blockDim.x * 3;
	}
		
	float randomFloat  = d_getRandomFloat(globalState, index);
	float randomFloat2 = d_getRandomFloat(globalState, index);
	statesOutput[index] = statesInput[index];

	if(state == SUCEPTABLE) {
		if(randomFloat < infectionProbability) {
			if(closestDistance >= 0 && closestDistance < infectionRadiusValue) {
				setActorInfected(index, baseColoursLength, colours, statesOutput);

				// Make infection radius visible.
				setColourRaw(index, infectionRadiusConditionalColours, infectionRadiusConditionalBaseColoursLength, INFECTION_RADIUS_RED, INFECTION_RADIUS_GREEN, INFECTION_RADIUS_BLUE, INFECTION_RADIUS_ALPHA);
			}
		}
	} else if(state == INFECTED) {
		if(randomFloat < deathProbability) {
			setActorDead(index, baseColoursLength, colours, statesOutput);

			// Don't show infection radius for dead actors.
			setColourRaw(index, infectionRadiusConditionalColours, infectionRadiusConditionalBaseColoursLength, 0, 0, 0, 0);
			setColourRaw(index, infectionRadiusColours, infectionRadiusBaseColoursLength, 0, 0, 0, 0);

			// Stop dead actors from moving.
			Entity::clearRotationMatrix(rotationMatrices + (index * ROTATION_MATRIX_LENGTH));
		} else if(randomFloat2 < recoveryProbability) {
			setActorRecovered(index, baseColoursLength, colours, statesOutput);

			setColourRaw(index, infectionRadiusConditionalColours, infectionRadiusConditionalBaseColoursLength, 0, 0, 0, 0);
		}
	}
}



EntityStates::EntityStates(Entity * entity, bool prepareForCuda) :
	entity(entity),
	preparedForCuda(prepareForCuda),
	statesLength(entity->numEntities),
	statesBytes(statesLength * sizeof(int)),
	initialInfectedNodeIndex(statesLength-1)
{
	hostInputStates = new int[statesLength];

	for(int n = 0;n<statesLength;n++) {
		setActorSuceptable(n,entity->baseColoursLength,entity->colours,hostInputStates);
	}

	setActorInfected(initialInfectedNodeIndex,entity->baseColoursLength,entity->colours,hostInputStates);
	
	if(prepareForCuda) {
		cutilSafeCall(cudaMalloc((void**) &inputStates, statesBytes));
		pushInputToDevice();

		cutilSafeCall(cudaMalloc((void**) &outputStates, statesBytes));
	} else {
		inputStates = hostInputStates;
		outputStates = new int[statesLength];
	}

	entity->bufferColourData = true;
}

EntityStates::~EntityStates() {
	delete[] hostInputStates;
	if(this->preparedForCuda) {
		cudaFree(this->inputStates);
		cudaFree(this->outputStates);
	} else {
		delete[] outputStates;
	}
}

void EntityStates::pushInputToDevice() {
	cutilSafeCall(cudaMemcpy(inputStates, hostInputStates, statesBytes, cudaMemcpyHostToDevice));
}

void EntityStates::reset() {
	pushInputToDevice();
	entity->bufferColourData = true;
}

void gpuTransitionStates(int gridSize, int blockSize, EntityStates * states, GLfloat * rotationMatrices, Entity * infectionRadiusConditional, Entity * infectionRadius, GLfloat infectionRadiusValue, float infectionProbability, float deathProbability, float recoveryProbability) {
	transitionStates<<<gridSize, blockSize>>>(cudaRandomStates, states->entity->colours, states->entity->baseColoursLength, states->inputStates, states->outputStates, states->entity->centers, rotationMatrices, infectionRadiusConditional->colours, infectionRadiusConditional->baseColoursLength, infectionRadius->colours, infectionRadius->baseColoursLength, infectionRadiusValue, infectionProbability, deathProbability, recoveryProbability);
	getLastCudaError("Kernel execution failed");
	cutilSafeCall(cudaThreadSynchronize());

	// Get ready for the next transition.
	int * aux = states->inputStates;
	states->inputStates = states->outputStates;
	states->outputStates = aux;
}

void listCudaAttributes() {
	int result;

	cudaDeviceGetAttribute(&result,cudaDevAttrMaxThreadsPerBlock,0);
	printf("Maximum threads per block: %d\n",result);

	cudaDeviceGetAttribute(&result,cudaDevAttrMaxBlockDimX,0);
	printf("Maximum x dimension of block: %d\n",result);

	cudaDeviceGetAttribute(&result,cudaDevAttrMaxGridDimX,0);
	printf("Maximum x dimension of grid: %d\n",result);	
}

int lastNumberActors = -1;

void initialiseStructures(int numberActors, int gridSize, int blockSize, GLfloat mapMin, GLfloat mapMax, GLfloat infectionRadius, GLfloat ** deviceRotationMatrices, GLfloat ** hostRotationMatrices, Entity ** actorsInfectionRadius, Entity ** actorsInfectionRadiusConditional, Entity ** actors, EntityStates ** actorStateMachine) {	
	bool changeInNumberActors = (lastNumberActors < 0 || numberActors != lastNumberActors);
	lastNumberActors = numberActors;

	prepareRandomNumberGeneratorHost(gridSize, blockSize);
		
	const int rotationMatriceLength = numberActors * ROTATION_MATRIX_LENGTH;
	const int rotationMatriceBytes = rotationMatriceLength * sizeof(GLfloat);

	if(changeInNumberActors) {
		delete *actors;
		delete *actorStateMachine; 
		delete *actorsInfectionRadius;
		delete *actorsInfectionRadiusConditional;
		cutilSafeCall(cudaFree(*deviceRotationMatrices));
		delete[] *hostRotationMatrices;

		*actors = Entity::buildCube(numberActors);
		*actorStateMachine = new EntityStates(*actors,true);
	
		*actorsInfectionRadius = Entity::buildDefaultSphere(numberActors);
		*actorsInfectionRadiusConditional = Entity::buildDefaultSphere(numberActors);

		*hostRotationMatrices = new GLfloat[rotationMatriceLength];
		cutilSafeCall(cudaMalloc((void**) deviceRotationMatrices, rotationMatriceBytes));
	} else {
		(*actors)->uninitialiseForCuda(false);
		(*actorsInfectionRadius)->uninitialiseForCuda(false);
		(*actorsInfectionRadiusConditional)->uninitialiseForCuda(false);
		(*actorStateMachine)->reset();
	}

	for(int n = 0;n<numberActors;n++) {
		GLfloat randX = getRandomFloat(mapMin,mapMax), randY = getRandomFloat(mapMin,mapMax), randZ = getRandomFloat(mapMin,mapMax);
		GLfloat randMoveX = getRandomFloat(-MAX_MOVE_SPEED,MAX_MOVE_SPEED), randMoveY = getRandomFloat(-MAX_MOVE_SPEED,MAX_MOVE_SPEED), randMoveZ = getRandomFloat(-MAX_MOVE_SPEED,MAX_MOVE_SPEED);
		GLfloat randRotateX = getRandomFloat(-ROTATION_SPEED_MAX,ROTATION_SPEED_MAX), randRotateY = getRandomFloat(-ROTATION_SPEED_MAX,ROTATION_SPEED_MAX), randRotateZ = getRandomFloat(-ROTATION_SPEED_MAX,ROTATION_SPEED_MAX);

		(*actorsInfectionRadius)->setPosition(n, randX, randY, randZ, infectionRadius, infectionRadius, infectionRadius);
		(*actorsInfectionRadius)->setColour(n, INFECTION_RADIUS_RED, INFECTION_RADIUS_GREEN, INFECTION_RADIUS_BLUE, INFECTION_RADIUS_ALPHA);

		(*actorsInfectionRadiusConditional)->setPosition(n, randX, randY, randZ, infectionRadius, infectionRadius, infectionRadius);
		
		if(n != (*actorStateMachine)->initialInfectedNodeIndex) {
			(*actorsInfectionRadiusConditional)->setColour(n,0,0,0,0);
		} else {
			(*actorsInfectionRadiusConditional)->setColour(n, INFECTION_RADIUS_RED, INFECTION_RADIUS_GREEN, INFECTION_RADIUS_BLUE, INFECTION_RADIUS_ALPHA);
		}

		(*actors)->setPosition(n,randX, randY, randZ, ACTOR_SIZE,ACTOR_SIZE,ACTOR_SIZE);
		Entity::getRotationMatrix(randRotateX, randRotateY, randRotateZ, randMoveX, randMoveY, randMoveZ, (*hostRotationMatrices) + n * ROTATION_MATRIX_LENGTH);
	}
	cutilSafeCall(cudaMemcpy(*deviceRotationMatrices, *hostRotationMatrices, rotationMatriceBytes, cudaMemcpyHostToDevice));
}