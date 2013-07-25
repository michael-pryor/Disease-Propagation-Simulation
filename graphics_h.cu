#include <GL/glew.h>
#include <helper_cuda.h>
#include <cuda_gl_interop.h>

class Entity;
struct EntityStates;

__device__ __host__ void transformRaw(int entity, GLfloat * centers, GLfloat * vertices, const int baseVerticesLength, const GLfloat * rotationMatrix);
__device__ __host__ void setPositionRaw(int entity, const GLfloat * baseVertices, const int baseVerticesLength, GLfloat * vertices, GLfloat * centers, GLfloat x, GLfloat y, GLfloat z, GLfloat sizeMultiplierX, GLfloat sizeMultiplierY, GLfloat sizeMultiplierZ);
__device__ __host__ void setColourRaw(int entity, GLfloat * colours, int baseColoursLength, GLfloat red, GLfloat green, GLfloat blue, GLfloat alpha);
__device__ __host__ GLfloat getDistanceRaw(int entity1, int entity2, const GLfloat * centers, bool precise=true);
__device__ __host__ GLfloat getDistanceRawEx(int entity1, GLfloat x2, GLfloat y2, GLfloat z2, const GLfloat * centers, bool precise=true);

__host__ __device__ void setActorSuceptable(int entity, int baseColoursLength, GLfloat * colours, int * states);
__host__ __device__ void setActorInfected(int entity, int baseColoursLength, GLfloat * colours, int * states);
__host__ __device__ void setActorRecovered(int entity, int baseColoursLength, GLfloat * colours, int * states);
__host__ __device__ void setActorDead(int entity, int baseColoursLength, GLfloat * colours, int * states) ;

void gpuApplyRotationMatrix(int gridSize, int blockSize, Entity * entity, const GLfloat * rotationMatrices, const int halfMapSize, const int sizeMultiplierX, const int sizeMultiplierY, const int sizeMultiplierZ);
void gpuTransitionStates(int gridSize, int blockSize, EntityStates * states, GLfloat * rotationMatrices, Entity * infectionRadiusConditional, Entity * infectionRadius, GLfloat infectionRadiusValue, float infectionProbability, float deathProbability, float recoveryProbability);
void gpuMatchPosition(int gridSize, int blockSize, Entity * entity, Entity * sourceEntity, const int sizeMultiplierX, const int sizeMultiplierY, const int sizeMultiplierZ);

void initialiseStructures(int numberActors, int gridSize, int blockSize, GLfloat mapMin, GLfloat mapMax, GLfloat infectionRadius, GLfloat ** deviceRotationMatrices, GLfloat ** hostRotationMatrices, Entity ** actorsInfectionRadius, Entity ** actorsInfectionRadiusConditional, Entity ** actors, EntityStates ** actorStateMachine);

void listCudaAttributes();

class Entity {
	// These variables are swapped between host and device as necessary.
	GLfloat * vertices;
	GLuint * indices;
	GLfloat * colours;
	GLfloat * centers;
	const GLfloat * baseVertices;

	// Always host.
	GLfloat * hostVertices;
	GLfloat * hostColours;
	GLfloat * hostCenters;
	const GLfloat * hostBaseVertices;
	const GLuint * baseIndices;

	// Always device.
	GLfloat * deviceCenters;
	const GLfloat * deviceBaseVertices;

public:
	const int baseIndicesLength;
	const int baseVerticesLength; // number of elements
	const int baseVerticesNum; // number of actual vertices where each element is a dimension e.g. x,y,z of a vertice.
	const int baseVerticesBytes;


	const int numEntities;

	const int baseColoursLength;
	const int centersLength, centersBytes;

	const int indicesLength, verticesLength, coloursLength;
	const int indicesBytes, verticesBytes, coloursBytes;
	const int verticesNum;
private:

	int verticeBufferId, colourBufferId;

public:
	const bool cleanupBase;
	const GLenum drawMode;
private:

	bool bufferColourData, bufferVerticeData;
	bool initialisedForCuda, mappedCudaResources;
	cudaGraphicsResource * resources[2];
	cudaStream_t cudaStream;

	static const int COLOUR_ELEMENTS = 4; // RGB and alpha.

	static GLfloat * defaultSphereVertices;
	static GLuint * defaultSphereIndices;
	static int defaultSphereVerticeSize, defaultSphereIndiceSize;
public:
	Entity(const GLfloat * baseVertices, int baseVerticesBytes, const GLuint * baseIndices, int baseIndicesBytes, int numEntities, GLenum drawMode = GL_TRIANGLES, bool cleanupBase = false);
	~Entity();
	
	__device__ __host__ static void clearRotationMatrix(GLfloat * rotationMatrix);
	static void getRotationMatrix(GLfloat angleX, GLfloat angleY, GLfloat angleZ, GLfloat moveX, GLfloat moveY, GLfloat moveZ, GLfloat * rotationMatrix);
	__device__ __host__ static void transformVector(const GLfloat * vectorInput, GLfloat * vectorOutput, const GLfloat * rotationMatrix);
	__device__ __host__ static void transformVectorNoRotate(const GLfloat * vectorInput, GLfloat * vectorOutput, const GLfloat * matrix);

	static void buildSphereRaw(GLfloat ** verticesOutput, int * verticeLengthOutput, GLuint ** indicesOutput, int * indiceLengthOutput, float radius, unsigned int rings, unsigned int sectors);
	static Entity * buildSphere(float radius, unsigned int rings, unsigned int sectors, int numEntities);
	static Entity * buildDefaultSphere(int numEntities);
	static Entity * buildCube(int numEntities);

	void transform(int entity, const GLfloat * rotationMatrix);

	void setPosition(int entity, GLfloat x, GLfloat y, GLfloat z, GLfloat sizeMultiplierX, GLfloat sizeMultiplierY, GLfloat sizeMultiplierZ);
	void setColour(int entity, GLfloat red, GLfloat green, GLfloat blue, GLfloat alpha);

	void draw(GLfloat cameraDistance, GLfloat cameraAngleX, GLfloat cameraAngleY);

	void initialiseForCuda();
	void uninitialiseForCuda(bool full);

	void prepareCudaKernel();
	void unprepareCudaKernel();

	GLfloat getDistance(int entity1, int entity2, bool precise=false);

	friend void ::gpuApplyRotationMatrix(int gridSize, int blockSize, Entity * entity, const GLfloat * rotationMatrices, const int halfMapSize, const int sizeMultiplierX, const int sizeMultiplierY, const int sizeMultiplierZ);
	friend void ::gpuTransitionStates(int gridSize, int blockSize, EntityStates * states, GLfloat * rotationMatrices, Entity * infectionRadiusConditional, Entity * infectionRadius, GLfloat infectionRadiusValue, float infectionProbability, float deathProbability, float recoveryProbability);
	friend void ::gpuMatchPosition(int gridSize, int blockSize, Entity * entity, Entity * sourceEntity, const int sizeMultiplierX, const int sizeMultiplierY, const int sizeMultiplierZ);
	friend struct EntityStates;
};

struct EntityStates {
	int * hostInputStates;
	int * inputStates, * outputStates;

	Entity * const entity;
	const bool preparedForCuda;

	const int statesLength, statesBytes;
	const int initialInfectedNodeIndex;

	EntityStates(Entity * entity, bool prepareForCuda);
	~EntityStates();

	void pushInputToDevice();
	void reset();
};