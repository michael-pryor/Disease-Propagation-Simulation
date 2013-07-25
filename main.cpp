#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <string>

#include "global.h"
#include "graphics_h.cu"

int MAP_MAX;
int MAP_MIN;
int MAP_SIZE;

int NUM_ACTORS = 1024;
int BLOCK_SIZE = 256;
int GRID_SIZE = (NUM_ACTORS / BLOCK_SIZE);

float INFECTION_PROB = 1.0f;
float DEATH_PROB = 0.0005f;
float RECOVERY_PROB = 0.0001f;
GLfloat INFECTION_RADIUS = 100.0f;

// Keep track of mouse activity.
float mouseX = 0, mouseY = 0;
bool mouseLeftDown=false,mouseRightDown=false,mouseMiddleDown=false;

// Keep track of camera positions/angles.
float cameraAngleX = 0;
float cameraAngleY = 0;
float cameraDistance = 0;

int screenWidth, screenHeight;

bool paused = true, resizingMap = false, showInfectionRadius = false,
	 showConditionalInfectionRadius = false, resizingInfectionRadius = false,
	 autoToggledInfectionRadius = false, changingNumberOfActors = false,
	 showHelp = false, showingText = true;
int changingProbability = 0;


int potentialNewNumberOfActors = NUM_ACTORS;
int potentialNewBlockSize = BLOCK_SIZE;
int potentialNewGridSize = GRID_SIZE;

GLfloat * actorRotationMatrices = 0;
GLfloat * d_actorRotationMatrices = 0;
Entity * actorsInfectionRadiusConditional = 0;
Entity * actorsInfectionRadius = 0;
Entity * actors = 0;
EntityStates * actorStateMachine = 0;
int step = 0;

int fpsTimerMs = 0;
int framesCount = 0;

GLfloat textSpaceMultiplier;

void initialise(int numberActors) {
	initialiseStructures(numberActors, GRID_SIZE, BLOCK_SIZE, MAP_MIN, MAP_MAX, INFECTION_RADIUS, &d_actorRotationMatrices, &actorRotationMatrices, &actorsInfectionRadius, &actorsInfectionRadiusConditional, &actors, &actorStateMachine);
}

void updateMapSize(int newMapSize) {
	int originalMapSize = MAP_SIZE;
	int changeInMapSize = MAP_SIZE - newMapSize;
	MAP_SIZE = newMapSize;
	MAP_MAX = MAP_SIZE / 2;
	MAP_MIN = -MAP_MAX;

	GLfloat size = float(MAP_SIZE) / 2.0f;

	GLfloat additionalSpace = ACTOR_SIZE * 2;
	size += additionalSpace;
}


void reset() {
	initialise(NUM_ACTORS);
	step = 0;
}

void keyPressed (unsigned char key, int x, int y) {
	// pause
	switch(key) {
		case(27):
			exit(0);
			break;

		case('p'):
			paused = !paused;
			break;

		case('e'):
			resizingMap = !resizingMap;
			resizingInfectionRadius = false;
			changingNumberOfActors = false;
			changingProbability = 0;
			break;

		case('r'):
			resizingInfectionRadius = !resizingInfectionRadius;
			resizingMap = false;
			changingNumberOfActors = false;
			changingProbability = 0;

			if(autoToggledInfectionRadius && showConditionalInfectionRadius) {
				showConditionalInfectionRadius = false;
			}

			if(resizingInfectionRadius && !showInfectionRadius && !showConditionalInfectionRadius) {
				showConditionalInfectionRadius = true;
				autoToggledInfectionRadius = true;
			}
			break;

		case('t'):
			reset();
			break;

		case('f'):
			autoToggledInfectionRadius = false;
			if(showInfectionRadius && !showConditionalInfectionRadius) {
				showInfectionRadius = false;
				showConditionalInfectionRadius = true;
			} else if(!showInfectionRadius && showConditionalInfectionRadius) {
				showConditionalInfectionRadius = false;
			} else if(!showInfectionRadius && !showConditionalInfectionRadius) {
				showInfectionRadius = true;
			}
			break;

		case('v'):
			if(changingNumberOfActors) {
				NUM_ACTORS = potentialNewNumberOfActors;
				BLOCK_SIZE = potentialNewBlockSize;
				GRID_SIZE = potentialNewGridSize;
				reset();
			} else {
				potentialNewNumberOfActors = NUM_ACTORS;
				potentialNewBlockSize = BLOCK_SIZE;
				potentialNewGridSize = GRID_SIZE;
			}

			changingNumberOfActors = !changingNumberOfActors;
			resizingInfectionRadius = false;
			resizingMap = false;
			changingProbability = 0;
			break;

		case('b'):
			changingNumberOfActors = false;
			resizingInfectionRadius = false;
			resizingMap = false;
			changingProbability += 1;
			changingProbability %= 4; // wrap around, 4 options (0,1,2,3).
			break;

		case('h'):
			showHelp = !showHelp;
			break;

		case('j'):
			showingText = !showingText;
			break;
	}
} 

// Respond to mouse clicks.
void onMouseStateChange(int button, int state, int x, int y) {
	mouseX = x;
	mouseY = y;

	bool isDown = state == GLUT_DOWN;
	if(button == GLUT_LEFT_BUTTON)	{
		mouseLeftDown = isDown;
	} else if(button == GLUT_RIGHT_BUTTON) {
		mouseRightDown = isDown;
	} else if(button == GLUT_MIDDLE_BUTTON)	{
		mouseMiddleDown = isDown;
	}
}

void adjustNumberOfActors(int numActors, int & blockSize, int & gridSize, int & outputNumActors, bool roundUp) {
	if(numActors < 1) {
		numActors = 1;
	}

	// In this case we just have 1 block.
	if(numActors < 32) {
		blockSize = numActors;
		gridSize = 1;
		outputNumActors = numActors;
		return;
	}

	// We pick the highest power of 2 possible up to out maximum block size and above warp size (which is 32).
	int currentBlockSize = MAX_BLOCK_SIZE;
	int remainder;
	while(currentBlockSize >= 32) {
		remainder = numActors % currentBlockSize;
		if(remainder == 0) {
			blockSize = currentBlockSize;
			gridSize = numActors / blockSize;
			outputNumActors = numActors;
			return;
		}

		currentBlockSize /= 2;
	}

	// We round up to the nearest multiple of 32 and try again. This time we know
	// for certain it will succeed with smallest block size of 32.
	numActors -= remainder;
	if(roundUp) {
		numActors += 32;
	}

	adjustNumberOfActors(numActors, blockSize, gridSize, outputNumActors, roundUp);
}


void changeProbability(int mouseChangeY, float & probability, bool fast) {
	if(fast) {
		probability += float(mouseChangeY) / 1000.0f;
	} else {
		probability += float(mouseChangeY) / 10000.0f;
	}

	if(probability > 1.0f) {
		probability = 1.0f;
	} else if(probability < 0) {
		probability = 0;
	}
}

// Respond to mouse movement.
void onMouseMove(int x, int y) {
	int mouseChangeX = mouseX - x, mouseChangeY = mouseY - y;
	int previousMouseX = mouseX, previousMouseY = mouseY;
	mouseX = x;
	mouseY = y;

	if(resizingMap) {
		updateMapSize(MAP_SIZE + mouseChangeY);
		return;
	}
	
	if(resizingInfectionRadius) {
		INFECTION_RADIUS += float(mouseChangeY) / 10.0f;
		return;
	}

	if(changingNumberOfActors) {
		potentialNewNumberOfActors += mouseChangeY;
		adjustNumberOfActors(potentialNewNumberOfActors, potentialNewBlockSize, potentialNewGridSize, potentialNewNumberOfActors, mouseChangeY > 0);
		return;
	}

	if(changingProbability == 1) {
		changeProbability(mouseChangeY, INFECTION_PROB, mouseRightDown);
		return;
	} else if(changingProbability == 2) {
		changeProbability(mouseChangeY, DEATH_PROB, mouseRightDown);
		return;
	} else if(changingProbability == 3) {
		changeProbability(mouseChangeY, RECOVERY_PROB, mouseRightDown);
		return;
	}

	if(mouseLeftDown) {
		cameraAngleY += previousMouseX - mouseX;
		cameraAngleX += previousMouseY - mouseY;
	}
	if(mouseRightDown) {
		cameraDistance -= (previousMouseY - mouseY) * 2.0f;
	}
}

void text(GLfloat x, GLfloat y, GLfloat red, GLfloat green, GLfloat blue, const char * text)
{
	glColor3f(red, green, blue);
	glRasterPos2f(x, y);
	size_t length = strlen(text);
	for(size_t n = 0; n < length; n++) {
		glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, text[n]);
	}
}

void onDisplay()
{
	// compute FPS.
	int newFpsTimerMs = glutGet(GLUT_ELAPSED_TIME);
	int difference = newFpsTimerMs - fpsTimerMs;
	framesCount += 1;
	static int lastFramesCount = 0;
	if(difference > 1000) {
		lastFramesCount = framesCount;
		fpsTimerMs = newFpsTimerMs;
		framesCount = 0;
	}
	
	actors->prepareCudaKernel();
	actorsInfectionRadius->prepareCudaKernel();
	actorsInfectionRadiusConditional->prepareCudaKernel();

	if(!paused)
	{
		gpuApplyRotationMatrix(GRID_SIZE, BLOCK_SIZE, actors, d_actorRotationMatrices, MAP_MAX, ACTOR_SIZE, ACTOR_SIZE, ACTOR_SIZE);
		gpuTransitionStates(GRID_SIZE, BLOCK_SIZE, actorStateMachine, d_actorRotationMatrices, actorsInfectionRadiusConditional, actorsInfectionRadius, INFECTION_RADIUS, INFECTION_PROB, DEATH_PROB, RECOVERY_PROB);

		step += 1;
	}

	if(showInfectionRadius) {
		gpuMatchPosition(GRID_SIZE, BLOCK_SIZE, actorsInfectionRadius, actors, INFECTION_RADIUS, INFECTION_RADIUS, INFECTION_RADIUS);
	}

	if(showConditionalInfectionRadius) {
		gpuMatchPosition(GRID_SIZE, BLOCK_SIZE, actorsInfectionRadiusConditional, actors, INFECTION_RADIUS, INFECTION_RADIUS, INFECTION_RADIUS);
	}

	actors->unprepareCudaKernel();
	actorsInfectionRadius->unprepareCudaKernel();
	actorsInfectionRadiusConditional->unprepareCudaKernel();

	// Reset.
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

	// Draw world cube.
	glPushMatrix();

	glTranslatef(0, 0, -cameraDistance);
	glRotatef(cameraAngleX, 1, 0, 0);   // pitch
	glRotatef(cameraAngleY, 0, 1, 0);   // heading
 
    glLineWidth(5);
	glColor3f(1,1,1);
    glBegin(GL_LINE_STRIP);
    glVertex3f(MAP_MIN,MAP_MIN,MAP_MIN);
	glVertex3f(MAP_MAX,MAP_MIN,MAP_MIN);
	glVertex3f(MAP_MAX,MAP_MIN,MAP_MAX);
	glVertex3f(MAP_MIN,MAP_MIN,MAP_MAX);
	glVertex3f(MAP_MIN,MAP_MIN,MAP_MIN);

	glVertex3f(MAP_MIN,MAP_MAX,MAP_MIN);
	glVertex3f(MAP_MIN,MAP_MAX,MAP_MAX);
	glVertex3f(MAP_MIN,MAP_MIN,MAP_MAX);
	glVertex3f(MAP_MIN,MAP_MAX,MAP_MAX);
	glVertex3f(MAP_MAX,MAP_MAX,MAP_MAX);

	glVertex3f(MAP_MAX,MAP_MIN,MAP_MAX);
	glVertex3f(MAP_MAX,MAP_MAX,MAP_MAX);
	glVertex3f(MAP_MAX,MAP_MAX,MAP_MIN);
	glVertex3f(MAP_MAX,MAP_MIN,MAP_MIN);
	glVertex3f(MAP_MAX,MAP_MAX,MAP_MIN);
	glVertex3f(MAP_MIN,MAP_MAX,MAP_MIN);
    glEnd();

    glPopMatrix();

	actors->draw(cameraDistance, cameraAngleX, cameraAngleY);
	
	if(showInfectionRadius) {
		actorsInfectionRadius->draw(cameraDistance, cameraAngleX, cameraAngleY);
	}
	if(showConditionalInfectionRadius) {
		actorsInfectionRadiusConditional->draw(cameraDistance, cameraAngleX, cameraAngleY);
	}
	
	if(showingText) {
		glPushMatrix();
		glLoadIdentity();

		const GLfloat textSpaceY = 4 * textSpaceMultiplier;
		const GLfloat textPositionX = -0.98f;

		const GLfloat textColourRed = 1, textColourGreen = 0.5f, textColourBlue = 0;

		char strNum[256];
		static std::string stringObject; // static means we can reuse allocated memory.

		const GLfloat buttonTextPositionY = 1.0f-(textSpaceY*1.5f);
		text(textPositionX,buttonTextPositionY,1,1,1,"(P): Pause, (T): Reset, (F): Display infection radius, (R): Resize infection radius, (E): Resize container");

		const GLfloat buttonTextPositionTwoY = buttonTextPositionY - textSpaceY;
		text(textPositionX,buttonTextPositionTwoY,1,1,1,"(V): Change number of actors, (B): Change probabilities, (H): Help, (J): Hide text");

		const GLfloat stateTextPositionY = buttonTextPositionTwoY - textSpaceY;
		if(resizingMap) {
			text(textPositionX,stateTextPositionY,textColourRed,textColourGreen,textColourBlue,"Resizing container cube (click and drag)");
		} else if(resizingInfectionRadius) {
			text(textPositionX,stateTextPositionY,textColourRed,textColourGreen,textColourBlue,"Changing infection radius (click and drag)");
		} else if(changingNumberOfActors) {
			text(textPositionX,stateTextPositionY,textColourRed,textColourGreen,textColourBlue,"Changing number of actors (click and drag, triggers reset)");
		} else if(changingProbability == 1) {
			text(textPositionX,stateTextPositionY,textColourRed,textColourGreen,textColourBlue,"Changing probability of infection (click and drag, right click = fast)");
		} else if(changingProbability == 2) {
			text(textPositionX,stateTextPositionY,textColourRed,textColourGreen,textColourBlue,"Changing probability of death while infected (click and drag, right click = fast)");
		} else if(changingProbability == 3) {
			text(textPositionX,stateTextPositionY,textColourRed,textColourGreen,textColourBlue,"Changing probability of recovery while infected (click and drag, right click = fast)");
		} else {
			text(textPositionX,stateTextPositionY,1,1,1,"Navigating world (left click to rotate, right click to zoom)");
		}

		const GLfloat displayStateTextPositionY =  stateTextPositionY - (textSpaceY*2);
		if(showInfectionRadius) {
			text(textPositionX,displayStateTextPositionY,textColourRed,textColourGreen,textColourBlue,"Showing actors and infection radius of all actors");
		} else if(showConditionalInfectionRadius) {
			text(textPositionX,displayStateTextPositionY,textColourRed,textColourGreen,textColourBlue,"Showing actors and infection radius of infected actors");
		} else {
			text(textPositionX,displayStateTextPositionY,1,1,1,"Showing actors");
		}

		const GLfloat displayPauseStatePositionY = displayStateTextPositionY - textSpaceY;

		sprintf(strNum,"%d",step);
		if(paused) {
			stringObject = "Timestep ";
			stringObject.append(strNum);
			stringObject.append(" - Paused");

			text(textPositionX,displayPauseStatePositionY,textColourRed,textColourGreen,textColourBlue,stringObject.c_str());
		} else {
			stringObject = "Timestep ";
			stringObject.append(strNum);

			text(textPositionX,displayPauseStatePositionY,1,1,1,stringObject.c_str());
		}

		const GLfloat displayProbabilitiesPositionY = displayPauseStatePositionY - textSpaceY;
		stringObject = "Probability of infection: ";
		sprintf(strNum,"%.4f",INFECTION_PROB);
		stringObject.append(strNum);

		stringObject.append(", death: ");
		sprintf(strNum,"%.4f",DEATH_PROB);
		stringObject.append(strNum);

		stringObject.append(", recovery: ");
		sprintf(strNum,"%.4f",RECOVERY_PROB);
		stringObject.append(strNum);

		if(changingProbability > 0) {
			text(textPositionX,displayProbabilitiesPositionY,textColourRed,textColourGreen,textColourBlue,stringObject.c_str());
		} else {
			text(textPositionX,displayProbabilitiesPositionY,1,1,1,stringObject.c_str());
		}


		const GLfloat displayNumActorsPositionY = displayProbabilitiesPositionY - textSpaceY;
		stringObject = "Number of actors: ";

		if(changingNumberOfActors) {
			sprintf(strNum,"%d",potentialNewNumberOfActors);
		} else {
			sprintf(strNum,"%d",NUM_ACTORS);
		}
		stringObject.append(strNum);

		stringObject.append(", Block size: ");

		if(changingNumberOfActors) {
			sprintf(strNum,"%d",potentialNewBlockSize);
		} else {
			sprintf(strNum,"%d",BLOCK_SIZE);
		}
		stringObject.append(strNum);

		stringObject.append(", Grid size: ");

		if(changingNumberOfActors) {
			sprintf(strNum,"%d",potentialNewGridSize);
		} else {
			sprintf(strNum,"%d",GRID_SIZE);
		}
		stringObject.append(strNum);

		if(!changingNumberOfActors) {
			text(textPositionX,displayNumActorsPositionY,1,1,1,stringObject.c_str());
		} else {
			text(textPositionX,displayNumActorsPositionY,textColourRed,textColourGreen,textColourBlue,stringObject.c_str());
		}

		const GLfloat displayFpsTextPositionY = displayNumActorsPositionY - textSpaceY;
		stringObject = "Vertices: ";
		sprintf(strNum, "%-8d", actors->verticesNum + actorsInfectionRadius->verticesNum + actorsInfectionRadiusConditional->verticesNum);
		stringObject.append(strNum);

		stringObject.append(" Distance computations: ");
		// Subtract numEntities because we don't do distance to self. i.e. node 3 to 3.
		//
		// Although distance is represented as a bidirectional graph, the way our CUDA
		// kernel works means we compute distance from 1 to 2 and distance from 2 to 1 for example,
		// if we didn't do this we could halve the computation but it is hard to to do this
		// in CUDA and probably would end up causing other inefficiencies.
		sprintf(strNum, "%-10d", actors->numEntities * actors->numEntities - actors->numEntities);
		stringObject.append(strNum);

		stringObject.append(" Frames per second: ");
		sprintf(strNum, "%-8d", lastFramesCount);
		stringObject.append(strNum);

		text(textPositionX,displayFpsTextPositionY,1,1,1,stringObject.c_str());

		GLfloat displayHelpTextPositionY = displayFpsTextPositionY - (textSpaceY * 3);

		const GLfloat helpColourBlue = 0.6f, helpColourGreen = 1.0f, helpColourRed = 1.0f;

		if(showHelp) {
			static const int helpTextLength = 6;
			static const char * helpText[helpTextLength];
			static bool init = false;
			if(!init) {
				helpText[0] = "Help:";
				helpText[1] = "Red actors are infected and have an infection radius.";
				helpText[2] = "Suceptable actors within this radius may become infected.";
				helpText[3] = "Blue nodes are recovered actors which were once infected but survived and are now immune to the disease.";
				helpText[4] = "Infected nodes may die, once dead they are hidden and have no further interaction.";
				helpText[5] = "State changes may occur each timestep with a probability.";
				init = true;
			}

			for(int n = 0;n<helpTextLength;n++) {
				text(textPositionX,displayHelpTextPositionY,helpColourRed,helpColourGreen,helpColourBlue,helpText[n]);
				displayHelpTextPositionY -= textSpaceY * 0.6;
			}
		}
		glPopMatrix();
	}
	// Swap buffers into action.
	glutSwapBuffers();
	glutPostRedisplay();
}
void onRedrawTimer(int millisec)
{
	glutTimerFunc(millisec, onRedrawTimer, millisec);
	glutPostRedisplay();
}



void cutilSafeCall(int result);
int main(int argc, char **argv){
	printf("%d arguments\n",argc);
	for(int n = 0;n<argc;n++) {
		printf("Argument %d: %s\n",n,argv[n]);
	}
	printf("\n");

	listCudaAttributes();

	glewInit();
	glutInit(&argc, argv);
		
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	
	
	if(argc >= 2) {
		screenHeight = atoi(argv[1]);
		screenWidth = screenHeight;
	} else {
		screenHeight = glutGet(GLUT_SCREEN_HEIGHT);
		screenWidth = glutGet(GLUT_SCREEN_WIDTH);
		if(screenHeight > screenWidth) {
			screenHeight = screenWidth;
		} else {
			screenWidth = screenHeight;
		}
	}

	printf("window height is: %d, window width is: %d\n",screenHeight, screenWidth);

	textSpaceMultiplier = 10.0f / float(screenHeight);

	glutInitWindowSize(screenWidth,screenHeight);
	glutCreateWindow("Disease Propagation Simulation");
	
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glEnable(GL_ALPHA_TEST);
	glAlphaFunc(GL_GREATER, 0.1f);

	gluPerspective(45,1.0,1.0,40000.0);
	
	cameraDistance = 5000;

	glutMouseFunc(onMouseStateChange);
	glutMotionFunc(onMouseMove);
	glutKeyboardFunc(keyPressed);
	glutDisplayFunc(onDisplay);
	
	glewInit();

	updateMapSize(2000);
	
	initialise(NUM_ACTORS);
	
	glutTimerFunc(10, onRedrawTimer, 10);             // redraw only every given millisec
	glutMainLoop();

	return 0;
}
