#include "BYJ_Stepper.h"

#define BUTTON A0
#define TIGHT_POSITION 1
#define RELAX_POSITION -1

BYJ_Stepper motor1(8, 9, 10, 11);
#define SPEED 500
#define ROUND_STEP 5000

void setup() {
  Serial.begin(115200);
  pinMode(BUTTON, INPUT_PULLUP);
  Serial.println("initializing...");
  motor1.setSpeed(500);
  motor1.currentPos();
}

// motor testing 
void loop() {
  Serial.println("Assign test mode, 1: rotate | 2: set direction | 3: set tight/relax position | 4: go full tight/relax | 5: go somewhere");
  while (Serial.available() == 0); 
  int test_mode = Serial.read();
  char buf = Serial.read(); // read trash "\n"
  // int input_buf[3] = {0};
  int destination;
  Serial.print("Execute: ");
  // delay(1000);
  // test 1: the step function and position memory
  switch (test_mode) {
    case '0':
      Serial.println("case 0");
      motor1.info();
      break;
    case '1':
      Serial.println("case 1");
      motor1.currentPos();
      motor1.step(500);
      break;
    case '2': 
      Serial.println("case 2");
      motor1.setDirection(BUTTON);
      break;
    case '3':
      Serial.println("case 3");
      motor1.setPosition(BUTTON, TIGHT_POSITION);
      motor1.setPosition(BUTTON, RELAX_POSITION);
      break;
    case '4': 
      Serial.println("case 4");
      motor1.fullTight();
      break;
    case '5': 
      Serial.println("case 5");
      motor1.fullRelax();
      break;
    case '6':
      Serial.println("case 6");
      Serial.print("Assign the rotation integer degree (0~180): ");
      // read rotate degree, if not number, read again
      do{
        int i = 0;
        destination = 0;
        buf = 'n';
        while (Serial.available() == 0);
        while (i < 3) { // press enter to end typing
          buf = Serial.read();
          if (buf >= '0'  && buf <= '9') {
            destination *= 10;
            destination += buf - '0';
            i++;
          }
          else; // trash!!!!
        }
        Serial.print("The destination is set to ");
        Serial.println(motor1.degree2Steps(destination));
      } while (destination > 180);
      // degree to steps
      motor1.goTo(motor1.degree2Steps(destination));
      break;
    case '7': // release the rope
      Serial.println("case 7");
      motor1.setPosition(BUTTON, RELAX_POSITION);
      break;
    case '8': // release the rope
      Serial.println("case 8");
      motor1.setPosition(BUTTON, TIGHT_POSITION);
      break;
    default: break;
  }
  Serial.println("**************************************************************\n");
}
