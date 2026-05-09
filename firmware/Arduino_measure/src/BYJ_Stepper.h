#ifndef BYJ_Stepper_h
#define BYJ_Stepper_h

#include "Arduino.h"

class BYJ_Stepper {
  public:
    // Constructor: Define the 4 control pins
    BYJ_Stepper(int p1, int p2, int p3, int p4);

    // Set speed in steps per second
    void setSpeed(long stepsPerSec);

    // set the tighten direction of the motor, press the limit button if need to reverse the direction
    void setDirection(int limitPin);

    // Move a specific number of steps (positive or negative)
    void step(int steps);

    // go to the assigned absolute position
    void goTo(int position);

    // go to fully relaxed or tightened position
    void fullRelax();
    void fullTight();

    // set home / maximum position of the motor
    // set home position: set_position = 1, the tightened direction
    // set max position: set_position = -1, the relax direction
    void setPosition(int limitPin, int set_position);

    void currentPos();

    long int degree2Steps(int degree);

    void info();

  private:
    void writeStep(int stage);
    int _pins[4];
    long _stepDelay; // Delay in microseconds between steps
    int _currentStep;
    long int _tightPosition;
    long int _relaxPosition;
    int _direction;
    long int _currentPosition;
};


#endif
