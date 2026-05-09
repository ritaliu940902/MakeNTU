#include "HardwareSerial.h"
#include "Arduino.h"
#include "BYJ_Stepper.h"

#define BUTTON A0

BYJ_Stepper::BYJ_Stepper(int p1, int p2, int p3, int p4) {
  _pins[0] = p1; _pins[1] = p2; _pins[2] = p3; _pins[3] = p4;
  for (int i = 0; i < 4; i++) {
    pinMode(_pins[i], OUTPUT);
  }
  _currentStep = 0;
  _direction = 1;
  _tightPosition = 2500;
  _relaxPosition = -2500;
  _currentPosition = 0;
  setSpeed(1); // Default speed
}

void BYJ_Stepper::setSpeed(long stepsPerSec) {
  // Calculate delay in microseconds between steps
  Serial.print("The speed is set to ");
  _stepDelay = 1000 / stepsPerSec;
  Serial.println(stepsPerSec);
}

// set the tighten direction of the motor, press the limit button if need to reverse the direction
void BYJ_Stepper::setDirection(int limitPin) {
  Serial.println("Press if not tightened");
  step(3000);
  Serial.println(digitalRead(limitPin));
  if (digitalRead(limitPin) != HIGH) {
    Serial.println("direction changed!");
    _direction = -_direction;
    step(3000);
  }
  else {
    Serial.println("direction unchanged!");
    _direction = _direction;
    step(-3000);
  }
}

void BYJ_Stepper::step(int steps) {
  int absSteps = abs(steps);
  int nextStep = (steps > 0) ?  _direction: -_direction;

  for (int i = 0; i < absSteps; i++) {
    _currentStep += nextStep;
    _currentPosition += nextStep;

    // Keep _currentStep between 0 and 7 (for 8-step sequence)
    if (_currentStep > 7) _currentStep = 0;
    if (_currentStep < 0) _currentStep = 7;

    writeStep(_currentStep);
    // Serial.print("current step: ");
    // Serial.println(_currentStep);
    delay(_stepDelay);
  }
  // Serial.println("complete!");
  // Serial.print("Current position: ");
  // Serial.println(_currentPosition);
}

void BYJ_Stepper::goTo(int position) {
  currentPos();
  int steps = position - _currentPosition;
  Serial.print("Few steps away: ");
  Serial.println(steps);
  step(steps);
  currentPos();
}

void BYJ_Stepper::fullTight() {
  goTo(_tightPosition);
}

void BYJ_Stepper::fullRelax() {
  goTo(_relaxPosition);
}

// Print current position
void BYJ_Stepper::currentPos() {
  Serial.print("Current position: ");
  Serial.println(_currentPosition);
}

// Convert degree to steps
long int BYJ_Stepper::degree2Steps(int degree) {
  if (_tightPosition - _relaxPosition != 0) {
    long int steps = round(degree * (_tightPosition - _relaxPosition) / 180);
    Serial.print("Convert degree ");
    Serial.print(degree);
    Serial.print(" to steps: ");
    Serial.println(steps);
    return steps;
  }
  else {
    setPosition(BUTTON, 1);  // set tighten position
    setPosition(BUTTON, -1); // set relax position
    return degree2Steps(degree);
  }
}


void BYJ_Stepper::info() {
  Serial.println("The information of motor");
  Serial.print("Speed:            ");
  Serial.println(1000 / _stepDelay); // Delay in microseconds between steps
  Serial.print("direction:        ");
  Serial.println(_direction);
  Serial.print("Current position: ");
  Serial.println(_currentPosition);
  Serial.print("Tight position:   ");
  Serial.println(_tightPosition);
  Serial.print("Relax positiob:   ");
  Serial.println(_relaxPosition);
}


// The 8-step sequence for ULN2003/28BYJ-48
void BYJ_Stepper::writeStep(int stage) {
  bool sequence[8][4] = {
    {1,0,0,0}, {1,1,0,0}, {0,1,0,0}, {0,1,1,0},
    {0,0,1,0}, {0,0,1,1}, {0,0,0,1}, {1,0,0,1}
  };
  for (int i = 0; i < 4; i++) {
    digitalWrite(_pins[i], sequence[stage][i]);
  }
}

// set home / maximum position of the motor
// set home position: set_position = 1, the tightened direction
// set max position: set_position = -1, the relax direction
void BYJ_Stepper::setPosition(int limitPin, int set_position) {
  if (set_position == -1 || set_position == 1) {
    if (set_position == 1) Serial.println("Press until its fully relaxed...");
    else Serial.println("Press until its fully tightened...");
    int nextStep = _direction * set_position;
    // Move one step at a time until the switch is pressed (LOW)
    while (digitalRead(limitPin) != 0) {
        step(nextStep); // Direction should be 1 or -1
        delay(1);        // Small stability delay
    }

    // Optional: Back off slightly from the switch to "zero" the position
    // step(nextStep * -50);

    // set the critical position
    if (set_position > 0) {
      _tightPosition = _currentPosition;
      Serial.print("Set tight position to ");
      Serial.println(_currentPosition);
    }
    else {
      _relaxPosition = _currentPosition;
      Serial.print("Set relax position to ");
      Serial.println(_currentPosition);
    }
  }
  else Serial.println("Wrong setting");
  delay(1000);
}
