#include <Arduino.h>
#include <Servo.h>

// =========================
// Serial protocol
// =========================
// PC/Raspberry Pi -> Arduino:
//   START       Move state servo to measuring position and rotate BYJ turntable to 90 deg.
//   MEASURE     Measure resistor and reply with MEAS,<ohm>.
//   SORT,1..5   Move sorting SG90 to box 1..5, release, then reset.
//   REJECT      Move sorting SG90 to the reject box, release, then reset.
//   RESET       Return servos and turntable to the initial position.
//
// Arduino -> PC/Raspberry Pi:
//   READY
//   CAMERA_READY
//   MEAS,<ohm>
//   DONE
//   ERR,<reason>

// =========================
// Pin map
// =========================
const int PIN_REF_R1 = 2;       // 22 ohm reference resistor control
const int PIN_REF_R2 = 3;       // 220 ohm reference resistor control
const int PIN_REF_R3 = 4;       // 2.2k ohm reference resistor control
const int PIN_MEASURE_ADC = A0; // voltage divider node

const int PIN_TRIGGER_LEGACY = 7; // optional pulse for old GPIO trigger/debug
const int PIN_SORT_SERVO = 9;     // SG90: sorting gate
const int PIN_STATE_SERVO = 10;   // SG90: state / release controller

const int PIN_BYJ_IN1 = 5;
const int PIN_BYJ_IN2 = 6;
const int PIN_BYJ_IN3 = 11;
const int PIN_BYJ_IN4 = 12;

// =========================
// Hardware config
// =========================
const long SERIAL_BAUD = 9600;

const float REF_R1_OHM = 22.0;
const float REF_R2_OHM = 220.0;
const float REF_R3_OHM = 2200.0;

const int SORT_BOX_ANGLES[5] = {30, 60, 90, 120, 150};
const int REJECT_ANGLE = 0; // change this if your reject box is at another angle

const int STATE_IDLE_ANGLE = 90;
const int STATE_MEASURE_ANGLE = 150;
const int STATE_RELEASE_ANGLE = 30;

const int SERVO_STEP_DELAY_MS = 10;
const int RELEASE_DELAY_MS = 2000;
const int STATE_SETTLE_MS = 500;

// 28BYJ-48 usually needs 4096 half-steps per output shaft revolution.
// If your turntable gear ratio is different, tune this value.
const long TURNTABLE_STEPS_PER_REV = 4096;
const int TURNTABLE_PHOTO_DEGREE = 90;
const int TURNTABLE_STEP_DELAY_MS = 2;

// =========================
// Runtime state
// =========================
Servo sortServo;
Servo stateServo;

enum MachineState {
  STATE_IDLE,
  STATE_READY_TO_SORT,
  STATE_RELEASING
};

MachineState currentState = STATE_IDLE;

int currentSortAngle = 0;
int currentStateServoAngle = STATE_IDLE_ANGLE;

const int turntablePins[4] = {
  PIN_BYJ_IN1,
  PIN_BYJ_IN2,
  PIN_BYJ_IN3,
  PIN_BYJ_IN4
};
int turntableStage = 0;
long turntablePositionSteps = 0;

// =========================
// Servo helpers
// =========================
void moveServoSmooth(Servo &servo, int &currentAngle, int targetAngle) {
  targetAngle = constrain(targetAngle, 0, 180);

  if (targetAngle > currentAngle) {
    for (int angle = currentAngle; angle <= targetAngle; angle++) {
      servo.write(angle);
      delay(SERVO_STEP_DELAY_MS);
    }
  } else {
    for (int angle = currentAngle; angle >= targetAngle; angle--) {
      servo.write(angle);
      delay(SERVO_STEP_DELAY_MS);
    }
  }

  currentAngle = targetAngle;
}

void enterIdleState() {
  currentState = STATE_IDLE;
  moveServoSmooth(stateServo, currentStateServoAngle, STATE_IDLE_ANGLE);
}

void enterMeasureState() {
  currentState = STATE_READY_TO_SORT;
  moveServoSmooth(stateServo, currentStateServoAngle, STATE_MEASURE_ANGLE);
  delay(STATE_SETTLE_MS);
}

void releasePartAndReset() {
  currentState = STATE_RELEASING;
  moveServoSmooth(stateServo, currentStateServoAngle, STATE_RELEASE_ANGLE);
  delay(RELEASE_DELAY_MS);
  enterIdleState();
}

void pulseLegacyTrigger() {
  digitalWrite(PIN_TRIGGER_LEGACY, HIGH);
  delay(100);
  digitalWrite(PIN_TRIGGER_LEGACY, LOW);
}

void sortToAngle(int targetAngle) {
  if (currentState != STATE_READY_TO_SORT) {
    Serial.println("ERR,NOT_READY_TO_SORT");
    return;
  }

  moveServoSmooth(sortServo, currentSortAngle, targetAngle);
  delay(STATE_SETTLE_MS);
  releasePartAndReset();
  Serial.println("DONE");
}

// =========================
// BYJ turntable helpers
// =========================
void writeTurntableStep(int stage) {
  const byte sequence[8][4] = {
    {1, 0, 0, 0},
    {1, 1, 0, 0},
    {0, 1, 0, 0},
    {0, 1, 1, 0},
    {0, 0, 1, 0},
    {0, 0, 1, 1},
    {0, 0, 0, 1},
    {1, 0, 0, 1}
  };

  for (int i = 0; i < 4; i++) {
    digitalWrite(turntablePins[i], sequence[stage][i]);
  }
}

void releaseTurntableCoils() {
  for (int i = 0; i < 4; i++) {
    digitalWrite(turntablePins[i], LOW);
  }
}

void stepTurntable(long steps) {
  int direction = (steps >= 0) ? 1 : -1;
  long count = labs(steps);

  for (long i = 0; i < count; i++) {
    turntableStage += direction;
    if (turntableStage > 7) turntableStage = 0;
    if (turntableStage < 0) turntableStage = 7;

    writeTurntableStep(turntableStage);
    turntablePositionSteps += direction;
    delay(TURNTABLE_STEP_DELAY_MS);
  }

  releaseTurntableCoils();
}

long degreeToTurntableSteps(int degree) {
  return ((long)degree * TURNTABLE_STEPS_PER_REV) / 360L;
}

void moveTurntableToDegree(int degree) {
  long targetSteps = degreeToTurntableSteps(degree);
  stepTurntable(targetSteps - turntablePositionSteps);
}

// =========================
// Measurement helpers
// =========================
void disconnectReferenceResistors() {
  pinMode(PIN_REF_R1, INPUT);
  pinMode(PIN_REF_R2, INPUT);
  pinMode(PIN_REF_R3, INPUT);
}

float measureResistanceWithReference(int activePin, float refValue) {
  disconnectReferenceResistors();

  pinMode(activePin, OUTPUT);
  digitalWrite(activePin, HIGH);
  delay(10);

  long adcTotal = 0;
  const int samples = 8;
  for (int i = 0; i < samples; i++) {
    adcTotal += analogRead(PIN_MEASURE_ADC);
    delay(3);
  }

  pinMode(activePin, INPUT);

  float adc = (float)adcTotal / samples;
  if (adc <= 0.5) return 0.0;
  if (adc >= 1022.5) return -1.0;

  return refValue * (adc / (1023.0 - adc));
}

float measureAutoRange() {
  disconnectReferenceResistors();

  pinMode(PIN_REF_R2, OUTPUT);
  digitalWrite(PIN_REF_R2, HIGH);
  delay(10);

  long adcTotal = 0;
  const int samples = 8;
  for (int i = 0; i < samples; i++) {
    adcTotal += analogRead(PIN_MEASURE_ADC);
    delay(3);
  }

  pinMode(PIN_REF_R2, INPUT);

  float testAdc = (float)adcTotal / samples;
  if (testAdc > 850.0) {
    return measureResistanceWithReference(PIN_REF_R3, REF_R3_OHM);
  }
  if (testAdc < 150.0) {
    return measureResistanceWithReference(PIN_REF_R1, REF_R1_OHM);
  }
  return measureResistanceWithReference(PIN_REF_R2, REF_R2_OHM);
}

// =========================
// Command handlers
// =========================
void handleStart() {
  enterMeasureState();
  moveTurntableToDegree(TURNTABLE_PHOTO_DEGREE);
  pulseLegacyTrigger();
  Serial.println("CAMERA_READY");
}

void handleMeasure() {
  float measuredOhm = measureAutoRange();

  if (measuredOhm < 0.0) {
    Serial.println("ERR,MEASURE_OPEN");
    return;
  }

  Serial.print("MEAS,");
  Serial.println(measuredOhm, 2);
}

void handleSortCommand(const String &cmd) {
  int commaIndex = cmd.indexOf(',');
  if (commaIndex < 0 || (unsigned int)commaIndex == cmd.length() - 1) {
    Serial.println("ERR,BAD_SORT_CMD");
    return;
  }

  int box = cmd.substring(commaIndex + 1).toInt();
  if (box < 1 || box > 5) {
    Serial.println("ERR,BAD_SORT_BOX");
    return;
  }

  sortToAngle(SORT_BOX_ANGLES[box - 1]);
}

void handleReset() {
  enterIdleState();
  moveTurntableToDegree(0);
  Serial.println("DONE");
}

void handleCommand(String cmd) {
  cmd.trim();
  cmd.toUpperCase();

  if (cmd.length() == 0) return;

  if (cmd == "START" || cmd == "M") {
    handleStart();
    return;
  }

  if (cmd == "MEASURE") {
    handleMeasure();
    return;
  }

  if (cmd.startsWith("SORT,")) {
    handleSortCommand(cmd);
    return;
  }

  if (cmd == "REJECT" || cmd == "R") {
    sortToAngle(REJECT_ANGLE);
    return;
  }

  // Legacy one-character sorting commands for quick manual testing.
  if (cmd.length() == 1 && cmd[0] >= '1' && cmd[0] <= '5') {
    int box = cmd[0] - '0';
    sortToAngle(SORT_BOX_ANGLES[box - 1]);
    return;
  }

  if (cmd == "RESET") {
    handleReset();
    return;
  }

  Serial.println("ERR,UNKNOWN_CMD");
}

// =========================
// Arduino entry points
// =========================
void setup() {
  Serial.begin(SERIAL_BAUD);
  Serial.setTimeout(100);

  disconnectReferenceResistors();

  pinMode(PIN_TRIGGER_LEGACY, OUTPUT);
  digitalWrite(PIN_TRIGGER_LEGACY, LOW);

  for (int i = 0; i < 4; i++) {
    pinMode(turntablePins[i], OUTPUT);
    digitalWrite(turntablePins[i], LOW);
  }

  sortServo.attach(PIN_SORT_SERVO);
  stateServo.attach(PIN_STATE_SERVO);

  sortServo.write(currentSortAngle);
  stateServo.write(currentStateServoAngle);
  delay(500);

  Serial.println("READY");
}

void loop() {
  if (Serial.available() <= 0) return;

  String cmd = Serial.readStringUntil('\n');
  handleCommand(cmd);
}
