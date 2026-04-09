#include <AccelStepper.h>

// ==========================================
//               PIN DEFINITIONS
// ==========================================
// MOTOR 1 (Use GPIO 26 & 27)
#define M1_DIR_PIN 2
#define M1_STEP_PIN 3

// MOTOR 2 (Use GPIO 14 & 12 - Safe ESP32 pins)
#define M2_DIR_PIN 4
#define M2_STEP_PIN 5

#define MOTOR_INTERFACE_TYPE 1

// ==========================================
//             CALIBRATION DATA
// ==========================================
// 1. Steps per rotation (TB6600 1/16 microstep = 3200)
const float STEPS_PER_REV = 3200.0; 
// 2. MM per rotation
const float MM_PER_REV = 125.0;

// Tuning Ratios (Adjust these if motors drift)
const float TUNING_FWD = 1.0; 
const float TUNING_BWD = 1.0; 

// Auto-Calculate Steps per MM
const float STEPS_PER_MM_FWD = (STEPS_PER_REV / MM_PER_REV) * TUNING_FWD;
const float STEPS_PER_MM_BWD = (STEPS_PER_REV / MM_PER_REV) * TUNING_BWD;

// ==========================================
//             OBJECT CREATION
// ==========================================
AccelStepper stepper1(MOTOR_INTERFACE_TYPE, M1_STEP_PIN, M1_DIR_PIN);
AccelStepper stepper2(MOTOR_INTERFACE_TYPE, M2_STEP_PIN, M2_DIR_PIN);

void setup() {
  Serial.begin(115200);
  
  // --- SETUP MOTOR 1 ---
  stepper1.setMaxSpeed(4000);
  stepper1.setAcceleration(2000);

  // --- SETUP MOTOR 2 ---
  stepper2.setMaxSpeed(4000);
  stepper2.setAcceleration(2000);

  Serial.println("--- Dual-Motor Controller (ESP32) ---");
  Serial.println("Format: distance1, distance2");
  Serial.println("Example: '100, 50' moves M1 100mm and M2 50mm");
}

void loop() {
  // Check for Serial Input
  if (Serial.available() > 0) {
    
    // Parse two numbers separated by a comma or space
    float mm_input_1 = -Serial.parseFloat();
    float mm_input_2 = -Serial.parseFloat();

    // --- MOTOR 1 LOGIC ---
    if (mm_input_1 != 0) {
      long steps1 = 0;
      if (mm_input_1 > 0) steps1 = mm_input_1 * STEPS_PER_MM_FWD;
      else                steps1 = mm_input_1 * STEPS_PER_MM_BWD;
      
      stepper1.move(steps1);
      Serial.print("M1: "); Serial.print(mm_input_1); Serial.print("mm ");
    }

    // --- MOTOR 2 LOGIC ---
    if (mm_input_2 != 0) {
      long steps2 = 0;
      if (mm_input_2 > 0) steps2 = mm_input_2 * STEPS_PER_MM_FWD;
      else                steps2 = mm_input_2 * STEPS_PER_MM_BWD;
      
      stepper2.move(steps2);
      Serial.print("| M2: "); Serial.print(mm_input_2); Serial.print("mm");
    }

    if (mm_input_1 != 0 || mm_input_2 != 0) {
      Serial.println(); // New line for cleanliness
    }
    
    // Clear buffer
    while(Serial.available() > 0) { Serial.read(); }
  }

  // CRITICAL: run() must be called as fast as possible in the loop
  stepper1.run();
  stepper2.run();
}