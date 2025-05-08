// ---------- Pin & timing configuration ----------
const int   PULSE_PIN       = 3;                // your trigger pin
const int LED_PIN = 13;                         // led pin for debugging
const unsigned long PULSE_WIDTH_MS = 10;       // pulse duration
const unsigned long PERIOD_MS      = 1000UL/15; // ~66 ms period for 15 Hz

// ---------- State variables ----------
bool running      = false;    // true once 's' received, false once 'e'
bool pulseActive  = false;    // are we currently driving the pin HIGH?
unsigned long lastPulseTime = 0; // timestamp when last pulse started
unsigned long pulseOnTime    = 0; // timestamp when pin was set HIGH

void setup() {
  pinMode(PULSE_PIN, OUTPUT);
  digitalWrite(PULSE_PIN, LOW);
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);
  Serial.begin(115200);
}

void loop() {
  unsigned long now = millis();

  // 1) Check for start/stop commands
  if (Serial.available() > 0) {
    char cmd = Serial.read();
    if (cmd == 's') {
      running = true;
      // schedule an immediate pulse on next iteration
      lastPulseTime = now - PERIOD_MS;
    }
    else if (cmd == 'e') {
      running     = false;
      pulseActive = false;
      digitalWrite(PULSE_PIN, LOW);
      digitalWrite(LED_PIN, LOW);
    }
    else if (cmd == 'i') {
      running = false;
      pulseActive = true;
      digitalWrite(PULSE_PIN, HIGH);
      digitalWrite(LED_PIN, HIGH);
      pulseOnTime = now;
      lastPulseTime =  now;
    }
  }

  // 2) If we’re in the middle of a pulse, see if it’s time to turn off
  if (pulseActive && now - pulseOnTime >= PULSE_WIDTH_MS) {
    digitalWrite(PULSE_PIN, LOW);
    digitalWrite(LED_PIN, LOW);
    pulseActive = false;
    // lastPulseTime remains the timestamp of when this pulse began
  }

  // 3) If not currently pulsing, and we're “running,” and period elapsed → start next pulse
  if (!pulseActive && running && now - lastPulseTime >= PERIOD_MS) {
    digitalWrite(PULSE_PIN, HIGH);
    digitalWrite(LED_PIN, HIGH);
    pulseOnTime    = now;
    lastPulseTime  = now;
    pulseActive    = true;
  }
}

