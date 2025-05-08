// config
const int PULSE_PIN = 3;
const unsigned long PULSE_WIDTH = 5; 

// set up
void setup() {
  pinMode(PULSE_PIN, OUTPUT);
  digitalWrite(PULSE_PIN, LOW);
  Serial.begin(115200);
}

void loop() {
  if (Serial.available()) {
    char c = Serial.read();
    if (c == 't') {
      // send 1 pulse
      digitalWrite(PULSE_PIN, HIGH);
      delay(PULSE_WIDTH);
      digitalWrite(PULSE_PIN, LOW);
    }
  }

}
