// Teensy 4.1 - 50 Hz square pulse on pin 5
const int pulsePin = 3;        // Output pin
const int ledPin = 13;
const int pulseDelay = 25;     // Half-period: 10 ms = 50 Hz
void setup() {
  pinMode(pulsePin, OUTPUT);   // Set the pin as output
  pinMode(ledPin, OUTPUT);
}
void loop() {
  digitalWrite(pulsePin, HIGH);  // Set HIGH
  digitalWrite(ledPin, HIGH);  // Set HIGH
  delay(pulseDelay);             // Wait 10 ms
  digitalWrite(pulsePin, LOW);   // Set LOW
  digitalWrite(ledPin, LOW);  // Set HIGH
  delay(pulseDelay);             // Wait 10 ms
}