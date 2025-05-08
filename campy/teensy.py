import serial 
import os 


class Teensy:
    """
    Interfaces with the Teensy device over a serial port.
    The send_reward method sends a command to trigger tone and water reward.
    """
    def __init__(self, port):
        self.port = port
        try:
            self.conn = serial.Serial(port, 115200)
            self.conn.flushInput()
        except Exception as e:
            raise Exception(f"Unable to open serial port {port}: {e}")

    def send_reward(self):
        # Send command (e.g., "p") to trigger the reward pulse.
        self.conn.write(b'p')

    def send_start_signal(self):

        self.conn.write(b's')
    
    def send_stop_signal(self):
        
        self.conn.write(b'e')

    def send_single_trigger(self):

        self.conn.write(b'i')

    def __del__(self):
        if hasattr(self, 'conn') and self.conn.isOpen():
            self.conn.close()
        print(f"Closed serial port {self.port}")