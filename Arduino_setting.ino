#include <Servo.h>
Servo LockServo;
int data;
int ServoLock = 13;

void setup() 
{
  Serial.begin(9600);
  pinMode(ServoLock, OUTPUT);
  LockServo.attach(9);
}

void loop() 
{
  if (Serial.available())
  {
    data = Serial.read();
    if (data == '0')
    {
      LockServo.write(0);
      digitalWrite (ServoLock, LOW);
    }

    if (data == '1')
    {
     digitalWrite (ServoLock, HIGH);
     LockServo.write(90);
   }
  }
}
