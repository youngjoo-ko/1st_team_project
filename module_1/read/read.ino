#include <SoftwareSerial.h>
#include <math.h>

SoftwareSerial my_serial(10, 11);


int flag = 0;
float data[2];
char char1[10];



void setup() {
  Serial.begin(38400);
  my_serial.begin(38400);
}


void recvData(int i)
{

  byte m = my_serial.readBytesUntil(',', char1, 10);
  data[m] = '\0';
  data[i] = atof(char1);
  //Serial.println(char1);

  if (i == 0) {
    Serial.print("distance=");
    Serial.println(data[i], 2);
  }
  if (i == 1)
  {
    Serial.print("degree=");
    Serial.println(data[i], 2);
    Serial.println("=======================");
  }

}


void loop() {

  if (my_serial.available())
  {
    if (flag == 0)
    {
      recvData(0);
      flag = 1;
    }
    if (flag == 1)
    {
      recvData(1);
      flag = 0;
    }
 }
}
