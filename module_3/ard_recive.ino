#include <SoftwareSerial.h>
SoftwareSerial mySerial(5, 6);

void setup(){
  Serial.begin(38400);
  mySerial.begin(38400);
}

void loop(){
    char ch = Serial.read(); 
    
    if (ch == 'F'){
      mySerial.write('F');
    }

    else if (ch == 'W'){
      mySerial.write('W');
    }
    
    else if (ch == 'S'){
      mySerial.write('S');
    }
   
    else if (ch == 'L'){
      mySerial.write('L');
    }
    
    else if (ch == 'R'){
      mySerial.write('R');
    }

//  if (mySerial.available())
//    Serial.write(mySerial.read());
//  if (Serial.available())
//    mySerial.write(Serial.read());
}
