// 서보모터 사용을 위한 호출
#include <SoftwareSerial.h>
SoftwareSerial mySerial(12, 13);
#include <Servo.h>

Servo SB1;
int servo_pin = 4;

// 뒷바퀴속도(0~255)
int pwm = 3;

// 뒷바퀴방향(전진,후진)
int dir_pin = 8;

int degree = 0;
int power = 0;
int dir = 0;

void setup() {
  
  // 서보모터핀
  SB1.attach(servo_pin);
  
  // 뒷바퀴속도
  pinMode(pwm, OUTPUT);

  // 뒷바퀴방향
  pinMode(dir_pin, OUTPUT);
  
  //시리얼 통신 설정
  Serial.begin(38400);
  mySerial.begin(38400);
  
}  
void loop() {
  
  if (mySerial.available() > 0){
      char ch = mySerial.read();
      Serial.println(ch);
      
       if (ch == 'L'){
//          Serial.println(ch);
            dir = 0;
            degree = 90;
            power = 25;
        }
       else if (ch == 'R'){
            //Serial.println(ch);
            dir = 0;
            degree = 140;
            power = 25;
       }
       else if (ch == 'F'){
//          Serial.println(ch);
            dir = 0;
            degree = 115;
            power = 25;
       }
       else if (ch == 'W'){
//          Serial.println(ch);
            dir = 0;
            degree = 115;
            power = 20;
       }
       else if (ch == 'S'){
            //Serial.println(ch);
            dir = 0;
            degree = 115;
            power = 0;
       }
       else if (ch == 'B'){
            //Serial.println(ch);
            dir = 1;
            degree = 115;
            power = 25;
       }
      }
  SB1.write(degree);
  digitalWrite(dir_pin, dir);
  analogWrite(pwm, power);
}
