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

/////////////////////////////////////////

void setup() {
  // put your setup code here, to run once:
  //////////////////// 핀모드 설정////////////////////////
  
  // 서보모터핀
  SB1.attach(servo_pin);
  
  // 뒷바퀴속도
  pinMode(pwm, OUTPUT);

  // 뒷바퀴방향
  pinMode(dir_pin, OUTPUT);
  
  ////////////////////////////////////////////

  //시리얼 통신 설정
  Serial.begin(38400);
  mySerial.begin(38400);
  
}  
void loop() {
  
  if (mySerial.available() > 0){
      char ch = mySerial.read();
      Serial.println(ch);
      
       if (ch == 'L'){
            SB1.write(90);
            digitalWrite(dir_pin, LOW);
            analogWrite(pwm, 20);
            delay(40);
        }
       else if (ch == 'R'){
            SB1.write(140);
            digitalWrite(dir_pin, LOW);
            analogWrite(pwm, 20);
            delay(40);
       }
       else if (ch == 'F'){
            SB1.write(115);
            Serial.print('F');
            digitalWrite(dir_pin, LOW);
            analogWrite(pwm, 20);
            delay(40);
       }
       else if (ch == 'S'){
            SB1.write(115);
            Serial.print('S');
            analogWrite(pwm, 0);
            delay(40);
       }
      }
  else{
    SB1.write(115);
    analogWrite(pwm, 0);
    delay(40);
  }
}
