
#include<SoftwareSerial.h>

#if (ARDUINO >= 100)
 #include <Arduino.h>
#else
 #include <WProgram.h>
#endif

#include <ros.h>
#include <geometry_msgs/Point.h>

SoftwareSerial my_serial(10,11);
ros::NodeHandle  nh;

float data[2];



void msg_cb1(const geometry_msgs::Point& msg)
{
    data[0] = msg.x;
    data[1] = msg.y;
    my_serial.print(data[0], 6);
    my_serial.print(',');
    my_serial.print(data[1] * 57.295792, 6);
    my_serial.print(',');
  }

ros::Subscriber<geometry_msgs::Point> sub1("min_position",&msg_cb1);



void setup() {
  my_serial.begin(38400); 
  nh.initNode();
  nh.subscribe(sub1);
}

void loop() {

  //delay(10);
  nh.spinOnce();
  delay(25);
 


}
