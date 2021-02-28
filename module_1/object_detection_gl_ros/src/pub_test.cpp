
#include <ros/ros.h>
#include <sensor_msgs/LaserScan.h>
#include "visualization_msgs/MarkerArray.h"

#include "std_msgs/Float64MultiArray.h"


#include "object_detection.h"
#include <cmath> // 추가 
#include "object_detection_gl_ros/obj_array.h" // 추가 



//int idx = 0;
float min_x= 0.0;
float min_y = 0.0;
float min_theta = 0.0;
float min_distance = 10000.0;
float radian = 180 / M_PI;
//float angle = 0.0;
//float angle_max = 0.0;


void msgCallback_1(const sensor_msgs::LaserScan::ConstPtr& laser_msg)
{

	//angle_max = laser_msg -> angle_max;
	//angle = 180 / angle_max; // 1라디안과 같은 degree
	//ROS_INFO("angle=%f", angle);

}


void msgCallback_2(const visualization_msgs::MarkerArray::ConstPtr& marker_msg)
{	
	/*int size = marker_msg -> markers.size();
	ROS_INFO("obj_size=%d", size);
	/*for (int i=0; i<size; i++)
	{	
		ROS_INFO("x of %d = %f" ,i , marker_msg -> markers[i].pose.position.x);
		ROS_INFO("y of %d = %f" ,i , marker_msg -> markers[i].pose.position.y);
	}*/

}



void msgCallback_3(const object_detection_gl_ros::obj_array::ConstPtr& obj_msg)
{

	int size = 0;
	
    	float x = 0.0;
	float y = 0.0;
	float theta = 0.0;
	float distance =0.0;
	float degree=0.0;
	float min_degree = 0.0;
	//float angle = 57.295792;

	
	size = obj_msg -> xy.size();
	ROS_INFO("number of obj=%d", size);


	for(int i=0; i< size; i++){

		x = obj_msg -> xy[i].x; // 퍼블리시 노드에서 가져온 물체의 x좌표
		y = obj_msg -> xy[i].y; // 퍼블리시 노드에서 가져온 물체의 y좌표
		theta = atan2(y,x); // 라이다와 물체 사이의 각도(theta) 
		distance = sqrt((x*x) + (y*y)); // 객체와 라이다 사이의 거리
		degree = theta * radian; // 라이다와 물체 사이의 각도(degree) 
		//degree = floor(t * angle);
		
		//ROS_INFO("(x,y) of %d = (%.2f, %.2f)", i,  x , y);
	
		ROS_INFO("radian of %d = %.2f" ,i, theta);
		ROS_INFO("degree of %d = %.2f", i , degree);
		ROS_INFO("distance of %d = %.2f", i , distance);		

		if(distance < min_distance)
		{
			
			min_distance = distance;
			min_theta = theta;
		}

		ROS_INFO("min_theta of %d = %.2f" ,i, min_theta);
		ROS_INFO("min_distance(obj_number = %d) = %.2f", i, min_distance);
		ROS_INFO(" ");
	}
	
}



int main(int argc, char **argv)
{
	ros::init(argc, argv, "pub_test"); // 노드명 초기화
	ros::NodeHandle nh; // ROS 시스템과 통신을 위한 노드핸들 선언

	ros::Subscriber obj_sub=nh.subscribe("position", 1, msgCallback_3);
	ros::spinOnce();

	ros::Publisher arduino_pub=nh.advertise<std_msgs::Float64MultiArray>("min_position",1);

	std_msgs::Float64MultiArray msg;
	ros::Rate loop_rate(50);  


	while(ros::ok()){
		msg.data[0] = min_distance;
		msg.data[1] = min_theta;
		
		arduino_pub.publish(msg); 
		
		ros::spinOnce();
		loop_rate.sleep(); 

	}
	
	//ros::spin();
	
	return 0;

}

