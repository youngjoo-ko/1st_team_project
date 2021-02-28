
#include <ros/ros.h>
#include <sensor_msgs/LaserScan.h>
#include "visualization_msgs/MarkerArray.h"
#include "geometry_msgs/Pose2D.h"
#include "object_detection.h"
#include <cmath> // 추가 
#include "object_detection_gl_ros/obj_array.h" // 추가 



int idx = 0;
float min_x= 0.0;
float min_y = 0.0;
float min_theta = 0.0;


void msgCallback_3(const object_detection_gl_ros::obj_array::ConstPtr& obj_msg)
{
	

	int size = 0;
	float min_distance = 10000.0;
    	float x = 0.0;
	float y = 0.0;
	float theta = 0.0;
	float distance =0.0;
	float degree=0;
	
	size = obj_msg -> xy.size();
	ROS_INFO("number of obj=%d", size);


	for(int i=0; i< size; i++){

		x = obj_msg -> xy[i].x; // 퍼블리시 노드에서 가져온 물체의 x좌표
		y = obj_msg -> xy[i].y; // 퍼블리시 노드에서 가져온 물체의 y좌표
		theta = atan2(y,x); // 라이다와 물체 사이의 각도(theta) 
		distance = sqrt((x*x) + (y*y)); // 객체와 라이다 사이의 거리
		degree = t * 57.295792; // 라이다와 물체 사이의 각도(degree) 
	
		ROS_INFO("(x,y) of %d = (%.2f, %.2f)", i,  x , y);
		ROS_INFO("radian of %d = %.2f" ,i, theta);
		ROS_INFO("distance of %d = %.2f", i , distance);		

		if(distance < min_distance)
		{
			idx = i;
			min_x = x;
			min_y = y;
			min_distance = distance;
			min_theta = theta;
		}


		ROS_INFO("min_distance(obj_number = %d) = %.2f", idx, min_distance);
		ROS_INFO(" ");
	}
	
}



int main(int argc, char **argv)
{
	ros::init(argc, argv, "object_detection_sub_node2"); // 노드명 초기화
	ros::NodeHandle nh; // ROS 시스템과 통신을 위한 노드핸들 선언

	ros::Subscriber obj_sub=nh.subscribe("position", 1, msgCallback_3);


	ros::Publisher arduino_pub=nh.advertise<geometry_msgs::Pose2D>("min_position",1);
	ros::Rate loop_rate(50); // 0.5 초 
	geometry_msgs::Pose2D msg;

	while(ros::ok()){
		msg.x = min_x;
		msg.y = min_y;
		msg.theta= min_theta;
		
		arduino_pub.publish(msg); 
		loop_rate.sleep(); 
		ros::spinOnce();

	}
	
	ros::spin();
	
	return 0;

}

