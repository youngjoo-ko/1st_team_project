
#include <ros/ros.h>
#include <sensor_msgs/LaserScan.h>
#include "visualization_msgs/MarkerArray.h"
#include "geometry_msgs/Pose2D.h"
#include "std_msgs/UInt16.h"
#include "object_detection.h"
#include <cmath> // 추가 
#include "object_detection_gl_ros/obj_array.h" // 추가 







void msgCallback_3(const object_detection_gl_ros::obj_array::ConstPtr& obj_msg)
{	

	int size = 0;
	int idx = 0;

	float r =0.0;
	float degree=0.0;
	float angle = 57.295792;
	float x = 0.0;
	float y = 0.0;
	float t = 0.0;

	ros::NodeHandle nh2;
	ros::Publisher arduino_pub=nh2.advertise<geometry_msgs::Pose2D>("arduino_position",1);
	//ros::Publisher arduino_pub2=nh2.advertise<std_msgs::UInt16.h>("idx",1);
	ros::Rate loop_rate(10); 
	geometry_msgs::Pose2D position_msg;
	//std_msgs::UInt16 idx_msg;

	size = obj_msg -> xy.size();
	ROS_INFO("number of obj=%d", size);


	for(int i=0; i< size; i++){

		//idx = i;
		x = obj_msg -> xy[i].x; // 퍼블리시 노드에서 가져온 물체의 x좌표
		y = obj_msg -> xy[i].y; // 퍼블리시 노드에서 가져온 물체의 y좌표
		t = atan2(y,x); // 라이다와 물체 사이의 각도(theta) 
		r = sqrt((x*x) + (y*y)); // 객체와 라이다 사이의 거리
		degree = t * angle; // 라이다와 물체 사이의 각도(degree) 

		ROS_INFO("(x,y) of %d = (%lf, %lf)", i,  x , y);
		ROS_INFO("radian of %d = %lf" ,i, t);
		ROS_INFO("degree of %d = %lf", i , degree);
		ROS_INFO("distance of %d = %lf", i , r);		
		ROS_INFO(" ");
	
		position_msg.x = x;
		position_msg.y = y;
		position_msg.theta= t;
		//idx_msg.data = idx;
		arduino_pub.publish(position_msg);
		//arduino_pub2.publish(idx_msg);
		//loop_rate.sleep(); 
		//ros::spinOnce();

		//ROS_INFO("min_distance(obj_number = %d) = %lf", idx, min_r);

	}
	loop_rate.sleep(); 
	ros::spinOnce();
	
}


//void msgCallback_4(const visualization_msgs::MarkerArray::ConstPtr& ard_msg){

//}


int main(int argc, char **argv)
{
	ros::init(argc, argv, "project_sub_node"); // 노드명 초기화
	ros::NodeHandle nh; // ROS 시스템과 통신을 위한 노드핸들 선언
	//ros::AsyncSpinner spinner(0);
	//spinner.start();

	ros::Subscriber obj_sub=nh.subscribe("position", 1, msgCallback_3);
	//ros::Subscriber arduino_sub=nh.subscribe("arduino", 1, msgCallback_4);

	
	//ros::Publisher arduino_pub2=nh.advertise<std_msgs::UInt16>("idx",1);
	//ros::Rate loop_rate(50); // 0.5 초 

	
	//std_msgs::UInt16 idx_msg;

	//while(ros::ok()){
		
		//idx_msg.data = idx;
		//ROS_INFO("obj_number = %d , position=(%f, %f)", idx, position_msg.x , position_msg.y);
		//ROS_INFO(" ");
		
		//arduino_pub2.publish(idx_msg);
		//loop_rate.sleep(); 
		//ros::spinOnce();

	//}
	
	ros::spin();
	//ros::waitForShutdown();
	
	return 0;

}

