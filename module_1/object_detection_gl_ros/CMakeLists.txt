cmake_minimum_required(VERSION 2.8.3)
project(object_detection_gl_ros)

EXECUTE_PROCESS( COMMAND uname -m COMMAND tr -d '\n' OUTPUT_VARIABLE ARCHITECTURE )
message( STATUS "Architecture: ${ARCHITECTURE}" )
set( LIBRARY_PATH "${PROJECT_SOURCE_DIR}/lib/${ARCHITECTURE}" )
set( GL_LIBRARY "${LIBRARY_PATH}/libobject_detection.so" "${LIBRARY_PATH}/libgl_driver.so" "${LIBRARY_PATH}/libserial.so" )

add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
  visualization_msgs 
  geometry_msgs
  std_msgs
  message_generation #추가 
  rosserial_arduino
  rosserial_client
)

# 사용할 메세지 파일 선언
add_message_files(
FILES
obj_array.msg
#obj_position.msg

 )

# 의존하는 메세지를 설정하는 옵션(std_msgs가 설치되어 있지 않으면 빌드 도중에 에러 발생 가능)
generate_messages(
DEPENDENCIES
std_msgs)



###################################
## catkin specific configuration ##
###################################
catkin_package(
  INCLUDE_DIRS include
  #LIBRARIES object_detection_gl_ros  # 변경 
  CATKIN_DEPENDS roscpp sensor_msgs visualization_msgs geometry_msgs std_msgs  rosserial_arduino rosserial_client
  
)

###########
## Build ##
###########

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  #${GL_LIBRARY} # 추가 
)


add_executable(object_detection_gl_ros_node
  src/object_detection_gl_ros_node.cpp
)
#add_executable(object_detection_sub_node
#  src/object_detection_sub_node.cpp
#)


add_executable(object_detection_sub_node
  src/object_detection_sub_node.cpp
)





target_link_libraries(object_detection_gl_ros_node
  ${catkin_LIBRARIES}
  ${GL_LIBRARY}

)


target_link_libraries(object_detection_sub_node
  ${catkin_LIBRARIES}
  ${GL_LIBRARY}
)




add_dependencies(object_detection_gl_ros_node
  ${${PROJECT_NAME}_EXPORTED_TARGETS} 
  ${catkin_EXPORTED_TARGETS}
  ${GL_LIBRARY}
)

add_dependencies(object_detection_sub_node
  ${${PROJECT_NAME}_EXPORTED_TARGETS} 
  ${catkin_EXPORTED_TARGETS}
  ${GL_LIBRARY}
)

#add_dependencies(object_detection_sub_node
#  ${${PROJECT_NAME}_EXPORTED_TARGETS} 
#  ${catkin_EXPORTED_TARGETS}
#  ${GL_LIBRARY}
#)

