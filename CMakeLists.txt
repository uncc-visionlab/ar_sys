cmake_minimum_required(VERSION 2.8.3)
project(ar_sys)

set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")

find_package(catkin REQUIRED COMPONENTS
	cv_bridge
	dynamic_reconfigure
	geometry_msgs
	image_transport
	roscpp
	tf
	visualization_msgs
		message_generation
)

find_package(OpenCV REQUIRED)

add_message_files(
		FILES
		SingleCorner.msg
		ArucoCornerMsg.msg
		OneMarker.msg
)

generate_messages(
		DEPENDENCIES
		std_msgs
)


###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if you package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
	INCLUDE_DIRS include
	CATKIN_DEPENDS cv_bridge dynamic_reconfigure geometry_msgs image_transport visualization_msgs roscpp tf
	DEPENDS OpenCV
)

include_directories(
	include
	${catkin_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
)

## Declare a cpp library
add_executable(single_board src/single_board.cpp
	src/utils.cpp)
add_dependencies(single_board ${catkin_EXPORTED_TARGETS} ar_sys_generate_messages_cpp)
target_link_libraries(single_board ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})

add_executable(charuco src/charuco.cpp
	src/utils.cpp)
add_dependencies(charuco ${catkin_EXPORTED_TARGETS} ar_sys_generate_messages_cpp)
target_link_libraries(charuco ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})

add_executable(multi_boards src/multi_boards.cpp
	src/utils.cpp)
add_dependencies(multi_boards ${catkin_EXPORTED_TARGETS} ar_sys_generate_messages_cpp)
target_link_libraries(multi_boards ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})

add_executable(system_viewer src/system_viewer.cpp)
add_dependencies(system_viewer ${catkin_EXPORTED_TARGETS})
target_link_libraries(system_viewer ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})

install(DIRECTORY include/
	DESTINATION ${CATKIN_GLOBAL_INCLUDE_DESTINATION}
	FILES_MATCHING PATTERN "*.h"
)
install(DIRECTORY camera_info/
	DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/camera_info
	FILES_MATCHING PATTERN "*.yaml"
)
install(DIRECTORY data/
	DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/data
)
install(DIRECTORY launch/
	DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
	FILES_MATCHING PATTERN "*.launch"
)
install(TARGETS single_board multi_boards system_viewer
	ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
	LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
	RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

