cmake_minimum_required(VERSION 3.5)
project(omega_haptic_device)

set(CMAKE_CXX_STANDARD 17)
add_compile_options(-std=c++17)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(ohrc_msgs REQUIRED)
find_package(Threads REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

link_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Omega.msg"
  "msg/Gripper.msg"
  DEPENDENCIES std_msgs geometry_msgs
)

include_directories(
  include
)

set(COMMON_DEPS
  rclcpp
  std_msgs
  geometry_msgs
  ohrc_msgs
  tf2
  tf2_ros
  tf2_eigen
  tf2_geometry_msgs
)

add_executable(omega_driver_test
  src/omega_driver_test.cpp
  src/omega_driver.cpp
)
ament_target_dependencies(omega_driver_test ${COMMON_DEPS})
target_link_libraries(omega_driver_test dhd drd usb-1.0 Threads::Threads)

add_executable(omega_driver_left
  src/omega_driver_left.cpp
  src/omega_driver.cpp
)
ament_target_dependencies(omega_driver_left ${COMMON_DEPS})
target_link_libraries(omega_driver_left dhd drd usb-1.0 Threads::Threads)

add_executable(omega_driver_right
  src/omega_driver_right.cpp
  src/omega_driver.cpp
)
ament_target_dependencies(omega_driver_right ${COMMON_DEPS})
target_link_libraries(omega_driver_right dhd drd usb-1.0 Threads::Threads)

install(TARGETS
  omega_driver_test
  omega_driver_left
  omega_driver_right
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
