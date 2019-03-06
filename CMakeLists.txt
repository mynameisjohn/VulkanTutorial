cmake_minimum_required(VERSION 2.8)

# https://vulkan-tutorial.com/Introduction
project(VulkanTutorial)

set(CMAKE_CXX_STANDARD 17)

find_package(Vulkan)

set(GLFW_DIR $ENV{SDK}/glfw)
set(GLM_DIR $ENV{SDK}/glm)
set(STB_DIR $ENV{SDK}/stb)

include_directories(${Vulkan_INCLUDE_DIRS} ${GLM_DIR} ${STB_DIR} ${GLFW_DIR}/include)
link_directories(${GLFW_DIR}/lib)

add_executable(VulkanTutorial main.cpp shader.vert shader.frag)
target_link_libraries(VulkanTutorial ${Vulkan_LIBRARIES} glfw3)