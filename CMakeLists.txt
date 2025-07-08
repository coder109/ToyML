cmake_minimum_required(VERSION 3.10)
project(MyProject)

include_directories(include)

file(GLOB SOURCES "src/*.c" "test.c")

add_executable(${PROJECT_NAME} ${SOURCES})