cmake_minimum_required(VERSION 3.10)
project(ToyML)

include_directories(include)

file(GLOB SOURCES "src/*.c" "test.c")

link_libraries(m)

add_executable(${PROJECT_NAME} ${SOURCES})

target_link_libraries(${PROJECT_NAME} m)