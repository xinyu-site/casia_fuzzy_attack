project(SampleClient)
cmake_minimum_required(VERSION 3.18)

# Enable C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# COMPILER FLAGS (default: Release)
# use "-DCMAKE_BUILD_TYPE=Debug" in cmake for a Debug-build
IF(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
   SET(CMAKE_BUILD_TYPE Release)
ENDIF(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
message(STATUS "Build type is " ${CMAKE_BUILD_TYPE})

# COMPILER FLAGS
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -fpermissive")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -fpermissive")
set(CMAKE_CXX_FLAGS_RELEASE "-O2")
set(CMAKE_CXX_FLAGS_DEBUG "-O0")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -s")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -s")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if (UNIX)
	add_definitions(-D_LINUX)
endif()

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

link_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/lib
)

add_executable(SampleClient
${CMAKE_CURRENT_SOURCE_DIR}/SampleClient.cpp
)

target_link_libraries(SampleClient
nokov_sdk
)

if (WIN32)
target_link_libraries(SampleClient
wsock32)
add_custom_command(TARGET SampleClient PRE_LINK
  COMMAND ${CMAKE_COMMAND} -E copy_directory  ${CMAKE_CURRENT_SOURCE_DIR}/bin 
  $<TARGET_FILE_DIR:SampleClient>
  )
endif()

#add_custom_command(TARGET SampleClient POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${EXECUTABLE_OUTPUT_PATH}/SampleClient.exe ${CMAKE_CURRENT_SOURCE_DIR}/bin)
