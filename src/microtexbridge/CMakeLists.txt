cmake_minimum_required(VERSION 3.16)

project(veusz_microtexbridge LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(MICROTEX_SRC "" CACHE PATH "Path to the MicroTeX source tree")
set(MICROTEX_LIB "" CACHE FILEPATH "Path to the built libLaTeX.a")

if(NOT MICROTEX_SRC)
  message(FATAL_ERROR "MICROTEX_SRC is required")
endif()
if(NOT MICROTEX_LIB)
  message(FATAL_ERROR "MICROTEX_LIB is required")
endif()

find_package(Qt6 COMPONENTS Core Gui Widgets Svg REQUIRED)
find_library(TINYXML2_LIB tinyxml2 REQUIRED)

add_library(microtexbridge SHARED microtex_bridge.cpp)
target_compile_definitions(microtexbridge PRIVATE BUILD_QT)
target_include_directories(microtexbridge PRIVATE "${MICROTEX_SRC}/src")
target_link_libraries(microtexbridge PRIVATE
  "${MICROTEX_LIB}"
  Qt6::Core
  Qt6::Gui
  Qt6::Widgets
  Qt6::Svg
  "${TINYXML2_LIB}"
)
