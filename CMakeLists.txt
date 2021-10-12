CMAKE_MINIMUM_REQUIRED(VERSION 3.3)
project(hanalearn)
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra -Wno-register -fPIC -march=native -Wfatal-errors")

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/rela)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/hanabi-learning-environment)

find_package(Torch REQUIRED)

pybind11_add_module(
  hanalearn
  ${CMAKE_CURRENT_SOURCE_DIR}/rlcc/utils.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/rlcc/r2d2_actor.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/rlcc/hand_dist.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/rlcc/hybrid_model.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/rlcc/sparta.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/rlcc/pybind.cc
)
target_link_libraries(hanalearn PUBLIC hanabi)
target_link_libraries(hanalearn PUBLIC rela_lib)
target_include_directories(hanalearn PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
