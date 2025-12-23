cmake_minimum_required(VERSION 3.16)
project(parallel_image_pipeline CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(APPLE)
  set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include")
  set(OpenMP_CXX_LIB_NAMES "omp")
  set(OpenMP_omp_LIBRARY "/opt/homebrew/opt/libomp/lib/libomp.dylib")
endif()

find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)

# SERIAL
add_executable(serial_app serial/main.cpp)
target_link_libraries(serial_app PRIVATE ${OpenCV_LIBS})

# OPENMP (3 yöntem)
add_executable(openmp_m1 openmp/main1.cpp)
target_link_libraries(openmp_m1 PRIVATE ${OpenCV_LIBS} OpenMP::OpenMP_CXX)

add_executable(openmp_m2 openmp/main2.cpp)
target_link_libraries(openmp_m2 PRIVATE ${OpenCV_LIBS} OpenMP::OpenMP_CXX)

add_executable(openmp_m3 openmp/main3.cpp)
target_link_libraries(openmp_m3 PRIVATE ${OpenCV_LIBS} OpenMP::OpenMP_CXX)
#  MPI 
add_executable(mpi_app mpi/main.cpp)
target_link_libraries(mpi_app PRIVATE ${OpenCV_LIBS} MPI::MPI_CXX)

target_include_directories(mpi_app PRIVATE ${MPI_CXX_INCLUDE_DIRS})

