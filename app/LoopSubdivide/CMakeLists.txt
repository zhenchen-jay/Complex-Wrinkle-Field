################################################################################
# General Information
################################################################################

cmake_minimum_required(VERSION 3.1)
project(LoopSundivision)

# Add your project files
# Create an executable
file(GLOB SRCFILES  
    ../../dep/SecStencils/*.h
    ../../include/*.h
    ../../include/*.hpp
    ../../include/MeshLib/*.h
    ../../include/Visualization/*.h
    ../../include/Optimization/*.h
    ../../include/IntrinsicFormula/*.h
    ../../include/ComplexLoop/*.h
    *.h

    ../../dep/SecStencils/*.cpp
    ../../src/*.cpp
    ../../src/MeshLib/*.cpp
    ../../src/Visualization/*.cpp
    ../../src/Optimization/*.cpp
    ../../src/IntrinsicFormula/*.cpp
    ../../src/ComplexLoop/*.cpp
    *.cpp)
    
add_executable(
        ${PROJECT_NAME}_bin
        ${SRCFILES}
        )

set_target_properties(${PROJECT_NAME}_bin PROPERTIES CXX_STANDARD 17)
set_target_properties(${PROJECT_NAME}_bin PROPERTIES CXX_STANDARD_REQUIRED ON)

# Link settings
target_link_libraries( ${PROJECT_NAME}_bin polyscope Eigen3::Eigen )
target_include_directories(${PROJECT_NAME}_bin PUBLIC ${SPECTRA_INCLUDE_DIRS})

if(BUILD_WITH_TBB)
  target_link_libraries( ${PROJECT_NAME}_bin polyscope geometry-central TBB::tbb igl::core igl::opengl_glfw igl::opengl_glfw_imgui igl::png igl::triangle ${SUITESPARSE_LIBRARIES} ${METIS_LIBRARY})
endif()