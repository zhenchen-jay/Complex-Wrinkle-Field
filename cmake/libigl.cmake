
if(TARGET igl::core)
return()
endif()

include(FetchContent)
FetchContent_Declare(
libigl
GIT_REPOSITORY https://github.com/libigl/libigl.git
GIT_TAG 3370a3e9caab4708e02f00a5ae34f6aaba27a428
)
FetchContent_MakeAvailable(libigl)