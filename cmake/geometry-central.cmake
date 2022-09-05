include(FetchContent)
FetchContent_Declare(
        geometry-central
        GIT_REPOSITORY "https://github.com/nmwsharp/geometry-central.git"
        GIT_TAG "cf0531dd15b72b4e314f7a0269156d94227db562"
)
FetchContent_MakeAvailable(geometry-central)