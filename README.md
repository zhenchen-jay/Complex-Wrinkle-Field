# Complex Wrinkle Field Evolution
This repository is the implementation of the paper: "Complex Wrinkle Field Evolution".

## To download
```
git clone https://github.com/zhenchen-jay/Complex-Wrinkle-Field.git 
```

## Dependencies
- [Libigl](https://github.com/libigl/libigl.git)
- [Polyscope](https://github.com/nmwsharp/polyscope.git)
- [Geometry-Central](https://github.com/nmwsharp/geometry-central.git) 
- [TBB](https://github.com/wjakob/tbb.git)
- [Spectra](https://github.com/yixuan/spectra.git)
- [Suite Sparse](https://people.engr.tamu.edu/davis/suitesparse.html)

All the dependencies are solved by Fetcontent, except Suite Sparse and Spectra (see below for instructions for these two libraries). 

## build with spectra
In order to build with Spectra with the same Eigen version of libigl, please comment out line 24-26 of the /build/_deps/spectra-src/CMakeLists.txt:
```
# find_package(Eigen3 NO_MODULE REQUIRED)
# set_package_properties(Eigen3 PROPERTIES TYPE REQUIRED PURPOSE "C++ vector data structures")
# message(STATUS "Found Eigen3 Version: ${Eigen3_VERSION} Path: ${Eigen3_DIR}")
```

## build with Suite-Sparse
This part is tricky, for linux, you should use 
```
sudo apt-get update -y
sudo apt-get install -y libsuitesparse-dev
```

For macOS, this can be done with [Homebrew](https://brew.sh/):
```
brew install suite-sparse
```

For windows, please follow the guidence provided in [suitesparse-metis-for-windows](https://github.com/jlblancoc/suitesparse-metis-for-windows).


## Build and Run
```
mkdir build
cd build
cmake ..
make -j4
./bin/WrinkleInterpolationGui_bin -i ../data/bunny_localRotation/data.json
```

## Results
Once you run the command, a GUI will pop up. Then, click on "Solve" and the local rotation bunny results, as shown in the paper, will be displayed.

![bunny_local_rotation](https://user-images.githubusercontent.com/29785561/188839142-906f3b2e-1051-458d-9c80-bd189e9bca07.gif)


## Other features
There are two extra executable programs provided: 
- WrinkleInterpolationCli_bin: a command line version of our interpolation app, which can be run by
```
./bin/WrinkleInterpolationCli_bin -i ../data/bunny_localRotation/data.json -r 
```
where -r option forces the program to reoptimize the problem

- UserWrinkleDesign_bin: a tool for users to design wrinkles by themselves by assign frequency field and amplitude. This can be executed with
```
./bin/UserWrinkleDesign_bin -i ../data/bunny_localRotation/design.json 
``` 
where the bunny wrinkles can be extracted by clicking on "run smoothest vector field", with "freq" setting to 62.83185 (20 pi) and "amp" equal to "0.01". Note that this app has not undergone full testing for robustness and there may be some GUI crash bugs.


## Issues
When compiling on MacOS with C++17, you may encounter this issue: 
```
build/_deps/comiso-src/ext/gmm-4.2/include/gmm/gmm_domain_decomp.h:84:2: error: ISO C++17 does not allow 'register' storage class specifier [-Wregister]
```
To solve this, please replace `register double` by `double`.  
