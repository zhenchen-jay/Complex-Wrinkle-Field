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
- Running the command with a JSON file from the "data" folder (e.g.,   `data/bunny_localRotation/data.json`) will open a GUI that displays the interpolated results between two given keyframes.
- You can click the "Solve" button to re-run the algorithm with different parameters, such as the number of frames.
- For visual efficiency, we only upsample twice to get the final wrinkles. You can increase this number ("upsample level" option in the GUI) for a better wrinkle appearance. 

## Other features
Two additional executable programs are provided:
- WrinkleInterpolationCli_bin: A command line version of the interpolation app, which can be run with:
```
./bin/WrinkleInterpolationCli_bin -i "some json file" -r 
```
The `-r` option forces the program to re-optimize the problem. Without `-r`, the program generates the corresponding (upsampled) wrinkled mesh sequence and save under `render/wrinkledMesh`. Note that this process may take a while (depends on the upsampling level)

- WrinkleInterpolationTwistedCylinder_bin: A specialized GUI designed for the twisted cylinder example shown in the paper. It is used like WrinkleInterpolationGui_bin:
```
./bin/WrinkleInterpolationTwistedCylinder_bin -i "some json file"
```
The only difference is that it also takes a sequence of base meshes as input.


## Issues
When compiling on macOS with C++17, you may encounter this issue:
```
build/_deps/comiso-src/ext/gmm-4.2/include/gmm/gmm_domain_decomp.h:84:2: error: ISO C++17 does not allow 'register' storage class specifier [-Wregister]
```
To solve this, please replace `register double` by `double` in the file `build/_deps/comiso-src/ext/gmm-4.2/include/gmm/gmm_domain_decomp.h`

## Contact
If you need any assistance, please feel free to contact me at zhenjaychen@gmail.com.
