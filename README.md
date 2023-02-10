# Funny Wrinkle Edition Tool

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

All the dependencies are solved by Fetcontent, except Suite Sparse and Spectra (need Eigen installed). 
For macOS, please  make sure TBB was installed using homebrew by `brew install tbb@2020`. (The newest brew installed TBB will have some cmake issue.)

## Spectra build
In order to build with Spectra, you should install eigen as:
```
git clone https://gitlab.com/libeigen/eigen.git
cd eigen
mkdir build
cd build
cmake ..
make install
```

## Suite Sparse build
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


## To build this, try
```
mkdir build
cd build
cmake ..
make -j4
./bin/staticWrinkleEditor_bin ../data/bunny_localRotation/data.json
```

## Results
Once you run that command, you will see a gui pop up, then click solve, you will get our local rotation bunny results shown in the paper.

![bunny_local_rotation](https://user-images.githubusercontent.com/29785561/188839142-906f3b2e-1051-458d-9c80-bd189e9bca07.gif)

## Issues
When compiling in MacOS with C++17, you may encounter this issue: 
```
build/_deps/comiso-src/ext/gmm-4.2/include/gmm/gmm_domain_decomp.h:84:2: error: ISO C++17 does not allow 'register' storage class specifier [-Wregister]
```
To solve this, please replace `register double` by `double`.  
