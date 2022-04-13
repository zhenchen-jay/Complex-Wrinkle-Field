# Funny Wrinkle Edition Tool
To download
```
git clone git@github.com:csyzzkdcz/PhaseInterpolation_polyscope.git -b refactor --recurse-submodule 
```

To build this, try
```
mkdir build
cd build
cmake -DCMAKE_CXX_COMPILER=`which g++-11` ..
make -j
./bin/userDesignerApp_bin "your mesh file"
```