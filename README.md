# Funny Wrinkle Edition Tool

## To download
```
git clone git@github.com:csyzzkdcz/PhaseInterpolation_polyscope.git 
```

## Dependencies
- [Libigl](https://github.com/libigl/libigl.git)
- [Polyscope](https://github.com/nmwsharp/polyscope.git)
- [Geometry-Central](https://github.com/nmwsharp/geometry-central.git) 
- [TBB](https://github.com/wjakob/tbb.git)
- [Spectra](https://github.com/yixuan/spectra.git)
- [Suite Sparse](https://people.engr.tamu.edu/davis/suitesparse.html)
All the dependencies are solved by Fetcontent, except Suite Sparse. 


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