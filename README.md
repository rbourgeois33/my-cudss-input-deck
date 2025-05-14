How to use:

Download cuDSS as a tarball: https://developer.nvidia.com/cudss-downloads?target_os=Linux&target_arch=x86_64&Distribution=Agnostic&cuda_version=12

then go into /cuDSS/libcudss-linux-x86_64-0.5.0.16_cuda12-archive/examples (adapt link)

copy this repo's files in there, adapt the CMakeLists.txt follow the build instructions in the example's README

to run:

./cudss_simple_from_file 3 general full matrix.mtx rhs.mtx

see the source code for options
