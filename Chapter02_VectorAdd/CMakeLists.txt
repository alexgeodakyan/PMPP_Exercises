# 1. Define the minimum version of CMake required
cmake_minimum_required(VERSION 3.18)

# 2. Name your project and tell CMake to look for C++ and CUDA compilers
# "PMPP_Ch02" is a user-defined label; you can change this.
project(PMPP_Ch02 LANGUAGES CXX CUDA)

# 3. Create the executable
# 'vector_add' is the name of the program that will be created.
# 'main.cu' is the name of your source file (ensure it matches your file name).
add_executable(vector_add main.cu)

# 4. Optional: Tell CMake to use a specific GPU architecture (RTX 4050 is sm_89)
set_target_properties(vector_add PROPERTIES CUDA_ARCHITECTURES native)