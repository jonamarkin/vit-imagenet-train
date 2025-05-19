#!/bin/bash
set -e  # Exit on any error

# Clean old build
if [ -d "build" ]; then
    rm -rf build
fi

mkdir build
cd build

# Export MPI profiling path (if needed)
export MPIP="-f /home/j.markin/hpcproject/ff_programs/resnet_train/mpip_results/ib0"

# === Symlinks ===
ln -s /opt/intel/oneapi/compiler/2023.2.1/linux/compiler/lib/intel64_lin/libiomp5.so ./libomp.so
ln -s /home/j.markin/libjpeg-turbo-3.1.0/build/libjpeg.so.62 ./libjpeg.so.62

# === Copy data and models ===
cp ../resnet18_cifar100.pt .
cp ../resnet152.pt .
cp -R ../data .

# === Compilation ===
echo "Compiling train_model.cpp..."

TORCH_DIR=$HOME/hpcproject/pytorch-install
VISION_DIR=$HOME/hpcproject/vision-install/share/cmake/TorchVision
MKL_INCLUDE=$MKLROOT/include
MKL_LIB=$MKLROOT/lib/intel64

CXX=mpiicpx
CC=mpiicx
CXXFLAGS="-std=c++20 -O3 -Wall -DNDEBUG -pthread -finline-functions"
INCLUDES="-I${TORCH_DIR}/include \
          -I${TORCH_DIR}/include/torch/csrc/api/include \
          -I${MKL_INCLUDE} \
          -I$HOME/fastflow \
          -I/home/j.markin/lib/cereal/include \
          -I/home/j.markin/libjpeg-turbo-3.1.0/build \
          -I$HOME/local/include \
          -I${VISION_DIR}/include"

LIBS="-L${TORCH_DIR}/lib \
      -ltorch -ltorch_cpu -lc10 \
      -L${MKL_LIB} \
      -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 \
      -ldl -lpthread -lm \
      -L${VISION_DIR} \
      -Wl,-rpath,${TORCH_DIR}/lib" 

# Compile
$CXX $CXXFLAGS $INCLUDES ../train_model.cpp -o train_resnet $LIBS

echo "✅ Build complete."

# === Run ===
# echo "🚀 Launching mpirun..."
# I_MPI_OFI_PROVIDER=psm2 \
# I_MPI_DEBUG=5 \
# LD_PRELOAD=/home/j.markin/mpiP_build/lib/libmpiP.so \
# nohup mpirun -iface ib0 \
#   --host node01-ib0,node02-ib0,node03-ib0,node04-ib0,node05-ib0,node06-ib0,node07-ib0,node08-ib0,node09-ib0,node12-ib0 \
#   -n 6 -ppn 1 ./train_resnet --DFF_Config=../resnet.json \
#   > ../ib0_logs/28_workers.log 2>&1 &


for i in {1..5}; do
  echo "▶️ Run $i..."

  I_MPI_OFI_PROVIDER=psm2 \
  I_MPI_DEBUG=5 \
  LD_PRELOAD=/home/j.markin/mpiP_build/lib/libmpiP.so \
  nohup mpirun --iface ib0 \
    --host node01-ib0,node02-ib0,node03-ib0,node04-ib0,node05-ib0,node06-ib0,node07-ib0,node08-ib0,node09-ib0,node12-ib0,node13-ib0,node14-ib0,node15-ib0,node16-ib0,node17-ib0,node18-ib0,node19-ib0,node20-ib0,node21-ib0,node22-ib0,node23-ib0,node24-ib0,node25-ib0,node26-ib0,node27-ib0,node28-ib0,node29-ib0,node30-ib0,node31-ib0,node32-ib0  \
    -n 30 -ppn 1 ./train_resnet --DFF_Config=../resnet.json \
    > ../ib0_logs/28_workers_run_${i}.log 2>&1 

  sleep 2  # optional: wait between launches (adjust if needed)
done