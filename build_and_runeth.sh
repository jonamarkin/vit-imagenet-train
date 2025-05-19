#!/bin/bash
set -e  # Exit on any error

# Clean old build
if [ -d "build" ]; then
    rm -rf build
fi

mkdir build
cd build

# Export MPI profiling path (if needed)
export MPIP="-f /home/j.markin/hpcproject/ff_programs/resnet_train/mpip_results/eth"

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
# I_MPI_OFI_PROVIDER=tcp \
# I_MPI_DEBUG=5 \
# LD_PRELOAD=/home/j.markin/mpiP_build/lib/libmpiP.so \
# nohup mpirun \
#   --host node01-eth1,node02-eth1,node03-eth1,node04-eth1,node05-eth1,node06-eth1 \
#   -n 6 -ppn 1 ./train_resnet --DFF_Config=../resneteth.json \
#   > ../eth_logs/4_workers_run_1.log 2>&1 &

# === Run 5 times ===
echo "🚀 Launching mpirun 5 times..."

for i in {1..5}; do
  echo "▶️ Run $i..."

  I_MPI_OFI_PROVIDER=tcp \
  I_MPI_DEBUG=5 \
  LD_PRELOAD=/home/j.markin/mpiP_build/lib/libmpiP.so \
  nohup mpirun \
    --host node01-eth1,node02-eth1,node03-eth1,node04-eth1,node05-eth1,node06-eth1,node07-eth1,node08-eth1,node09-eth1,node12-eth1,node13-eth1,node14-eth1,node15-eth1,node16-eth1,node17-eth1,node18-eth1,node19-eth1,node20-eth1,node21-eth1,node22-eth1,node23-eth1,node24-eth1,node25-eth1,node26-eth1,node27-eth1,node28-eth1,node29-eth1,node30-eth1,node31-eth1,node32-eth1 \
    -n 30 -ppn 1 ./train_resnet --DFF_Config=../resneteth.json \
    > ../eth_logs/28_workers_run_${i}.log 2>&1 

  sleep 2  # optional: wait between launches (adjust if needed)
done
