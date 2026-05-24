# Third-party Dependencies

## SpInfer

Sparse matrix multiplication library for accelerating FFN outlier refinement.

### Repository Information
- **Upstream**: https://github.com/xxyux/SpInfer.git
- **Our Fork**: https://github.com/zzzhh123-create/25Eurosys-SpInfer.git
- **Branch**: `lightx2v-integration`

### Setup Instructions
cd third_party
git clone https://github.com/zzzhh123-create/25Eurosys-SpInfer.git
cd SpInfer
git checkout lightx2v-integration
git submodule update --init --recursive
export SpInfer_HOME=$(pwd)
source Init_SpInfer.sh
cd third_party/FasterTransformer && git apply ../ft_spinfer.patch
cd ../sputnik && git apply ../sputnik.patch
cd ../../build
make -j
ls -lh libSpMM_API.so

### Current Version
Last updated: 2026-05-24

Commit: [run 'cd SpInfer && git rev-parse HEAD' to get current commit]

### Modifications
Document any custom modifications made for LightX2V integration here.
