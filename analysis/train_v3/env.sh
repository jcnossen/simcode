# source env.sh <gpu>: simcode-py311, the frozen SIMCODE snapshot + the torch-psf snapshot of the v3 PSF fit, one GPU
source ~/miniconda3/bin/activate simcode-py311
export TV3=/home/jc/simcode2/out/train_v3 SRC=/home/jc/simcode2/out/train_v3/src_e8fc62f4 TPSF=/home/jc/simcode2/out/psf_v3/tpsf_64871e5
export PYTHONPATH=$SRC:$TPSF TORCH_DISABLE_NATIVE_JIT=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$1 TQDM_DISABLE=1
