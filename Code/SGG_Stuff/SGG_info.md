replace files for IETrans-SGG.pytorch

IETrans-SGG.pytorch/maskrcnn_benchmark/csrc/cuda



updated instructions:

make venv with python=3.8

for apex, set head with (git checkout 2386a912164b0c5cfcd8be7a2b890fbac5607c82)

use cuda 11.7 (add to path with export PATH=/usr/local/cuda-11.7/bin:$PATH)

replace files that have THC to AT


