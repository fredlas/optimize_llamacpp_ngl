import os
import struct
import subprocess
import sys


def try_llama_main(ngl, llama_main_path, model_path, desired_context,
                   split_mode, main_gpu, tensor_split):
  print(f'trying -ngl {ngl}{tensor_split}')
  cmd = f'{llama_main_path} -m {model_path} -c {desired_context} --split-mode {split_mode}{tensor_split} --main-gpu {main_gpu} --flash-attn -t 4 -ngl {ngl} -n 1 --prompt "if this sentence is in stdout then our ngl was ok"'
  process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
  stdout, stderr = process.communicate()
  return 'if this sentence is in stdout then our ngl was ok' in stdout.decode()


def best_tensor_split(vram_per_gpu, MiB_per_layer, ngl):
  vram_left = vram_per_gpu[:]
  remaining_layers = ngl
  # HACK these overhead estimates are very rough guesses, optimized for 8x22B.
  # too conservative for smaller models and GPUs? TODO use CUDA0etc buffer size from initial guess
  for i in range(0, len(vram_left)):
    vram_left[i] -= 500
  vram_left[main_gpu_ind] -= 1000
  layers = [0] * len(vram_left)
  while remaining_layers > 0:
    target_ind = vram_left.index(max(vram_left))
    layers[target_ind] += 1
    remaining_layers -= 1
    vram_left[target_ind] -= MiB_per_layer
  # if we think it easily fits, then just let llama.cpp do the default
  if all(x > 2 * MiB_per_layer for x in vram_left):
    return ''
  return ' --tensor-split ' + ','.join(map(str, layers))


def binary_search_most_ngl_possible(low, high, llama_main_path, model_path,
                                    desired_context, main_gpu, vram_per_gpu, MiB_per_layer):
  works = {}
  ts_strings = {}
  while low < high:
    mid = (low + high) // 2
    ts_strings[mid] = best_tensor_split(vram_per_gpu, MiB_per_layer, mid)
    works[mid] = try_llama_main(mid, llama_main_path, model_path, desired_context,
                                'layer', main_gpu, ts_strings[mid])
    if works[mid]:
      low = mid + 1
    else:
      high = mid
  # if our answer is below the range we evaluated, indicate not found
  return low - 1, works.get(low - 1, False), ts_strings.get(low - 1, '')


def nvidia_smi_query_quantity(to_query):
  cmd = f'nvidia-smi --query-gpu={to_query} --format=csv,noheader'
  process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
  stdout, stderr = process.communicate()
  output_lines = stdout.decode().split()
  output_ints = [int(s) for s in output_lines if s.isdigit()]
  return output_ints


if len(sys.argv) < 4 or not sys.argv[1].isdigit():
  print(f'Usage: python3 {sys.argv[0]} desired_context /path/to/llama.cpp/main /path/to/model.gguf')
  print(f'\ne.g.: python3 {sys.argv[0]} 8192 ~/llama.cpp/main ~/models/model.gguf')
  exit(1)
desired_context = int(sys.argv[1])
llama_main_path = sys.argv[2]
model_path = sys.argv[3]
if not os.path.isfile(llama_main_path):
  print(f'file does not exist: {llama_main_path}')
  exit(1)
main_help_process = subprocess.Popen(f'{llama_main_path} --help', stdout=subprocess.PIPE, shell=True)
main_help_stdout, main_help_stderr = main_help_process.communicate()
mh = main_help_stdout.decode()
if not ('--split-mode' in mh and '--reverse-prompt' in mh):
  print(f'this file does not appear to be a working llama.cpp "main" executable: {llama_main_path}')
  exit(1)
if not os.path.isfile(model_path):
  print(f'file does not exist: {model_path}')
  exit(1)

pcie_lanes_per_gpu = nvidia_smi_query_quantity('pcie.link.width.current')
vram_per_gpu = nvidia_smi_query_quantity('memory.total') # in MiB

nvidia_smi_process = subprocess.Popen('nvidia-smi --query-gpu=name --format=csv,noheader',
                                      stdout=subprocess.PIPE, shell=True)
nv_stdout, nv_stderr = nvidia_smi_process.communicate()
gpu_names = [s for s in nv_stdout.decode().split('\n') if s != '']
vram_bandwidth_table = { 'NVIDIA GeForce RTX 4090': 1008,
                         'NVIDIA GeForce RTX 4080': 737,
                         'NVIDIA GeForce RTX 4070 Ti': 672,
                         'NVIDIA GeForce RTX 4070': 504,
                         'NVIDIA GeForce RTX 4060 Ti': 288,
                         'NVIDIA GeForce RTX 4060': 272,
                         'NVIDIA GeForce RTX 3090': 936,
                         'NVIDIA GeForce RTX 3080 Ti': 912,
                         'NVIDIA GeForce RTX 3080': 912,
                         'NVIDIA GeForce RTX 3070 Ti': 608,
                         'NVIDIA GeForce RTX 3070': 448,
                         'NVIDIA GeForce RTX 3060 Ti': 448,
                         'NVIDIA GeForce RTX 3060': 360,
                         'NVIDIA GeForce RTX 2080 Ti': 616,
                         'NVIDIA GeForce RTX 2080': 448,
                         'NVIDIA GeForce RTX A6000': 768,
                         'NVIDIA GeForce RTX A4000': 448,
                         'Tesla P40': 346,
                         'Tesla P100': 732 }
gpu_bandwidths = [vram_bandwidth_table.get(name, 222) for name in gpu_names]
for i in range(0, len(gpu_bandwidths)):
  if gpu_bandwidths[i] == 222:
    print(f'WARNING! Unrecognized GPU name {gpu_names[i]}. Look up your GPU VRAM bandwidth and put it in the vram_bandwidth_table map in this script (and please open a pull request!), or you might get a suboptimal value for --main-gpu if you have heterogeneous GPUs.')

# Pick main GPU: pick highest VRAM bandwidth GPU, with PCIe lanes as tiebreaker
best_bw_inds = [i for i, x in enumerate(gpu_bandwidths) if x == max(gpu_bandwidths)]
main_gpu_ind = best_bw_inds[0]
best_pcie = pcie_lanes_per_gpu[main_gpu_ind]
for gpu_ind in best_bw_inds:
  if pcie_lanes_per_gpu[gpu_ind] > best_pcie:
    main_gpu_ind = gpu_ind


model_layers = 0
model_is_MoE = False
with open(model_path, 'rb') as model_file:
  data = model_file.read(4096)
  ind_block_count = data.find(b'.block_count')
  if ind_block_count == -1:
    print('\nfailed to determine model n_layer from model file block_count')
    exit(1)
  model_layers = struct.unpack('<i', data[ind_block_count+16:ind_block_count+20])[0]
  ind_expert_count = data.find(b'.expert_count')
  if ind_expert_count > -1:
    expert_count = struct.unpack('<i', data[ind_expert_count+17:ind_expert_count+21])[0]
    model_is_MoE = (expert_count > 1)

# initial guess, hopefully we can mostly skip binary search
model_size_MiB = 1 + os.path.getsize(model_path) // (1024 * 1024)
MiB_per_layer = 1 + model_size_MiB // model_layers
print(f'model has {model_layers} layers, file size {model_size_MiB}MiB, model_is_MoE: {model_is_MoE}')
# we can be pretty sure that if x * mem_per_layer > VRAM, we ain't fittin x layers in VRAM
ngl_upperlimit = min(model_layers, sum(vram_per_gpu) // MiB_per_layer)
# HACK very crude estimates based on observation
total_overhead_guess = 11000 if model_size_MiB > 60000 else (5000 if model_size_MiB > 30000 else 2000)
ngl_lowerlimit = min(model_layers, (sum(vram_per_gpu) - total_overhead_guess) // MiB_per_layer)

print(f'binary search bounds: low {ngl_lowerlimit} high {ngl_upperlimit+1}')
max_ngl_possible, found, tensor_split = binary_search_most_ngl_possible(
  ngl_lowerlimit, ngl_upperlimit+1, llama_main_path,
  model_path, desired_context, main_gpu_ind, vram_per_gpu, MiB_per_layer)
if not found:
  print('good ngl not found within guessed binary search area, trying down to 0')
  max_ngl_possible, found, tensor_split = binary_search_most_ngl_possible(
    0, ngl_lowerlimit, llama_main_path, model_path, desired_context, main_gpu_ind,
    vram_per_gpu, MiB_per_layer)
  if not found:
    print("Not even -ngl 0 works. It looks like you can't run this model. Sorry.")
    exit(1)

split_mode = 'layer' if model_is_MoE or max_ngl_possible < model_layers else 'row'
# TODO do we need to confirm --split-mode row can run?
print(f'\nRun llama.cpp with these arguments:\n-m {model_path} -c {desired_context} --split-mode {split_mode}{tensor_split} --main-gpu {main_gpu_ind} --flash-attn -t 4 -ngl {max_ngl_possible}')
