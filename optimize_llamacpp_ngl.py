import os
import subprocess
import sys

model_layers = 0
model_is_MoE = False

def try_llama_main(ngl, llama_main_path, model_path, desired_context, split_mode, main_gpu):
  print(f'trying -ngl {ngl}')
  cmd = f'{llama_main_path} -m {model_path} -c {desired_context} --split-mode {split_mode} --main-gpu {main_gpu} --flash-attn -t 4 -ngl {ngl} -n 1 --prompt "if this sentence is in stdout then our ngl was ok"'
  process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
  stdout, stderr = process.communicate()
  output = stdout.decode()
  global model_layers
  global model_is_MoE
  if model_layers == 0:
    errlines = stderr.decode().split('\n')
    for line in errlines:
      if 'n_layer' in line:
        start_ind = line.find('=') + 1
        model_layers = int(line[start_ind:].strip())
      if 'n_expert' in line:
        start_ind = line.find('=') + 1
        n_expert = int(line[start_ind:].strip())
        model_is_MoE = (n_expert > 1)
  return 'if this sentence is in stdout then our ngl was ok' in output


def binary_search_most_ngl_possible(low, high, llama_main_path, model_path, desired_context, split_mode, main_gpu):
  while low < high:
    mid = (low + high) // 2
    if try_llama_main(mid, llama_main_path, model_path, desired_context, split_mode, main_gpu):
      low = mid + 1
    else:
      high = mid
  return low - 1


def nvidia_smi_query_quantity(to_query):
  cmd = f'nvidia-smi --query-gpu={to_query} --format=csv,noheader'
  process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
  stdout, stderr = process.communicate()
  output_lines = stdout.decode().split()
  output_ints = [int(s) for s in output_lines if s.isdigit()]
  return output_ints


if len(sys.argv) < 4 or not sys.argv[1].isdigit():
  print(f'Usage: python3 {sys.argv[0]} desired_context /path/to/llama.cpp/main /path/to/model.gguf [ngl_guess]')
  print(f'\ne.g.: python3 {sys.argv[0]} 8192 ~/llama.cpp/main ~/models/model.gguf 40')
  exit(1)
desired_context = int(sys.argv[1])
llama_main_path = sys.argv[2]
model_path = sys.argv[3]
ngl_guess = 30
if len(sys.argv) >= 5:
  ngl_guess = int(sys.argv[4])
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
vram_per_gpu = nvidia_smi_query_quantity('memory.total')

nvidia_smi_process = subprocess.Popen('nvidia-smi --query-gpu=name --format=csv,noheader', stdout=subprocess.PIPE, shell=True)
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

# fill model_layers and model_is_MoE with an initial guessed trial
ngl_guess_works = try_llama_main(ngl_guess, llama_main_path, model_path, desired_context, 'layer', 0)
ngl_low = ngl_guess+1 if ngl_guess_works else 0
ngl_high = model_layers+1 if ngl_guess_works else ngl_guess

# TODO should use the CUDA0etc buffer size output of the initial guess to inform a second guess

print(f'model has {model_layers} layers, binary search bounds: low {ngl_low} high {ngl_high}')

max_ngl_possible = binary_search_most_ngl_possible(ngl_low, ngl_high, llama_main_path, model_path, desired_context, 'layer', main_gpu_ind)
print(f'done! best feasible -ngl is {max_ngl_possible}')

# TODO wiggle around with --tensor-split to find an extra layer or two

if max_ngl_possible == model_layers and not model_is_MoE:
  print(f'\nRun llama.cpp with these arguments:\n-m {model_path} -c {desired_context} --split-mode row --main-gpu {main_gpu_ind} --flash-attn -t 4 -ngl {max_ngl_possible}')
else:
  print(f'\nRun llama.cpp with these arguments:\n-m {model_path} -c {desired_context} --split-mode layer --main-gpu {main_gpu_ind} --flash-attn -t 4 -ngl {max_ngl_possible}')

# TODO do we need to confirm --split-mode row can run?
