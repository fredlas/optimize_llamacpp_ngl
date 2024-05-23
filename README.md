For the time being, llama.cpp can't automatically determine its `-ngl` (number of layers to offload to GPU) param, and users have to figure it out by trial and error. This script automates that process, giving you the optimal command line flags for llama.cpp to run a given model on your machine.

There should be no dependencies - just a working llama.cpp main compiled for CUDA. Run as:

`python3 optimize_llamacpp_ngl.py desired_context /path/to/llama.cpp/main /path/to/model.gguf [ngl_guess]`

e.g.

`python3 optimize_llamacpp_ngl.py 8192 /path/to/llama.cpp/main /path/to/model.gguf 40`

This script is written for nvidia cards. If you have AMD or Intel cards and would like to add support, I welcome PRs. I have tested it on a multiple nvidia GPU machine, but not with heterogeneous models - I do think that should work fine, though.

## TODO
* Should be able to use the report `--split-mode layer` gives on CUDA buffer sizes to make better guesses than blind binary search
* Tweaking `--tensor-split` can sometimes fit an extra layer or two
* Need to confirm whether `--split-mode layer` fitting all layers always guarantees `--split-mode row` is possible
