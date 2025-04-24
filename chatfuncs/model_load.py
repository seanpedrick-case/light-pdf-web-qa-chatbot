import torch

# Currently set gpu_layers to 0 even with cuda due to persistent bugs in implementation with cuda
if torch.cuda.is_available():
    torch_device = "cuda"
    gpu_layers = 100
else: 
    torch_device =  "cpu"
    gpu_layers = 0

print("Running on device:", torch_device)
threads = 8 #torch.get_num_threads()
print("CPU threads:", threads)

# Qwen 2 0.5B (small, fast) Model parameters
temperature: float = 0.1
top_k: int = 3
top_p: float = 1
repetition_penalty: float = 1.15
flan_alpaca_repetition_penalty: float = 1.3
last_n_tokens: int = 64
max_new_tokens: int = 1024
seed: int = 42
reset: bool = False
stream: bool = True
threads: int = threads
batch_size:int = 256
context_length:int = 2048
sample = True

# Bedrock parameters
max_tokens = 4096


class CtransInitConfig_gpu:
    def __init__(self,
                 last_n_tokens=last_n_tokens,
                 seed=seed,
                 n_threads=threads,
                 n_batch=batch_size,
                 n_ctx=max_tokens,
                 n_gpu_layers=gpu_layers):

        self.last_n_tokens = last_n_tokens
        self.seed = seed
        self.n_threads = n_threads
        self.n_batch = n_batch
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        # self.stop: list[str] = field(default_factory=lambda: [stop_string])

    def update_gpu(self, new_value):
        self.n_gpu_layers = new_value

class CtransInitConfig_cpu(CtransInitConfig_gpu):
    def __init__(self):
        super().__init__()
        self.n_gpu_layers = 0

gpu_config = CtransInitConfig_gpu()
cpu_config = CtransInitConfig_cpu()


class CtransGenGenerationConfig:
    def __init__(self, temperature=temperature,
                 top_k=top_k,
                 top_p=top_p,
                 repeat_penalty=repetition_penalty,
                 seed=seed,
                 stream=stream,
                 max_tokens=max_new_tokens
                 ):
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repeat_penalty = repeat_penalty
        self.seed = seed
        self.max_tokens=max_tokens
        self.stream = stream

    def update_temp(self, new_value):
        self.temperature = new_value