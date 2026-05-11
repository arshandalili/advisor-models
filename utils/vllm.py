import subprocess
import sys
import time
import urllib.request


def start_vllm_server(
    model_to_serve_name: str,
    served_model_name: str = "advisor_model",
    tensor_parallel_size: int = 4,
    max_model_len: int = 32768,
    port: int = 8000,
    gpu_memory_utilization: float = 0.9,
) -> subprocess.Popen:
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model_to_serve_name,
        "--served-model-name",
        served_model_name,
        "--tensor-parallel-size",
        str(tensor_parallel_size),
        "--max-model-len",
        str(max_model_len),
        "--port",
        str(port),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--disable-log-requests",
    ]
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    # Wait for server to be ready (up to 10 minutes)
    url = f"http://127.0.0.1:{port}/health"
    for _ in range(120):
        time.sleep(5)
        try:
            with urllib.request.urlopen(url, timeout=3):
                print(f"vLLM server ready at port {port}")
                return process
        except Exception:
            pass
        if process.poll() is not None:
            output = process.stdout.read().decode() if process.stdout else ""
            raise RuntimeError(f"vLLM server exited early:\n{output}")

    raise RuntimeError("vLLM server did not become ready within 10 minutes")
