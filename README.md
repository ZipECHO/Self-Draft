# Self-Draft

## How to run

1. Install the dependencies
```bash
pip install -r requirements.txt
```

2. Download the dataset
```bash
cd data
bash download.sh
```

2. Run the script
```bash
# run gsm-100
CUDA_VISIBLE_DEVICES=1 python main.py --question-file data/gsm/test.jsonl --sample-number 100 --model-path meta-llama/Llama-2-7b-chat-hf
# run mt-bench
CUDA_VISIBLE_DEVICES=1 python main.py --question-file data/mt-bench/mt-bench.jsonl --sample-number -1 --model-path meta-llama/Llama-2-7b-chat-hf
# run humaneval
CUDA_VISIBLE_DEVICES=1 python main.py --question-file openai/openai_humaneval --sample-number -1 --model-path meta-llama/CodeLlama-7b-hf
# run mbpp
CUDA_VISIBLE_DEVICES=1 python main.py --question-file google-research-datasets/mbpp --sample-number 100 --model-path meta-llama/CodeLlama-7b-hf
```