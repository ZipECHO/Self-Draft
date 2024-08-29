mkdir gsm
wget -O gsm/test.jsonl https://github.com/openai/grade-school-math/blob/master/grade_school_math/data/test.jsonl

mkdir mt-bench
wget https://raw.githubusercontent.com/lm-sys/FastChat/v0.2.31/fastchat/llm_judge/data/mt_bench/question.jsonl -O mt-bench/mtbench.jsonl

