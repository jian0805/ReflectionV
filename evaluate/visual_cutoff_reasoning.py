import json
from tqdm import tqdm

import torch
from vllm import LLM, SamplingParams
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

try:
    from .evaluate import process_dict, answer_verify, load_dataset, format_options
except ImportError:
    from evaluate import process_dict, answer_verify, load_dataset, format_options

split_dict = {
    "MMMU": ["validation"], "MMMU_Pro": ["standard (4 options)/test"],
    "M3CoT": ["test"], "MathVista": ["testmini"], 
    "MME": ["test"], "MathVerse": ["testmini"], "MathVision": ["test"]
}

def cutoff_image_reasoning(sample, prompt, response_text, step=50): 
    response_token_ids = processor.tokenizer.encode(response_text)

    input_list = []
    for n in range(0, len(response_token_ids), step):
        response_seg = processor.tokenizer.decode(response_token_ids[:n], skip_special_tokens=True)
        messages = [{"role": "user", "content": prompt}]
        messages.append({"role": "assistant", "content": response_seg})
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        input_list.append({"prompt": prompt[:-11]})

    outputs = model.generate(input_list, sampling_params=sampling_params, use_tqdm=False)

    acc_dict = {}
    for index, n in enumerate(range(0, len(response_token_ids), step)):
        response = outputs[index].outputs[0].text
        acc = answer_verify(response, sample["gt_answer"])

        acc_dict[n] = {"id": sample["id"], "acc": acc, "response": response, "cutoff_tokens": n}

    return acc_dict 
        


if __name__ == "__main__":
    # load model
    vlm_path = ""
    model = LLM(vlm_path, gpu_memory_utilization=0.95, trust_remote_code=True,
                tensor_parallel_size=1, limit_mm_per_prompt={"image": 7})
    sampling_params = SamplingParams(n=1, temperature=0.5, max_tokens=2048)
    vlm_template = """You FIRST think about the reasoning process as an internal monologue and then provide the final answer.
 The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \boxed{}. Qustion:\n"""

    min_pixels = 256 * 28 * 28
    max_pixels = 1024 * 28 * 28
    processor = AutoProcessor.from_pretrained(vlm_path, min_pixels=min_pixels, max_pixels=max_pixels)
    processor.tokenizer.padding_side = "left"

    # load dataset
    dataset_name = "MMMU"
    dataset_path = ""
    dataset_dict = load_dataset(dataset_name, dataset_path)
    dataset = dataset_dict[split_dict[dataset_name][0]]

    # 随机抽取 500 个样本
    out_file = ""

    # process dataset
    with open(out_file, "w") as f:
        for sample in tqdm(dataset):
            sample = process_dict[dataset_name](sample)

            if sample["options"] is not None:
                question = sample['question'] + "" + format_options(sample["options"])
            else:
                question = sample['question']
            
            prompt_str = vlm_template + question

            content_list = []
            for i, content in enumerate(prompt_str.split("<image>")):
                if i != 0:
                    content_list.append({"type": "image"})

                if content:
                    content_list.append({"type": "text", "text": content})

            messages = [{"role": "user", "content": content_list}]
            
            text = processor.apply_chat_template(messages, tokenize=False, 
                                                 add_generation_prompt=True)

            
            # 先生成 response, 再生成 attentions
            output_text = model.generate({"prompt": text, "multi_modal_data": {"image": sample["images"]},}, 
                                         sampling_params=sampling_params, use_tqdm=False)[0].outputs[0].text

            cutoff_dict = cutoff_image_reasoning(sample, prompt_str, output_text)

            # write sample to file
            sample["visual_cutoff_acc"] = cutoff_dict
            sample.pop("images")
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
