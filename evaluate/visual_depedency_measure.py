import json, os
import argparse
from tqdm import tqdm

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

try:
    from .evaluate import process_dict, answer_verify, load_dataset, format_options
except ImportError:
    from evaluate import process_dict, answer_verify, load_dataset, format_options


def hellinger_distance(P: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
    """计算两个概率矩阵的逐行Hellinger距离"""
    sqrt_P = torch.sqrt(P)
    sqrt_Q = torch.sqrt(Q)
    return (1 / torch.sqrt(torch.tensor(2.0))) * torch.norm(sqrt_P - sqrt_Q, p=2, dim=-1)

def visual_depedency_measure(model, processor, prompt, response_text, step=5):
    """
    Calculate the visual dependency measure.
    :param messages: list of messages
    :param text_response: text response
    :param step: step size
    :return: visual dependency measure
    """
    # w & w/o image input
    processor.padding_side = "left"

    content_list = []
    for i, content in enumerate(prompt_str.split("<image>")):
        if i != 0:
            content_list.append({"type": "image"})

        if content:
            content_list.append({"type": "text", "text": content})

    messages_with_image = [{"role": "user", "content": content_list}]
    messages_without_image = [{"role": "user", "content": prompt_str}]

    response_token_ids = processor.tokenizer.encode(response_text)[:500]

    vdm_dict = {}
    for n in range(0, len(response_token_ids), step):
        response_seg = processor.tokenizer.decode(response_token_ids[:n], skip_special_tokens=True)

        messages_with_image.append({"role": "assistant", "content": response_seg})
        messages_without_image.append({"role": "assistant", "content": response_seg})

        prompt_list = processor.apply_chat_template([messages_with_image, messages_without_image], 
                                                    tokenize=False, add_generation_prompt=False)
        
        inputs = processor(text=prompt_list, images=sample["images"], padding=True, 
                           return_tensors="pt",).to(model.device)
        
        generate_dict = model.generate(**inputs, min_new_tokens=step, max_new_tokens=step, 
                                       output_logits=True, return_dict_in_generate=True)
        
        logit = torch.stack(generate_dict.logits)
        probabilities1 = torch.nn.functional.softmax(logit[:, 0, :], dim=-1)
        probabilities2 = torch.nn.functional.softmax(logit[:, 1, :], dim=-1)
        distance = hellinger_distance(probabilities1, probabilities2)
        
        vdm_dict[n] = float(distance.mean().item())
    
    return vdm_dict

def id_filter(sample, filter_list):
    """
    Filter the id list based on the filter list.
    :param id_list: list of ids
    :param filter_list: list of ids to be filtered
    :return: filtered id list
    """
    return sample["id"] not in filter_list
        
def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess data for multimodal reasoning")
    parser.add_argument("--dataset_name", type=str, default="MMMU", help="dataset name")
    parser.add_argument("--model_path", type=str, default="", help="model path")
    parser.add_argument("--data_dir", type=str, default="", help="data dir")
    parser.add_argument("--output_file", type=str, default="", help="split name")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    # load model
    vlm_path = args.model_path
    vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(vlm_path, torch_dtype=torch.bfloat16, device_map="auto", 
                                                             attn_implementation="flash_attention_2")
    vlm_template = """You FIRST think about the reasoning process as an internal monologue and then provide the final answer.
 The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \boxed{}. Qustion:\n"""

    min_pixels = 256 * 28 * 28
    max_pixels = 256 * 28 * 28
    processor = AutoProcessor.from_pretrained(vlm_path, min_pixels=min_pixels, max_pixels=max_pixels)
    processor.tokenizer.padding_side = "left"

    # load dataset
    dataset_name = args.dataset_name
    dataset_path = args.data_dir
    dataset_dict = load_dataset(dataset_name, dataset_path)
    dataset = dataset_dict.get(list(dataset_dict.keys())[0])
    dataset = dataset.shuffle(seed=715).select(range(min(500, len(dataset))))

    # 随机抽取 100 个样本
    out_file = args.output_file
    filter_list = []
    # 如果文件存在，则读取文件中的 id 列表
    if os.path.exists(out_file):
        with open(out_file, "r") as f:
            exist_data = [json.loads(line) for line in f.readlines()]
            filter_list = [sample["id"] for sample in exist_data]
    dataset = dataset.filter(lambda x: id_filter(x, filter_list), num_proc=8)

    # 数据生成
    out_file = args.output_file

    # process dataset
    with open(out_file, "a") as f:
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

            inputs = processor(text=[text], images=sample["images"], padding=True, 
                               return_tensors="pt",).to(vlm.device)
            
            # 先生成 response, 再生成 attentions
            generate_dict = vlm.generate(**inputs, max_new_tokens=512, do_sample=True, 
                                        temperature=0.5, output_logits=True, 
                                        return_dict_in_generate=True)

            # attention 矩阵
            input_len = inputs["input_ids"].shape[1]
            output_text = processor.batch_decode(
                generate_dict.sequences[:, input_len:], 
                skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            
            vdm_dict = visual_depedency_measure(
                vlm, processor, prompt_str, output_text, step=5
            )

            # write sample to file
            sample["vdm"] = vdm_dict
            sample["response"] = output_text
            sample.pop("images")
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

        

        

        

        
        