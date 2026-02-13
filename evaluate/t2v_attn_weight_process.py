import json
import argparse
from tqdm import tqdm

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info



from data_process.evaluate import process_dict, answer_verify, load_dataset, format_options


def calculate_text2visual_attn_weights(weights, input_ids, processor, input_len, num_heads=28):
    """
    Calculate the attention weights for each head.
    :param attention_weights: (batch_size, num_heads, num_tokens, num_tokens)
    :param num_heads: number of attention heads
    :return: list of attention weights for each head
    """
    _, visual_token_indices = (input_ids == processor.tokenizer.convert_tokens_to_ids('<|image_pad|>')).nonzero(as_tuple=True)
    respones_token_indices = torch.arange(input_len, input_ids.shape[1]).to(visual_token_indices.device)

    t2v_attn_w_dict = {}
    for n, layer_weight in enumerate(weights):
        layer_weight = layer_weight[0].sum(dim=0) # (num_tokens, num_tokens)
        t2v_weight = layer_weight[respones_token_indices,:][:, visual_token_indices]
        t2v_attn_w_list = []
        for k in range(len(respones_token_indices)):
            t2v_weight = layer_weight[respones_token_indices[k], :][visual_token_indices]
            t2v_weight = float(t2v_weight.sum() / (t2v_weight > 0).sum().item())
            t2v_attn_w_list.append(t2v_weight)

        t2v_attn_w_dict[n] = t2v_attn_w_list

    return t2v_attn_w_dict

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
    vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(vlm_path, torch_dtype=torch.bfloat16, device_map="auto")
    vlm_template = """You FIRST think about the reasoning process as an internal monologue and then provide the final answer.
 The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \boxed{}. Qustion:\n"""

    min_pixels = 256*28*28
    max_pixels = 512*28*28
    processor = AutoProcessor.from_pretrained(vlm_path, min_pixels=min_pixels, max_pixels=max_pixels)

    # load dataset
    dataset_name = args.dataset_name
    dataset_path = args.data_dir
    dataset_dict = load_dataset(dataset_name, dataset_path)
    dataset = dataset_dict.get(list(dataset_dict.keys())[0])

    out_file = args.output_file

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

            inputs = processor(text=[text], images=sample["images"], padding=True, 
                            return_tensors="pt",).to(vlm.device)
            
            # 先生成 response, 再生成 attentions
            generate_dict = vlm.generate(**inputs, max_new_tokens=800, do_sample=True, 
                                        temperature=0.5, output_logits=True, 
                                        return_dict_in_generate=True)

            # attention 矩阵
            input_len = inputs["input_ids"].shape[1]
            output_text = processor.batch_decode(
                generate_dict.sequences[:, input_len:], 
                skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

            messages.append({"role": "assistant", "content": output_text})
            text = processor.apply_chat_template(messages, tokenize=False, 
                                                add_generation_prompt=False)
            inputs = processor(text=[text], images=sample["images"], padding=True, 
                            return_tensors="pt",).to(vlm.device)

            
            generate_dict = vlm(**inputs, output_attentions=True, return_dict=True) # weight 28 * (1, num_head, num_token, num_token)
                                                                                    # logit_dict (1, num_token, vocab_size)
            
            t2v_attn_w_dict = calculate_text2visual_attn_weights(
                generate_dict.attentions, inputs["input_ids"], 
                processor, input_len)

            # write sample to file
            sample["t2v_weights"] = t2v_attn_w_dict
            sample["response"] = output_text
            sample.pop("images")
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

        

        

        