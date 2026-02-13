import argparse
import os, json, string, re
from tqdm import tqdm
from math_verify import LatexExtractionConfig, parse, verify
from latex2sympy2_extended import NormalizationConfig
from PIL import Image

# from vllm import LLM, SamplingParams
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from datasets import concatenate_datasets, load_from_disk
from qwen_vl_utils import process_vision_info


def format_options(options):

    # 用于生成字母编号（从 A 开始）
    alphabet = string.ascii_uppercase
    

    # str2list
    if isinstance(options, str):
        options = eval(options)

    # 创建选项字符串
    formatted_options = []
    if len(options) == 0:
        return ""
    
    for idx, option in enumerate(options):
        # 使用字母编号和选项内容构建格式化字符串
        formatted_options.append(f"({alphabet[idx]}) {option}")
    
    # 将所有选项连接成一个字符串，用逗号和空格分隔
    return "\n".join(formatted_options)

def scale_min_size(image, min_size=28, resample=Image.LANCZOS):
    """
    将图像缩放至至少满足 min_size 的最小边尺寸
    """
    # 获取原始尺寸
    width, height = image.size
    
    # 判断是否需要缩放
    if width >= min_size and height >= min_size:
        return image
    
    # 计算缩放比例[3,5](@ref)
    ratio = max(min_size / width, min_size / height)
    new_width = int(round(width * ratio))
    new_height = int(round(height * ratio))
    
    # 执行缩放[1,7](@ref)
    return image.resize((new_width, new_height), resample=resample)

split_dict = {
    "MMMU": ["validation"], "MMMU_Pro": ["standard (4 options)/test"],
    "M3CoT": ["test"], "MathVista": ["testmini"], 
    "MME": ["test"], "MathVerse": ["testmini"], "MathVision": ["testmini"]
}

CAT_SHORT2LONG = {
    'acc': 'Accounting',
    'agri': 'Agriculture',
    'arch': 'Architecture_and_Engineering',
    'art': 'Art',
    'art_theory': 'Art_Theory',
    'bas_med': 'Basic_Medical_Science',
    'bio': 'Biology',
    'chem': 'Chemistry',
    'cli_med': 'Clinical_Medicine',
    'cs': 'Computer_Science',
    'design': 'Design',
    'diag_med': 'Diagnostics_and_Laboratory_Medicine',
    'econ': 'Economics',
    'elec': 'Electronics',
    'ep': 'Energy_and_Power',
    'fin': 'Finance',
    'geo': 'Geography',
    'his': 'History',
    'liter': 'Literature',
    'manage': 'Manage',
    'mark': 'Marketing',
    'mate': 'Materials',
    'math': 'Math',
    'mech': 'Mechanical_Engineering',
    'music': 'Music',
    'phar': 'Pharmacy',
    'phys': 'Physics',
    'psy': 'Psychology',
    'pub_health': 'Public_Health',
    'socio': 'Sociology'
}


def process_mmmu(sample):
    images = []
    image_columns = ["image_" + str(n) for n in range(1, 8)]
    for col in image_columns:
        if sample[col] is not None:
            images.append(sample[col])
    
    image_tokens = ["<image " + str(n) + ">" for n in range(1, 8)]
    for img_token in image_tokens:
        sample['question'] = sample['question'].replace(img_token, img_token + " <image>", 1)
    
    # 如果 sample['question'] 中的 "<image>" 的数量小于 images 的数量， 在末尾添加 "<image>\n"
    if "<image>" not in sample['question']:
        sample['question'] = "<image>\n" + sample['question']
    
    if sample['question'].count("<image>") < len(images):
        sample['question'] += "<image>\n" * (len(images) - sample['question'].count("<image>"))

    if 'question_type' not in sample:
        sample['question_type'] = 'multi_choice'
    sample = {'id': sample['id'], 'question': sample['question'],
                'options': sample['options'], 'gt_answer': sample['answer'], 
                'images': images, 'question_type': sample['question_type']}
    return sample

def process_m3cot(sample):
    if "<image>" not in sample['question']:
        sample['question'] = "<image>\n" + sample['question']
    sample = {'id': sample['id'], 'question': sample['question'],
              'options': sample['choices'], 'gt_answer': sample['answer'], 
              'images': [sample['image']], 'question_type': sample['category']}
    return sample

def process_mme(sample):
    if "<image>" not in sample['question']:
        sample['question'] = "<image>\n" + sample['question']
    sample = {'id': sample['question_id'], 'question': sample['question'],
                'options': None, 'gt_answer': sample['answer'], 
                'images': [sample['image']], 'question_type': sample['category']}
    return sample

def process_mathvista(sample):
    if "<image>" not in sample['question']:
        sample['question'] = "<image>\n" + sample['question']
    
    if sample['question_type'] == "multi_choice":
        for n, option in enumerate(sample['choices']):
            if option == sample['answer']:
                sample['answer'] = string.ascii_uppercase[n]

    sample['decoded_image'] = scale_min_size(sample['decoded_image'])
    sample = {'id': sample['pid'], 'question': sample['question'] , 'options': sample['choices'],
                'gt_answer': sample['answer'], 'images': [sample['decoded_image']],
                'question_type': sample['question_type']}
    return sample

def process_mathverse(sample):
    if "<image>" not in sample['question']:
        sample['question'] = "<image>\n" + sample['question']
    # question = re.sub(r'^Hint: ', '', sample['query'])
    sample = {'id': sample['sample_index'], 'question': sample["question"] , 'options': None,
                'gt_answer': sample['answer'], 'images': [sample['image']],
                'question_type': sample['question_type']}
    return sample


def process_mathvision(sample):
    if "<image1>" in sample['question']:
        sample['question'] = sample['question'].replace("<image1>", "<image>\n", 1)
    else:
        sample['question'] = "<image>\n" + sample['question']
    # question = re.sub(r'^Hint: ', '', sample['query'])
    sample = {'id': sample['id'], 'question': sample["question"] , 'options': sample['options'],
                'gt_answer': sample['answer'], 'images': [sample['decoded_image']],
                'question_type': "multi_choice"}
    return sample

process_dict = {
    "MMMU": process_mmmu, "MMMU_Pro": process_mmmu, "M3CoT": process_m3cot,
    "MathVista": process_mathvista, "MME": process_mme,
    "MathVerse": process_mathverse, "MathVision": process_mathvision
}


def load_dataset(dataset_name, data_path):
    # dataset
    splits = split_dict[dataset_name]
    dataset_dict = {}
    for split in splits:
        if dataset_name == "MMMU":

            sub_dataset_list = []
            for subject in CAT_SHORT2LONG.values():
                sub_dataset = load_from_disk(f"{data_path}/{subject}/{split}")
                sub_dataset_list.append(sub_dataset)
            
            dataset = concatenate_datasets(sub_dataset_list)   
            
        elif dataset_name == "MMMU_Pro":
            dataset = load_from_disk(f"{data_path}/{split}")
        
        else:
            # 是否存在 default split
            if os.path.exists(f"{data_path}/default/{split}"):
                dataset = load_from_disk(f"{data_path}/default/{split}")
            else:
                dataset = load_from_disk(f"{data_path}/{split}")
        
        dataset_dict[split] = dataset

    return dataset_dict

def answer_verify(text, gold_answer):  # 默认选项数为 4
    try:
        answer = text.split("</think>")[1]
        # 正则匹配提取 "\boxed{}" 中的内容
        if "\\boxed{" not in answer and "boxed{" in answer:
            answer = answer.replace("boxed{", "\\boxed{", 1)
        

    except:
        return False
    
    parsed_answer = parse(
        answer, extraction_config=[
            LatexExtractionConfig(
                normalization_config=NormalizationConfig(
                    nits=False,
                    malformed_operators=False,
                    basic_latex=True,
                    equations=True,
                    boxed="all",
                    units=True,
                ),
                # Ensures that boxed is tried first
                boxed_match_priority=0,
                try_extract_without_anchor=False,
            )
        ],
        extraction_mode="first_match",
    )

    gold = parse("\\boxed{" +  gold_answer + "}", extraction_config=[
            LatexExtractionConfig(
                normalization_config=NormalizationConfig(
                    nits=False,
                    malformed_operators=False,
                    basic_latex=True,
                    equations=True,
                    boxed="all",
                    units=True,
                ),
                # Ensures that boxed is tried first
                boxed_match_priority=0,
                try_extract_without_anchor=False,
            )
        ],
        extraction_mode="first_match",
    )

    correct = verify(gold, parsed_answer)

    return correct


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess data for multimodal reasoning")
    parser.add_argument("--dataset_name", type=str, default="MMMU", help="dataset name")
    parser.add_argument("--prompt_template", type=str, default="", help="prompt template")
    parser.add_argument("--model_path", type=str, default="", help="model path")
    parser.add_argument("--data_dir", type=str, default="", help="data dir")
    parser.add_argument("--output_file", type=str, default="", help="split name")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # load dataset
    dataset_name = args.dataset_name
    data_dir = args.data_dir
    dataset_dict = load_dataset(dataset_name, data_dir)
    dataset = dataset_dict.get(list(dataset_dict.keys())[0])

    # dataset = dataset.shuffle(seed=42).select(range(10))

    # processer
    model = LLM(args.model_path, gpu_memory_utilization=0.95, trust_remote_code=True,
                tensor_parallel_size=1,  limit_mm_per_prompt={"image": 7})
    sampling_params = SamplingParams(n=1, temperature=0.5, max_tokens=2048)
    processor = AutoProcessor.from_pretrained(args.model_path)

    # process multimodal input
    sys_prompt = "You are a helpful assistant."
    prompt_temp = args.prompt_template
    if prompt_temp == "":
        prompt_temp = """You FIRST think about the reasoning process as an internal monologue and then provide the final answer.
    The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \\boxed{}. Qustion:\n"""

    mm_inputs = []
    for sample in tqdm(dataset):
        sample = process_dict[dataset_name](sample)
        if sample["options"] is not None:
            question = sample['question'] + "" + format_options(sample["options"])
        else:
            question = sample['question']
    
        question = prompt_temp + question

        content_list = []
        for i, content in enumerate(question.split("<image>")):
            if i != 0:
                content_list.append({"type": "image"})

            if content:
                content_list.append({"type": "text", "text": content})

        messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": content_list}]

        prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

        try:
            input_data = processor(sample["images"], [prompt], add_special_tokens=True, return_tensors="pt")
        except:
            continue

        mm_inputs.append({"prompt": prompt, "multi_modal_data": {"image": sample["images"]},})

    outputs = model.generate(mm_inputs, sampling_params=sampling_params)
    
    acc_list = []
    for i, output in enumerate(outputs):
        response = output.outputs[0].text
        sample = process_dict[dataset_name](dataset[i])
        acc = answer_verify(response, sample["gt_answer"])
        print(response)
        acc_list.append({"id": sample["id"],"acc": acc, "response": response})
    
    # 计算准确率
    acc_count = 0
    for acc in acc_list:
        if acc["acc"]:
            acc_count += 1
    accuracy = acc_count / len(acc_list)
    print(f"Accuracy: {accuracy:.4f}")
    
    # save to json
    with open(args.output_file, "w") as f:
        json.dump(acc_list, f, indent=4)