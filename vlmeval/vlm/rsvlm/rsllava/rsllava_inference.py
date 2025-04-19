import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import argparse
import json
from tqdm import tqdm
import yaml
from PIL import Image
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


class RSLLaVA:
    def __init__(self, model_path=None, **kwargs):
        from vlmeval.vlm.rsvlm.rsllava.llava.model.builder import load_pretrained_model
        from vlmeval.vlm.rsvlm.rsllava.llava.utils import disable_torch_init
        from vlmeval.vlm.rsvlm.rsllava.llava.mm_utils import get_model_name_from_path
        self.model_path = model_path
        self.model_base = kwargs['model_base']
        assert model_path is not None and self.model_base is not None, \
            "model_path and model_base must be provided!"

        self.questions = None
        self.model_args = None
        # questions = generate_example(args.question_file)
        # Model
        disable_torch_init()
        model_path = os.path.expanduser(self.model_path)
        model_name = get_model_name_from_path(model_path)
        tokenizer, model, image_processor, context_len = load_pretrained_model(self.model_path, self.model_base,
                                                                               model_name)
        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.context_len = context_len
        self.get_args()

    def get_args(self):
        self.model_args = yaml.safe_load(open("./rsvlm_config/rsllava.yaml", "rb"))
        self.model_args = argparse.Namespace(**self.model_args)

    def generate(self, save_root, dataset, model_name, dataset_name):
        import torch
        from vlmeval.vlm.rsvlm.rsllava.llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, \
            DEFAULT_IM_START_TOKEN, \
            DEFAULT_IM_END_TOKEN
        from vlmeval.vlm.rsvlm.rsllava.llava.conversation import conv_templates, SeparatorStyle
        from vlmeval.vlm.rsvlm.rsllava.llava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria
        from vlmeval.utils.rsvlm_tool import generate_example, transfer_json_to_tsv
        from vlmeval.smp import LMUDataRoot
        args = self.model_args
        args.answers_save_dir = save_root
        data_root = LMUDataRoot()
        file_pth = os.path.join(data_root, dataset_name.strip() + ".tsv")
        args.question_file = file_pth
        print("====={}=====".format(os.path.basename(file_pth)))
        self.questions = generate_example(dataset, data_root, dataset_name)
        # self.questions = get_chunk(self.questions, args.num_chunks, args.chunk_idx)
        answer_file_name = "{}_{}.json".format(model_name, dataset_name)
        answers_file = os.path.expanduser(os.path.join(args.answers_save_dir, answer_file_name))
        os.makedirs(os.path.dirname(answers_file), exist_ok=True)
        # ans_file = open(answers_file, "w")
        got_ids = set()
        if os.path.exists(answers_file):
            with open(answers_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    line = json.loads(line)
                    got_ids.add(line['question_id'])
        ans_file = open(answers_file, "a+")
        for line in tqdm(self.questions):
            idx = line["question_id"]
            if idx in got_ids:
                continue
            image_file = line["image"]
            qs = line["text"]  # + "\nAnswer with the option's letter (A, B, C or D) from the given choices directly."
            cur_prompt = qs
            if self.model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            # print(prompt)
            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(
                0).cuda()

            image = Image.open(os.path.join(args.image_folder, image_file))
            # try:
            image_tensor = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            # except Exception as e:
            #     print("error:", image_file)
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    images=image_tensor.unsqueeze(0).half().cuda(),
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    # no_repeat_ngram_size=3,
                    max_new_tokens=1024,
                    use_cache=True)

            input_token_len = input_ids.shape[1]
            n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
            outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            outputs = outputs.strip()
            # print(outputs)
            ans_file.write(json.dumps({"image_name": os.path.basename(image_file),
                                       "question_id": line["question_id"],
                                       "answer": outputs,
                                       "ground_truth": line['ground_truth'],
                                       "gt_value": line['gt_value'],
                                       "negative": str(line['negative']),
                                       "pre_answer": str(line['pre_answer']),
                                       "category": line['category'], }) + "\n")
            ans_file.flush()
        ans_file.close()
        transfer_json_to_tsv(answers_file, dataset, args.answers_save_dir, dataset_name)
