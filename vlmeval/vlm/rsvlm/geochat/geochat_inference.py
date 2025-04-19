import argparse
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import pickle
import json
from PIL import Image
import math
from tqdm import tqdm
import yaml


# random.seed(8)
def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


class GeoChat:
    def __init__(self, model_path=None, **kwargs):
        from vlmeval.vlm.rsvlm.geochat.model.builder import load_pretrained_model
        from vlmeval.vlm.rsvlm.geochat.utils import disable_torch_init
        from vlmeval.vlm.rsvlm.geochat.mm_utils import get_model_name_from_path

        assert model_path is not None, "model_path must be provided!"
        self.model_path = model_path
        self.model_base = None
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
        self.model_args = yaml.safe_load(open("./rsvlm_config/geochat.yaml", "rb"))
        self.model_args = argparse.Namespace(**self.model_args)

    def generate(self, save_root, dataset, model_name, dataset_name):
        import torch
        from vlmeval.vlm.rsvlm.geochat.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, \
            DEFAULT_IM_END_TOKEN
        from vlmeval.vlm.rsvlm.geochat.conversation import conv_templates, SeparatorStyle
        from vlmeval.vlm.rsvlm.geochat.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria

        from vlmeval.smp import LMUDataRoot
        from vlmeval.utils import generate_example, transfer_json_to_tsv
        args = self.model_args
        args.answers_save_dir = save_root
        data_root = LMUDataRoot()
        file_pth = os.path.join(data_root, dataset_name.strip() + ".tsv")
        args.question_file = file_pth
        print("====={}=====".format(os.path.basename(file_pth)))

        self.questions = generate_example(dataset, data_root, dataset_name)

        self.questions = get_chunk(self.questions, args.num_chunks, args.chunk_idx)
        answer_file_name = "{}_{}.json".format(model_name, dataset_name)
        answers_file = os.path.expanduser(os.path.join(args.answers_save_dir, answer_file_name))
        os.makedirs(os.path.dirname(answers_file), exist_ok=True)
        got_ids = set()
        if os.path.exists(answers_file):
            with open(answers_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    line = json.loads(line)
                    got_ids.add(line['question_id'])
        ans_file = open(answers_file, "a+")

        for i in tqdm(range(0, len(self.questions), args.batch_size)):
            input_batch = []
            input_image_batch = []
            count = i
            image_folder = []
            batch_end = min(i + args.batch_size, len(self.questions))

            for j in range(i, batch_end):
                if self.questions[j]['question_id'] in got_ids:
                    continue
                image_file = self.questions[j]['image']
                qs = self.questions[j]['text']

                if self.model.config.mm_use_im_start_end:
                    qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
                else:
                    qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
                conv = conv_templates[args.conv_mode].copy()
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                # print(prompt)
                input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX,
                                                  return_tensors='pt').unsqueeze(
                    0).cuda()
                input_batch.append(input_ids)

                image = Image.open(os.path.join(args.image_folder, image_file))
                # image = image.resize((800, 800))
                image_folder.append(image)

                stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                keywords = [stop_str]
                stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
            if len(input_batch) == 0:
                continue
            max_length = max(tensor.size(1) for tensor in input_batch)

            final_input_list = [torch.cat(
                (torch.zeros((1, max_length - tensor.size(1)), dtype=tensor.dtype, device=tensor.get_device()),
                 tensor),
                dim=1) for tensor in input_batch]
            final_input_tensors = torch.cat(final_input_list, dim=0)
            image_tensor_batch = \
                self.image_processor.preprocess(image_folder, crop_size={'height': 504, 'width': 504},
                                                size={'shortest_edge': 504},
                                                return_tensors='pt')['pixel_values']
            # image_tensor_batch = image_processor.preprocess(image_folder, do_center_crop=False, do_resize=False,
            #                                                 crop_size={'height': 800, 'width': 800},
            #                                                 size={'shortest_edge': 800}, return_tensors='pt')[
            #     'pixel_values']
            err_count = 0
            try:
                with torch.inference_mode():
                    output_ids = self.model.generate(final_input_tensors, images=image_tensor_batch.half().cuda(),
                                                     do_sample=False,
                                                     temperature=args.temperature, top_p=args.top_p, num_beams=1,
                                                     max_new_tokens=256,
                                                     length_penalty=2.0, use_cache=True)

                input_token_len = final_input_tensors.shape[1]
                n_diff_input_output = (final_input_tensors != output_ids[:, :input_token_len]).sum().item()
                if n_diff_input_output > 0:
                    print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
                outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)
            except Exception as e:
                outputs = [f'An error occurred: {e}'] * len(final_input_list)
                with open(answer_file_name + "_error.txt", 'a+') as f:
                    for k in range(0, len(final_input_list)):
                        f.writelines(self.questions[count + k]["question_id"])
                err_count += 1
                assert err_count < 10, "Inference error over 10 times !"
            for k in range(0, len(final_input_list)):
                output = outputs[k].strip()
                if output.endswith(stop_str):
                    output = output[:-len(stop_str)]
                output = output.strip()

                ans_file.write(json.dumps({
                    "image_name": os.path.basename(image_file),
                    "question_id": self.questions[count]["question_id"],
                    "answer": output,
                    "ground_truth": self.questions[count]['ground_truth'],
                    "gt_value": self.questions[count]['gt_value'],
                    "negative": str(self.questions[count]['negative']),
                    "pre_answer": str(self.questions[count]['pre_answer']),
                    "category": self.questions[count]['category'],
                }) + "\n")
                count = count + 1
                ans_file.flush()
        ans_file.close()
        transfer_json_to_tsv(answers_file, dataset, args.answers_save_dir, dataset_name)
