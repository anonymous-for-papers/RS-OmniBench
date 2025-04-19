import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import re
import math
import requests
import json
import argparse
import yaml
from PIL import Image
from tqdm import tqdm


# import shortuuid


def convt_qa(conversations):
    values = [conversation['value'] for conversation in conversations]
    query = values[0]
    answer = values[1]
    return query, answer


class VHMModel:
    def __init__(self, model_path=None, **kwargs):
        from vlmeval.vlm.rsvlm.vhm.vhm.utils import disable_torch_init

        from vlmeval.vlm.rsvlm.vhm.vhm_model import VHM

        self.model_path = model_path
        assert model_path is not None, \
            "model_path must be provided!"
        self.model_base = None
        self.questions = None
        self.model_args = None
        model_path_map = {
            'VHM': self.model_path
        }
        # questions = generate_example(args.question_file)
        # Model
        disable_torch_init()
        self.model = VHM(name="VHM", model_path_map=model_path_map)
        self.get_args()

    def get_args(self):
        self.model_args = yaml.safe_load(open("./rsvlm_config/rsllava.yaml", "rb"))
        self.model_args = argparse.Namespace(**self.model_args)

    def generate(self, save_root, dataset, model_name, dataset_name):
        import torch
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
        for sample in tqdm(self.questions):
            idx = sample['question_id']
            if idx in got_ids:
                continue
            question = '{VQA}' + ' ' + sample[
                "text"] + "Answer with the letter from the given options."  # '{IT}' +' ' + sample["text"]
            image_path = sample["image"]
            err_count = 0
            try:
                with torch.inference_mode():
                    outputs = self.model.generate(image_path, question)
                outputs = outputs
            except Exception as e:
                outputs = f'An error occurred: {e}'
                err_count += 1
                print(outputs)
                assert err_count < 10, "Inference error over 10 times !"

            # print("answer:{}, pred:{}".format(sample["ground_truth"], outputs))
            ans_file.write(json.dumps({"image_name": os.path.basename(image_path),
                                       "question_id": sample["question_id"],
                                       "answer": outputs,
                                       "ground_truth": sample['ground_truth'],
                                       "gt_value": sample['gt_value'],
                                       "negative": str(sample['negative']),
                                       "pre_answer": str(sample['pre_answer']),
                                       "category": sample['category'], }) + "\n")

        ans_file.flush()
        transfer_json_to_tsv(answers_file, dataset, args.answers_save_dir, dataset_name)
