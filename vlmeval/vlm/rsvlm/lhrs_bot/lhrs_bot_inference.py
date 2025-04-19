import json
import logging
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import argparse
import numpy as np
import yaml

from PIL import Image
from tqdm import tqdm

logger = logging.getLogger("eval")


class LHRSBot:
    def __init__(self, model_path=None, **kwargs):
        import torch
        from vlmeval.vlm.rsvlm.lhrs_bot.lhrs.models import build_model
        from vlmeval.vlm.rsvlm.lhrs_bot.lhrs.utils import type_dict
        from transformers import CLIPImageProcessor
        self.model_path = model_path
        self.model_base = kwargs['model_base']
        assert model_path is not None and self.model_base is not None, \
            "model_path and model_base must be provided!"
        self.questions = None
        self.model_args = None
        # questions = generate_example(args.question_file)
        # Model
        self.get_args()
        config = self.model_args
        config.text.path = self.model_base
        logger.info(f"Creating model")
        self.model = build_model(config, activate_modal=("rgb", "text"))
        self.vis_transform = CLIPImageProcessor.from_pretrained(config.rgb_vision.vit_name)
        self.tokenizer = self.model.text.tokenizer
        self.dtype = type_dict[config.dtype]
        self.model.to(self.dtype)

        # load model
        if self.model_path is not None:
            logger.info(f"Loading pretrained checkpoint from {self.model_path}")
            if getattr(self.model, "custom_load_state_dict", False):
                msg = self.model.custom_load_state_dict(self.model_path)
            else:
                ckpt = torch.load(self.model_path, map_location="cpu")
                msg = self.model.load_state_dict(ckpt["model"], strict=False)
            if msg is not None:
                logger.info(f"After loading, missing keys: {msg.missing_keys}, unexpected keys: {msg.unexpected_keys}")
                logger.info(str(self.model))

        if config.accelerator == "gpu":
            self.device = torch.device("cuda")
        else:
            self.device = torch.device(config.accelerator)
        self.model.to(self.device)
        self.model.eval()

    def get_args(self):
        from vlmeval.vlm.rsvlm.lhrs_bot.lhrs.CustomTrainer import init_distributed
        from vlmeval.vlm.rsvlm.lhrs_bot.lhrs.CustomTrainer.utils import ConfigArgumentParser, str2bool
        import torch
        import torch.backends.cudnn as cudnn
        import torch.distributed as dist
        import ml_collections
        import wandb
        config = yaml.safe_load(open("./rsvlm_config/lhrs_bot.yaml", "rb"))
        config = ml_collections.config_dict.ConfigDict(config)
        # config = argparse.Namespace(**config)
        config.rank, config.local_rank, config.world_size = init_distributed()
        config.is_distribute = config.world_size > 1
        config.adjust_norm = False
        print(config)

        os.makedirs(config.output, exist_ok=True)

        if config.is_distribute:
            seed = config.seed + dist.get_rank()
        else:
            seed = config.seed

        torch.manual_seed(seed)
        np.random.seed(seed)
        cudnn.benchmark = True

        if config.rank == 0:
            path = os.path.join(config.output, "lhrs_config.json")
            with open(path, "w") as f:
                configDict = dict(config.to_dict())
                json.dump(configDict, f, indent=4)
            logger.info(f"Full config saved to {path}")
            logger.info(config)

        if config.wandb and config.rank == 0:
            wandb.init(config=config.to_dict(), entity=config.entity, project=config.project)
            config = wandb.config

        self.model_args = config

    def generate(self, save_root, dataset, model_name, dataset_name):
        from vlmeval.smp import LMUDataRoot
        from vlmeval.utils import generate_example, transfer_json_to_tsv
        import torch
        from vlmeval.vlm.rsvlm.lhrs_bot.lhrs.Dataset.conversation import default_conversation
        from vlmeval.vlm.rsvlm.lhrs_bot.lhrs.models import (
            DEFAULT_IM_END_TOKEN,
            DEFAULT_IM_START_TOKEN,
            DEFAULT_IMAGE_TOKEN,
            IMAGE_TOKEN_INDEX,
            tokenizer_image_token,
        )
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
        got_ids = set()
        if os.path.exists(answers_file):
            with open(answers_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    line = json.loads(line)
                    got_ids.add(line['question_id'])
        ans_file = open(answers_file, "a+")
        with torch.no_grad():
            for sample in tqdm(self.questions, desc="Evaluating"):
                question = sample["text"]
                image_path = sample["image"]
                answer = sample["ground_truth"]
                idx = sample["question_id"]
                if idx in got_ids:
                    continue

                image = Image.open(image_path).convert("RGB")
                # r_size = vis_transform.size["shortest_edge"]
                # image = image.resize((r_size, r_size))
                image_tensor = self.vis_transform(image, return_tensors="pt")["pixel_values"].to(self.device).to(
                    self.dtype)
                if args.tune_im_start:
                    question = (
                            DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + question
                    )
                else:
                    question = DEFAULT_IMAGE_TOKEN + "\n" + question

                inp = (
                        question + "Answer with the letter from the given options"
                )
                choices_logits_list = []
                dummy_conv = default_conversation.copy()
                dummy_conv.append_message(dummy_conv.roles[0], inp)
                dummy_conv.append_message(dummy_conv.roles[1], None)
                test_ques = dummy_conv.get_prompt()
                dummy_input = (
                    tokenizer_image_token(
                        dummy_conv.get_prompt(), self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                    )
                    .unsqueeze(0)
                    .to(self.device)
                )
                with torch.autocast(
                        device_type="cuda" if args.accelerator == "gpu" else "cpu",
                        enabled=args.enable_amp,
                        dtype=self.dtype,
                ):
                    output_ids = self.model.generate(
                        dummy_input,
                        images=image_tensor,
                        do_sample=False,
                        num_beams=1,
                        temperature=1.0,
                        top_p=1.0,
                        max_new_tokens=10,
                    )

                    outputs = self.model.text.tokenizer.batch_decode(output_ids, skip_special_tokens=False)
                    outputs = outputs[0].split("<|eot_id|>")[0]
                    output = outputs.strip().replace("<s>", "").replace("</s>", "").strip()
                # print(test_ques)
                # print("answer:", answer)
                # print("output:", output, outputs)
                ans_file.write(json.dumps({"image_name": os.path.basename(image_path),
                                           "question_id": sample["question_id"],
                                           "answer": output,
                                           "ground_truth": sample['ground_truth'],
                                           "gt_value": sample['gt_value'],
                                           "negative": str(sample['negative']),
                                           "pre_answer": str(sample['pre_answer']),
                                           "category": sample['category'], }) + "\n")
            ans_file.flush()
            transfer_json_to_tsv(answers_file, dataset, args.answers_save_dir, dataset_name)
