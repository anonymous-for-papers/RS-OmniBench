from vlmeval.vlm import *
from vlmeval.api import *
from functools import partial

from vlmeval.vlm.rsvlm.vhm.vhm_model import VHM

PandaGPT_ROOT = None
MiniGPT4_ROOT = None
TransCore_ROOT = None
Yi_ROOT = None
OmniLMM_ROOT = None
Mini_Gemini_ROOT = None
VXVERSE_ROOT = None
VideoChat2_ROOT = None
VideoChatGPT_ROOT = None
PLLaVA_ROOT = None
RBDash_ROOT = None
VITA_ROOT = None
LLAVA_V1_7B_MODEL_PTH = 'Please set your local path to LLaVA-7B-v1.1 here, the model weight is obtained by merging LLaVA delta weight based on vicuna-7b-v1.1 in https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md with vicuna-7b-v1.1. '

video_models = {}
o1_key = 'XXX'  # noqa: E501
o1_apis = {
    'o1': partial(
        GPT4V,
        model="o1-2024-12-17",
        key=o1_key,
        api_base='OFFICIAL',
        temperature=0,
        img_detail='high',
        retry=10,
        verbose=False,
    ),
}
api_models = {
    # GPT
    'GPT4V': partial(GPT4V, model='gpt-4-1106-vision-preview', temperature=0, img_size=512, img_detail='low', retry=10,
                     verbose=False),
    'GPT4V_HIGH': partial(GPT4V, model='gpt-4-1106-vision-preview', temperature=0, img_size=-1, img_detail='high',
                          retry=10, verbose=False),
    'GPT4V_20240409': partial(GPT4V, model='gpt-4-turbo-2024-04-09', temperature=0, img_size=512, img_detail='low',
                              retry=10, verbose=False),
    'GPT4V_20240409_HIGH': partial(GPT4V, model='gpt-4-turbo-2024-04-09', temperature=0, img_size=-1, img_detail='high',
                                   retry=10, verbose=False),
    'GPT4o': partial(GPT4V, model='gpt-4o-2024-05-13', temperature=0, img_size=512, img_detail='low', retry=10,
                     verbose=False),
    'GPT4o_HIGH': partial(GPT4V, model='gpt-4o-2024-05-13', temperature=0, img_size=-1, img_detail='high', retry=10,
                          verbose=False),
    'GPT4o_20240806': partial(GPT4V, model='gpt-4o-2024-08-06', temperature=0, img_size=-1, img_detail='high', retry=10,
                              verbose=False),
    'GPT4o_20241120': partial(GPT4V, model='gpt-4o-2024-11-20', temperature=0, img_size=-1, img_detail='high', retry=10,
                              verbose=False),
    'GPT4o_MINI': partial(GPT4V, model='gpt-4o-mini-2024-07-18', temperature=0, img_size=-1, img_detail='high',
                          retry=10, verbose=False),
    # Gemini
    'GeminiPro1-0': partial(GeminiProVision, model='gemini-1.0-pro', temperature=0, retry=10),
    # now GeminiPro1-0 is only supported by vertex backend
    'GeminiPro1-5': partial(GeminiProVision, model='gemini-1.5-pro', temperature=0, retry=10),
    'GeminiFlash1-5': partial(GeminiProVision, model='gemini-1.5-flash', temperature=0, retry=10),
    'GeminiFlash2-0': partial(GeminiProVision, model='gemini-2.0-flash', temperature=0, retry=10),
    'GeminiPro2-0': partial(GeminiProVision, model='gemini-2.0-pro-exp', temperature=0, retry=10),
    'GeminiPro1-5-002': partial(GPT4V, model='gemini-1.5-pro-002', temperature=0, retry=10),  # Internal Use Only
    'GeminiFlash1-5-002': partial(GPT4V, model='gemini-1.5-flash-002', temperature=0, retry=10),  # Internal Use Only
    # Claude
    'Claude3V_Opus': partial(Claude3V, model='claude-3-opus-20240229', temperature=0, retry=10, verbose=False),
    'Claude3V_Sonnet': partial(Claude3V, model='claude-3-sonnet-20240229', temperature=0, retry=10, verbose=False),
    'Claude3V_Haiku': partial(Claude3V, model='claude-3-haiku-20240307', temperature=0, retry=10, verbose=False),
    'Claude3-5V_Sonnet': partial(Claude3V, model='claude-3-5-sonnet-20240620', temperature=0, retry=10, verbose=False),
    'Claude3-5V_Sonnet_20241022': partial(Claude3V, model='claude-3-5-sonnet-20241022', temperature=0, retry=10,
                                          verbose=False),
    'Claude3-5V_Sonnet_20241022_tem07': partial(Claude3V, model='claude-3-5-sonnet-20241022', temperature=0.7, retry=10,
                                                verbose=False),
}

xtuner_series = {
    'llava-internlm2-7b': partial(LLaVA_XTuner, llm_path='internlm/internlm2-chat-7b',
                                  llava_path='xtuner/llava-internlm2-7b', visual_select_layer=-2,
                                  prompt_template='internlm2_chat'),
    'llava-internlm2-20b': partial(LLaVA_XTuner, llm_path='internlm/internlm2-chat-20b',
                                   llava_path='xtuner/llava-internlm2-20b', visual_select_layer=-2,
                                   prompt_template='internlm2_chat'),
    'llava-internlm-7b': partial(LLaVA_XTuner, llm_path='internlm/internlm-chat-7b',
                                 llava_path='xtuner/llava-internlm-7b', visual_select_layer=-2,
                                 prompt_template='internlm_chat'),
    'llava-v1.5-7b-xtuner': partial(LLaVA_XTuner,
                                    llm_path='lmsys/vicuna-7b-v1.5',
                                    llava_path='xtuner/llava-v1.5-7b-xtuner',
                                    visual_select_layer=-2, prompt_template='vicuna'),
    'llava-v1.5-13b-xtuner': partial(LLaVA_XTuner,
                                     llm_path='lmsys/vicuna-13b-v1.5',
                                     llava_path='xtuner/llava-v1.5-13b-xtuner',
                                     visual_select_layer=-2, prompt_template='vicuna'),
    'llava-llama-3-8b': partial(LLaVA_XTuner, llm_path='xtuner/llava-llama-3-8b-v1_1',
                                llava_path='xtuner/llava-llama-3-8b-v1_1', visual_select_layer=-2,
                                prompt_template='llama3_chat'),
}

llava_series = {
    'llava_v1.5_7b': partial(LLaVA, model_path='liuhaotian/llava-v1.5-7b'),
    'llava_v1.5_13b': partial(LLaVA, model_path='liuhaotian/llava-v1.5-13b'),
    'llava_v1_7b': partial(LLaVA, model_path=LLAVA_V1_7B_MODEL_PTH),
    'sharegpt4v_7b': partial(LLaVA, model_path='Lin-Chen/ShareGPT4V-7B'),
    'sharegpt4v_13b': partial(LLaVA, model_path='Lin-Chen/ShareGPT4V-13B'),
    'llava_next_vicuna_7b': partial(LLaVA_Next,
                                    model_path='llava-hf/llava-v1.6-vicuna-7b-hf'),
    'llava_next_vicuna_13b': partial(LLaVA_Next,
                                     model_path='llava-hf/llava-v1.6-vicuna-13b-hf'),
    'llava_next_mistral_7b': partial(LLaVA_Next, model_path='llava-hf/llava-v1.6-mistral-7b-hf'),
    'llava_next_yi_34b': partial(LLaVA_Next,
                                 model_path='llava-hf/llava-v1.6-34b-hf'),
    'llava_next_llama3': partial(LLaVA_Next, model_path='llava-hf/llama3-llava-next-8b-hf'),
    'llava_next_72b': partial(LLaVA_Next, model_path='llava-hf/llava-next-72b-hf'),
    'llava_next_110b': partial(LLaVA_Next, model_path='llava-hf/llava-next-110b-hf'),
    'llava_next_qwen_32b': partial(LLaVA_Next2, model_path='lmms-lab/llava-next-qwen-32b'),
    'llava_next_interleave_7b': partial(LLaVA_Next, model_path='llava-hf/llava-interleave-qwen-7b-hf'),
    'llava_next_interleave_7b_dpo': partial(LLaVA_Next, model_path='llava-hf/llava-interleave-qwen-7b-dpo-hf'),
    'llava-onevision-qwen2-0.5b-ov-hf': partial(
        LLaVA_OneVision_HF, model_path='llava-hf/llava-onevision-qwen2-0.5b-ov-hf'
    ),
    'llava-onevision-qwen2-0.5b-si-hf': partial(
        LLaVA_OneVision_HF, model_path='llava-hf/llava-onevision-qwen2-0.5b-si-hf'
    ),
    'llava-onevision-qwen2-7b-ov-hf': partial(LLaVA_OneVision_HF, model_path='llava-hf/llava-onevision-qwen2-7b-ov-hf'),
    'llava-onevision-qwen2-7b-si-hf': partial(LLaVA_OneVision_HF, model_path='llava-hf/llava-onevision-qwen2-7b-si-hf'),
    'llava_onevision_qwen2_0.5b_si': partial(LLaVA_OneVision, model_path='lmms-lab/llava-onevision-qwen2-0.5b-si'),
    'llava_onevision_qwen2_7b_si': partial(LLaVA_OneVision, model_path='lmms-lab/llava-onevision-qwen2-7b-si'),
    'llava_onevision_qwen2_72b_si': partial(LLaVA_OneVision, model_path='lmms-lab/llava-onevision-qwen2-72b-si'),
    'llava_onevision_qwen2_0.5b_ov': partial(LLaVA_OneVision, model_path='lmms-lab/llava-onevision-qwen2-0.5b-ov'),
    'llava_onevision_qwen2_7b_ov': partial(LLaVA_OneVision, model_path='lmms-lab/llava-onevision-qwen2-7b-ov'),
    'llava_onevision_qwen2_72b_ov': partial(LLaVA_OneVision, model_path='lmms-lab/llava-onevision-qwen2-72b-ov-sft'),
    'Aquila-VL-2B': partial(LLaVA_OneVision, model_path='BAAI/Aquila-VL-2B-llava-qwen'),
    'llava_video_qwen2_7b': partial(LLaVA_OneVision, model_path='lmms-lab/LLaVA-Video-7B-Qwen2'),
    'llava_video_qwen2_72b': partial(LLaVA_OneVision, model_path='lmms-lab/LLaVA-Video-72B-Qwen2'),
    'varco-vision-hf': partial(LLaVA_OneVision_HF, model_path='NCSOFT/VARCO-VISION-14B-HF'),
}

internvl_series = {
    'InternVL-Chat-V1-1': partial(InternVLChat, model_path='OpenGVLab/InternVL-Chat-V1-1', version='V1.1'),
    'InternVL-Chat-V1-2': partial(InternVLChat, model_path='OpenGVLab/InternVL-Chat-V1-2', version='V1.2'),
    'InternVL-Chat-V1-2-Plus': partial(
        InternVLChat, model_path='OpenGVLab/InternVL-Chat-V1-2-Plus', version='V1.2'
    ),
    # InternVL1.5 series
    'InternVL-Chat-V1-5': partial(
        InternVLChat,
        model_path='OpenGVLab/InternVL-Chat-V1-5',
        version='V1.5',
    ),
    'Mini-InternVL-Chat-2B-V1-5': partial(
        InternVLChat, model_path='OpenGVLab/Mini-InternVL-Chat-2B-V1-5', version='V1.5'
    ),
    'Mini-InternVL-Chat-4B-V1-5': partial(
        InternVLChat, model_path='OpenGVLab/Mini-InternVL-Chat-4B-V1-5', version='V1.5'
    ),
    # InternVL2 series
    'InternVL2-1B': partial(InternVLChat,
                            model_path='OpenGVLab/InternVL2-1B',
                            version='V2.0'),
    'InternVL2-2B': partial(InternVLChat,
                            model_path='OpenGVLab/InternVL2-2B',
                            version='V2.0'),
    'InternVL2-4B': partial(InternVLChat,
                            model_path='OpenGVLab/InternVL2-4B',
                            version='V2.0'),
    'InternVL2-8B': partial(InternVLChat,
                            model_path='OpenGVLab/InternVL2-8B',
                            version='V2.0'),
    'InternVL2-26B': partial(InternVLChat,
                             model_path='OpenGVLab/InternVL2-26B',
                             version='V2.0'),
    'InternVL2-40B': partial(
        InternVLChat, model_path='OpenGVLab/InternVL2-40B', version='V2.0'
    ),
    'InternVL2-76B': partial(InternVLChat, model_path='OpenGVLab/InternVL2-Llama3-76B', version='V2.0'),
    # InternVL2 MPO series
    'InternVL2-8B-MPO': partial(InternVLChat, model_path='OpenGVLab/InternVL2-8B-MPO', version='V2.0'),
    'InternVL2-8B-MPO-CoT': partial(
        InternVLChat, model_path='OpenGVLab/InternVL2-8B-MPO', version='V2.0', use_mpo_prompt=True
    ),
    # InternVL2.5 series
    'InternVL2_5-1B': partial(InternVLChat, model_path='OpenGVLab/InternVL2_5-1B', version='V2.0'),
    'InternVL2_5-2B': partial(InternVLChat, model_path='OpenGVLab/InternVL2_5-2B', version='V2.0'),
    'InternVL2_5-4B': partial(InternVLChat, model_path='OpenGVLab/InternVL2_5-4B', version='V2.0'),
    'InternVL2_5-8B': partial(InternVLChat, model_path='OpenGVLab/InternVL2_5-8B', version='V2.0'),
    'InternVL2_5-26B': partial(InternVLChat, model_path='OpenGVLab/InternVL2_5-26B', version='V2.0'),
    'InternVL2_5-38B': partial(InternVLChat, model_path='OpenGVLab/InternVL2_5-38B', version='V2.0'),
    'InternVL2_5-78B': partial(InternVLChat, model_path='OpenGVLab/InternVL2_5-78B', version='V2.0'),
    # InternVL2.5-MPO series
    'InternVL2_5-1B-MPO': partial(
        InternVLChat, model_path='OpenGVLab/InternVL2_5-1B-MPO', version='V2.0', use_mpo_prompt=True
    ),
    'InternVL2_5-2B-MPO': partial(
        InternVLChat, model_path='OpenGVLab/InternVL2_5-2B-MPO', version='V2.0', use_mpo_prompt=True
    ),
    'InternVL2_5-4B-MPO': partial(
        InternVLChat, model_path='OpenGVLab/InternVL2_5-4B-MPO', version='V2.0', use_mpo_prompt=True
    ),
    'InternVL2_5-8B-MPO': partial(
        InternVLChat, model_path='OpenGVLab/InternVL2_5-8B-MPO', version='V2.0', use_mpo_prompt=True
    ),
    'InternVL2_5-26B-MPO': partial(
        InternVLChat, model_path='OpenGVLab/InternVL2_5-26B-MPO', version='V2.0', use_mpo_prompt=True
    ),
    'InternVL2_5-38B-MPO': partial(
        InternVLChat, model_path='OpenGVLab/InternVL2_5-38B-MPO', version='V2.0', use_mpo_prompt=True
    ),
    'InternVL2_5-78B-MPO': partial(
        InternVLChat, model_path='OpenGVLab/InternVL2_5-78B-MPO', version='V2.0', use_mpo_prompt=True
    ),
}

yivl_series = {
    'Yi_VL_6B': partial(Yi_VL, model_path='01-ai/Yi-VL-6B', root=Yi_ROOT),
    'Yi_VL_34B': partial(Yi_VL, model_path='01-ai/Yi-VL-34B', root=Yi_ROOT),
}

xcomposer_series = {
    'XComposer': partial(XComposer, model_path='internlm/internlm-xcomposer-vl-7b'),
    'sharecaptioner': partial(ShareCaptioner, model_path='Lin-Chen/ShareCaptioner'),
    'XComposer2': partial(XComposer2,
                          model_path='internlm/internlm-xcomposer2-vl-7b'),
    'XComposer2_1.8b': partial(XComposer2, model_path='internlm/internlm-xcomposer2-vl-1_8b'),
    'XComposer2_4KHD': partial(XComposer2_4KHD, model_path='internlm/internlm-xcomposer2-4khd-7b'),
    'XComposer2d5': partial(XComposer2d5, model_path='internlm/internlm-xcomposer2d5-7b'),
}

instructblip_series = {
    'instructblip_7b': partial(InstructBLIP, name='instructblip_7b'),
    'instructblip_13b': partial(InstructBLIP, name='instructblip_13b'),
}

deepseekvl_series = {
    'deepseek_vl_7b': partial(DeepSeekVL,
                              model_path='deepseek-ai/deepseek-vl-7b-chat'),
    'deepseek_vl_1.3b': partial(DeepSeekVL, model_path='deepseek-ai/deepseek-vl-1.3b-chat'),
}

deepseekvl2_series = {
    'deepseek_vl2_tiny': partial(DeepSeekVL2, model_path='deepseek-ai/deepseek-vl2-tiny'),
    'deepseek_vl2_small': partial(DeepSeekVL2, model_path='deepseek-ai/deepseek-vl2-small'),
    'deepseek_vl2': partial(DeepSeekVL2, model_path='deepseek-ai/deepseek-vl2'),
}

llama_series = {
    'Llama-3.2-11B-Vision-Instruct': partial(llama_vision, model_path='meta-llama/Llama-3.2-11B-Vision-Instruct'),
    'LLaVA-CoT': partial(llama_vision, model_path='Xkev/Llama-3.2V-11B-cot'),
    'Llama-3.2-90B-Vision-Instruct': partial(llama_vision, model_path='meta-llama/Llama-3.2-90B-Vision-Instruct'),
}
rsvlm = {
    'GeoChat': partial(GeoChat, model_path='MBZUAI/geochat-7B'),
    "RSLLaVA": partial(RSLLaVA, model_path='BigData-KSU/RS-llava-v1.5-7b-LoRA',
                       model_base='Intel/neural-chat-7b-v3-3'),
    "VHM": partial(VHMModel, model_path='FitzPC/vhm_7B'),
    "LHRSBot": partial(LHRSBot, model_path='/data/parameters/LHRS-Bot/Stage2/FINAL.pt',
                       model_base='meta-llama/Llama-2-7b-chat-hf'),
}
supported_VLM = {}

model_groups = [
    rsvlm, o1_apis, api_models,
    xtuner_series, llava_series, internvl_series, yivl_series,
    xcomposer_series, instructblip_series,
    deepseekvl_series, deepseekvl2_series, video_models, llama_series
]

for grp in model_groups:
    supported_VLM.update(grp)
