## <a id="top"></a>Quick Start for RS-OmniBench

<p align="left">
  <a href="#preparedataset"><b>Prepare the Dataset</b></a> |
  <a href="#installation"><b>Installation</b></a> |
  <a href="#configuration"><b>Configuration</b></a> |
  <a href="#evaluation"><b>Evaluation</b></a>

[//]: # (  <a href="#extract"><b>Compute Accuracy</b></a> |)
</p>

### <a id="preparedataset"></a>Step 0: Prepare the Dataset
Our full dataset has not been released yet. 
Once the paper is accepted, we will make it publicly available immediately. 
In the meantime, you can refer to our sample dataset located in the [`dataset`](./dataset) directory. 
The format of the full dataset will be exactly the same as the sample dataset. 
After downloading the data, please place it in our `LMUData` directory.

### <a id="installation"></a>Step 1: Installation and Setup API Keys

#### Installation
```bash
git clone https://github.com/anonymous-for-papers/RS-OmniBench.git
cd RS-OmniBench
pip install -e .
```

#### Setup API Keys
To use proprietary LVLMs for evaluating our dataset, you need to set up the API keys for the following models in the `.env` file located under the project directory (`$VLMEvalKit`). You can modify the values of these keys as needed.

```bash
# .env file，place it under $VLMEvalKit
# API Keys of Proprietary VLMs
# Gemini w. Google Cloud Backends
GOOGLE_API_KEY=
# OpenAI API
OPENAI_API_KEY=
OPENAI_API_BASE=
# Claude API
ANTHROPIC_API_KEY =

# You can set up a proxy during the evaluation phase to route all API calls through this proxy.
EVAL_PROXY=
# Set LMUData Path for your datasets
LMUData= ./LMUData
```
Configuration Details:
- GOOGLE_API_KEY: API key for Gemini model with Google Cloud backends.
- OPENAI_API_KEY: API key for OpenAI services (e.g., GPT-4, ChatGPT).
- OPENAI_API_BASE: The base URL for OpenAI API (if applicable).
- ANTHROPIC_API_KEY: API key for the Claude model by Anthropic.
- EVAL_PROXY: Optional proxy to route API calls through during the evaluation phase.
- LMUData: Path to the directory containing your datasets (e.g., /data/LMUData).

Make sure to fill in the appropriate API keys and paths to ensure smooth evaluation. The .env file allows for easy modification of these values.
### 🛠️<a id="configuration"></a>Step 2: Configuration

All VLMs are configured in `vlmeval/config.py`. For some VLMs, you need to configure the model weight root (e.g., LHRS-Bot and InstructBLIP) before conducting the evaluation.

#### Configuration Process

1. **Model Selection**: During evaluation, you should use the model name specified in `supported_VLM` in `vlmeval/config.py` to select the VLM.
   
2. **InstructBLIP Configuration**: For InstructBLIP, you also need to modify the configuration files in `vlmeval/vlm/misc` to configure the LLM path and checkpoint (`ckpt`) path.

#### VLMs Requiring Configuration

Each VLM evaluated in this project may require its own specific runtime environment. Please follow the steps below to ensure proper configuration before running any evaluation:

- **Code Preparation & Installation**:
  - [InstructBLIP](https://github.com/salesforce/LAVIS)
  - [LLaVA & LLaVA-Next & Yi-VL](https://github.com/haotian-liu/LLaVA)
  - [InternVL2](https://github.com/OpenGVLab/InternVL.git)
  - [DeepSeek-VL](https://github.com/deepseek-ai/DeepSeek-VL)
  - [XComposer2](https://github.com/InternLM/InternLM-XComposer.git)
  - [ShareGPT4V](https://github.com/ShareGPT4Omni/ShareGPT4V)
  - [GeoChat](https://github.com/mbzuai-oryx/GeoChat)
  - [VHM](https://github.com/opendatalab/VHM)
  - [RS-LLaVA](https://github.com/BigData-KSU/RS-LLaVA)
  - [LHRS-Bot](https://github.com/NJU-LHRS/LHRS-Bot)
  
- **Manual Weight Preparation**:
  - For LHRS-Bot, you will need to modify the configuration files in `vlmeval/config.py` to set the model weight path.
  - For InstructBLIP, you will need to modify the configuration files in `vlmeval/vlm/misc` to set the LLM path and checkpoint (`ckpt`) path.
- **Transformers Version Recommendation:**

  Note that some VLMs may not be able to run under certain transformer versions, we recommend the following settings to evaluate each VLM:
    - **Please use** `transformers==4.31.0` **for**: `GeoChat`, `VHM`.
    - **Please use** `transformers==4.33.0` **for**: `InternLM-XComposer Series`, `InstructBLIP series`.
    - **Please use** `transformers==4.35.0` **for**: `RS-LLaVA`.
    - **Please use** `transformers==4.36.1` **for**: `LHRS-Bot`.
    - **Please use** `transformers==4.37.0` **for**: `LLaVA series`, `ShareGPT4V series`, `LLaVA (XTuner)`, `Yi-VL Series`, `DeepSeek-VL series`, `InternVL series`.
    - **Please use** `transformers==latest` **for**: `LLaVA-Next series`.

Make sure to complete the necessary configurations before starting the evaluation to ensure a smooth process.

### <a id="evaluation"></a>Step 3: Evaluation
We use `run.py` to perform the evaluation. To execute `run.py`, you can call the script `$VLMEvalKit/run.sh`. Below are the available arguments and their descriptions:
- `--data (list[str])`: Specifies the dataset names, for example, `image_level_iamge_captioning_region_specific`. Provide the datasets you want to use for evaluation.
- `--model (list[str])`: Specifies the VLM names that are supported in VLMEvalKit (defined in `supported_VLM` in `vlmeval/config.py`). Provide the VLM models you wish to evaluate.
- `--mode (str, default='all', choices=['all', 'infer'])`: Sets the evaluation mode. **In this study, we set `mode=infer`**.
  - When `mode='all'`, it will perform both inference and evaluation.
  - When `mode='infer'`, it will only perform the inference.
- `--api-nproc (int, default=4)`: Specifies the number of threads for OpenAI API calls. The default is 4 threads, but you can adjust based on your system.
- `--work-dir (str, default='./outputs')`: Defines the directory where the evaluation results will be saved. By default, it saves the results in the current directory (`./outputs`).

Make sure to adjust the parameters based on your setup before running the evaluation.  

**Example `run.sh` Execution**

```bash
# Run with specific dataset and model
CUDA_VISIBLE_DEVICES=4 \
/{$your_env}/python run.py \
    --data image_level_image_captioning_image_wide \
          image_level_image_classification_image_wide \
          image_level_image_vqa_image_wide \
          image_level_image_retrieval_image_wide \
          region_level_image_od_image_wide \
          region_level_image_vg_image_wide \
          pixel_level_image_segmentation_image_wide \
          image_level_image_captioning_region_specific \
          image_level_image_classification_region_specific \
          image_level_image_vqa_region_specific \
          region_level_image_vg_region_specific \
          region_level_image_od_region_specific \
          pixel_level_image_segmentation_region_specific \
    --model GeoChat \
    --verbose \
    --mode infer \
    --reuse
```

[//]: # (### <a id="extract"></a>Step 4: Compute Accuracy)

#### 🔝 [Back to Top](#top)