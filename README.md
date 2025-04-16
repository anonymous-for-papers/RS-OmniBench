# <a id="top"></a>RS-OmniBench: A Comprehensive Benchmark for Evaluating Large Vision-Language Models in Remote Sensing Imagery

<p align="left">
  <a href="#quickstart"><b>Quick Start</b></a> |
  <a href="#introduction"><b>Introduction</b></a> |
  <a href="#evaluation"><b>Evaluation Results</b></a> |
  <a href="#leaderboard"><b>LeaderBoard</b></a> |
  <a href="#OmniBench"><b>RS-OmniBench Dataset</b></a>
</p>

## <a id="news"></a>📢 News
We are excited to announce that the dataset and code for this project will be made publicly available 
immediately upon the acceptance of the paper. Stay tuned for updates!
- `2025-04-16`: We have released sample examples for RS-OmniBench dataset covering **53 subtasks**, 
which are available in the [`dataset`](./dataset) directory. For the description of the key fields in the dataset's TSV files, please refer to [`RS-OmniBench Dataset`](#OmniBench).  
**The complete dataset will be made publicly available immediately upon the acceptance of the paper.**
## <a id="introduction"></a>💡 Introduction
RS-OmniBench is a comprehensive, large-scale benchmark designed to offer a holistic evaluation of Large Vision-Language Models (LVLMs) in the context of remote sensing imagery. The benchmark includes seven meta-tasks and 53 subtasks, addressing image-level, region-level, and pixel-level understanding.

With a total of 175K multiple-choice visual questions, RS-OmniBench rigorously evaluates the recognition, localization, and reasoning capabilities of LVLMs in remote sensing imagery.

![overview](assets/theme.png)

## <a id="evaluation"></a>📊 Evaluation Results
- An overview of 22 LVLMs evaluated in this study. This table is provided in the **Supplementary Materials** of the paper.  
<img src="assets/table3.png" alt="overview" width="500"/>

- Quantitative results for 22 LVLMs across seven core meta-tasks 
from three perspectives (image-level, region-level, and pixel-level) 
are summarized. Accuracy is the metric. The overall accuracy score 
is calculated across all data in RS-OmniBench. 
Bold indicates the best performance in each column, 
and underline represents the second best.  
<img src="assets/table1.png" alt="overview"/>

- Results of 22 LVLMs across 53 diverse subtasks: Horizontal axis representing subtasks ('IW-' for image-wide, 'RS-' for
region-specific) and vertical axis representing models. 
Current LVLMs exhibit limitations across a wide range of subtasks (blue region).  
<img src="assets/53subs_res.png" alt="overview"/>

- Model performance comparison on seven meta-tasks.
Bold indicates the best performance in each column, and
underline represents the second best. This table is provided in the **Supplementary Materials** of the paper.  
<img src="assets/table2.png" alt="overview" width="600"/>


## <a id="leaderboard"></a>🎖️ Leaderboard
-  The leaderboard displays the performance of the 22 LVLMs evaluated in our study. 
On the left, we visualize the results of representative LVLMs across seven meta-tasks, while the right side highlights the overall ranking.  
<img src="assets/leaderboard.png" alt="overview" width="600"/>

## <a id="omnibench"></a>🌐 RS-OmniBench Dataset
- We have released sample examples for the **53 subtasks**, 
which are available in the <a href="./dataset">`dataset`</a> directory. 
The complete dataset will be made publicly available immediately upon the acceptance of the paper. 
- The key fields of our dataset's TSV files are outlined below as an example. 
This provides a clear overview of the structure and contents of the data.

| **Field Name**     | **Description**                                                                                                                         | **Example**                                                 |
|--------------------|-----------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------|
| `id`               | Unique identifier for each entry in the dataset.                                                                                        | `bb4be5f8-c472-560f-a14b-87dcd2f0aeb6`                      |
| `question`         | Content of the question.                                                                                                                 | `-`                                                         |
| `image`            | Base64-encoded binary content of the image.                                                                                              | `iVBORw0KGgoAAAANSUhEUgAA...` (base64 string)              |
| `A`, `B`, `C`, `D` | Corresponding choices and their content.                                                                                               | `-`                                                         |
| `answer`           | The correct answer for the question.                                                                                                    | `A`                                                         |
| `level`            | Corresponding task level: `image_level`, `region_level`, or `pixel_level`.                                                                | `image_level`                                               |
| `core_task`        | Meta-task name.                                                                                                                         | `image_captioning`                                          |
| `type`             | Task type: `image_wide` or `region_specific`.                                                                                             | `region_specific`                                           |
| `subtask`          | Subtask name.                                                                                                                           | `RS-D-IC`                                                   |
| `category`         | A combination of `level\|core_task\|type\|subtask`.                                                                                         | `image_level\|image_captioning\|region_specific\|RS-D-IC`     |
| `negative`         | For `image_wide` type, `negative=-1`. For `region_specific` type, `negative=True/False` representing whether it's a negative sample.      | `False`                                                     |
| `pre_answer`       | `pre_answer` is valid only when `negative=True`, representing the incorrect global distractor choice as described in the paper.           | `D`                                                         |

## <a id="quickstart"></a>🚀 Quick Start
**📰 Important Notice:**  
The full code and reproduction workflow will be made publicly available immediately upon the acceptance of the paper. Stay tuned for updates!

[//]: # (To get started quickly, follow these steps:)

[//]: # ()
[//]: # (1. Clone the repository)

[//]: # (2. Install the necessary dependencies)

[//]: # (3. Run the example script)

#### 🔝 [Back to Top](#top)