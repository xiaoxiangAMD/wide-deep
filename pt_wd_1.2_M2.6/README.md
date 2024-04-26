# Content
  
This project provide W&D recommendation model using criteo dataset in CTR prediction.


## Available Datasets
You could download criteo dataset from [Criteo Display Advertising Challenge](https://www.kaggle.com/datasets/mrkmakr/criteo-dataset), then unzip data and store dataset on local disk as ./data/train.txt

## Available Models

| Model | Reference |
|-------|-----------|
| Wide&Deep | [HT Cheng, et al. Wide & Deep Learning for Recommender Systems, 2016.](https://arxiv.org/abs/1606.07792) |

## Steps to run

0. env
    ```bash
    pip install -r requirements.txt
    ```

1. train and evalute the fp32 model as pth
    ```bash
    bash run_model.sh
    ```
2. evalute the fp32 pytorch model
    ```bash
    bash run_inference.sh
    ```
3. produce fp32  onnx using pth
    ```bash
    bash run_to_onnx.sh
    ```
4. test fp32/fp16 onnx's using migraphx
    ```bash
    bash run_migraphx_onnx.sh
    ```
5. test fp32/fp16 onnx's using onnxruntime
    ```bash
    bash run_inference_onnx.sh
    bash run_inference_fp16_onnx.sh
    ```

### Performance
We evaluate the model's auc:

|dlrm  model | auc(%)|
|----|----|
|Pytorch model| 79.09 |
|FP32 onnx model| 79.09 |
|FP16 onnx model| 79.09 |
