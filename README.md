# Look In Your Minds: Show Chain of Evidence in Iterative Retrieval Augmented Generation

This repository contains the code and models for the paper *"Look In Your Minds: Show Chain of Evidence in Iterative Retrieval Augmented Generation"*.

## Usage
### 1. Obtain Pretrained Models
Download our trained models from https://www.modelscope.cn/anonymou1111/CoE-8B

### 2. Prepare Dataset
Download the CoE-Wiki dataset from https://www.modelscope.cn/datasets/anonymou1111/CoE-Wiki

### 3. Setup Environment
```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
```

### 4. Configure Dataset
Replace the dataset configuration:
```bash
cp path/to/CoE/dataset_info.json data/dataset_info.json
cp path/to/CoE-Wiki/screenshots_test.json data/screenshots_test.json
cp path/to/CoE-Wiki/screenshots_train.json data/screenshots_train.json
cp -r path/to/CoE-Wiki/screenshots_did data/screenshots_did
```

### 5. Run Inference
```bash
chmod +x predict.sh
./predict.sh
```

### 5. Train From Scratch
```bash
chmod +x train.sh
./train.sh
```

## License
This project uses models licensed under https://modelscope.cn/terms and datasets licensed under https://creativecommons.org/licenses/by-sa/4.0/.

## Acknowledgements
Special thanks to the developers of:
- https://github.com/hiyouga/LLaMA-Factory
- https://www.modelscope.cn
- https://huggingface.co
```
