# 准备局部Bench(MME-RealWorld-Lite)
## Step 1 准备数据文件
### Step 1.1 下载数据
```bash
wget https://huggingface.co/datasets/yifanzhang114/MME-RealWorld-Lite/blob/main/data.zip
```
### Step 1.2 解压数据
```bash
unzip data.zip
```

## Step 2 json文件转换
这一步把本地存储好的数据换成通用的推理格式
### Step 2.1 把image字段换成列表
⚠️因为原始数据庞大，MME只在Lite集合上做过测试，Lite集合的处理首先处理了image字段，保证其能够打开。
```bash
# 确保已经确认全局Bench数据的image的路径，然后按照实际情况调整这个py脚本中的三个目录
# NEW_DIR=存储图片的根目录
# ORIGINAL_JSON="MME_RealWorld_Lite.json"  # 可以改
# OUTPUT_JSON="MME-RealWorld_Lite_new.json" # 可以改
python preprocess_image.py
```
❓判断图片路径有没有对上？打开`MME-RealWorld_Lite_new.json`，看看文件中的img路径能不能打开。

### Step 2.2 统一推理json格式
⚠️如下是示例，建议改成绝对路径，存储在你预期的目录下.
```bash
python unify_format.py --input_file MME-RealWorld_new.json --output_file MME-RealWorld_unified.json
```

## Step 3 推理和评测
从这里参照 `推理和评测流程`


# 准备全局Bench（MME-RealWorld）
## Step 1 准备数据文件
这一步把官方准备好的数据存储到本地

### Step 1.1 下载数据
```bash
cd MME-RealWorld-Lite
cd data
hf download yifanzhang114/MME-RealWorld --repo-type dataset --local-dir ./MME-RealWorld-total # 这会在data下创建一个MME-RealWolrd目录
```

### Step 1.2 解压数据
```bash
cd MME-RealWorld-total
ls #这时应该会看到原始数据文件
```

接着，在MME-RealWorld-total下，把如下文件写入unzip_file.sh。
```bash
#!/bin/bash
# Function to process each set of split files
process_files() {
    local part="$1"
    
    # Extract the base name of the file
    local base_name=$(basename "$part" .tar.gz.part_aa)
    
    # Merge the split files into a single archive
    cat "${base_name}".tar.gz.part_* > "${base_name}.tar.gz"
    
    # Extract the merged archive
    tar -xzf "${base_name}.tar.gz"
    
    # Remove the individual split files
    rm -rf "${base_name}".tar.gz.part_*

    rm -rf "${base_name}.tar.gz"
}

export -f process_files

# Find all .tar.gz.part_aa files and process them in parallel
find . -name '*.tar.gz.part_aa' | parallel process_files

# Wait for all background jobs to finish
wait
```

然后执行这个文件
```bash
# 在终端运行
nohup  bash unzip_file.sh >> unfold.log 2>&1 &
```
wyh: 上面是官方给的解压文件的脚本，因为数据量很大，我暂时没有解压过，所以不敢保证这一步的正确性orz

## Step 2 json文件转换
这一步把本地存储好的数据换成通用的推理格式
### Step 2.1 把image字段换成列表
⚠️因为原始数据庞大，MME只在Lite集合上做过测试，Lite集合的处理首先处理了image字段，保证其能够打开。
```bash
# 确保已经确认全局Bench数据的image的路径，然后按照实际情况调整这个py脚本中的三个目录
# NEW_DIR= // 如果我没推断错，应该是在data/MME-Realworld-Total/img下
# ORIGINAL_JSON="MME_RealWorld.json" 
# OUTPUT_JSON="MME-RealWorld_new.json"
python preprocess_image.py
```
❓判断图片路径有没有对上？打开`MME-RealWorld_new.json`看看文件能不能打开。

### Step 2.2 统一推理json格式
⚠️如下是示例，建议改成绝对路径，存储在你预期的目录下.
```bash
python unify_format.py --input_file MME-RealWorld_new.json --output_file MME-RealWorld_unified.json
```

## Step 3 推理和评测
从这里参照 `推理和评测流程`

# 推理和评测流程
## 推理
模型在这一步生成推理结果

```bash
bash inference.sh
```
## 评测
记得修改里面的输入文件和输出文件。
因为MME-RealWorld似乎都是多选题，所以这里暂时没有上LLM来judge.
```bash 
bash judge.sh
```
