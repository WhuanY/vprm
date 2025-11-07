## Data Preparation

The MME-RealWorld dataset from [Hugging Face](https://huggingface.co/datasets/yifanzhang114/MME-RealWorld)

For downloading images to disk, you should do the following steps:
1. Move `download_image_to_this_dir.sh` to your desired image directory.
For example, if you want to download the images to `/path/to/MME-RealWorld/data/images`, you should run the following command:
```sh
mkdir -p /path/to/MME-RealWorld/data/images
mv download_image_to_this_dir.sh /path/to/MME-RealWorld/data/images/download_image_to_this_dir.sh
```

2. Run the `download_image_to_this_dir.sh` script to download the images.
```sh
bash /path/to/MME-RealWorld/data/images/download_image_to_this_dir.sh
```
this script will download the images to `/path/to/MME-RealWorld/data/images`.

## Inference

1. Run the `inference.sh` script to run the inference. Note that you should chaneg the image_base_dir variable to you image directory.

## Judge

1. Run the `judge.sh` script to judge the inference results.
```sh
bash judge.sh
```