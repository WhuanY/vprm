# image_base_dir 是存放图像的目录
python unify_format_lite.py \
--input_file data/MME-RealWorld-Lite.json \
--output_file data/MME-RealWorld-Lite_unified.json \
--image_base_dir "/mnt/minyingqian/MME-RealWorld-Lite-data/data/imgs" > data/unifyfmt.log 2>&1 &