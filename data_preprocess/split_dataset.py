import os
import random
import shutil
import yaml
# 源目录A和目标目录B、C


config =yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
source_dir = config['dataset']['txt_folder'] + config['dataset']['name']
target_dir_b = source_dir + '_v'
target_dir_c = source_dir + '_t'
ratio = min_pkt_number = config['preprocess']['split_ratio']
# 获取源目录中的所有子目录
sub_dirs = [os.path.join(source_dir, name) for name in os.listdir(source_dir) if
            os.path.isdir(os.path.join(source_dir, name))]


for sub_dir in sub_dirs:
    sub_dir_name = os.path.basename(sub_dir)
    target_sub_dir_b = os.path.join(target_dir_b, sub_dir_name)
    target_sub_dir_c = os.path.join(target_dir_c, sub_dir_name)
    os.makedirs(target_sub_dir_b, exist_ok=True)
    os.makedirs(target_sub_dir_c, exist_ok=True)

# 移动文件到目标目录
for sub_dir in sub_dirs:
    sub_dir_files = os.listdir(sub_dir)
    sub_dir_file_count = len(sub_dir_files)
    target_sub_dir_b = os.path.join(target_dir_b, os.path.basename(sub_dir))
    target_sub_dir_c = os.path.join(target_dir_c, os.path.basename(sub_dir))

    num_files_b = sub_dir_file_count // ratio
    # num_files_c = 0
    num_files_c = sub_dir_file_count // ratio

    for i in range(num_files_b):
        filename = random.choice(sub_dir_files)
        sub_dir_files.remove(filename)
        source_file = os.path.join(sub_dir, filename)
        dest_file = os.path.join(target_sub_dir_b, filename)
        shutil.move(source_file, dest_file)
    for i in range(num_files_c):
        filename = random.choice(sub_dir_files)
        sub_dir_files.remove(filename)
        source_file = os.path.join(sub_dir, filename)
        dest_file = os.path.join(target_sub_dir_c, filename)
        shutil.move(source_file, dest_file)

