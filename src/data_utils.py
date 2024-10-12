import os
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models, datasets
import pandas as pd
import json
import torch
from transformers import BertTokenizer
import random
import shutil
from PIL import Image
data_transforms = {
    # 分成三部分，一部分是训练
    'train': transforms.Compose([transforms.RandomRotation(45), # 随机旋转 -45度到45度之间
                                 transforms.CenterCrop(224), # 从中心处开始裁剪
                                 transforms.RandomHorizontalFlip(p = 0.5), # 50%的概率随机水平翻转
                                 transforms.RandomVerticalFlip(p = 0.5), # 50%的概率随机垂直翻转
                                 transforms.ColorJitter(brightness = 0.2, contrast = 0.1, saturation = 0.1, hue = 0.1), # 参数1为亮度，参数2为对比度，参数3为饱和度，参数4为色相
                                 transforms.RandomGrayscale(p = 0.25), # 25%概率转换为灰度图，三通道RGB。灰度图转换以后也是三个通道，但是只是RGB是一样的
                                 transforms.ToTensor(),
                                 transforms.RandomErasing(p=1,scale = (0.02,0.33),ratio =(0.3,3.3),value = (254/255,0,0)), # 随机遮挡
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # 均值，标准差
                                ]),
    # resize成256 * 256 再选取 中心 224 * 224，然后转化为向量，最后正则化
    'valid': transforms.Compose([transforms.Resize(256),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # 均值和标准差和训练集相同
                                ]),
    'test':transforms.Compose([transforms.Resize(256),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # 均值和标准差和训练集相同
                                ])
}


def read_txt(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
        f.close()
    return text


def dict_dump(id, text, label, img_path, img_num):
    tmp_dict = {}
    tmp_dict["id"] = id
    tmp_dict["text"] = text
    tmp_dict["label"] = label
    tmp_dict["img_path"] = img_path
    tmp_dict["img_num"] = img_num
    with open(f"./dataset/new/{id}.json", "w", encoding="utf-8") as json_file:
        json.dump(tmp_dict, json_file, indent=2, ensure_ascii=False)
        json_file.close()


def load_json(json_file):
    with open(json_file, 'r', encoding="utf-8") as json_file:
        data = json.load(json_file)
        json_file.close()
    return data


def load_data(mode="train"):
    file_dir = f"./dataset/raw/{mode}"
    filenames = os.listdir(file_dir)
    filepaths = [os.path.join(file_dir, name) for name in filenames]
    return [load_json(file) for file in filepaths]

def label_img(img_dir="./dataset/image"):
    #df = pd.read_excel("./text_data/img_label.xlsx")
    #img_label_dict = {row['img_id']: row['label'] for index, row in df.iterrows()}
    for filename in os.listdir(img_dir):
        if filename.endswith(".jpg"):
            img_id = os.path.splitext(filename)[0]
            img_id = filename.split("_")[0]
            #label = img_label_dict[int(img_id)]
            old_name = os.path.join(img_dir, filename)
            new_name = os.path.join(img_dir, f"{img_id}.jpg")
            os.rename(old_name, new_name)


def map_json(img_dir="./dataset/image", text_dir="./dataset/text_test", text_img_path="./dataset/text_img.xlsx"):
    sheet = pd.read_excel(text_img_path)

    img_path_list = []
    img_num_list = []

    text_id_list = sheet['id']
    img_filename_list = sheet['img_id'].tolist()
    img_filename_list = [filename.replace('，', ',') for filename in img_filename_list]
    label_list = sheet['label']

    text_filename_list = [str(name)+".txt" for name in text_id_list]
    text_filepath_list = [os.path.join(
        text_dir, text_filename) for text_filename in text_filename_list]
    text_list = [read_txt(filepath) for filepath in text_filepath_list]

    for imgid in img_filename_list:
        if isinstance(imgid, str):
            img_name_list = imgid.split(",")
            img_name_list = [img+".jpg" for img in img_name_list]
            img_filepath_list = [os.path.join(
                img_dir, img_filename) for img_filename in img_name_list]
        elif isinstance(imgid, int):
            img_filepath_list = [os.path.join(img_dir, str(imgid)+".jpg")]
        img_path_list.append(img_filepath_list)
        img_num_list.append(int(len(img_filepath_list)))

    for index in range(len(text_filename_list)):
        id = text_id_list[index]
        text = text_list[index]
        label = label_list[index]
        img_path = img_path_list[index]
        img_num = img_num_list[index]
        dict_dump(str(id), text, str(label), img_path, str(img_num))


def split_dataset(source_folder, train_ratio, test_ratio, val_ratio, dest_folder):
    files = os.listdir(source_folder)
    random.shuffle(files)

    total_files = len(files)
    train_size = int(total_files * train_ratio)
    test_size = int(total_files * test_ratio)

    train_files = files[:train_size]
    test_files = files[train_size:train_size + test_size]
    val_files = files[train_size + test_size:]

    move_files(train_files, source_folder, dest_folder, 'train')
    move_files(test_files, source_folder, dest_folder, 'test')
    move_files(val_files, source_folder, dest_folder, 'valid')


def move_files(file_list, source_folder, dest_folder, subfolder):
    for file in file_list:
        source_path = os.path.join(source_folder, file)
        dest_path = os.path.join(dest_folder, subfolder, file)
        shutil.move(source_path, dest_path)


def excel_text(text_img_path="./dataset/text_img.xlsx"):
    df = pd.read_excel(text_img_path)
    texts = df["text"].tolist()
    ids = df["id"].tolist()
    for id,text in zip(ids,texts):
        with open(f"./dataset/text_test/{id}.txt","w",encoding="utf-8-sig") as f:
            f.write(text)
            f.close()


class Bert_Dataset(Dataset):
    def __init__(self, texts, labels, max_len):
        self.texts = texts
        self.labels = labels
        self.max_len = max_len
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        token_type_ids = inputs['token_type_ids']
        return {
            'input_ids': torch.tensor(input_ids),
            'attention_mask': torch.tensor(attention_mask),
            'token_type_ids': torch.tensor(token_type_ids),
            'label': torch.tensor(label, dtype=torch.int64)
        }

class MULTI_DATASET(Dataset):
    def __init__(self, texts, labels, max_len,chunk_size, img_path_list, mode="test"):
        self.texts = texts
        self.labels = labels
        self.max_len = max_len
        self.chunk_size = chunk_size
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        self.imgs = []

        for img_paths in img_path_list:
            imgs = [Image.open(img_path) for img_path in img_paths]
            self.imgs.append(imgs)

        self.mode = mode

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        imgs = self.imgs[idx]
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        token_type_ids = inputs['token_type_ids']
        image_tensor_list = [data_transforms[self.mode]
                           (img) for img in imgs]
        padded_image_tensor = torch.zeros((self.chunk_size, 3, 224, 224), dtype=torch.float32)
        padded_image_tensor[:len(image_tensor_list)] = torch.stack(image_tensor_list)
        return {
            'input_ids': torch.tensor(input_ids),
            'attention_mask': torch.tensor(attention_mask),
            'token_type_ids': torch.tensor(token_type_ids),
            'label': torch.tensor(label, dtype=torch.int64),
            'img_inputs': padded_image_tensor
        }

def get_img_dataloader(batch_size,shuffle):
    img_dir = "./dataset/image"
    image_datasets = {x: datasets.ImageFolder(os.path.join(img_dir, x), data_transforms[x])
                      for x in ['train', 'valid', 'test']}
    
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=shuffle)
                   for x in ['train', 'valid', 'test']}

    return dataloaders
    



if __name__ == "__main__":
    source_folder = './dataset/new'
    dest_folder = './dataset/raw'
    #label_img()
    map_json()
    split_dataset(source_folder, 0.8, 0.1, 0.1, dest_folder)
