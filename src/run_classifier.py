import argparse
import os
import torch
import torch.nn as nn
from data_utils import MULTI_DATASET,load_data,Bert_Dataset,get_img_dataloader
from model_utils import device_load,load_model,evaluate,train,load_dict
import config
from transformers import AutoModel
from torchvision import transforms, models
from torch.utils.data import DataLoader
from modeling import BERT,RESNET152,DECISION
import torch.optim as optim
import sys

def decision_making(text_model,text_model_path,img_model,img_model_path):
    test_texts, test_img_paths, test_labels = [data["text"] for data in load_data(
        mode="all")], [data["img_path"] for data in load_data(mode="all")], [int(data["label"]) for data in load_data(mode="all")]
    test_dataset = MULTI_DATASET(
        test_texts, test_labels, max_len=config.max_len, chunk_size=config.chunk_size,img_path_list=test_img_paths)
    test_loader = DataLoader(
    test_dataset, batch_size=config.batch_size, shuffle=False)
    if os.path.exists(text_model_path) and os.path.exists(img_model_path):
        bert = load_dict(
            text_model_path, text_model)
        resnet = load_dict(
            img_model_path, img_model)
    for weight in range(11):
        weight = weight*0.1
        decision = DECISION(test_loader, bert, resnet, weight=weight,device=device)
        acc,precision,recall,f1 = decision.desionfusion()
        print("="*20)
        print(f"weight:{weight}")
        print('Accuracy: {:.3f}\tPrecision: {:.3f}\tRecall: {:.3f}\tF1: {:.3f}'.format(acc, precision, recall, f1))

if __name__ == "__main__":
    random_seed = 42
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode_select", type=str, default="decision")
    parser.add_argument("--patience", type=int, default=config.patience)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--batch_size", type=int, default=config.batch_size)
    parser.add_argument("--epochs", type=int, default=config.epochs)
    parser.add_argument("--dropout", type=float, default=config.dropout)
    parser.add_argument("--weight", type=float, default=0.0)
    args = parser.parse_args()

    mode_select = args.mode_select
    lr = args.lr
    batch_size = args.batch_size
    patience = args.patience
    total_epoch = args.epochs
    dropout = args.dropout
    weight = args.weight

    model_file = f"{mode_select}_checkpoint.pth"
    result_file = f"{mode_select}_result.txt"

    device = device_load()
    # =================================================定义模型====================================================#
    bert = BERT(dropout=dropout)
    resnet = RESNET152(dropout=dropout)
    #=========================================prepare my data================================================#
    if mode_select == "decision":
        decision_making(text_model = bert,
                        text_model_path = "text_checkpoint.pth",
                        img_model = resnet,
                        img_model_path = "img_checkpoint.pth")
        sys.exit()
    elif mode_select == "text":
        train_texts, train_labels = [data["text"] for data in load_data(
        mode="train")], [int(data["label"]) for data in load_data(mode="train")]
        test_texts, test_labels = [data["text"] for data in load_data(
            mode="test")], [int(data["label"]) for data in load_data(mode="test")]
        val_texts, val_labels = [data["text"] for data in load_data(
            mode="valid")], [int(data["label"]) for data in load_data(mode="valid")]
        model = bert
        train_dataset = Bert_Dataset(
        train_texts, train_labels, max_len=config.max_len)
        train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True)

        test_dataset = Bert_Dataset(
            test_texts, test_labels, max_len=config.max_len)
        test_loader = DataLoader(
            test_dataset, batch_size=config.batch_size, shuffle=True)

        val_dataset = Bert_Dataset(val_texts, val_labels, max_len=config.max_len)
        val_loader = DataLoader(
            val_dataset, batch_size=config.batch_size, shuffle=True)
    elif mode_select == "img":
        train_loader=get_img_dataloader(batch_size=config.batch_size,shuffle=True)["train"]
        val_loader = get_img_dataloader(batch_size=config.batch_size,shuffle=True)["valid"]
        test_loader = get_img_dataloader(batch_size=config.batch_size,shuffle=True)["test"]
        model = resnet
     

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    if os.path.exists(model_file):
        current_epoch, model, optimizer = load_model(
            model_file, model, optimizer)
        model.to(device)
        acc, precision, recall, f1, epoch_loss = evaluate(mode_select,
            model, criterion, test_loader, current_epoch)
    else:
        model.to(device)
        train(mode_select,model, optimizer, scheduler, criterion, train_loader, val_loader, total_epoch, patience, model_file, result_file)