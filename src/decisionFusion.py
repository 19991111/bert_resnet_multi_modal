import torch
import torch.nn as nn
import time
import os
import numpy as np
from torch.utils.data import DataLoader
from data_utils import MULTI_DATASET, load_data
from function import load_dict,device
import config
from transformers import BertModel
from Resnet import RESNET152
from sklearn.metrics import confusion_matrix

class BERT(nn.Module):
    def __init__(self,bert_model):
        super(BERT, self).__init__()
        self.bert = bert_model
        self.fc = nn.Sequential(nn.Linear(self.bert.config.hidden_size, 2),
                                nn.ReLU(),
                                nn.Dropout(config.dropout),
                                nn.Softmax(dim=1))
    def forward(self,input_ids,attention_mask,token_type_ids):
        bert_outputs = self.bert(input_ids,attention_mask,token_type_ids)
        bert_pooler_outputs = bert_outputs.pooler_output
        outputs = self.fc(bert_pooler_outputs)
        return outputs


class DECISION():
    def __init__(self, test_loader, bert, resnet, weight):
        self.test_loader = test_loader
        self.bert = bert.to(device)
        self.resnet = resnet.to(device)
        self.weight = weight

    def desionfusion(self):
        since = time.time()
        total = 0
        correct = 0

        all_preds= []
        all_labels = []

        self.bert.eval()
        self.resnet.eval()
        
        with torch.no_grad():
            for batch in self.test_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                token_type_ids = batch['token_type_ids'].to(device)
                img_inputs_list = batch['img_inputs'].to(device)
                label = batch['label'].to(device)

                img_outputs_list = []
                for i in range(img_inputs_list.size()[0]):
                    img_inputs = img_inputs_list[i]
                    img_outputs = [self.resnet(img).tolist()[0]  for _,img in enumerate(torch.chunk(img_inputs, config.chunk_size, dim=0))]
                    max_inner_list = max(img_outputs, key=lambda x: max(x))
                    img_output = torch.tensor(max_inner_list).to(device)
                    img_outputs_list.append(img_output)

                img_outputs = torch.stack(img_outputs_list, dim=0)

                text_outputs = self.bert(
                    input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                )
                
                decision_outputs = torch.add(
                    torch.mul(text_outputs, self.weight), torch.mul(img_outputs, 1 - self.weight))
                _, predicted = torch.max(decision_outputs.data, 1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(label.cpu().numpy())

                total += label.size(0)
                correct += (predicted == label).sum().item()

            acc = correct/total
            precision = precision_score(all_labels, all_preds, average='macro')
            recall = recall_score(all_labels, all_preds, average='macro')
            f1 = f1_score(all_labels, all_preds, average='macro')
            time_elapsed = time.time() - since
            print('Time elapsed {:.0f}m {:.0f}s'.format(
                    time_elapsed // 60, time_elapsed % 60))
            print('Weight: {:.1f}\tAccuracy: {:.4f}'.format(weight,acc))
        return acc


if __name__ == "__main__":
    bert_model_path = "./bert_checkpoint.pth"
    resnet_model_path = "./resnet_checkpoint.pth"
    # =================================================随机种子====================================================#
    random_seed = 42
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    # =================================================加载数据====================================================#
    test_texts, test_img_paths, test_labels = [data["text"] for data in load_data(
        mode="all")], [data["img_path"] for data in load_data(mode="all")], [int(data["label"]) for data in load_data(mode="all")]
    test_dataset = MULTI_DATASET(
        test_texts, test_labels, max_len=config.max_len, chunk_size=config.chunk_size,img_path_list=test_img_paths)
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size, shuffle=False)
    # =================================================定义模型====================================================#
    bert_model = BertModel.from_pretrained('BERT', output_hidden_states=True)
    bert = BERT(bert_model=bert_model)
    resnet = RESNET152()

    if os.path.exists(bert_model_path) and os.path.exists(resnet_model_path):
        bert = load_dict(
            bert_model_path, bert)
        resnet = load_dict(
            resnet_model_path, resnet)
        for weight in range(11):
            weight = weight*0.1
            decision = DECISION(test_loader, bert, resnet, weight=weight)
            acc = decision.desionfusion()
    else:
        print("check your checkpoint")
    
    
