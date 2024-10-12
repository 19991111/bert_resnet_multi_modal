import torch
import torch.nn as nn
import time
import numpy as np
from torchvision import transforms, models, datasets
from model_utils import set_parameter_requires_grad
from sklearn.metrics import recall_score, f1_score,precision_score
import config
from transformers import AutoModel,AutoConfig

class RESNET152(nn.Module):
    def __init__(self,dropout):
        super(RESNET152, self).__init__()
        self.model_ft = models.resnet152(pretrained=config.use_pretrained)
        set_parameter_requires_grad(self.model_ft, config.feature_extracting)
        self.fc = nn.Sequential(nn.Dropout(dropout),
                                nn.Linear(self.model_ft.fc.out_features, 2),
                                nn.ReLU(),
                                nn.Softmax(dim=1))

    def forward(self, inputs):
        resnet152_outputs = self.model_ft(inputs)
        outputs = self.fc(resnet152_outputs)
        return outputs

class BERT(nn.Module):
    def __init__(self,dropout):
        super(BERT, self).__init__()
        self.bert = AutoModel.from_pretrained("bert-base-chinese")
        self.fc = nn.Sequential(nn.Dropout(dropout),
                                nn.Linear(self.bert.config.hidden_size, 2),
                                nn.ReLU(),
                                nn.Softmax(dim=1))
    def forward(self,input_ids,attention_mask,token_type_ids):
        bert_outputs = self.bert(input_ids,attention_mask,token_type_ids)
        bert_pooler_outputs = bert_outputs.last_hidden_state[:, 0, :]
        outputs = self.fc(bert_pooler_outputs)
        return outputs
    
class DECISION():
    def __init__(self, test_loader, bert, resnet, weight,device):
        self.test_loader = test_loader
        self.bert = bert.to(device)
        self.resnet = resnet.to(device)
        self.weight = weight
        self.device = device

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
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                token_type_ids = batch['token_type_ids'].to(self.device)
                img_inputs_list = batch['img_inputs'].to(self.device)
                label = batch['label'].to(self.device)

                img_outputs_list = []
                for i in range(img_inputs_list.size()[0]):
                    img_inputs = img_inputs_list[i]
                    img_outputs = [self.resnet(img).tolist()[0]  for _,img in enumerate(torch.chunk(img_inputs, config.chunk_size, dim=0))]
                    max_inner_list = max(img_outputs, key=lambda x: max(x))
                    img_output = torch.tensor(max_inner_list).to(self.device)
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
            print('Weight: {:.1f}\tAccuracy: {:.4f}'.format(self.weight,acc))
        return acc,precision,recall,f1