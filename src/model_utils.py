import torch
import time
from sklearn.metrics import recall_score, f1_score, precision_score


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def device_load():
    train_on_gpu = torch.cuda.is_available()
    if not train_on_gpu:
        print('CUDA is not available.   Training on CPU ...')
    else:
        print('CUDA is available! Training on GPU ...')
        device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    return device


def save_result(result_path, epoch, learning_rate, train_loss, val_loss, val_acc, val_precision, val_recall, val_f1):
    f = open(result_path, 'a+', encoding='utf-8')
    f.writelines('Epoch: {}\tlearningrate: {}\ttrain_loss: {:.3f}\tval_loss: {:.3f}\tacc: {:.3f}\tPrecision: {:.3f}\tRecall: {:.3f}\tF1: {:.3f}'.format(
        epoch, learning_rate, train_loss, val_loss, val_acc, val_precision, val_recall, val_f1))
    f.writelines('\n')


def save_model(file, epoch, model, optimizer):
    checkpoint = {
        'epochID': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(checkpoint, file)
    return checkpoint


def load_model(file, model, optimizer):
    checkpoint = torch.load(file)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    current_epoch = checkpoint['epochID'] + 1
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.cuda()
    return current_epoch, model, optimizer


def load_dict(file, model):
    checkpoint = torch.load(file)
    model.load_state_dict(checkpoint['state_dict'])
    return model

def get_text_outputs(batch,model,device):
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    token_type_ids = batch['token_type_ids'].to(device)
    labels = batch['label'].to(device)
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids)
    return labels,outputs

def get_img_outputs(batch,model,device):
    inputs = batch[0]
    labels = batch[1]
    inputs = inputs.to(device)
    labels = labels.to(device)
    outputs = model(inputs)
    return labels,outputs


def train(mode,model, optimizer, scheduler, criterion, train_loader, val_loader, epochs, patience, model_file, result_file):
    # =================================================初始化===================================================#
    early_stopping_patience = patience
    early_stopping_counter = 0
    best_val_loss = float('inf')

    for epoch in range(epochs):
        scheduler.step()
        train_loss = 0.0
        val_loss = 0.0
        print('Epoch {}/{}'.format(epoch+1, epochs))
        print('-' * 20)
    # =================================================模型训练===================================================#
        model.train()
        for batch in train_loader:
            if mode == "text":
                labels,outputs = get_text_outputs(batch,model,device)
            elif mode == "img":
                labels,outputs = get_img_outputs(batch,model,device)
            optimizer.zero_grad()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()  
        train_loss += loss.item()
        train_loss = train_loss/len(train_loader)
    # =================================================模型验证===================================================#
        val_acc, val_precision, val_recall, val_f1, val_loss = evaluate(mode,
            model, criterion, val_loader, epoch)
        print('Optimizer learning rate : {:.7f}'.format(
            optimizer.param_groups[0]['lr']))
        save_result(result_file, epoch,
                    optimizer.param_groups[0]['lr'], train_loss, val_loss, val_acc, val_precision, val_recall, val_f1)
    # =================================================设置早停====================================================#
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model_file, epoch, model, optimizer)
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= early_stopping_patience:
            print("Early stopping triggered.")
            break


def evaluate(mode,model, criterion, loader, epoch):
    since = time.time()
    model.eval()

    all_preds = []
    all_labels = []

    tmp_loss = 0.0
    total = 0
    correct = 0
    with torch.no_grad():
        for batch in loader:
            if mode == "text":
                labels,outputs = get_text_outputs(batch,model,device)
            elif mode == "img":
                labels,outputs = get_img_outputs(batch,model,device)

            loss = criterion(outputs, labels)
            
        tmp_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
            
        acc = correct / total
        precision = precision_score(all_labels, all_preds, average='macro')
        recall = recall_score(all_labels, all_preds, average='macro')
        f1 = f1_score(all_labels, all_preds, average='macro')
        epoch_loss = tmp_loss / len(loader)

        time_elapsed = time.time() - since
        print('Time elapsed {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Epoch: {}\tValidation Loss: {:.3f}\tAccuracy: {:.3f}\tPrecision: {:.3f}\tRecall: {:.3f}\tF1: {:.3f}'.format(
            epoch, epoch_loss, acc, precision, recall, f1))
        
    return acc, precision, recall, f1, epoch_loss

device = device_load()
