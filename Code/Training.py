import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.utils.data as data
import matplotlib.pyplot as plt
from functions import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import itertools
import warnings
from multiprocessing import freeze_support
warnings.filterwarnings("ignore", category=DeprecationWarning)
def train(log_interval, model, device, train_loader, optimizer, epoch):
    con_encoder, rnn_decoder = model
    con_encoder.train()
    rnn_decoder.train()
    losses = []
    scores = []
    N_count = 0  
    for batch_idx, (X, y) in enumerate(train_loader):
        X, y = X.to(device), y.to(device).view(-1, )

        N_count += X.size(0)

        optimizer.zero_grad()
        output = rnn_decoder(con_encoder(X))   

        loss = F.cross_entropy(output, y)
        losses.append(loss.item())

        # to compute accuracy
        y_pred = torch.max(output, 1)[1]  
        step_score = accuracy_score(y.cpu().data.squeeze().numpy(), y_pred.cpu().data.squeeze().numpy())
        scores.append(step_score)       

        loss.backward()
        optimizer.step()
        if (batch_idx + 1) % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accu: {:.2f}%'.format(
                epoch + 1, N_count, len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader), loss.item(), 100 * step_score))

    return losses, scores


def validation(model, device, optimizer, valid_loader):
    con_encoder, rnn_decoder = model
    con_encoder.eval()
    rnn_decoder.eval()

    test_loss = 0
    all_y = []
    all_y_pred = []
    with torch.no_grad():
        for X, y in valid_loader:
            X, y = X.to(device), y.to(device).view(-1, )

            output = rnn_decoder(con_encoder(X))

            loss = F.cross_entropy(output, y, reduction='sum')
            test_loss += loss.item()              
            y_pred = output.max(1, keepdim=True)[1] 
            all_y.extend(y)
            all_y_pred.extend(y_pred)

    test_loss /= len(valid_loader.dataset)

    all_y = torch.stack(all_y, dim=0)
    all_y_pred = torch.stack(all_y_pred, dim=0)
    test_score = accuracy_score(all_y.cpu().data.squeeze().numpy(), all_y_pred.cpu().data.squeeze().numpy())

    # show information
    print('\nTest set ({:d} samples): Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(len(all_y), test_loss, 100* test_score))

    if (epoch+1) % 5 == 0:
        torch.save(con_encoder.state_dict(), os.path.join(save_model_path, 'con_encoder_epoch{}.pth'.format(epoch + 1)))  
        torch.save(rnn_decoder.state_dict(), os.path.join(save_model_path, 'rnn_decoder_epoch{}.pth'.format(epoch + 1)))  
        torch.save(optimizer.state_dict(), os.path.join(save_model_path, 'optimizer_epoch{}.pth'.format(epoch + 1)))     
        print("Epoch {} model saved!".format(epoch + 1))

    return test_loss, test_score



def plot_confusion_matrix(con_mat, labels, title='Confusion Matrix', cmap=plt.cm.get_cmap('Blues'), normalize=False):
    plt.imshow(con_mat, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    marks = np.arange(len(labels))
    nlabels = []
    for k in range(len(con_mat)):
        n = sum(con_mat[k])
        nlabel = '{0}(n={1})'.format(labels[k],n)
        nlabels.append(nlabel)
    plt.xticks(marks, labels)
    plt.yticks(marks, labels)

    thresh = con_mat.max() / 2.
    if normalize:
        for i, j in itertools.product(range(con_mat.shape[0]), range(con_mat.shape[1])):
            plt.text(j, i, '{:.1f}%'.format(con_mat[i, j] * 100 / n), horizontalalignment="center", color="white" if con_mat[i, j] > thresh else "black")
    else:
        for i, j in itertools.product(range(con_mat.shape[0]), range(con_mat.shape[1])):
            plt.text(j, i, con_mat[i, j], horizontalalignment="center", color="white" if con_mat[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig("Results/NTNU/NTNU_plot_confusion_matrix.png", dpi=600)
    plt.show()
    
    
def test(model, device, optimizer, test_loader, action_names):
    con_encoder, rnn_decoder = model
    con_encoder.eval()
    rnn_decoder.eval()

    test_loss = 0
    all_y = []
    all_y_pred = []
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device).view(-1, )
            output = rnn_decoder(con_encoder(X))
            loss = F.cross_entropy(output, y, reduction='sum')
            test_loss += loss.item()                # sum up batch loss
            y_pred = output.max(1, keepdim=True)[1]  # (y_pred != output) get the index of the max log-probability
            all_y.extend(y.tolist())
            all_y_pred.extend(list(itertools.chain.from_iterable(y_pred.tolist())))
    test_loss /= len(test_loader.dataset)
    print(all_y)
    print(all_y_pred)

    confusion = confusion_matrix(all_y, all_y_pred)
    plot_confusion_matrix(confusion, labels=action_names)
if __name__ == '__main__':
    freeze_support()
    data_path=r"NTU\NTU_Frames"
    save_model_path = "Results/NTNU/Models"
    con_fc_hidden1, con_fc_hidden2 = 1024, 512
    con_embed_dim = 512      
    con_embed_dim2= 1024      
    res_size = 224  
    dropout_p = 0.3          
    RNN_hidden_layers = 3
    RNN_hidden_nodes = 128
    RNN_FC_dim = 256
    k = 26            
    epochs = 50       
    batch_size = 8  
    learning_rate = 0.00004
    log_interval = 2   
    begin_frame, end_frame, skip_frame = 1, 39, 3
    use_cuda = torch.cuda.is_available()                  
    device = torch.device("cuda" if use_cuda else "cpu")   
    params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 8, 'pin_memory': True} if use_cuda else {}
    action_names = os.listdir(r"NTU\NTU_actions")
    le = LabelEncoder()
    le.fit(action_names)
    print('Total classes===', len(action_names))
    action_category = le.transform(action_names).reshape(-1, 1)
    enc = OneHotEncoder()
    enc.fit(action_category)
    actions = []
    fnames = os.listdir(data_path)
    all_names = []
    for f in fnames:
        loc = f.find('(')
        actions.append(f[:loc])
        all_names.append(f)
    all_X_list = all_names                  
    all_y_list = labels2cat(le, actions)   
    all_train_list, test_list, all_train_label, test_label = train_test_split(all_X_list, all_y_list, test_size=0.10, stratify=all_y_list, random_state=22)
    train_list, valid_list, train_label, valid_label = train_test_split(all_train_list, all_train_label, test_size=0.20, stratify=all_train_label, random_state=22)
    transform = transforms.Compose([ transforms.Resize([520, 960]),
                                    transforms.ToTensor()])
    selected_frames = np.arange(begin_frame, end_frame, skip_frame).tolist()
    train_set, valid_set, test_set = Dataset_CRNN(data_path, train_list, train_label, selected_frames, transform=transform), \
                                     Dataset_CRNN(data_path, valid_list, valid_label, selected_frames, transform=transform), \
                                     Dataset_CRNN(data_path, test_list, test_label, selected_frames, transform=transform)
    train_loader = data.DataLoader(train_set, **params)
    valid_loader = data.DataLoader(valid_set, **params)
    test_loader = data.DataLoader(test_set, **params)
    con_encoder = ActconEncoder(fc_hidden1=con_fc_hidden1, fc_hidden2=con_fc_hidden2, drop_p=dropout_p, con_embed_dim=con_embed_dim).to(device)
    rnn_decoder = DecoderRNN(con_embed_dim=con_embed_dim2, h_RNN_layers=RNN_hidden_layers, h_RNN=RNN_hidden_nodes, 
                              h_FC_dim=RNN_FC_dim, drop_p=dropout_p, num_classes=k).to(device)
    
    if torch.cuda.device_count() > 1:
        con_encoder = nn.DataParallel(con_encoder)
        rnn_decoder = nn.DataParallel(rnn_decoder)

        crnn_params = list(con_encoder.module.fc1.parameters()) + list(con_encoder.module.bn1.parameters()) + \
                      list(con_encoder.module.fc2.parameters()) + list(con_encoder.module.bn2.parameters()) + \
                      list(con_encoder.module.fc3.parameters()) + list(rnn_decoder.parameters())

    elif torch.cuda.device_count() == 1:
        print("Using", torch.cuda.device_count(), "GPU!")
        crnn_params = list(con_encoder.fc1.parameters()) + list(con_encoder.bn1.parameters()) + \
                      list(con_encoder.fc2.parameters()) + list(con_encoder.bn2.parameters()) + \
                      list(con_encoder.fc3.parameters()) + list(con_encoder.Con_fc1.parameters()) + list(con_encoder.Con_bn1.parameters()) + \
                      list(con_encoder.Con_fc2.parameters()) + list(con_encoder.Con_bn2.parameters()) + \
                      list(con_encoder.Con_fc3.parameters()) + list(rnn_decoder.parameters())

    optimizer = torch.optim.Adam(crnn_params, lr=learning_rate)


    epoch_train_losses = []
    epoch_train_scores = []
    epoch_test_losses = []
    epoch_test_scores = []

    for epoch in range(epochs):
        print("---------", epoch)
        train_losses, train_scores = train(log_interval, [con_encoder, rnn_decoder], device, train_loader, optimizer, epoch)
        epoch_test_loss, epoch_test_score = validation([con_encoder, rnn_decoder], device, optimizer, valid_loader)

    test([con_encoder, rnn_decoder], device, optimizer, test_loader, action_names)
    
    fig = plt.figure(figsize=(10, 4))
    plt.subplot(121)
    plt.plot(np.arange(1, epochs + 1), A[:, -1])  
    plt.plot(np.arange(1, epochs + 1), C)         
    plt.title("model loss")
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend(['train', 'test'], loc="upper left")
    # 2nd figure
    plt.subplot(122)
    plt.plot(np.arange(1, epochs + 1), B[:, -1])  
    plt.plot(np.arange(1, epochs + 1), D)         
    plt.title("training scores")
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.legend(['train', 'test'], loc="upper left")
    title = "Results/NTNU/fig_UCF101_CRNN.png"
    plt.savefig(title, dpi=600)
    # plt.close(fig)
    plt.show()
