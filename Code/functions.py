import os
import numpy as np
from PIL import Image
from torch.utils import data
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm
from torchvision.models.optical_flow import Raft_Large_Weights
import torchextractor as tx
from torchvision.utils import flow_to_image
import torchvision.transforms as transforms
weights = Raft_Large_Weights.DEFAULT
R_transform = weights.transforms()

import torchvision.models.video.resnet
import timm

transform = transforms.Compose([ transforms.Resize([224, 224])])
def labels2cat(label_encoder, list):
    return label_encoder.transform(list)

def labels2onehot(OneHotEncoder, label_encoder, list):
    return OneHotEncoder.transform(label_encoder.transform(list).reshape(-1, 1)).toarray()

def onehot2labels(label_encoder, y_onehot):
    return label_encoder.inverse_transform(np.where(y_onehot == 1)[1]).tolist()

def cat2labels(label_encoder, y_cat):
    return label_encoder.inverse_transform(y_cat).tolist()

class Dataset_3Dcon(data.Dataset):
    def __init__(self, data_path, folders, labels, frames, transform=None):
        self.data_path = data_path
        self.labels = labels
        self.folders = folders
        self.transform = transform
        self.frames = frames

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.folders)

    def read_images(self, path, selected_folder, use_transform):
        X = []
        for i in self.frames:
            image = Image.open(os.path.join(path, selected_folder, 'image_{:08d}.jpg'.format(i))).convert('L')

            if use_transform is not None:
                image = use_transform(image)

            X.append(image.squeeze_(0))
        X = torch.stack(X, dim=0)

        return X

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        folder = self.folders[index]

        # Load data
        X = self.read_images(self.data_path, folder, self.transform).unsqueeze_(0)  
        y = torch.LongTensor([self.labels[index]])                             
        return X, y


class Dataset_CRNN(data.Dataset):
    "Characterizes a dataset for PyTorch"
    def __init__(self, data_path, folders, labels, frames, transform=None):
        "Initialization"
        self.data_path = data_path
        self.labels = labels
        self.folders = folders
        self.transform = transform
        self.frames = frames

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.folders)

    def read_images(self, path, selected_folder, use_transform):
        X = []

        for i in self.frames:
            image = Image.open(os.path.join(path, selected_folder, '{:08d}.jpg'.format(i)))

            if use_transform is not None:
                image = use_transform(image)

            X.append(image)
        X = torch.stack(X, dim=0)

        return X

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        folder = self.folders[index]

        # Load data
        X = self.read_images(self.data_path, folder, self.transform)     
        y = torch.LongTensor([self.labels[index]])                  

        # print(X.shape)
        return X, y

def Conv3d_final_prediction(model, device, loader):
    model.eval()

    all_y_pred = []
    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(tqdm(loader)):
            # distribute data to device
            X = X.to(device)
            output = model(X)
            y_pred = output.max(1, keepdim=True)[1]  
            all_y_pred.extend(y_pred.cpu().data.squeeze().numpy().tolist())

    return all_y_pred


def CRNN_final_prediction(model, device, loader):
    con_encoder, rnn_decoder = model
    con_encoder.eval()
    rnn_decoder.eval()

    all_y_pred = []
    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(tqdm(loader)):
            # distribute data to device
            X = X.to(device)
            output = rnn_decoder(con_encoder(X))
            y_pred = output.max(1, keepdim=True)[1]  
            all_y_pred.extend(y_pred.cpu().data.squeeze().numpy().tolist())

    return all_y_pred

def conv3D_output_size(img_size, padding, kernel_size, stride):
    # compute output shape of conv3D
    outshape = (np.floor((img_size[0] + 2 * padding[0] - (kernel_size[0] - 1) - 1) / stride[0] + 1).astype(int),
                np.floor((img_size[1] + 2 * padding[1] - (kernel_size[1] - 1) - 1) / stride[1] + 1).astype(int),
                np.floor((img_size[2] + 2 * padding[2] - (kernel_size[2] - 1) - 1) / stride[2] + 1).astype(int))
    return outshape

class ActconEncoder(nn.Module):
    def __init__(self, fc_hidden1=512, fc_hidden2=512, drop_p=0.3, con_embed_dim=300):
        super(ActconEncoder, self).__init__()

        self.fc_hidden1, self.fc_hidden2 = fc_hidden1, fc_hidden2
        self.drop_p = drop_p

        self.swin = timm.create_model("swin_base_patch4_window7_224", pretrained=True)
        self.swin = nn.Sequential(*list(self.swin.children())[:-3])  
        '''
        for param in self.resnet[-1].parameters():
            param.requires_grad = True
        for param in self.resnet[-2].parameters():
            param.requires_grad = True
        for param in self.resnet[-3].parameters():
            param.requires_grad = True'''

        #Contextual_Model_o= torch.hub.load('facebookresearch/deit:main', 'deit_tiny_patch16_224', pretrained=True) 
        #self.Contextual_Model = nn.Sequential(*list(Contextual_Model_o.children())[:-3])
        self.Action_Model = models.optical_flow.raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False)
        #self.Action_Model = tx.Extractor(Action_Model, ["update_block.flow_head"])
        
        self.MPL= nn.MaxPool2d(kernel_size=3, stride=3)
        self.MPL2= nn.MaxPool2d(kernel_size=3, stride=2)
        self.flow_img = nn.AvgPool2d(kernel_size = 3, stride = 1, padding = 0, ceil_mode=False, count_include_pad=True)
        self.GAP= nn.AdaptiveAvgPool2d((5, 5))
        self.fc1 = nn.Linear(110720, fc_hidden1)
        self.Con_fc1 = nn.Linear(1400, fc_hidden1)
        self.Con_bn1 = nn.BatchNorm1d(fc_hidden1, momentum=0.01)
        self.bn1 = nn.BatchNorm1d(fc_hidden1, momentum=0.01)
        self.Con_fc2 = nn.Linear(fc_hidden1, fc_hidden2)
        self.fc2 = nn.Linear(fc_hidden1, fc_hidden2)
        self.bn2 = nn.BatchNorm1d(fc_hidden2, momentum=0.01)
        self.Con_bn2 = nn.BatchNorm1d(fc_hidden2, momentum=0.01)
        self.Con_fc3 = nn.Linear(fc_hidden2, con_embed_dim)
        self.fc3 = nn.Linear(fc_hidden2, con_embed_dim)
        #self.values=0
        
    def forward(self, x_3d):
        con_embed_seq = []
        for t in range(x_3d.size(1)-1):
            # ResNet con
            
            transformed_img = transform(x_3d[:, t, :, :, :])
            #Con_features = self.resnet(transformed_img)  # ResNet
            Con_features = self.swin(transformed_img)
            # mean_activation = Con_features.mean(dim=1, keepdim=True)
            # max_activation = Con_features.max(dim=1, keepdim=True).values
            # Con_features = torch.max(mean_activation, max_activation).squeeze(0)
            with torch.no_grad():
                
                img1_batch, img2_batch= R_transform(x_3d[:, t, :, :, :], x_3d[:, t+1, :, :, :])
                #model_output, x= self.Action_Model(img1_batch, img2_batch )
                x= self.Action_Model(img1_batch, img2_batch )
                predicted_flows = x[-1]
             
                #values = list(x.values())
                #x =torch.tensor(x)
                #x = values[-1][:]

                #x = flow_to_image(x)
         


                #x = x.view(x.size(0), -1)      

            # FC layers
            #x= self.MPL(x)
            #x= x.to(torch.float32)
            #x= self.flow_img(x)
            x=self.MPL(predicted_flows)
            #print(x.shape)
            #Con_x=self.MPL2(Con_features)
            Con_x= self.GAP(Con_features)
     
            Con_x = Con_x.view(x.size(0), -1)
            
            x = x.view(x.size(0), -1)

            
            Con_x = self.Con_bn1(self.Con_fc1(Con_x))
            Con_x = F.relu(Con_x)
            x = self.bn1(self.fc1(x))
            x = F.relu(x)
            
            x = self.bn2(self.fc2(x))
            x = F.relu(x)
            Con_x = self.Con_bn2(self.Con_fc2(Con_x))
            Con_x = F.relu(Con_x)
            x = F.dropout(x, p=self.drop_p, training=self.training)
            Con_x = F.dropout(Con_x, p=self.drop_p, training=self.training)
            x = self.bn2(self.fc3(x))
            x = F.relu(x)
            
            Con_x = self.Con_bn2(self.Con_fc3(Con_x))
            Con_x = F.relu(Con_x)
            x = F.dropout(x, p=self.drop_p, training=self.training)
            Con_x = F.dropout(Con_x, p=self.drop_p, training=self.training)
    
            concatenated = torch.cat((x, Con_x), dim=1)
        
            

            con_embed_seq.append(concatenated)

   
        con_embed_seq = torch.stack(con_embed_seq, dim=0).transpose_(0, 1)

        return con_embed_seq
    
class DecoderRNN(nn.Module):
    def __init__(self, con_embed_dim=300, h_RNN_layers=2, h_RNN=256, h_FC_dim=128, drop_p=0.3, num_classes=50):
        super(DecoderRNN, self).__init__()

        self.RNN_input_size = con_embed_dim
        self.h_RNN_layers = h_RNN_layers   
        self.h_RNN = h_RNN                 
        self.h_FC_dim = h_FC_dim
        self.drop_p = drop_p
        self.num_classes = num_classes

        self.biLSTM1 = nn.LSTM(
            input_size=self.RNN_input_size,
            hidden_size=self.h_RNN,        
            num_layers=1,       
            bidirectional=True,     
            batch_first=True,      
        )

        self.biLSTM2 = nn.LSTM(
            input_size=self.h_RNN*2,   
            hidden_size=self.h_RNN, 
            num_layers=1, 
            bidirectional=True, 
            batch_first=True,)

        self.fc1 = nn.Linear(self.h_RNN*2, self.h_FC_dim)  
        self.fc2 = nn.Linear(self.h_FC_dim, self.num_classes)

    def forward(self, x_RNN):
        
        self.biLSTM1.flatten_parameters()
        x_RNN, _ = self.biLSTM1(x_RNN, None)
        self.biLSTM2.flatten_parameters()
        RNN_out, (h_n, h_c) = self.biLSTM2(x_RNN, None)  
        x = self.fc1(RNN_out[:, -1, :])   
        x = F.relu(x)
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.fc2(x)

        return x



