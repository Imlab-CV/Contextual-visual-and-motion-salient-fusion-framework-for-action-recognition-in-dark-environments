import os
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import matplotlib.pyplot as plt
from functions import *
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score
import pandas as pd
import pickle
import cv2
import matplotlib

matplotlib.rcParams['font.family'] = 'Times New Roman'


def find_image(input_path):
    path_dir = input_path
    input_list = os.listdir(path_dir)
    return input_list
    
if __name__ == '__main__':

    data_path = os.path.join( 'Results/NTNU/Test_data/')    
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
    batch_size = 8
    begin_frame, end_frame, skip_frame = 1, 39, 5


    action_names = os.listdir(r"Test\NTU\NTU_Video_Data")
    le = LabelEncoder()
    le.fit(action_names)
    list(le.classes_)
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
    use_cuda = torch.cuda.is_available()                   
    device = torch.device("cuda" if use_cuda else "cpu")   
    params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    transform = transforms.Compose([transforms.Resize([520, 960]),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    selected_frames = np.arange(begin_frame, end_frame, skip_frame).tolist()
    all_data_params = {'batch_size': batch_size, 'shuffle': False, 'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    all_data_loader = data.DataLoader(Dataset_CRNN(data_path, all_X_list, all_y_list, selected_frames, transform=transform), **all_data_params)


    con_encoder = ActconEncoder(fc_hidden1=con_fc_hidden1, fc_hidden2=con_fc_hidden2, drop_p=dropout_p, con_embed_dim=con_embed_dim).to(device)
    rnn_decoder = DecoderRNN(con_embed_dim=con_embed_dim2, h_RNN_layers=RNN_hidden_layers, h_RNN=RNN_hidden_nodes, 
                             h_FC_dim=RNN_FC_dim, drop_p=dropout_p, num_classes=k).to(device)

    con_encoder.load_state_dict(torch.load(os.path.join(save_model_path, 'con_encoder.pth')))
    rnn_decoder.load_state_dict(torch.load(os.path.join(save_model_path, 'rnn_decoder.pth')))
    print('CRNN model reloaded!')
    from torchinfo import summary
    summary(rnn_decoder)
    print('Predicting all {} videos:'.format(len(all_data_loader.dataset)))
    all_y_pred = CRNN_final_prediction([con_encoder, rnn_decoder], device, all_data_loader)
    
    path = data_path
    action_names = os.listdir(r"Test\NTU\NTU_Video_Data")
    all_file = find_image(path)
    print(all_file)
    count = 0
    os.makedirs("Results/NTNU/Prediction/", exist_ok=True)
    
    for file in all_file:
        all_image = find_image(path+file)
        os.makedirs("Results/NTNU/Prediction/"+file, exist_ok=True)
        for image in all_image:
            img = cv2.imread(path+file+"/"+image, cv2.IMREAD_COLOR)
            
            cv2.putText(img, action_names[all_y_pred[count]],
                        (100,100), cv2.FONT_HERSHEY_SIMPLEX,
                        1.1, (0, 0, 255), 2)    
            
            cv2.imwrite("Results/NTNU/Prediction/"+file+"/"+image, img)
            
        count += 1
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    




