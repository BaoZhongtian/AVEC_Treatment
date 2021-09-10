import os
from Loader import loader_audio, loader_text
from Model.HierarchyBLSTM import HierarchyBLSTM
from Template import Template_Basic_Train_Method

cuda_flag = True
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

if __name__ == '__main__':
    train_loader, val_loader = loader_audio(appoint_part='Part')
    model = HierarchyBLSTM(feature_number=40, first_attention_type='Vanilla', second_attention_type='Self')
    Template_Basic_Train_Method(model, train_loader, val_loader,
                                save_path='C:/ProjectData/AVECResult-Self/Audio-Vanilla-Self-BLSTM/')
