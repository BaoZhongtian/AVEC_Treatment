import os
from Loader import loader_audio, loader_text
from Model.HierarchyTCN import HierarchyTCN
from Template import Template_Basic_Train_Method

cuda_flag = True
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

if __name__ == '__main__':
    train_loader, val_loader = loader_audio(appoint_part='PART')
    model = HierarchyTCN(feature_number=40, first_attention_type='Average', second_attention_type='Self')
    Template_Basic_Train_Method(model, train_loader, val_loader,
                                save_path='C:/ProjectData/AVECResult-Self/Audio-Average-Self-TCN/')
