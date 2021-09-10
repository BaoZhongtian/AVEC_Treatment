import os
import numpy
import tqdm
import torch
import json
from transformers import BertTokenizer, BertModel

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    embedding = BertModel.from_pretrained('bert-base-uncased')
    embedding.cuda()

    load_path = 'C:/ProjectData/AVEC2017_Treatment/Step1_CopyFiles/'
    save_path = 'C:/ProjectData/AVEC2017_Treatment/Text_BertResult_Part/'
    for part_name in os.listdir(load_path):
        if not os.path.exists(os.path.join(save_path, part_name)): os.makedirs(os.path.join(save_path, part_name))
        for fold_name in tqdm.tqdm(os.listdir(os.path.join(load_path, part_name))[2::3]):
            transcript = numpy.genfromtxt(
                fname=os.path.join(load_path, part_name, fold_name, fold_name[:-1] + 'TRANSCRIPT.csv'), dtype=str,
                delimiter='\t')

            total_result = []
            for index in range(1, len(transcript)):
                if transcript[index][2] == 'Ellie': continue
                # print(transcript[index])
                # exit()
                text = transcript[index][-1]
                token_id = tokenizer.encode(text, return_tensors='pt')
                result, _ = embedding(token_id.cuda())
                result = result.squeeze().detach().cpu().numpy().tolist()
                total_result.append(result)
            json.dump(total_result, open(os.path.join(save_path, part_name, fold_name + '.json'), 'w'))
            # exit()

    # total_result = []
    # for sample in text:
    #     if sample == '' or sample == ' ': continue
    #     sample = sample.lower()
    #     total_result.append(model.get_vector(sample))
    # print(total_result)
    # print(numpy.shape(total_result))
