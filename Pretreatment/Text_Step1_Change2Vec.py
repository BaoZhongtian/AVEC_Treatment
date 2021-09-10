import os
import numpy
import tqdm
import torch
import json
from transformers import BertTokenizer, BertModel


class BertEmbeddingReveal(BertModel):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, encoder_hidden_states=None, encoder_attention_mask=None):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
        embedding_output = self.embeddings(input_ids=input_ids, position_ids=position_ids,
                                           token_type_ids=token_type_ids, inputs_embeds=inputs_embeds)
        return embedding_output


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    embedding = BertEmbeddingReveal.from_pretrained('bert-base-uncased')

    load_path = 'C:/ProjectData/AVEC2017_Treatment/Step1_CopyFiles/'
    save_path = 'C:/ProjectData/AVEC2017_Treatment/Text_Change2Vec_Part/'
    for part_name in os.listdir(load_path):
        if not os.path.exists(os.path.join(save_path, part_name)): os.makedirs(os.path.join(save_path, part_name))
        for fold_name in tqdm.tqdm(os.listdir(os.path.join(load_path, part_name))[1::3]):
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
                result = embedding(token_id).squeeze(0).detach().numpy().tolist()
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
