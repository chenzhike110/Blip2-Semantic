import torch
from transformers import BertTokenizer
from lavis.models.blip2_models.Qformer import BertConfig, BertLMHeadModel

import numpy as np
from PIL import Image

if __name__ == "__main__":
    # device = torch.device("cuda")
    # tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", truncation_side='left')
    # tokenizer.add_special_tokens({"bos_token": "[DEC]"})
    # encoder_config = BertConfig.from_pretrained("bert-base-uncased")
    # encoder_config.query_length = 32
    # Qformer = BertLMHeadModel.from_pretrained(
    #     "bert-base-uncased", config=encoder_config
    # ).to(device)
    # query_tokens = torch.nn.Parameter(
    #     torch.zeros(1, 32, encoder_config.hidden_size)
    # ).to(device)
    # # query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
    # prompt = "Question: "
    # input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    # outputs = Qformer.greedy_search(input_ids, do_sample=False, max_length=32, query_embeds=query_tokens)
    # result = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    # print(result)
    # data = np.load("/data2/czk/MVNet/dataset_merge/dataset_all_40995.npy")
    # np.save("trg_image.npy", data[4])
    data = np.load("src_img.npy")[32]
    img = Image.fromarray(data)
    img.save("1.jpg")