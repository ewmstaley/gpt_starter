'''
Copyright © 2024 The Johns Hopkins University Applied Physics Laboratory LLC
 
Permission is hereby granted, free of charge, to any person obtaining a copy 
of this software and associated documentation files (the “Software”), to 
deal in the Software without restriction, including without limitation the 
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or 
sell copies of the Software, and to permit persons to whom the Software is 
furnished to do so, subject to the following conditions:
 
The above copyright notice and this permission notice shall be included in 
all copies or substantial portions of the Software.
 
THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, 
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR 
IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''

import torch
from tqdm import tqdm
import numpy as np
from hftokenizer import HFTokenizer


def construct_dataset(data_txt_file, sequence_length=256):

    # construct tokenizer
    tokenizer = HFTokenizer()
    tokenizer.load()

    # tokenize the text and add <eos> at the env of each sample
    f = open(data_txt_file, "r")
    lines = f.readlines()
    tokenized_samples = []
    for line in tqdm(lines):
        # remove newlines
        line = line.replace("\n", "")
        line += "<|endoftext|>"
        tokenized_samples.append(np.array(tokenizer.encode(line)))

    # pack into sequences of length sequence_length.
    giant_list = np.concatenate(tokenized_samples, axis=0)
    print(giant_list.shape)
    length = (len(giant_list)//sequence_length)*sequence_length
    giant_list = giant_list[:length]
    data = giant_list.reshape((-1, sequence_length))

    # steal the beginning tokens and add to the end of prior sequences
    # this sacrifices one entry
    begs = data[:,0][1:] # all first tokens from second sequence onwards
    begs = np.expand_dims(begs, 1)
    data = data[:-1]
    data = np.concatenate([data, begs], axis=1)

    # some printouts to show its working
    print(data[0])
    print(data.shape)

    # shuffle
    np.random.shuffle(data)

    # save the tokenized and sequenced data
    with open('dataset.npy', 'wb') as f:
        np.save(f, data)



if __name__ == "__main__":
    construct_dataset("./data.txt", 256)