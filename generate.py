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

import numpy as np
import torch
from tqdm import tqdm
from gpt import GPTModel
from hftokenizer import HFTokenizer
from sampler import Sampler

'''
Example of using Sampler with our own GPT model trained on wikipedia data.
'''

# make sampling algorithm
samp = Sampler(top_p=0.9, frequency_penalty=1.2)

# load our model, make sure these settings match whatever we trained
model = GPTModel(d_model=512, n_heads=16, layers=8, vocab_size=10000, max_seq_len=256)
model.load_state_dict(torch.load("./model_weights.pt"))
model.eval()

# make tokenizer
tokenizer = HFTokenizer()
tokenizer.load()

# =================================================================

# some starting text
intial_text = "The Godfather is a film about"
token_ids = tokenizer.encode(intial_text)
token_ids = torch.tensor([token_ids])

# generate N more tokens. We are not using kv cache so this may be slow.
for i in tqdm(range(100)):

	# pass tokens through the model to get logits
	output = model(token_ids)[0,-1,:]

	# sample from the logits, take away batch dim
	token_ids_np = token_ids.data.cpu().numpy()[0]
	tok = samp(output.data.cpu().numpy(), token_ids_np)

	# add the resulting token id to our list
	token_ids_np = np.append(token_ids_np, tok)
	token_ids = torch.from_numpy(token_ids_np)

	# add back a batch size of 1
	token_ids = token_ids[None,:]

	# if we generated a stop token, stop!
	if tok == tokenizer.tokenizer.eos_token_id:
		break


token_ids = token_ids.data.cpu().numpy()[0]

# print out resulting ids
print(token_ids)

# print out the decoded text
print(tokenizer.decode(token_ids))