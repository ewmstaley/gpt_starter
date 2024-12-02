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

'''
Run this script on its own to train a tokenizer. During training, you can
create an instance of this class and load in the trained tokenization with
the load() method.

For module 6 onward we are using huggingface tokenizers in the interest of speed.
These are heavily optimized to tokenize large amounts of text quickly.

This tokenizer is similar to the tokenizer we trained in module 2, except:
- Uses Ġ to denote space (if you look in the outputs from saving)
- Byte-level BPE
- does a better job of preserving whitespace sequences

'''

from transformers import AutoTokenizer

class HFTokenizer():

	def __init__(self):
		self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
		self.tokenizer.eos_token = "<|endoftext|>"

	def train(self, datafile):
		self.tokenizer = self.tokenizer.train_new_from_iterator(
			open(datafile, "r").readlines(), 
			10000,
			limit_alphabet=500,
		)
		self.tokenizer.save_pretrained("./hftokenizer/")

	def load(self):
		self.tokenizer = AutoTokenizer.from_pretrained("./hftokenizer/")

	# string to token_ids
	def encode(self, string):
		return self.tokenizer(string)["input_ids"]

	# token_ids to string
	def decode(self, list_of_ids):
		return self.tokenizer.decode(list_of_ids)





if __name__ == "__main__":

	tokenizer = HFTokenizer()
	tokenizer.train("./data.txt")