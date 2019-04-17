import torch
import time
from pytorch_pretrained_bert import BertTokenizer, BertForSequenceClassification
import torch.optim as optim

def seq_to_100(words, seq_length):
    length = ( len(words) - 1 )
    total = length - 1
    end_word  = words[-1]
    words = words[:-1]
    while total < seq_length - 2:
        total = total + 1
        words.append(words[total%length])
    words.append(end_word)
    return words

def change_inputs(inputs):
    inputs[0][0] += 1
    return inputs

def batch_inputs(input, batchsize):
    ret = []
    for i in range(batchsize):
        ret.append(input)
    return ret

seq_length = 11
batchsize = 1
Iters = 1

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenized input
text = "Who was Jim Henson ? Jim Henson was a puppeteer"
tokenized_text = tokenizer.tokenize(text)
masked_index = 6
tokenized_text[masked_index] = '[MASK]'
assert tokenized_text == ['who', 'was', 'jim', 'henson', '?', 'jim', '[MASK]', 'was', 'a', 'puppet', '##eer']
segments_ids = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
attention_mask = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
next_sentence_label = [1]

tokenized_text = seq_to_100(tokenized_text, seq_length)
segments_ids = seq_to_100(segments_ids, seq_length)
attention_mask = seq_to_100(attention_mask, seq_length)

indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

indexed_tokens = batch_inputs(indexed_tokens, batchsize)
segments_ids = batch_inputs(segments_ids, batchsize)
attention_mask = batch_inputs(attention_mask, batchsize)
next_sentence_label = batch_inputs(next_sentence_label, batchsize)

tokens_tensor = torch.tensor(indexed_tokens)
segments_tensors = torch.tensor(segments_ids)
attention_mask = torch.tensor(attention_mask)
next_sentence_label = torch.tensor(next_sentence_label)

load_start = time.time()
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
model.eval()

# for layer in model.modules():
#    if isinstance(layer, torch.nn.Linear):
#         print(layer)
#         print(layer.weight)
#         print(layer.bias)

print(tokens_tensor)
print(segments_tensors)
print(attention_mask)

gpu = 1
if gpu:
    model = model.cuda()
    tokens_tensor = tokens_tensor.cuda()
    segments_tensors = segments_tensors.cuda()
    attention_mask = attention_mask.cuda()
    next_sentence_label = next_sentence_label.cuda()

for i in range(Iters):
    #tokens_tensor[0][0] += 1
    _ = model(tokens_tensor, segments_tensors, attention_mask, next_sentence_label)
    # import ipdb; ipdb.set_trace()

# optimizer = optim.SGD(model.parameters(), lr=0.0000001, momentum=0.9)
# for i in range(Iters):
#     optimizer.zero_grad()
#     next_sentence_loss = model(tokens_tensor, segments_tensors, attention_mask, next_sentence_label)
#     next_sentence_loss.backward()
#     optimizer.step()
