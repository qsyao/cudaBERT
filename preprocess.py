from pytorch_pretrained_bert.tokenization import BertTokenizer
from utils import Tagged_line

tokenlizer = None

'''
    Convert a input line from input_file to a tagged_line:
    record id_line
    process inputs_id, segments_id, mask
    record line_data(raw string in line from input_file to output)

    There is an example to process ./data/example.tsv

    process_line() will be called in engine.py 
'''

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, num_line, line_data, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.num_line = num_line
        self.line_data = line_data
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b

def init_tokenlizer(vocab_file, do_lower_case):
    global tokenizer
    tokenizer = BertTokenizer.from_pretrained(\
                    vocab_file, do_lower_case=do_lower_case)

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def convert_example_to_feature(example, max_seq_length):
    """Loads a data file into a list of `InputBatch`s."""
    tokens_a = tokenizer.tokenize(example.text_a)

    tokens_b = None
    if example.text_b:
        tokens_b = tokenizer.tokenize(example.text_b)

    if tokens_b:
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0:(max_seq_length - 2)]

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0   0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambigiously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    if tokens_b:
        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length


    return  Tagged_line(
                num_line = example.num_line,
                line_data=example.line_data,
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids)

def create_example(line, set_type, index):
    line = line.replace("\0", '').rstrip().split('\t')
    guid = "%s-%s" % (set_type, index)
    text_a = line[0]
    text_b = line[1]

    return  InputExample(index, line_data='\t'.join(line), \
                         guid=guid, text_a=text_a, text_b=text_b)

def process_line(args, line, index):
    example = create_example(line, "dev", index)
    eval_feature = convert_example_to_feature(
        example, args.max_seq_length)
    return eval_feature
