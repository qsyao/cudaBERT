import argparse

parser = argparse.ArgumentParser()

## Required parameters
parser.add_argument("--input_file_1",
                    default="./data/cuda_10w_output_test.tsv",
                    type=str)

parser.add_argument("--input_file_2",
                    default="./data/pytorch_10w_optimize.tsv",
                    type=str)

args = parser.parse_args()

def read_tsv(input_file):
    """Reads a tab separated value file."""
    with open(input_file, "r", encoding="utf8") as f:
        data = f.readlines()
        lines = []
        for line in data:
            line = line.replace("\0", '').rstrip()
            split_line = line.split('\t')
            lines.append(split_line)
        return lines

def make_false_tsv(false_list):
    with open("./data/preprocess_deepqa_train_10w.tsv", 'r', encoding='utf-8') as f:
        lines = f.readlines()
        false_lines = []
        for index in false_list:
            false_lines.append(lines[index])
    with open("./data/false_inputs.tsv", 'w', encoding='utf-8') as f:
        for line in false_lines:
            f.write(line)

def test():
    with open("./data/preprocess_deepqa_train_10w.tsv", 'r', encoding='utf-8') as f:
        lines = f.readlines()
        false_lines = lines[99200:]
    with open("./data/false_inputs.tsv", 'w', encoding='utf-8') as f:
        for line in false_lines:
            f.write(line)

if __name__ == "__main__":
    test()
    out_a = read_tsv(args.input_file_1)
    out_b = read_tsv(args.input_file_2)
    false_list = []
    assert(len(out_a) == len(out_b))
    total_loss = 0
    max_loss = 0
    for i in range(len(out_a)):
        assert(out_a[i][0] == out_b[i][0])
        loss = abs(float(out_a[i][3]) - float(out_b[i][3]))
        total_loss += loss
        max_loss = max(max_loss, loss)
        if loss > 0.1:
            print("Num {}, Loss {}, out_a {} out_b {}".format(i, loss, out_a[i][3], out_b[i][3]))
            false_list.append(i)
    print("L1_Loss between {} and {} : \n len: {}  Mean_L1_Loss: {}  Max_L1_LOSS: {}".\
            format(args.input_file_1, args.input_file_2, len(out_a), total_loss/len(out_a), max_loss))
    # make_false_tsv(false_list)
