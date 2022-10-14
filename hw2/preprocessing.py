# transform the data to swag format
import json
from argparse import ArgumentParser, Namespace
from pathlib import Path
import pandas as pd
import numpy as np


SPLITS = ["train", "valid"]



def main(args):
    path_para = args.context_path
    paragraphs = json.loads(path_para.read_text())
    if args.train_val:
        print("train.json & val.json -> swag_train.csv & swag_validation.csv & squad_train.csv & squad_validation.csv")

        paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
        data = {split: json.loads(path.read_text()) for split, path in paths.items()}
        train_data, val_data = data.values()

        swag_train_data = [[tr["question"], tr["question"], paragraphs[tr["paragraphs"][0]], paragraphs[tr["paragraphs"][1]], \
        paragraphs[tr["paragraphs"][2]], paragraphs[tr["paragraphs"][3]], tr["paragraphs"].index(tr["relevant"])] for tr in train_data]

        # squad_train_data = {"context": [paragraphs[tr["relevant"]] for tr in train_data], "question": [tr["question"] for tr in train_data], "answers": [tr["answer"] for tr in train_data]}
        squad_train_data = [[tr["id"], paragraphs[tr["relevant"]], tr["question"], json.dumps({"text": [tr["answer"]["text"]], "answer_start": [tr["answer"]["start"]]})] for tr in train_data]


        swag_val_data = [[val["question"], val["question"], paragraphs[val["paragraphs"][0]], paragraphs[val["paragraphs"][1]], \
        paragraphs[val["paragraphs"][2]], paragraphs[val["paragraphs"][3]], val["paragraphs"].index(val["relevant"])] for val in val_data]


        # squad_val_data = {"context": [paragraphs[val["relevant"]] for val in val_data], "question": [val["question"] for val in val_data], "answers": [val["answer"] for val in val_data]}
        squad_val_data = [[val["id"], paragraphs[val["relevant"]], val["question"], json.dumps({"text": [val["answer"]["text"]], "answer_start": [val["answer"]["start"]]})] for val in val_data]

        swag_train = pd.DataFrame(np.array(swag_train_data), columns=['sent1', 'sent2', 'ending0', 'ending1', 'ending2', 'ending3', 'label'])
        swag_validation = pd.DataFrame(np.array(swag_val_data), columns=['sent1', 'sent2', 'ending0', 'ending1', 'ending2', 'ending3', 'label'])

        squad_train = pd.DataFrame(np.array(squad_train_data), columns=['id', 'context', 'question', 'answers'])
        squad_validation = pd.DataFrame(np.array(squad_val_data), columns=['id', 'context', 'question', 'answers'])
        # squad_train = json.dumps(squad_train_data)
        # squad_validation = json.dumps(squad_val_data)
        swag_train.to_csv('./data/swag_train.csv', index=False)
        swag_validation.to_csv('./data/swag_validation.csv', index=False)
        
        # with open('./data/squad_train.json', 'w') as f:
        #     json.dump(squad_train_data, f, ensure_ascii=False)
        # with open('./data/squad_validation.json', 'w') as f:
        #     json.dump(squad_val_data, f, ensure_ascii=False)
        squad_train.to_csv('./data/squad_train.csv', index=False)
        squad_validation.to_csv('./data/squad_validation.csv', index=False)

    if args.test:
        print("test.json -> swag_test.csv")

        path = args.test_path
        test_data = json.loads(path.read_text())


        swag_test_data = [[test["question"], test["question"], paragraphs[test["paragraphs"][0]], paragraphs[test["paragraphs"][1]], \
        paragraphs[test["paragraphs"][2]], paragraphs[test["paragraphs"][3]]] for test in test_data]
    

        df_test = pd.DataFrame(np.array(swag_test_data), columns=['sent1', 'sent2', 'ending0', 'ending1', 'ending2', 'ending3'])

        df_test.to_csv(args.data_dir / 'swag_test.csv', index=False)

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./mydata/",
    )
    parser.add_argument(
        "--context_path",
        type=Path,
        help="Path to the context file.",
        required=True
    )
    parser.add_argument(
        "--test_path",
        type=Path,
        help="Directory to the test file.",
        required=True
    )
    parser.add_argument("--train_val", type=bool, default=False)
    parser.add_argument("--test", type=bool, default=True)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)