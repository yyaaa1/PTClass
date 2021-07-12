import torch
import argparse 
import os
os.chdir('/home/yangyi/PTClass')

import utils
#from training import trainer


def _parse_args():
    parser = argparse.ArgumentParser("flags for train PTClass")

    parser.add_argument(
        "--type_of_dataset", type=str, choices=['sentimental', 'news', 'general'], help="type of the dataset. \
        Choose from sentimental, news, general"
    )

    parser.add_argument(
        "--pretrained_lm", default='bert-base-uncased', type=str, help="either name of the pretrained model or a path"
    )
    # configs of dataset 
    parser.add_argument(
        '--dataset_name', default='imdb',help='name of the dataset'
    )
    parser.add_argument(
        '--dataset_dir', default='datasets/imdb/',help='dataset directory'
    )
    parser.add_argument(
        '--label_names_file', default='label_names.txt',help='file containing label names (under dataset directory)'
    )
    parser.add_argument(
        '--train_file', default='train.txt',help='unlabeled text corpus for training (under dataset directory); one document per line'
    )
    parser.add_argument(
        '--train_label_file', default='train_labels.txt',help='train corpus ground truth label; if provided, model will be evaluated during self-training'
    )
    parser.add_argument(
        '--test_file', default='test.txt',help='test corpus to conduct model predictions (under dataset directory); one document per line'
    )
    parser.add_argument(
        '--test_label_file', default='test_labels.txt',help='test corpus ground truth label; if provided, model will be evaluated during self-training'
    )
    parser.add_argument(
        '--max_len', type=int, default=512,help='length that documents are padded/truncated to'
    )
    # configs of pseudo labeling
    parser.add_argument(
        '--types_of_category_vocab_size', nargs='+', type=int, default=[20,15,10,5,1], help="selected category vocab sizes"
    )
    return parser.parse_args()


def main(args):
    # [1] data-preprocessing and determining the current scenario (LNA or LNE)
    datasets = utils.data_preprocessing(args)



    # [2] use prompt-based pseudo-labeling method to generate pseudo labels for the whole dataset
    model = LOTClassModel.from_pretrained(args.pretrained_lm,
                                          output_attentions=False,
                                          output_hidden_states=False,
                                          num_labels=self.num_class)

    pseudo_labeling = utils.pseudo_labeling_with_prompts(args,
                                                         model,
                                                         datasets)
    pseudo_labeling_with_prompts.prompt_labeling()
    


    # [5] training
    # trainer.train()


if __name__ == '__main__':
    args = _parse_args()
    main(args)
