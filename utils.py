import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from math import ceil
from multiprocessing import cpu_count
from joblib import Parallel, delayed
from transformers import BertTokenizer
from model import LOTClassModel
from tqdm import tqdm
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

# set up distributed training
def set_up_dist(model,
                dist_port,
                world_size,
                rank):
    dist.init_process_group(
        backend='nccl',
        init_method=f'tcp://localhost:{dist_port}',
        world_size=world_size,
        rank=rank
    )
    # create local model
    model = model.to(rank)
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    return model


# print error message based on CUDA memory error
def cuda_mem_error(err, mode, rank):
    if rank == 0:
        print(err)
        if "CUDA out of memory" in str(err):
            if mode == "eval":
                print(
                    f"Your GPUs can't hold the current batch size for evaluation, try to reduce `--eval_batch_size")
            else:
                print(
                    f"Your GPUs can't hold the current batch size for training, try to reduce `--train_batch_size")
    sys.exit(1)



# create dataset loader
def make_dataloader(rank,
                    data_dict,
                    batch_size,
                    world_size):
    if "labels" in data_dict:
        dataset = TensorDataset(
            data_dict["input_ids"], data_dict["attention_masks"], data_dict["labels"], data_dict['orig_index'])
    else:
        dataset = TensorDataset(
            data_dict["input_ids"], data_dict["attention_masks"], data_dict['orig_index'])
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataset_loader = DataLoader(
        dataset, sampler=sampler, batch_size=batch_size, shuffle=False)
    return dataset_loader


class data_preprocessing():
    # init parameters
    def __init__(self, args):
        self.tokenizer = BertTokenizer.from_pretrained(
            args.pretrained_lm, do_lower_case=True)
        self.vocab = self.tokenizer.get_vocab()
        self.mask_id = self.vocab[self.tokenizer.mask_token]
        self.max_len = args.max_len
        self.num_cpus = min(10, cpu_count() - 1) if cpu_count() > 1 else 1
        self.read_label_names(args.dataset_dir, args.label_names_file)
        self.read_data(args.dataset_dir, args.train_file, args.train_label_file,
                       args.test_file, args.test_label_file)

    # read text corpus and labels from files

    def read_data(self,
                  dataset_dir,
                  train_file,
                  train_label_file,
                  test_file,
                  test_label_file,
                  find_label_name=True,
                  label_name_loader_name="label_name_data.pt"):

        self.train_data, self.label_name_data = self.create_dataset(
            dataset_dir,
            train_file,
            train_label_file,
            "train.pt",
            find_label_name=find_label_name,
            label_name_loader_name=label_name_loader_name,
        )
        if test_file is not None:
            self.test_data = self.create_dataset(
                dataset_dir,
                test_file,
                test_label_file,
                "test.pt"
            )

    # convert a list of strings to token ids

    def encode(self,
               docs):

        encoded_dict = self.tokenizer.batch_encode_plus(docs, add_special_tokens=True, max_length=self.max_len, padding='max_length',
                                                        return_attention_mask=True, truncation=True, return_tensors='pt')
        input_ids = encoded_dict['input_ids']
        attention_masks = encoded_dict['attention_mask']
        return input_ids, attention_masks

    # find label name indices and replace out-of-vocab label names with [MASK]
    def label_name_in_doc(self, doc):
        doc = self.tokenizer.tokenize(doc)
        label_idx = -1 * torch.ones(self.max_len, dtype=torch.long)
        new_doc = []
        wordpcs = []
        idx = 1  # index starts at 1 due to [CLS] token
        for i, wordpc in enumerate(doc):
            wordpcs.append(wordpc[2:] if wordpc.startswith("##") else wordpc)
            if idx >= self.max_len - 1:  # last index will be [SEP] token
                break
            if i == len(doc) - 1 or not doc[i+1].startswith("##"):
                word = ''.join(wordpcs)
                if word in self.label2class:
                    label_idx[idx] = self.label2class[word]
                    # replace label names that are not in tokenizer's vocabulary with the [MASK] token
                    if word not in self.vocab:
                        wordpcs = [self.tokenizer.mask_token]
                new_word = ''.join(wordpcs)
                if new_word != self.tokenizer.unk_token:
                    idx += len(wordpcs)
                    new_doc.append(new_word)
                wordpcs = []
        if (label_idx >= 0).any():
            return ' '.join(new_doc), label_idx
        else:
            return None

    # find label name occurrences in the corpus
    def label_name_occurrence(self, docs):
        text_with_label = []
        label_name_idx = []
        for doc in docs:
            result = self.label_name_in_doc(doc)
            if result is not None:
                text_with_label.append(result[0])
                label_name_idx.append(result[1].unsqueeze(0))
        if len(text_with_label) > 0:
            encoded_dict = self.tokenizer.batch_encode_plus(text_with_label, add_special_tokens=True, max_length=self.max_len,
                                                            padding='max_length', return_attention_mask=True, truncation=True, return_tensors='pt')
            input_ids_with_label_name = encoded_dict['input_ids']
            attention_masks_with_label_name = encoded_dict['attention_mask']
            label_name_idx = torch.cat(label_name_idx, dim=0)
        else:
            input_ids_with_label_name = torch.ones(
                0, self.max_len, dtype=torch.long)
            attention_masks_with_label_name = torch.ones(
                0, self.max_len, dtype=torch.long)
            label_name_idx = torch.ones(0, self.max_len, dtype=torch.long)
        return input_ids_with_label_name, attention_masks_with_label_name, label_name_idx

    # read label names from file
    def read_label_names(self, dataset_dir, label_name_file):
        label_name_file = open(os.path.join(dataset_dir, label_name_file))
        label_names = label_name_file.readlines()
        self.label_name_dict = {i: [word.lower() for word in category_words.strip(
        ).split()] for i, category_words in enumerate(label_names)}
        print(f"Label names used for each class are: {self.label_name_dict}")
        self.label2class = {}
        self.all_label_name_ids = [self.mask_id]
        self.all_label_names = [self.tokenizer.mask_token]
        self.label_names_to_idx = {}
        for class_idx in self.label_name_dict:
            for word in self.label_name_dict[class_idx]:
                if class_idx in self.label_names_to_idx:
                    self.label_names_to_idx[class_idx].append(self.vocab[word])
                else:
                    self.label_names_to_idx[class_idx] = [self.vocab[word]]
                assert word not in self.label2class, f"\"{word}\" used as the label name by multiple classes!"
                self.label2class[word] = class_idx
                if word in self.vocab:
                    self.all_label_name_ids.append(self.vocab[word])
                    self.all_label_names.append(word)

    # convert dataset into tensors

    def create_dataset(self,
                       dataset_dir,
                       text_file,
                       label_file,
                       loader_name,
                       find_label_name=False,
                       label_name_loader_name=None):

        loader_file = os.path.join(dataset_dir, loader_name)
        if os.path.exists(loader_file):
            print(f"Loading encoded texts from {loader_file}")
            data = torch.load(loader_file)
        else:
            print(f"Reading texts from {os.path.join(dataset_dir, text_file)}")
            with open(os.path.join(dataset_dir, text_file), encoding="utf-8") as corpus:
                docs = [doc.strip() for doc in corpus.readlines()]
            print(f"Converting texts into tensors.")
            chunk_size = ceil(len(docs) / self.num_cpus)
            chunks = [docs[x:x+chunk_size]
                      for x in range(0, len(docs), chunk_size)]

            results = Parallel(n_jobs=self.num_cpus)(
                delayed(self.encode)(docs=chunk) for chunk in chunks)
            input_ids = torch.cat([result[0] for result in results])
            attention_masks = torch.cat([result[1] for result in results])
            print(f"Saving encoded texts into {loader_file}")
            if label_file is not None:
                print(
                    f"Reading labels from {os.path.join(dataset_dir, label_file)}")
                truth = open(os.path.join(dataset_dir, label_file))
                labels = [int(label.strip()) for label in truth.readlines()]
                labels = torch.tensor(labels)
                data = {"input_ids": input_ids,
                        "attention_masks": attention_masks,
                        "labels": labels}
            else:
                data = {"input_ids": input_ids,
                        "attention_masks": attention_masks}
            torch.save(data, loader_file)

        if find_label_name:
            loader_file = os.path.join(dataset_dir, label_name_loader_name)
            if os.path.exists(loader_file):
                print(f"Loading texts with label names from {loader_file}")
                label_name_data = torch.load(loader_file)
            else:
                print(
                    f"Reading texts from {os.path.join(dataset_dir, text_file)}")
                with open(os.path.join(dataset_dir, text_file), encoding="utf-8") as corpus:
                    docs = [doc.strip() for doc in corpus.readlines()]
                print("Locating label names in the corpus.")
                chunk_size = ceil(len(docs) / self.num_cpus)
                chunks = [docs[x:x+chunk_size]
                          for x in range(0, len(docs), chunk_size)]
                results = Parallel(n_jobs=self.num_cpus)(
                    delayed(self.label_name_occurrence)(docs=chunk) for chunk in chunks)
                input_ids_with_label_name = torch.cat(
                    [result[0] for result in results])
                attention_masks_with_label_name = torch.cat(
                    [result[1] for result in results])
                label_name_idx = torch.cat([result[2] for result in results])
                assert len(
                    input_ids_with_label_name) > 0, "No label names appear in corpus!"
                label_name_data = {"input_ids": input_ids_with_label_name,
                                   "attention_masks": attention_masks_with_label_name,
                                   "labels": label_name_idx}
                loader_file = os.path.join(dataset_dir, label_name_loader_name)
                print(f"Saving texts with label names into {loader_file}")
                torch.save(label_name_data, loader_file)
            return data, label_name_data
        else:
            return data


PREFIX_DCT = {
    'sentimental': [(u'It is [MASK] !', 3), (u'It is very [MASK] !', 4), (u'Overall, it was [MASK].', 5)
                    (u'It was very [MASK] !', 4), (u'Just [MASK] !', 2), (u' All in all, it was [MASK].', 7)]
    'news': [('[MASK] News:', 1), ('This is [MASK] News:', 3),
             ('It is [MASK] News:', 3), ('That is [MASK] News:', 3), ('[Category: [MASK]]', 4)]
    'general': [('[Category: [MASK]]', 4), ('[Categories: [MASK]]', 4), ('Categories: [MASK]', 3),
                ('Category: [MASK]', 3), ('[MASK]:', 1), ('The following article belongs to [MASK] category:', 6)]
}


class pseudo_labeling_with_prompts():
    def __init__(self,
                 args,
                 model,
                 datasets):
        self.label_names_to_idx = datasets.label_names_to_idx
        self.types_of_category_vocab_size = args.types_of_category_vocab_size
        self.tokenizer = BertTokenizer.from_pretrained(args.pretrained_lm,
                                                       do_lower_case=True)
        self.mask_token_id = vocab[tokenizer.mask_token]
        self.dataset_type = args.type_of_dataset
        self.dataset_name = args.dataset_name
        self.model = model

    def get_pseudo_labels(self,
                          all_category_vocabs,
                          pseudo_labels,
                          probs,
                          original_idx_tabel,
                          return_all_category_probs=False):
        all_category_probs = {}
        for name, category_vocab in all_category_vocabs.items():
            onehot = []
            for key in category_vocab.keys():
                cur_vocab = category_vocab[key]
                map_to_new_index = [
                    original_idx_tabel.index(i) for i in cur_vocab]
                value = torch.mean(
                    probs[:, torch.tensor(map_to_new_index)], dim=1)
                onehot.append(value.unsqueeze(1))
            pseudo_label = torch.argmax(torch.cat(onehot, dim=1), dim=1)
            pseudo_labels[name] = pseudo_labels.get(
                name, []) + [pseudo_label.cpu()]
            all_category_probs[name] = torch.cat(onehot, dim=1)
        if return_all_category_probs:
            return all_category_probs

    def dist_eval(self,
                  rank,
                  model,
                  dist_port,
                  world_size,
                  data_dict,
                  loader_name,
                  temp_dir,
                  batch_size,
                  all_category_vocabs,
                  all_unique_vocabs):
        if rank == 0:
            print('--------eval start------')

        def save_results(c_k,
                         all_hidden_states,
                         pseudo_labels,
                         all_labels,
                         all_sum_of_probs,
                         all_orig_idx,
                         loader_name,
                         rank):
            # concatenation
            c_k = torch.stack(c_k, dim=0).sum(dim=0)
            all_sum_of_probs = torch.stack(all_sum_of_probs, dim=0).sum(dim=0)
            all_hidden_states = torch.cat(all_hidden_states, dim=0)
            pseudo_labels = {key: torch.cat(val).long()
                             for key, val in pseudo_labels.items()}
            all_labels = torch.cat(all_labels).long()
            all_orig_idx = torch.cat(all_orig_idx).long()
            dct = {'all_labels': all_labels, 'all_hidden_states': all_hidden_states,
                   'all_pseudo_labels': pseudo_labels, 'c_k': c_k, 'all_sum_of_probs': all_sum_of_probs,
                   'orig_idx': all_orig_idx}
            # save results
            for name, val in dct.items():
                save_file = os.path.join(
                    temp_dir+f'/{name}', f"{rank}_"+loader_name)
                torch.save({name: val}, save_file)

        # init model
        model = set_up_dist(model,
                            dist_port,
                            world_size,
                            rank)

        # init variables
        c_k, all_hidden_states, all_labels, all_sum_of_probs, all_orig_idx = [], [], [], [], []
        pseudo_labels = OrderedDict()
        # create dataloader
        dataloader = make_dataloader(rank,
                                     data_dict,
                                     batch_size,
                                     world_size)

        model.eval()
        wrap_train_dataset_loader = tqdm(
            dataloader) if rank == 0 else dataloader
        try:
            for idx, batch in enumerate(wrap_train_dataset_loader):
                with torch.no_grad():
                    input_ids = batch[0].to(rank)
                    input_mask = batch[1].to(rank)
                    labels = batch[2].to(rank)
                    orig_idx = batch[3]
                    predictions, last_hidden_states = model(input_ids,
                                                            pred_mode="mlm",
                                                            token_type_ids=None,
                                                            return_hidd_stat=True,
                                                            attention_mask=input_mask)

                    predictions = predictions[:, mask_idx, :].squeeze(0)
                    probs = softmax_layer(predictions)
                    last_hidden_states = last_hidden_states[:,
                                                            mask_idx, :] + last_hidden_states[:, 0, :]
                    feas = torch.cat((last_hidden_states, torch.ones(
                        last_hidden_states.size(0), 1).to(rank)), 1)  # num_instances * (fea_size+1)
                    # num_instances * (fea_size+1)
                    feas = (feas / torch.norm(feas, p=2, dim=1).unsqueeze(-1))

                    selected_probs = probs[:, all_unique_vocabs]
                    selected_probs /= torch.norm(selected_probs,
                                                 p=2, dim=1).unsqueeze(-1)

                    all_sum_of_probs.append(
                        selected_probs.sum(dim=0).unsqueeze(-1).cpu())
                    c_k.append(torch.matmul(selected_probs.t(), feas).cpu())
                    all_orig_idx.append(orig_idx)

                    # generate pseudo labels using pre-defined category vocab
                    self.get_pseudo_labels(all_category_vocabs,
                                           pseudo_labels,
                                           selected_probs,
                                           all_unique_vocabs)

                    all_hidden_states.append(last_hidden_states.cpu())
                    all_labels.append(labels.cpu())
            if rank == 0:
                print('-----start saving-----')
            save_results(c_k,
                         all_hidden_states,
                         pseudo_labels,
                         all_labels,
                         all_sum_of_probs,
                         all_orig_idx,
                         loader_name,
                         rank)

        except RuntimeError as err:
            cuda_mem_error(err, "eval", rank)

    def dist_eval_acc_pseudo_labels(self,
                                    temp_dir):

        category_vocab_infos = torch.load(
            os.path.join(temp_dir, 'category_vocab_infos'))
        all_unique_vocabs = category_vocab_infos['all_unique_vocabs']
        all_category_vocabs = category_vocab_infos['all_category_vocabs']

        # init empty tensors for concatenation
        all_feas = torch.empty(0)
        labels = torch.empty(0)
        orig_idx = torch.empty(0)
        pseudo_labels = {}
        all_c_k = []
        sum_of_probs = []
        # get files under each folder
        c_k_files = sorted(os.listdir(temp_dir+'/c_k'))
        all_hidden_states_files = sorted(
            os.listdir(temp_dir+'/all_hidden_states'))
        all_pseudo_labels_files = sorted(
            os.listdir(temp_dir+'/all_pseudo_labels'))
        all_labels_files = sorted(os.listdir(temp_dir+'/all_labels'))
        all_sum_of_probs_files = sorted(
            os.listdir(temp_dir+'/all_sum_of_probs'))
        all_orig_idx_files = sorted(os.listdir(temp_dir+'/orig_idx'))
        # load datasets
        print('-------loading previous results-------')

        # load all_hidden_states, all_pseudo_labels, and all_labels
        for idx, filename in enumerate(tqdm(all_hidden_states_files)):
            all_hidden_states = torch.load(os.path.join(
                temp_dir+'/all_hidden_states', all_hidden_states_files[idx]), map_location='cpu')
            all_pseudo_labels = torch.load(os.path.join(
                temp_dir+'/all_pseudo_labels', all_pseudo_labels_files[idx]), map_location='cpu')
            all_labels = torch.load(os.path.join(
                temp_dir+'/all_labels', all_labels_files[idx]), map_location='cpu')
            all_orig_idx = torch.load(os.path.join(
                temp_dir+'/orig_idx', all_orig_idx_files[idx]), map_location='cpu')
            all_sum_of_probs = torch.load(os.path.join(
                temp_dir+'/all_sum_of_probs', all_sum_of_probs_files[idx]), map_location='cpu')
            c_k = torch.load(os.path.join(
                temp_dir+'/c_k', c_k_files[idx]), map_location='cpu')

            all_feas = torch.cat(
                [all_feas, all_hidden_states['all_hidden_states']], dim=0)
            pseudo_labels = {key: torch.cat([pseudo_labels.get(key, torch.empty(
                0)), val], dim=0) for key, val in all_pseudo_labels['all_pseudo_labels'].items()}
            labels = torch.cat([labels, all_labels['all_labels']], dim=0)
            orig_idx = torch.cat([orig_idx, all_orig_idx['orig_idx']], dim=0)
            all_c_k.append(c_k['c_k'])
            sum_of_probs.append(all_sum_of_probs['all_sum_of_probs'])

        torch.save(labels, os.path.join(temp_dir, 'labels.pt'))
        torch.save(orig_idx, os.path.join(temp_dir, 'orig_idx.pt'))
        # num_instances * (fea_size+1)
        all_feas = torch.cat((all_feas, torch.ones(all_feas.size(0), 1)), 1)
        # num_instances * (fea_size+1)
        all_feas = (all_feas / torch.norm(all_feas, p=2, dim=1).unsqueeze(-1))
        c_k = torch.stack(all_c_k, dim=0).sum(dim=0)
        sum_of_probs = torch.stack(sum_of_probs, dim=0).sum(dim=0)

        print('-------evaluation before clustering-------')
        for key, pseudo_label in pseudo_labels.items():
            print(f'\n{key} for each category')
            acc1 = (labels == pseudo_label).float().sum() / len(labels)
            print(f'acc: {acc1}')

        print('-------clustering start------')
        # first attain prototype representation(centriods)

        c_k = c_k / (1e-8+sum_of_probs)  # class_nums * fea_size
        # class_nums * fea_size
        c_k_normalize = c_k / torch.norm(c_k, p=2, dim=1).unsqueeze(-1)

        res = torch.empty(0)
        pseudo_labels = {}
        # num_instances * class_nums
        cosine_dis = torch.matmul(all_feas, c_k_normalize.t())

        all_category_probs = self.get_pseudo_labels(all_category_vocabs,
                                                    pseudo_labels,
                                                    cosine_dis,
                                                    all_unique_vocabs,
                                                    return_all_category_probs=True)
        torch.save(all_category_probs, os.path.join(
            temp_dir, 'all_category_probs'))
        pseudo_labels = {key: torch.cat(val).long()
                         for key, val in pseudo_labels.items()}
        print('-------evaluation after clustering-------')
        for key, pseudo_label in pseudo_labels.items():
            os.makedirs(os.path.join(temp_dir, f'pseudo_label_{key}'))
            saved_path = os.path.join(temp_dir, f'pseudo_label_{key}')
            torch.save(pseudo_label, os.path.join(
                saved_path, 'pseudo_labels.pt'))
            print(f'\n{key} for each category')
            acc1 = (labels == pseudo_label).float().sum() / len(labels)
            print(f'acc: {acc1}')

    def eval(self,
             world_size,
             model,
             dist_port,
             data_dict,
             loader_name,
             batch_size,
             all_category_vocabs,
             all_unique_vocabs):

        if not os.path.exists(DATASET):
            os.makedirs(DATASET)

        temp_dir = f'{DATASET}/tmp_{dist_port}'
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
            os.makedirs(os.path.join(temp_dir, 'all_labels'))
            os.makedirs(os.path.join(temp_dir, 'all_pseudo_labels'))
            os.makedirs(os.path.join(temp_dir, 'all_hidden_states'))
            os.makedirs(os.path.join(temp_dir, 'c_k'))
            os.makedirs(os.path.join(temp_dir, 'all_sum_of_probs'))
            os.makedirs(os.path.join(temp_dir, 'orig_idx'))
        torch.save({'all_unique_vocabs': all_unique_vocabs, 'all_category_vocabs': all_category_vocabs},
                   os.path.join(temp_dir, 'category_vocab_infos'))
        mp.spawn(self.dist_eval, nprocs=world_size, args=(model,
                                                          dist_port,
                                                          world_size,
                                                          data_dict,
                                                          loader_name,
                                                          temp_dir,
                                                          batch_size,
                                                          all_category_vocabs,
                                                          all_unique_vocabs))

        def load_train_dataset(self,
                               label_existing=True):
            train_dataset = torch.load(
                f'./datasets/{DATASET}_pet_selected/train.pt')
            return train_dataset

        # build new categorical vocabularies
        def get_categorical_vocab(self,
                                  top_n=5):
            print('--------building categorical vocabularies---------')
            new_category_vocab = {}
            if top_n > 1:
                cos = nn.CosineSimilarity(dim=1, eps=1e-6)
                for key, val in self.label_names_to_idx.items():
                    all_sim = []
                    for i in range(vocab_embed.shape[0]):
                        # print(val[0])
                        input1 = torch.tensor(vocab_embed[val[0]]).unsqueeze(0)
                        input2 = torch.tensor(vocab_embed[i]).unsqueeze(0)
                        all_sim.append(cos(input1, input2))
                    selected_sample = np.argsort(np.array(all_sim))[
                        ::-1][:top_n]
                    new_category_vocab[key] = np.array(selected_sample)
            else:
                for key, val in category_vocab.items():
                    new_category_vocab[key] = np.array(val[:top_n])

            print(new_category_vocab)
            keywords = [self.tokenizer.convert_ids_to_tokens(
                val) for _, val in new_category_vocab.items()]
            print(keywords)
            all_category_vocabs = {}
            all_unique_vocabs = set()
            for c_type in self.types_of_category_vocab_size:
                all_category_vocabs[str(c_type)+'_words'] = get_categorical_vocab(
                    category_vocab, top_n=c_type)

                for key, val in all_category_vocabs[str(c_type)+'_words'].items():
                    all_unique_vocabs = all_unique_vocabs | set(val)

            all_unique_vocabs = sorted(list(all_unique_vocabs))

            return all_category_vocabs, new_category_vocab

    def add_prefix(self,
                   prefix,
                   mask_idx,
                   train_dataset):
        # Add prefix
        prefix_tokens = self.tokenizer.encode_plus(prefix)
        # print(prefix_tokens)
        prefix_input_ids = torch.tensor(
            prefix_tokens['input_ids']).unsqueeze(0)
        prefix_input_ids = prefix_input_ids.repeat(
            train_dataset[label_key_name].shape[0], 1)
        updated_input_ids = torch.cat(
            [prefix_input_ids, train_dataset['input_ids'][:, 1:]], dim=1)[:, :512]
        prefix_attention_mask = torch.tensor(
            prefix_tokens['attention_mask']).unsqueeze(0)
        prefix_attention_mask = prefix_attention_mask.repeat(
            train_dataset[label_key_name].shape[0], 1)
        updated_attention_mask = torch.cat(
            [prefix_attention_mask, train_dataset['attention_masks'][:, 1:]], dim=1)[:, :512]
        updated_input_ids[:, mask_idx] = self.mask_token_id

        # index
        orig_index = torch.range(0, train_dataset[label_key_name].size(0)-1)

        updated_train_dataset = {'input_ids': updated_input_ids, 'attention_masks': updated_attention_mask,
                                 'labels': train_dataset[label_key_name], 'orig_index': orig_index}
        return updated_train_dataset

    def prompt_labeling(self,
                        eval_only=False):
        train_dataset = self.load_train_dataset()
        all_category_vocabs, new_category_vocab = self.get_categorical_vocab()
        selected_prefix = PREFIX_DCT[self.dataset_type]
        for prefix, mask_idx in selected_prefix:
            updated_train_dataset = self.add_prefix(prefix,
                                                    mask_idx,
                                                    train_dataset)
            # eval
            if not eval_only:
                dist_port = torch.randint(10000, 14000, (1,)).item()
                self.eval(world_size=4,
                          model=self.model,
                          dist_port=dist_port,
                          data_dict=updated_train_dataset,
                          loader_name='eval_results',
                          batch_size=32,
                          all_category_vocabs=all_category_vocabs,
                          all_unique_vocabs=all_unique_vocabs)
                temp_dir = f'{self.dataset_name}/tmp_{dist_port}'
                self.dist_eval_acc_pseudo_labels(temp_dir)
            else:
                dist_port = torch.randint(10000, 14000, (1,)).item()
                temp_dir = f'{self.dataset_name}/tmp_{dist_port}'
                self.dist_eval_acc_pseudo_labels(temp_dir)
