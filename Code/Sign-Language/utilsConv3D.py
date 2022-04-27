import cv2
import numpy as np
import spacy
import os
from torch.utils.data import Dataset
import pandas as pd
import torch
from torchtext.data.metrics import bleu_score
from transformers import AutoTokenizer
from torch.nn.utils.rnn import  pad_sequence
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T


# _________________________________________________________________________________________
OR_PATH = '/home/ubuntu/ASSINGMENTS/SignLanguage'
DATA_DIR = '/home/ubuntu/ASL'
spacy_en = spacy.load('en_core_web_sm')
channels = 3 #Gray scale frames
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
training_videos = [i[:-4] for i in os.listdir('/home/ubuntu/ASL/train_videos')]
test_videos = [i[:-4] for i in os.listdir('/home/ubuntu/ASL/test_videos')]
val_videos = [i[:-4] for i in os.listdir('/home/ubuntu/ASL/val_videos')]

# ____________________________________Video Processing_____________________________________
# For each video get a sequence of frames: array dim (#frames, size1, size2, channels)
# >>>>>read video with pytorch:

# need to pip install av
video = torchvision.io.read_video(DATA_DIR+'/test_videos/'+'-fZc293MpJk_2-1-rgb_front'+'.mp4')


# Helper function to get array

# ____________________________________Helper Functions_____________________________________
# Helper function to clean text
punctuations = """!()-[]{};:'"\,<>./?@#$%^&*_~"""
def cleaner(sentence, punctuations):
    no_punc = ''
    for char in sentence:
        if char not in punctuations:
            no_punc = no_punc + char

    return no_punc

def preprocess(batch):
    """ Process frames"""
    transforms = T.Compose(
        [
            T.Resize((100,100)),
            T.ConvertImageDtype(torch.float32),
            T.Normalize(mean=0.5, std=0.5),
        ]
    )
    batch = transforms(batch)
    return batch


transforms = [T.Resize((100,100)), T.ConvertImageDtype(torch.float32)]
frame_transform = T.Compose(transforms)


#  ____________________________________Vocabulary_____________________________________
# Idea obtained from : https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/Basics/custom_dataset_txt/loader_customtext.py

class Vocabulary:
    def __init__(self, annotations_path=OR_PATH+'/how2sign_realigned_train 2.csv'):
        """freq_threshold: frequency of words to build vocabulary
        itos: index to string
        stoi: string to index"""

        self.itos ={0:"<PAD>", 1:"<SOS>", 2:"<EOS>", 3:"<UNK>"}
        self.stoi = {"<PAD>":0, "<SOS>":1, "<EOS>":2, "<UNK>": 3}
        self.dataframe = pd.read_csv(annotations_path)

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer_eng(text):
        return [tok.text.lower() for tok in spacy_en.tokenizer(text)]

    def build_vocabulary(self):
        self.dataframe['clean_text'] = self.dataframe['SENTENCE'].apply(lambda x: cleaner(x,punctuations))
        sentences_list = self.dataframe['clean_text'].tolist()
        tokens_list = [self.tokenizer_eng(i) for i in sentences_list]
        idx = 4  # 3:<unk>
        for sentence in tokens_list:
            for token in sentence:
                if token not in self.stoi:
                    self.itos[idx] = token
                    self.stoi[token] = idx
                    idx += 1
                else:
                    pass
    def numericalize(self, text):
        tokenized_text = self.tokenizer_eng(text)
        return [
            self.stoi[token] if token in self.stoi else self.stoi['<UNK>']
            for token in tokenized_text
        ]

# ____________________________________Data Loader_____________________________________
class signvideosDataset(Dataset):
    def __init__(self, csv_file, root_dir, keyword, transform=None):
        # read entire annotations files to build vocabulary:
        self.annotations = pd.read_csv(csv_file)
        self.annotations['clean text'] = self.annotations['SENTENCE'].apply(lambda x: cleaner(x, punctuations))
        self.vocab = Vocabulary()
        self.vocab.build_vocabulary()
        self.keyword = keyword

        # Tet with 100 videos for training, validating and testing
        if self.keyword == "train":
            self.annotations = self.annotations[self.annotations['SENTENCE_NAME'].isin(training_videos)]
            self.annotations = self.annotations.sample(10)
        elif self.keyword == "test":
            self.annotations = self.annotations[self.annotations['SENTENCE_NAME'].isin(test_videos)]
            self.annotations = self.annotations.sample(10)
        elif self.keyword == 'val':
            self.annotations = self.annotations[self.annotations['SENTENCE_NAME'].isin(val_videos)]
            self.annotations = self.annotations.sample(10)
        else:
            print('PLEASE SPECIFY KEYWORD')
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):

        #Read path to process with processing function:
        video_path = os.path.join(self.root_dir, self.annotations.iloc[index, 3])
        #coordinates = torch.from_numpy(process_video(video_path+'.mp4')).float()
        vid, _, _ = torchvision.io.read_video(video_path+'.mp4')

        #video = torch.stack(video_frames, 0)
        if self.transform:
            # Size (#frames, H, W, C)
            frames_processed = self.transform(vid.view(vid.shape[0], vid.shape[-1], vid.shape[1], vid.shape[2]))
            # Size (#frames, C, h,w)
        else:
        # input shape: (N, Cin, Din, Hin, Win)
            frames_processed = preprocess(torch.stack([f for f in vid]))
        #frames_processed = frames_processed.view(frames_processed.shape[1], frames_processed.shape[-1], frames_processed.shape[0], frames_processed.shape[2], frames_processed.shape[3])

        # Reads sentence to process with tokenizer
        y_label = self.annotations.iloc[index, 6]
        numericalized_caption = [self.vocab.stoi["<SOS>"]]
        numericalized_caption += self.vocab.numericalize(y_label) #word->index
        numericalized_caption.append(self.vocab.stoi["<EOS>"])

        return frames_processed , torch.tensor(numericalized_caption)


class collate_batch:
    def __init__(self, pad_idx, frames_idx, device):
        self.pad_idx = pad_idx
        self.frames_ids = frames_idx
        self.device = device

    def __call__(self, batch):
        self.text_list = []
        self.frames_list = []
        for (_frames, _text) in batch:
            self.text_list.append(_text)
            self.frames_list.append(_frames)

        self.text_list = pad_sequence(self.text_list, batch_first=False, padding_value=self.pad_idx)
        self.frames_list = pad_sequence(self.frames_list, batch_first=False, padding_value=self.frames_ids)
        return self.frames_list, self.text_list

def get_loader(
        csv_file,
        root_dir,
        keyword,
        batch_size = 1,
        transform=frame_transform
):
    dataset = signvideosDataset(csv_file, root_dir, keyword, transform=transform)
    pad_idx = dataset.vocab.stoi['<PAD>']
    frames_ids = 1.0
    loader = DataLoader(dataset=dataset,
        batch_size= batch_size,
        collate_fn = collate_batch(pad_idx = pad_idx, frames_idx=frames_ids, device=device), num_workers=8)
    if keyword == 'train':
        return loader, dataset
    else:
        return loader

#batch_size = 1
#loader, dataset = get_loader(OR_PATH+'/how2sign_realigned_train 2.csv', root_dir=DATA_DIR+"/train_videos/", keyword='train', batch_size=batch_size, transform=frame_transform)
#for batch_idx, (inputs, labels) in enumerate(loader):

#    print(f'Batch number {batch_idx} \n Inputs Shape {inputs.shape} \n Labels Shape {labels.shape}')


