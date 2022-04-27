import random

import torch.nn as nn
import torch.optim as optim
from utils import *
import tqdm


# ______________________________________________Training configuration__________________________________________

num_epochs = 20
learning_rate = 0.001
batch_size = 1

# Model hyper-parameters
load_model = False
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

#_______________________________Paths__________________________________________
OR_PATH = '/home/ubuntu/ASSINGMENTS/SignLanguage'

DATA_DIR = '/home/ubuntu/ASL'

glove_path = '/home/ubuntu/ASSINGMENTS'

#_____________________________________________________Word2Vec___________________________________________
embeddings = {} #vocab store 400k words from glove and embeddings dim 50
with open(glove_path+'/glove.6B.50d.txt', 'rt') as fi:
  full_content = fi.read().strip().split('\n')
for i in range (len(full_content)):
  i_word = full_content[i].split(' ')[0]
  i_embeddings = np.array([[float(val) for val in full_content[i].split(' ')[1:]]])
  embeddings[i_word] = i_embeddings

# Append <SOS>, <EOS>, <PAD>, and <UNK> tokens to embbeddings:
#embedding_array = np.array(embeddings)
pad_emb_np = np.ones((1, embeddings['the'].shape[1])) #<PAD>
pad_sos = np.zeros((1, embeddings['the'].shape[1])) #<SOS>
pad_eos = np.zeros((1, embeddings['the'].shape[1])) #<EOS>
pad_unk = np.mean([j for i,j in enumerate(embeddings.values())], axis=0, keepdims=True) #<UNK> take average
#embedding_array = np.vstack((pad_emb_np,pad_sos,pad_eos,pad_unk, embedding_array)) # embedding array will be sent in model
embeddings['<PAD>'] = pad_emb_np
embeddings['<SOS>'] = pad_sos
embeddings['<EOS>'] = pad_eos
embeddings['<UNK>'] = pad_unk


# __________________________________________________Data________________________________________________________________

loader, dataset = get_loader(OR_PATH+'/how2sign_realigned_train 2.csv', root_dir=DATA_DIR+"/train_videos/",keyword='train', batch_size=batch_size)

val_loader = get_loader(OR_PATH+'/how2sign_realigned_val.csv', root_dir=DATA_DIR+"/val_videos/", keyword='val', batch_size=batch_size)

test_loader = get_loader(OR_PATH+'/how2sign_realigned_test.csv', root_dir=DATA_DIR+"/test_videos/", keyword= 'test', batch_size=1)


# _________________________________Model Definition___________________________________________
class video2vec(nn.Module):

    """This class will take a video and apply Convolution 3D to obtain a vector with
    the best features on the vide. This idea is based on:
    https://github.com/karolzak/conv3d-video-action-recognition/blob/master/python/c3dmodel.py"""

    def __init__(self):
        super(video2vec, self).__init__()
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.norm1 = nn.BatchNorm3d(64)
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.drop1 = nn.Dropout(0.5)

        self.conv2 = nn.Conv3d(64, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.norm2 = nn.BatchNorm3d(32)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.drop2 = nn.Dropout(0.5)

        self.conv3 = nn.Conv3d(32, 16, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.norm3 = nn.BatchNorm3d(16)
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4 = nn.Conv3d(16, 3, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.norm4 = nn.BatchNorm3d(3)
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5 = nn.Conv3d(3, 1, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.global_avg_pool = nn.AdaptiveAvgPool3d((456, 1, 1))
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(456, 1024)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.drop1(self.pool1(self.norm1(self.conv1(x))))
        x = self.drop2(self.pool2(self.norm2(self.conv2(x))))
        x = self.pool3(self.norm3(self.conv3(x)))
        x = self.pool4(self.norm4(self.conv4(x)))
        x = self.conv5(x)
        # Flatten
        x = self.flatten(self.global_avg_pool(x))
        x = self.activation(self.linear(x))
        return x


class Decoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size, num_layers, p):
        # input_size : size english vocabulary. Inputs will be words(one hot encoded) of target
        # output_size: same input_size. Output a tensor of idexes to map to a vocabulary
        # hidden_size: will be the same of the ouput of Conv3D: 256

        super(Decoder, self).__init__()
        self.dropout= nn.Dropout(p)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.embedding_size = embedding_size

        self.embedding = nn.Embedding(input_size, embedding_size) # english word -> embedding
        # embedding gives shape: (1,batch_Size,embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p)
        self.fc = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x, hidden):
        # x = [batch_size]
        # hidden = [n_layers*n_dir, batch_size, hid_dim]
        # cell = [n_layers*n_dir, batch_size, hid_dim]

        #x = x.unsqueeze(0) # x = [1, , batchsize]
        try:
            x = embeddings[dataset.vocab.itos[x.item()]] # embedding = [1, batch_size, emb_dim]
        except:
            x = embeddings['<UNK>'].squeeze(0)  # bug solved with squeezing 1 dimension that was added during embedding

        x = torch.from_numpy(x).float()  # gives word embedding -> vector 50 dimension every word
        x = x.to(device)
        x = x.unsqueeze(0)  # x = [1, 1, dim]

        outputs, (hidden, cell) = self.rnn(x, hidden)
        # outputs = [seq_len, batch_size, hid_dime * n_dir]
        # hidden = [n_layers*n_dir, batch_size, hid_dim]
        # cell = [n_layers * n_dir, batch_size, hid_dim]

        predictions = self.fc(outputs.squeeze(0)) #shape: (1, Batch_Size, length_target_vocab)
        # predictions = [batch_size, output_dim]

        #predictions = predictions.squeeze(0) #shape: (N, length_target_vocab)

        return predictions, hidden, cell

class video2text(nn.Module):
    def __init__(self, encoder, decoder):
        super(video2text, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target, teacher_force_ration=0.5):
        batch_size = source.shape[0] #[N, C, frames, H, W]
        target_len = target.shape[0]
        target_vocab_size = len(dataset.vocab.itos)

        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(device)
        #Encoder part:
        hidden = self.encoder(source)#vector(s) best features from a video Shape (1,1024)
        hidden = hidden.unsqueeze(0)
        hidden = hidden.unsqueeze(0)

        #Decoder part: where translation happens:
        x = target[0, :] #<SOS> token
        for t in range(1, target_len):
            output, hidden, cell = self.decoder(x, hidden)
            outputs[t] = output
            teacher_force = random.random() < teacher_force_ration
            best_guess = output.argmax(1)

            x = target[t] if teacher_force else best_guess

        return outputs

#_____________________________________________ Training hyper-parameters__________________________________________

input_size_decoder = len(dataset.vocab.itos)
output_size = len(dataset.vocab.itos)
decoder_embedding_size = 50
hidden_size = 1024
num_layers = 1
dec_dropout = 0.5

conv_net = video2vec().to(device)

decoder_net = Decoder(
    input_size_decoder,
    decoder_embedding_size,
    hidden_size,
    output_size,
    num_layers,
    dec_dropout
)

model = video2text(conv_net, decoder_net).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
pad_idx = dataset.vocab.stoi['<PAD>']
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

# _________________________________Model Training and Evaluation___________________________
def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    for batch_idx, (inputs, labels) in enumerate(tqdm.tqdm(iterator)):
        #with tqdm(total=len(iterator)) as pbar:
        #inputs = inputs.view(inputs.shape[1], inputs.shape[0], inputs.shape[-1])
        inputs = inputs.view(inputs.shape[1], inputs.shape[2], inputs.shape[0], inputs.shape[-2], inputs.shape[-1])
        inp_data = inputs.to(device)
        target = labels.to(device)
        optimizer.zero_grad()

        output = model(inp_data, target)
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)  # output[1:]
        target = target[1:].view(-1)  # target[1:]
        loss = criterion(output,target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss+= loss.item()
            #pbar.update(1)

    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(tqdm.tqdm(iterator)):
            #with tqdm(total=len(iterator)) as pbar:
            #inputs = torch.reshape(inputs, (inputs.shape[1], inputs.shape[0], inputs.shape[-1]))
            inp_data = inputs.to(device)
            target = labels.to(device)

            output = model(inp_data, target, 0)
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)  # output[1:]
            target = target[1:].view(-1)  # target[1:]
            loss = criterion(output, target)
            epoch_loss += loss.item()
            #pbar.update(1)
    return epoch_loss / len(iterator)


CLIP = 1

best_valid_loss = float('inf')

for epoch in range(num_epochs):
    print(f'Epoch {epoch}  training')
    train_loss = train(model, loader, optimizer, criterion, CLIP)
    print(f'Epoch {epoch}  training loss = {train_loss}')

    print(f'Epoch {epoch}  evaluating')
    valid_loss = evaluate(model, test_loader, criterion)
    print(f'Epoch {epoch}  test loss = {valid_loss}')
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'model_{}.pt'.format('SIGN2TEXT_500'))
        print("The model has been saved!")
        print(f'\tBest Train Loss: {train_loss:.3f}')
        print(f'\tBest Test Loss: {valid_loss:.3f}')
        try:
            sentences, translations = translate_video(model, test_loader, device, dataset) #dataset used for getting vocabulary
            print(f'Original sentences \n {sentences}')
            print(f' Translated Sentence \n {translations}')
        except:
            pass