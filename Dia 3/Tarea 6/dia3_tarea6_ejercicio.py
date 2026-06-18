"""
--------------------------------------------------
DOCUMENTACIÓN DEL SCRIPT: TRADUCCIÓN SECUENCIA A SECUENCIA (SEQ2SEQ) CON LSTM (PYTORCH)
--------------------------------------------------

Descripción general:
Este script implementa un modelo de traducción automática básico tipo Seq2Seq utilizando PyTorch.

El objetivo es traducir frases simples del inglés al francés usando una arquitectura Encoder-Decoder basada en LSTM.

El sistema incluye:
- Construcción de vocabularios manuales
- Tokenización y padding de secuencias
- Modelo Encoder-Decoder con LSTM
- Entrenamiento con teacher forcing
- Inferencia para traducción de frases nuevas

--------------------------------------------------
COMPONENTES PRINCIPALES
--------------------------------------------------

1. Dataset:
   - Conjunto pequeño de frases en inglés y sus equivalentes en francés
   - Dataset sintético de 5 ejemplos para demostración

2. Preprocesamiento:
   - Construcción manual de vocabulario (incluye tokens especiales):
     <PAD>, <SOS>, <EOS>, <UNK>
   - Tokenización palabra por palabra
   - Padding a longitud fija por idioma
   - Conversión a tensores PyTorch

3. Arquitectura del modelo:
   - Encoder (LSTM):
     - Embedding de palabras
     - LSTM multicapa
     - Devuelve hidden state y cell state

   - Decoder (LSTM):
     - Embedding de palabras objetivo
     - LSTM condicionado en estado del encoder
     - Capa lineal para predicción de vocabulario

   - Seq2Seq:
     - Conecta encoder y decoder
     - Implementa teacher forcing durante entrenamiento
     - Genera secuencias token por token en inferencia

4. Entrenamiento:
   - Loss: CrossEntropyLoss con ignore_index para <PAD>
   - Optimizer: Adam
   - Teacher forcing ratio: 0.5
   - Entrenamiento por épocas con DataLoader

5. Inferencia (traducción):
   - Codificación de oración de entrada
   - Generación autoregresiva token a token
   - Decodificación de índices a palabras

--------------------------------------------------
COMPONENTES DEL FLUJO SEQ2SEQ
--------------------------------------------------

1. Encoder:
   - Convierte secuencia de entrada en representación latente
   - Produce hidden state y cell state

2. Decoder:
   - Genera secuencia de salida usando estado del encoder
   - Predice palabra siguiente en cada paso temporal

3. Teacher Forcing:
   - Técnica de entrenamiento donde se usa la palabra real
     como entrada en lugar de la predicción del modelo

--------------------------------------------------
NOTA SOBRE EL DATASET
--------------------------------------------------

- Este ejemplo utiliza solo 5 pares de frases.
- No es un sistema de traducción real, sino una demostración estructural del modelo Seq2Seq.

--------------------------------------------------
FLUJO GENERAL DEL SCRIPT
--------------------------------------------------

1. Definición de frases inglés-francés
2. Construcción de vocabularios
3. Tokenización + padding
4. Creación de Dataset y DataLoader
5. Definición Encoder, Decoder y Seq2Seq
6. Entrenamiento con teacher forcing
7. Traducción de una frase de prueba
"""

# --------------------------------------------------
# IMPORTACIÓN DE LIBRERÍAS
# --------------------------------------------------
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np

# --------------------------------------------------
# DATASET DE EJEMPLO (INGLÉS → FRANCÉS)
# --------------------------------------------------
english_sentences = ["hello", "how are you", "good morning", "thank you", "good night"]
french_sentences = ["bonjour", "comment ça va", "bon matin", "merci", "bonne nuit"]

# --------------------------------------------------
# CONSTRUCCIÓN DE VOCABULARIOS
# --------------------------------------------------
def build_vocab(sentences):
    """
    Construye un vocabulario a partir de una lista de frases.

    Parámetros:
    sentences (list): lista de strings

    Retorno:
    dict: vocabulario palabra → índice
    """
    vocab = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}

    for sentence in sentences:
        for word in sentence.split():
            if word not in vocab:
                vocab[word] = len(vocab)

    return vocab

english_vocab = build_vocab(english_sentences)
french_vocab = build_vocab(french_sentences)

# --------------------------------------------------
# TOKENIZACIÓN + PADDING
# --------------------------------------------------
def tokenize(sentences, vocab, max_len):
    """
    Convierte frases en secuencias de índices y aplica padding.

    Parámetros:
    sentences (list): frases de entrada
    vocab (dict): vocabulario palabra→índice
    max_len (int): longitud máxima de secuencia

    Retorno:
    np.array: secuencias tokenizadas y normalizadas
    """
    tokenized = []

    for sentence in sentences:
        tokens = [vocab.get(word, vocab["<UNK>"]) for word in sentence.split()]
        tokens = [vocab["<SOS>"]] + tokens + [vocab["<EOS>"]]
        tokens += [vocab["<PAD>"]] * (max_len - len(tokens))
        tokenized.append(tokens)

    return np.array(tokenized)

max_len_eng = max(len(sentence.split()) for sentence in english_sentences) + 2
max_len_fr = max(len(sentence.split()) for sentence in french_sentences) + 2

english_data = tokenize(english_sentences, english_vocab, max_len_eng)
french_data = tokenize(french_sentences, french_vocab, max_len_fr)

# --------------------------------------------------
# DATASET PYTORCH
# --------------------------------------------------
class TranslationDataset(Dataset):
    """
    Dataset personalizado para pares de traducción (inglés → francés).
    """

    def __init__(self, src_data, tgt_data):
        self.src_data = src_data
        self.tgt_data = tgt_data

    def __len__(self):
        return len(self.src_data)

    def __getitem__(self, idx):
        return torch.tensor(self.src_data[idx]), torch.tensor(self.tgt_data[idx])

dataset = TranslationDataset(english_data, french_data)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# --------------------------------------------------
# ENCODER (LSTM)
# --------------------------------------------------
class Encoder(nn.Module):
    """
    Encoder LSTM que transforma la secuencia de entrada
    en un estado latente (hidden + cell).
    """

    def __init__(self, input_dim, embed_dim, hidden_dim, num_layers):
        super(Encoder, self).__init__()

        # Embedding de palabras de entrada
        self.embedding = nn.Embedding(input_dim, embed_dim)

        # LSTM encoder
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers,
            batch_first=True
        )

    def forward(self, x):
        embedded = self.embedding(x)
        outputs, (hidden, cell) = self.lstm(embedded)
        return hidden, cell

# --------------------------------------------------
# DECODER (LSTM)
# --------------------------------------------------
class Decoder(nn.Module):
    """
    Decoder LSTM que genera la secuencia de salida
    palabra por palabra.
    """

    def __init__(self, output_dim, embed_dim, hidden_dim, num_layers):
        super(Decoder, self).__init__()

        self.embedding = nn.Embedding(output_dim, embed_dim)

        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden, cell):
        # Entrada de un solo token
        x = x.unsqueeze(1)

        embedded = self.embedding(x)

        outputs, (hidden, cell) = self.lstm(embedded, (hidden, cell))

        predictions = self.fc(outputs.squeeze(1))

        return predictions, hidden, cell

# --------------------------------------------------
# MODELO SEQ2SEQ
# --------------------------------------------------
class Seq2Seq(nn.Module):
    """
    Modelo Seq2Seq que conecta Encoder y Decoder.
    """

    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        batch_size = src.size(0)
        tgt_len = tgt.size(1)
        tgt_vocab_size = self.decoder.fc.out_features

        outputs = torch.zeros(
            batch_size,
            tgt_len,
            tgt_vocab_size
        ).to(self.device)

        hidden, cell = self.encoder(src)

        input = tgt[:, 0]

        for t in range(1, tgt_len):
            output, hidden, cell = self.decoder(input, hidden, cell)

            outputs[:, t, :] = output

            top1 = output.argmax(1)

            input = tgt[:, t] if torch.rand(1).item() < teacher_forcing_ratio else top1

        return outputs

# --------------------------------------------------
# ENTRENAMIENTO
# --------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_dim = len(english_vocab)
output_dim = len(french_vocab)
embed_dim = 64
hidden_dim = 128
num_layers = 2

encoder = Encoder(input_dim, embed_dim, hidden_dim, num_layers)
decoder = Decoder(output_dim, embed_dim, hidden_dim, num_layers)
model = Seq2Seq(encoder, decoder, device).to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)

criterion = nn.CrossEntropyLoss(ignore_index=french_vocab["<PAD>"])

def train(model, dataloader, optimizer, criterion, device, num_epochs=20):
    """
    Entrena el modelo Seq2Seq.

    Parámetros:
    model: modelo Seq2Seq
    dataloader: batches de entrenamiento
    optimizer: optimizador
    criterion: función de pérdida
    device: CPU/GPU
    num_epochs: número de épocas
    """

    model.train()

    for epoch in range(num_epochs):
        epoch_loss = 0

        for src, tgt in dataloader:
            src, tgt = src.to(device), tgt.to(device)

            optimizer.zero_grad()

            output = model(src, tgt)

            output = output[:, 1:].reshape(-1, output.shape[2])
            tgt = tgt[:, 1:].reshape(-1)

            loss = criterion(output, tgt)

            loss.backward()

            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch+1},/{num_epochs}, Loss: {epoch_loss / len(dataloader):.4f}")

train(model, dataloader, optimizer, criterion, device)

# --------------------------------------------------
# INFERENCIA (TRADUCCIÓN)
# --------------------------------------------------
def translate_sentence(model, sentence, english_vocab, french_vocab, max_len_fr, device):
    """
    Traduce una oración del inglés al francés usando el modelo entrenado.
    """

    model.eval()

    tokens = [english_vocab.get(word, english_vocab["<UNK>"]) for word in sentence.split()]
    tokens = [english_vocab["<SOS>"]] + tokens + [english_vocab["<EOS>"]]

    src = torch.tensor(tokens).unsqueeze(0).to(device)

    with torch.no_grad():
        hidden, cell = model.encoder(src)

    tgt_vocab = {v: k for k, v in french_vocab.items()}
    tgt_indices = [french_vocab["<SOS>"]]

    for _ in range(max_len_fr):
        tgt_tensor = torch.tensor([tgt_indices[-1]]).to(device)

        output, hidden, cell = model.decoder(tgt_tensor, hidden, cell)

        pred = output.argmax(1).item()

        tgt_indices.append(pred)

        if pred == french_vocab["<EOS>"]:
            break

    translated_sentence = [tgt_vocab[i] for i in tgt_indices[1:-1]]

    return " ".join(translated_sentence)

# --------------------------------------------------
# PRUEBA DE TRADUCCIÓN
# --------------------------------------------------
sentence = "good night"
translation = translate_sentence(model, sentence, english_vocab, french_vocab, max_len_fr, device)

print(f"Translated Sentence: {translation}")