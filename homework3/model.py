import torch
from torch import nn
import pytorch_lightning as pl
from torchmetrics import Accuracy

class LitLSTMSentimentTagger(pl.LightningModule):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, classes):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size+1, embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=1)
        self.hidden2tag = nn.Linear(hidden_dim, classes)
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.accuracy = Accuracy()

    def forward(self, sentence, lengths):
        embedded_sentence = self.word_embeddings(sentence)
        out, _ = self.lstm(embedded_sentence)
#         out = out.gather(dim=0, index=lengths.unsqueeze(0)-1)
        lstm_out = []
        for i, lenght in enumerate(lengths):
            lstm_out.append(out[:, i][lenght-1])
        lstm_out = torch.stack(lstm_out)
        logits = self.hidden2tag(torch.relu(lstm_out))
        return logits.squeeze()

    def training_step(self, batch, batch_idx):
        x_true, lenghts, y_true = batch
        y_pred = self(x_true, lenghts)
        loss = self.criterion(y_pred, y_true)
        y_pred_labels = torch.sigmoid(y_pred.data) > 0.5
        self.accuracy(y_pred_labels, y_true.int())
        
        self.log("Loss/Train", loss, batch_size=lenghts.shape[0], prog_bar=True)
        self.log("Acc/Train", self.accuracy, batch_size=lenghts.shape[0], prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x_true, lenghts, y_true = batch
        y_pred = self(x_true, lenghts)
        loss = self.criterion(y_pred, y_true)
        y_pred_labels = torch.sigmoid(y_pred.data) > 0.5
        self.accuracy(y_pred_labels, y_true.long())

        self.log("Loss/Val", loss, batch_size=lenghts.shape[0], prog_bar=True)
        self.log("Acc/Val", self.accuracy, batch_size=lenghts.shape[0], prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01, weight_decay=1e-3)
        return optimizer