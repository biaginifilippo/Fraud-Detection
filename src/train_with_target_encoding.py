import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import sys


def target_encoding(train_df, val_df, test_df, column, target_col='label', smoothing=10):

    # Calcola la media globale del target sul training set
    global_mean = train_df[target_col].mean()

    # Calcola media e conteggio per ogni categoria nel training set
    agg = train_df.groupby(column)[target_col].agg(['mean', 'count'])

    # Applica smoothing: (count * mean + smoothing * global_mean) / (count + smoothing)
    agg['smoothed_mean'] = (agg['count'] * agg['mean'] + smoothing * global_mean) / (agg['count'] + smoothing)

    # Crea il mapping
    encoding_map = agg['smoothed_mean'].to_dict()

    # Applica l'encoding
    new_col_name = f'{column}_encoded'
    train_df[new_col_name] = train_df[column].map(encoding_map).fillna(global_mean)
    val_df[new_col_name] = val_df[column].map(encoding_map).fillna(global_mean)
    test_df[new_col_name] = test_df[column].map(encoding_map).fillna(global_mean)

    # Rimuovi la colonna originale
    train_df.drop(columns=[column], inplace=True)
    val_df.drop(columns=[column], inplace=True)
    test_df.drop(columns=[column], inplace=True)

    return train_df, val_df, test_df



def preprocessing_dataset():

    dataset = pd.read_csv('../data/dati_with_label.csv')
    dataset.dropna(axis=0, how='any', inplace=True)
    cols_importi = []
    cols_operazioni = []
    for column in dataset.columns:
        print(f"------------ {column} ------------")
        print(dataset[column].astype('category').values)
        if 'importo' in column:
            cols_importi.append(column)
        if 'operazioni' in column:
            cols_operazioni.append(column)

    dataset.drop(columns=['descrizione_professionale', 'stato_residenza'], inplace=True, axis=1)

    dataset.info()
    for column in cols_importi:
        print(f'-------- {column} --------')
        dataset[column] = np.log2(dataset[column] + 1)
        print(dataset[column].describe())
        bins = [-np.inf, 10, 13, 15, np.inf]
        labels = [
             '10', '13', '15', 'gt_15'
        ]

        # assegna i bin
        dataset['bin_col'] = pd.cut(dataset[column], bins=bins, labels=labels, right=True)

        # one-hot encoding denso
        onehot = pd.get_dummies(dataset['bin_col'], prefix=column)

        # unisci al dataset
        dataset = pd.concat([dataset, onehot], axis=1)
        dataset.drop(columns=[column, 'bin_col'], axis=1, inplace=True)

        print(len(dataset.columns))

    for column in cols_operazioni:
        print(f'-------- {column} --------')
        dataset[column] = np.log2(dataset[column] + 1)

    df_onehot = pd.get_dummies(dataset['segm_patr_de_crm'], prefix='seqm')
    dataset = pd.concat([dataset, df_onehot], axis=1)

    dataset.drop(columns=['segm_patr_de_crm'], axis=1, inplace=True)


    dataset.drop(columns=['fascia_eta'], axis=1, inplace=True)

    df_onehot = pd.get_dummies(dataset['paese_residenza'], prefix='paese_residenza')
    dataset = pd.concat([dataset, df_onehot], axis=1)

    dataset.drop(columns=['paese_residenza'], axis=1, inplace=True)

    print(len(dataset.columns))

    # dataset.drop(columns=['professione_macro', 'loc_geo'], axis=1, inplace=True)

    bool_cols = dataset.select_dtypes(bool).columns
    dataset[bool_cols] = dataset[bool_cols].astype(int)
    dataset_senza_label = dataset.drop(columns=['label'], axis=1)
    dataset = pd.concat([dataset_senza_label, dataset['label']], axis=1)
    dataset.to_csv('..\data\dataset_con_geo_profession_v2.csv', index=False)

    return

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layer_input = nn.Linear(33, 1024)
        self.layer_2 = nn.Linear(1024, 32)
        self.layer_out = nn.Linear(32, 2)

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.75)

    def forward(self, x):
        x = self.activation(self.layer_input(x))
        # x = self.dropout(self.activation(self.layer_1(x)))
        # x = self.dropout(self.activation(self.layer_1_bis(x)))
        # x = self.dropout(self.activation(self.layer_1_tris(x)))
        # x = self.dropout(self.activation(self.layer_1_quad(x)))
        # x = self.dropout(self.activation(self.layer_1_pent(x)))
        x = self.dropout(self.activation(self.layer_2(x)))
        # x = self.dropout(self.activation(self.layer_4(x)))
        # x = self.dropout(self.activation(self.layer_5(x)))
        x = self.layer_out(x)
        return x


def train_model(model, X_train, y_train, X_val=None, y_val=None,
                epochs=30, batch_size=64, lr=1e-3, device='cuda' if torch.cuda.is_available() else 'cpu'):

    model.to(device)
    epoch_val = 0
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                  torch.tensor(y_train, dtype=torch.long))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_loader = None
    if X_val is not None and y_val is not None:
        val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                    torch.tensor(y_val, dtype=torch.long))
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    weights = torch.tensor([1.0, 1.5]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    best_val_loss = 1
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for X_batch, y_batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}', leave=False):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()

            outputs = model(X_batch)

            loss = criterion(outputs, y_batch)

            loss.backward()

            optimizer.step()

            running_loss += loss.item() * X_batch.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        history['train_loss'].append(epoch_loss)

        if val_loader:
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for X_val_batch, y_val_batch in val_loader:
                    X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
                    outputs = model(X_val_batch)
                    loss = criterion(outputs, y_val_batch)
                    val_loss += loss.item() * X_val_batch.size(0)

                    preds = torch.argmax(outputs, dim=1)
                    correct += (preds == y_val_batch).sum().item()
                    total += y_val_batch.size(0)

            val_loss /= len(val_loader.dataset)
            if val_loss < best_val_loss+0.003:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                torch.save(model.state_dict(), "../models/rete con solo un hidden layer/pesi_migliori.pth")
                epoch_val = epoch
            val_acc = correct / total
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            print(
                f"Epoch {epoch + 1}/{epochs} - Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        else:
            print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {epoch_loss:.4f}")

    return history, epoch_val


if __name__ == '__main__':

    """in questa versione uso loc_geo e professione_macro con target encoding """
    model = Model()
    dataset = pd.read_csv('../data/dataset_con_geo_profession_v2.csv')
    X = dataset.drop(columns=['label'])
    y = dataset['label']

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    data_train = pd.concat([X_train, y_train], axis=1)
    data_val = pd.concat([X_val, y_val], axis=1)
    data_test = pd.concat([X_test, y_test], axis=1)

    data_train, data_val, data_test = target_encoding(data_train, data_val, data_test, 'professione_macro')
    data_train, data_val, data_test = target_encoding(data_train, data_val, data_test, 'loc_geo')

    X_train = data_train.drop(columns=['label'])
    y_train = data_train['label']
    X_val = data_val.drop(columns=['label'])
    y_val = data_val['label']
    X_test = data_test.drop(columns=['label'])
    y_test = data_test['label']


    X_train = torch.tensor(X_train.values, dtype=torch.float32)
    y_train = torch.tensor(y_train.values, dtype=torch.long)
    X_val = torch.tensor(X_val.values, dtype=torch.float32)
    y_val = torch.tensor(y_val.values, dtype=torch.long)
    X_test = torch.tensor(X_test.values, dtype=torch.float32)
    y_test = torch.tensor(y_test.values, dtype=torch.long)

    history, epoch_val = train_model(model, X_train, y_train, X_val, y_val, epochs=45, batch_size=512)

    print("test_finale")

    correct = 0
    model.load_state_dict(torch.load("../models/rete con solo un hidden layer/pesi_migliori.pth"))
    out = model(X_test)
    preds = torch.argmax(out, dim=1)
    correct += (preds == y_test).sum().item()
    accuracy = correct/len(X_test)
    plt.plot(range(len(history['train_loss'])), history['train_loss'])
    plt.plot(range(len(history['val_loss'])), history['val_loss'])
    plt.axvline(x=epoch_val, color='red', linestyle='--', label='Best Model')
    plt.savefig("loss.png")
    plt.show()

    print(accuracy)
    cm = confusion_matrix(y_test, preds, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=[0, 1])
    disp.plot()
    plt.savefig("cm.png")
    plt.show()











