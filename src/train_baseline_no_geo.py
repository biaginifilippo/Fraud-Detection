import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

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
        bins = [-np.inf, 0, 9, 10, 11, 12, 13, 14, 15, np.inf]
        labels = [
            'eq_0', '1_to_9', '10', '11', '12', '13', '14', '15', 'gt_15'
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

    df_onehot = pd.get_dummies(dataset['fascia_eta'], prefix='eta')
    dataset = pd.concat([dataset, df_onehot], axis=1)

    dataset.drop(columns=['fascia_eta'], axis=1, inplace=True)

    df_onehot = pd.get_dummies(dataset['paese_residenza'], prefix='paese_residenza')
    dataset = pd.concat([dataset, df_onehot], axis=1)

    dataset.drop(columns=['paese_residenza'], axis=1, inplace=True)

    print(len(dataset.columns))

    dataset.drop(columns=['professione_macro', 'loc_geo'], axis=1, inplace=True)
    bool_cols = dataset.select_dtypes(bool).columns
    dataset[bool_cols] = dataset[bool_cols].astype(int)
    dataset_senza_label = dataset.drop(columns=['label'], axis=1)
    dataset = pd.concat([dataset_senza_label, dataset['label']], axis=1)
    dataset.to_csv('dataset_senza_geo_profession.csv', index=False)

    return

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layer_input = nn.Linear(63, 64)
        self.layer_1 = nn.Linear(64, 128)
        self.layer_1_bis = nn.Linear(128, 256)
        self.layer_1_tris = nn.Linear(256, 256)
        self.layer_2 = nn.Linear(256, 128)
        self.layer_3 = nn.Linear(128, 64)
        self.layer_4 = nn.Linear(64, 32)
        self.layer_5 = nn.Linear(32, 16)
        self.layer_out = nn.Linear(16, 2)

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.01)

    def forward(self, x):
        x = self.activation(self.layer_input(x))
        x = self.dropout(self.activation(self.layer_1(x)))
        x = self.dropout(self.activation(self.layer_1_bis(x)))
        x = self.dropout(self.activation(self.layer_1_tris(x)))
        x = self.dropout(self.activation(self.layer_2(x)))
        x = self.dropout(self.activation(self.layer_3(x)))
        x = self.dropout(self.activation(self.layer_4(x)))
        x = self.dropout(self.activation(self.layer_5(x)))
        x = self.layer_out(x)
        return x


def train_model(model, X_train, y_train, X_val=None, y_val=None,
                epochs=30, batch_size=64, lr=1.1e-3, device='cuda' if torch.cuda.is_available() else 'cpu'):

    model.to(device)

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                  torch.tensor(y_train, dtype=torch.long))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_loader = None
    if X_val is not None and y_val is not None:
        val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                    torch.tensor(y_val, dtype=torch.long))
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

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
            val_acc = correct / total
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            print(
                f"Epoch {epoch + 1}/{epochs} - Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        else:
            print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {epoch_loss:.4f}")

    return history


if __name__ == '__main__':

    """in questa versione non uso loc_geo e professione_macro """
    model = Model()
    dataset = pd.read_csv('../data/dataset_senza_geo_profession.csv')
    X = dataset.drop(columns=['label'])
    y = dataset['label']

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    X_train = torch.tensor(X_train.values, dtype=torch.float32)
    y_train = torch.tensor(y_train.values, dtype=torch.long)
    X_val = torch.tensor(X_val.values, dtype=torch.float32)
    y_val = torch.tensor(y_val.values, dtype=torch.long)
    X_test = torch.tensor(X_test.values, dtype=torch.float32)
    y_test = torch.tensor(y_test.values, dtype=torch.long)

    history = train_model(model, X_train, y_train, X_val, y_val, epochs=12, batch_size=512)

    print("test_finale")

    correct = 0
    out = model(X_test)
    preds = torch.argmax(out, dim=1)
    correct += (preds == y_test).sum().item()
    accuracy = correct/len(X_test)

    print(accuracy)
    cm = confusion_matrix(y_test, preds, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=[0, 1])
    disp.plot()
    plt.show()