import numpy as np
import warnings
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
warnings.filterwarnings("ignore")

inputs = np.load('./data/inputs.npy')
labels = np.load('./data/labels.npy')
VOCAB_SIZE = int(inputs.max()) + 1

# Training params
EPOCHS = 20
CLIP = 5 # gradient clipping - to avoid gradient explosion (frequent in RNNs)
lr = 0.1
BATCH_SIZE = 32  # we follow mini batch gradient descent to ensure faster convergence

# Model params
EMBEDDING_DIM = 50
HIDDEN_DIM = 10
DROPOUT = 0.2

import syft as sy

labels = torch.tensor(labels).to(torch.int64)
inputs = torch.tensor(inputs).to(torch.int64)

# splitting training and test data
pct_test = 0.2

train_labels = labels[:-int(len(labels)*pct_test)]
train_inputs = inputs[:-int(len(labels)*pct_test)]

test_labels = labels[-int(len(labels)*pct_test):]
test_inputs = inputs[-int(len(labels)*pct_test):]

# Hook that extends the Pytorch library to enable all computations with pointers of tensors sent to other workers
hook = sy.TorchHook(torch)

# Creating 3 virtual workers
vatsal = sy.VirtualWorker(hook, id="vatsal")
ananya = sy.VirtualWorker(hook, id="ananya")
arjav = sy.VirtualWorker(hook, id="arjav")

# threshold indexes for dataset split (one third for Vatsal, one third half for Ananya, the left for Arjav)
train_idx = int(len(train_labels)/3)
test_idx = int(len(test_labels)/3)
train_idx2 = int(len(train_labels)*2/3)
test_idx2 = int(len(test_labels)*2/3)

# Sending training datasets to virtual workers
vatsal_train_dataset = sy.BaseDataset(train_inputs[:train_idx], train_labels[:train_idx]).send(vatsal)
ananya_train_dataset = sy.BaseDataset(train_inputs[train_idx:train_idx2], train_labels[train_idx:train_idx2]).send(ananya)
arjav_train_dataset = sy.BaseDataset(train_inputs[train_idx2:], train_labels[train_idx2:]).send(arjav)
vatsal_test_dataset = sy.BaseDataset(test_inputs[:test_idx], test_labels[:test_idx]).send(vatsal)
ananya_test_dataset = sy.BaseDataset(test_inputs[test_idx: test_idx2], test_labels[test_idx:test_idx2]).send(ananya)
arjav_test_dataset = sy.BaseDataset(test_inputs[test_idx2:], test_labels[test_idx2:]).send(arjav)

# Creating federated datasets, an extension of Pytorch TensorDataset class
federated_train_dataset = sy.FederatedDataset([vatsal_train_dataset, ananya_train_dataset, arjav_train_dataset])
federated_test_dataset = sy.FederatedDataset([vatsal_test_dataset, ananya_test_dataset, arjav_test_dataset])


# Creating federated dataloaders, an extension of Pytorch DataLoader class
federated_train_loader = sy.FederatedDataLoader(federated_train_dataset, shuffle=True, batch_size=BATCH_SIZE)
federated_test_loader = sy.FederatedDataLoader(federated_test_dataset, shuffle=False, batch_size=BATCH_SIZE)

from Selfmade_GRU import GRU

# Initiating the model
model = GRU(vocab_size=VOCAB_SIZE, hidden_dim=HIDDEN_DIM, embedding_dim=EMBEDDING_DIM, dropout=DROPOUT)

# Defining loss and optimizer
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=lr)

for e in range(EPOCHS):
    
    ######### Training ##########
    
    losses = []
    # Batch loop
    for inputs, labels in federated_train_loader:
        # Location of current batch
        worker = inputs.location
        # Initialize hidden state and send it to worker
        h = torch.Tensor(np.zeros((BATCH_SIZE, HIDDEN_DIM))).send(worker)
        # Send model to current worker
        model.send(worker)
        # Setting accumulated gradients to zero before backward step
        optimizer.zero_grad()
        # Output from the model
        output, _ = model(inputs, h)
        # Calculate the loss and perform backprop
        loss = criterion(output.squeeze(), labels.float())
        loss.backward()
        # Clipping the gradient to avoid explosion
        nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        # Backpropagation step
        optimizer.step() 
        # Get the model back to the local worker
        model.get()
        losses.append(loss.get())
    
    ######## Evaluation ##########
    
    # Model in evaluation mode
    model.eval()

    with torch.no_grad():
        test_preds = []
        test_labels_list = []
        eval_losses = []

        for inputs, labels in federated_test_loader:
            # get current location
            worker = inputs.location
            # Initialize hidden state and send it to worker
            h = torch.Tensor(np.zeros((BATCH_SIZE, HIDDEN_DIM))).send(worker)    
            # Send model to worker
            model.send(worker)
            
            output, _ = model(inputs, h)
            loss = criterion(output.squeeze(), labels.float())
            eval_losses.append(loss.get())
            preds = output.squeeze().get()
            test_preds += list(preds.numpy())
            test_labels_list += list(labels.get().numpy().astype(int))
            # Get the model back to the local worker
            model.get()
        
        score = roc_auc_score(test_labels_list, test_preds)
        test_preds_2 = []
        for i in range(0,len(test_preds)):
            if test_preds[i] < 0.5:
                test_preds_2.append(0)
            else:
                test_preds_2.append(1)
        acc = accuracy_score(test_labels_list, test_preds_2)
        f1 = f1_score(test_labels_list, test_preds_2)

    
    print("Epoch {}/{}...  \
    AUC_ROC Score: {:.3%}...  \
    Validation Accuracy: {:.3%}...  \
    F1 Score: {:.5f}...  \
    Training loss: {:.5f}...  \
    Validation loss: {:.5f}".format(e+1, EPOCHS, score, acc, f1, sum(losses)/len(losses), sum(eval_losses)/len(eval_losses)))
    
    model.train()