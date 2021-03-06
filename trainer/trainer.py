import sys
sys.path.append("trainer")

from datetime import datetime

import matplotlib.pyplot as plt
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from my_dataset import myDataset

def train():
    with wandb.init() as run:

        now = datetime.now().isoformat(timespec='seconds')
        
        run.name = f"sweep-{now}"
        print(run.name)
        
        hparams = run.config
        hparams.NOW = now
        
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print ("torch version ::: %s"%(torch.__version__))
        print ("device        ::: %s"%(device))

        mnist_train = myDataset(
            name="MNIST_train",
            data_path=hparams.DATA_PATH,
            train=True,
            transforms=transforms.ToTensor(),
            download=True
        )
        mnist_test = myDataset(
            name="MNIST_test",
            data_path=hparams.DATA_PATH,
            train=False,
            transforms=transforms.ToTensor(),
            download=True
        )
        print ("Dataset")

        tr_loader = DataLoader(
            mnist_train,
            batch_size=hparams.BATCH_SIZE,
            shuffle=True,
            num_workers=hparams.NUM_WORKERS
        )
        val_loader = DataLoader(
            mnist_test,
            batch_size=hparams.BATCH_SIZE,
            shuffle=True,
            num_workers=hparams.NUM_WORKERS
        )
        print ("DataLoader")


        class simpleNetwork(nn.Module):
            def __init__(self):
                super(simpleNetwork,self).__init__()

                self.flatten = nn.Flatten()
                self.fc = nn.Linear(1*28*28, 10)
                    
            def forward(self,x):
                x = self.flatten(x)
                x = self.fc(x)
                
                return x


        net = simpleNetwork().to(device)
        print("Model")

        loss = nn.CrossEntropyLoss()
        print ("loss")
        opt = optim.Adam(net.parameters(), lr=hparams.LEARNING_RATE)
        print ("opt")




        for epoch in range(hparams.EPOCHS):
            tr_loss_sum, tr_acc_sum = 0, 0
            net.train() 
            for X, y in tr_loader:
                X, y = X.to(device), y.to(device)
                
                y_pred = net(X)
                loss_out = loss(y_pred, y)
                
                opt.zero_grad()
                loss_out.backward()
                opt.step()
                
                tr_loss_sum += loss_out
                tr_acc_sum += (y_pred.argmax(axis=1)==y).sum().item()/len(y)
            tr_loss_avg = tr_loss_sum/len(tr_loader)
            tr_acc_avg = tr_acc_sum/len(tr_loader)
            print(f"tr_loss_avg ::: {tr_loss_avg}, tr_acc_avg ::: {tr_acc_avg}")
            wandb.log({"train_loss":tr_loss_avg, "train_acc":tr_acc_avg})

            val_loss_sum, val_acc_sum = 0, 0
            with torch.no_grad():
                net.eval()
                for X, y in val_loader:
                    X, y = X.to(device), y.to(device)
                    
                    y_pred = net(X)
                    loss_out = loss(y_pred, y)
                    
                    val_loss_sum += loss_out
                    val_acc_sum += (y_pred.argmax(axis=1)==y).sum().item()/len(y)
                val_loss_avg = val_loss_sum/len(val_loader)
                val_acc_avg = val_acc_sum/len(val_loader)
                print(f"val_loss_avg ::: {val_loss_avg}, val_acc_avg ::: {val_acc_avg}")
                wandb.log({"val_loss_avg":val_loss_avg, "val_acc_avg":val_acc_avg})


        print ("training finishes")



        with torch.no_grad():
            net.eval() # to evaluation mode 
            X, y = next(iter(val_loader))
            y_pred = net(X.to(device))
            y_pred = y_pred.argmax(axis=1)
                
            fig, ax = plt.subplots(5,5,figsize=(10,10))
            for i,(X,y_p) in enumerate(zip(X,y_pred)):
                div, mod = divmod(i,5)
                ax[div][mod].imshow(X.permute(1,2,0), cmap='gray')
                ax[div][mod].axis('off')
                ax[div][mod].set_title(f"Prediction:{y_p}")
                if i == 24:
                    break
        wandb.log({"plot": fig})
        plt.savefig(hparams.SAVEFIG_NAME)

        print ("result png")

        print("done")