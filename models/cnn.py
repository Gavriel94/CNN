import torch
import torch.nn as nn
import time
import wandb

class GenClassifier(nn.Module):
    """
    CNN with training and evaluation methods.
    """
    def __init__(self):
        super(GenClassifier, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=2),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            
            nn.Conv2d(16, 24, 2),
            nn.ReLU(),
            nn.BatchNorm2d(24),
            
            nn.Conv2d(24, 36, 3),
            nn.ReLU(),
            nn.BatchNorm2d(36),
            
            nn.Conv2d(36, 48, 3),
            nn.ReLU(),
            nn.BatchNorm2d(48),
            
            nn.Conv2d(48, 64, 3),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(3, 3),
            
            nn.Conv2d(64, 96, 3),
            nn.ReLU(),
            nn.BatchNorm2d(96),
            nn.MaxPool2d(3, 3),
            
            # Emulate global average pooling
            nn.AdaptiveAvgPool2d((1,1))
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(96, 112),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(112, 138),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(138, 72),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(72, 36),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(36, 7)
        )
        
        self.flatten = nn.Flatten()
        
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.classifier(x)
        x = x.squeeze(1)
        return x
    
    def run_training(self, 
                    training, 
                    validation, 
                    testing, 
                    optimizer,
                    scheduler, 
                    criterion,
                    device, 
                    epochs, 
                    verbose=True,
                    wandb_track=False):
        """
        Trains and evaluates the model.
        Returns metrics for analysis.

        Args:
            training (DataLoader): Training data.
            validation (DataLoader): Validation data.
            testing (DataLoader): Test data.
            optimizer (torch.optim.sgd): Backpropagation method.
            criterion (torch.nn.modules.loss): Loss function.
            epochs (int): Number of epochs the model is trained for.
            verbose (bool): Display metrics (default=True).
            wandb_track (bool): Analyse method using wandb.
        Returns:
            train_accuracy, train_loss, val_accuracy, val_loss 
                (list, list, list, list): Metrics saved during training and
                evaluation.
        """
        # Containers for training and evaluation metrics
        train_accuracy = []
        train_loss = []
        val_accuracy = []
        val_loss = []

        start_time = time.time()
        for epoch in range(epochs):
            epoch_start = time.time()
            print('-' * 49)
            print(f'|\t\t     Epoch {epoch + 1}      \t\t|')
            print('-' * 49)
            # Training loop
            t_loss, t_acc, learning_rate = self.__train(training,
                                                         optimizer,
                                                         scheduler, 
                                                         criterion,
                                                         device, 
                                                         verbose)  
            train_loss.append(t_loss)
            train_accuracy.append(t_acc)
            
            # Evaluation loop
            v_loss, v_acc = self.__evaluate(validation, criterion, device)
            val_loss.append(v_loss)
            val_accuracy.append(v_acc)

            print('-' * 49)
            print(f'| Validation Accuracy   : {v_acc:.8f}% accurate |')
            print('-' * 49)
            print(f'| Time Elapsed\t\t: {time.time() - epoch_start:.2f} seconds\t|')
            print('-' * 49)
            print()
            
            if wandb_track:
                # Log metrics to wandb
                wandb.log({
                'Epoch': epoch,
                'Training Accuracy': t_acc,
                'Training Loss': t_loss,
                'Validation Accuracy': v_acc, 
                'Validation Loss': v_loss,
                'Learning Rate': learning_rate
                })

        # Evaluate model on test data
        _, test_acc = self.__evaluate(testing, criterion, device)
        total_minutes = (time.time() - start_time).__round__()/60
        print('*' + '-' * 47 + '*')
        test_metrics = (
            f'*\t\tEvaluating Test Data\t\t*\n'
            f'*' + '-' * 47 + '*\n'
            f'* Test Accuracy\t\t: {test_acc:.8f}% accurate *\n'
            f'* Total Training Time\t: {total_minutes:.2f} minutes  \t*'
        )
        print(test_metrics)
        print('*' + '-' * 47 + '*')
        if wandb_track:
                wandb.finish()
        return train_accuracy, train_loss, val_accuracy, val_loss
    
    def __train(self,
                dataloader, 
                optimizer, 
                scheduler, 
                criterion,
                device, 
                verbose=True):
        """
        Training .
        
        Args:
            dataloader (DataLoader): Training data.
            model (nn.Module): The LSTM model being trained.
            optimizer (torch.optim.sgd): Backpropagation method.
            criterion (torch.nn.modules.loss): Loss function.
            epoch (int): The current epoch.
            verbose (Boolean): Display metrics (default=True).

        Returns:
            epoch_loss, epoch_accuracy, epoch_count 
                (float, float, int): loss, accuracy and number of 
                predictions made in one epoch.
        """
        self.train()
        # Accuracy and loss accumulated over epoch
        total_accuracy, total_loss, num_predictions = 0, 0, 0
        # Displays training metrics every quarter of epoch
        intervals = (len(dataloader) / 4).__round__()
        for idx, (label, img, _, _) in enumerate(dataloader):
            # Convert and move data 
            label=label.to(device, dtype=torch.long)
            img = img.to(device, dtype=torch.float32)
            # Make prediction
            prediction = self(img)
            # Select most likely class
            _, predicted_classes = torch.max(prediction, 1)
            # Calculate loss
            loss = criterion(prediction, label)
            batch_loss = loss.item()
            # Update weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Store metrics
            total_accuracy += (predicted_classes == label).sum().item()
            total_loss += batch_loss
            num_predictions += label.size(0)

            if verbose and idx % intervals == 0 and idx > 0:
                epoch_metrics = (
                    f'| {idx:5} / {len(dataloader):5} batches |' 
                    f' {(total_accuracy/num_predictions)*100:.8f}% accurate |'
                    )
                print(epoch_metrics)
        scheduler.step()
        # current lr used for wandb tracking
        current_lr = scheduler.get_last_lr()[0]
        average_accuracy_pct = (total_accuracy / num_predictions) * 100
        average_loss_per_sample = total_loss / num_predictions
        return average_loss_per_sample, average_accuracy_pct, current_lr
        
    def __evaluate(self, dataloader, criterion, device):
        """
        Evaluates the model on validation and test data.

        Args:
            dataloader (DataLoader): Validation or test data.
            criterion (torch.nn.modules.loss): Loss function.

        Returns:
            batch_loss, batch_accuracy, batch_count 
                (float, float, int): loss, accuracy and number of 
                predictions made over the data.
        """
        self.eval()
        total_accuracy, total_loss, num_predictions = 0, 0, 0
        with torch.no_grad():
            for idx, (label, img, _, _) in enumerate(dataloader):
                # Convert and move data
                label = label.to(device, dtype=torch.long)
                img = img.to(device, dtype=torch.float32)
                # Make prediction
                prediction = self(img)
                # Calculate loss
                loss = criterion(prediction, label)
                # Select most likely class
                _, predicted_classes = torch.max(prediction, 1)
                # Calculate metrics
                total_accuracy += (predicted_classes == label).sum().item()
                num_predictions += label.size(0)
                total_loss += loss.item()
        average_accuracy_pct = (total_accuracy / num_predictions) * 100
        average_loss_per_sample = total_loss / num_predictions
        return average_loss_per_sample, average_accuracy_pct