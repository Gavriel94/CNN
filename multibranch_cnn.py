import torch
import torch.nn as nn
import time
import wandb

class MultiBranchAgePrediction(nn.Module):
    def __init__(self):
        super(MultiBranchAgePrediction, self).__init__()

        self.branch = nn.Sequential(
            nn.Conv2d(3, 24, 2, stride=2, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(),

            nn.Conv2d(24, 48, 3, stride=1, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(),

            nn.Conv2d(48, 96, 3, stride=1, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(),

            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.classifier = nn.Sequential(
            nn.Linear(3*96, 144),
            nn.BatchNorm1d(144),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(144, 72),
            nn.BatchNorm1d(72),
            nn.ReLU(),
            nn.Dropout(0.25),

            nn.Linear(72, 14),
            nn.BatchNorm1d(14),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(14, 7)
            )
        
    def forward(self, original_x, bright_x, dark_x):
        out_original = self.branch(original_x).view(original_x.size(0), -1)
        out_bright = self.branch(bright_x).view(bright_x.size(0), -1)
        out_dark = self.branch(dark_x).view(dark_x.size(0), -1)
        out = torch.cat((out_original, out_bright, out_dark), dim=1)
        out = self.classifier(out)
        prediction = out.squeeze(1)
        return prediction

    def __train(self,
              dataloader, 
              model, 
              optimizer, 
              scheduler, 
              criterion,
              device, 
              verbose=True):
        """
        Used to train a neural network.
        Sets the model to training mode and initialises variables to keep 
        track of metrics during training.
        Iterates through label/text pairs from each dataset making label
        predictions. 
        Calculates loss and backpropagates parameter updates
        through the network to ideally reduce loss over epochs.

        Args:
            dataloader (DataLoader): DataLoader containing training
                data.
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
        model.train()
        # Accuracy and loss accumulated over epoch
        total_accuracy, total_loss, num_predictions = 0, 0, 0
        # Displays training metrics every quarter of epoch
        intervals = (len(dataloader) / 4).__round__()
        for idx, (label, 
                  img, 
                  bright_img, 
                  dark_img) in enumerate(dataloader):
            # Convert and move data 
            label=label.to(device, dtype=torch.long)
            img = img.to(device, dtype=torch.float32)
            bright_img = bright_img.to(device, dtype=torch.float32)
            dark_img = dark_img.to(device, dtype=torch.float32)
            # Make prediction
            prediction = model(img, bright_img, dark_img)
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

    def __evaluate(self, dataloader, model, criterion, device):
        """
        Used to evaluate model training.
        Works similarly to the training method, allowing the model
        to make predictions on labelled data, however no parameters are
        updated.

        Args:
            dataloader (DataLoader): DataLoader containing either validation
                or testing data.
            model (nn.Module): The LSTM model being trained.
            criterion (torch.nn.modules.loss): Loss function.

        Returns:
            batch_loss, batch_accuracy, batch_count 
                (float, float, int): loss, accuracy and number of 
                predictions made over the validation set.
        """
        model.eval()
        total_accuracy, total_loss, num_predictions = 0, 0, 0
        with torch.no_grad():
            for idx, (label, img, bright_img, dark_img) in enumerate(dataloader):
                # Convert and move data
                label = label.to(device, dtype=torch.long)
                img = img.to(device, dtype=torch.float32)
                bright_img = bright_img.to(device, dtype=torch.float32)
                dark_img = dark_img.to(device, dtype=torch.float32)
                # Make prediction
                prediction = model(img, bright_img, dark_img)
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

    def run_training(self, 
                     training, 
                     validation, 
                     testing, 
                     model, 
                     optimizer,
                     scheduler, 
                     criterion,
                     device, 
                     epochs, 
                     verbose=True,
                     wandb_track=False):
        """
        Wraps the training and evaluation functions in one method.
        At the end of each epoch, the model asseses the validation set.
        Once all epochs are complete performance is assesed on the test set.

        Args:
            training (DataLoader): DataLoader with training data.
            validation (DataLoader): DataLoader with validation data.
            testing (DataLoader): DataLoader with testing data.
            model (nn.Module): The LSTM model being trained.
            optimizer (torch.optim.sgd): Backpropagation method.
            criterion (torch.nn.modules.loss): Loss function.
            epochs (int): Number of epochs the model is trained for.
            verbose (Boolean): Display metrics (default=True).

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
        # Time saved for calculating final processing time
        start_time = time.time()
        for epoch in range(epochs):
            epoch_start = time.time()
            print('-' * 49)
            print(f'|\t\t     Epoch {epoch + 1}      \t\t|')
            print('-' * 49)
            # Process training data
            t_loss, t_acc, learning_rate = self.__train(training, 
                                    model, 
                                    optimizer,
                                    scheduler, 
                                    criterion,
                                    device, 
                                    verbose)  
            # Store training metrics
            train_loss.append(t_loss)
            train_accuracy.append(t_acc)
            # Evaluate validation data
            v_loss, v_acc = self.__evaluate(validation, model, criterion, device)
            # Store evaluation metrics
            val_loss.append(v_loss)
            val_accuracy.append(v_acc)
            # Log metrics to wandb

            print('-' * 49)
            print(f'| Validation Accuracy   : {v_acc:.8f}% accurate |')
            print('-' * 49)
            print(f'| Time Elapsed\t\t: {time.time() - epoch_start:.2f} seconds\t|')
            print('-' * 49)
            print()
            if wandb_track:
                wandb.log({
                'Epoch': epoch,
                'Training Accuracy': t_acc,
                'Training Loss': t_loss,
                'Validation Accuracy': v_acc, 
                'Validation Loss': v_loss,
                'Learning Rate': learning_rate
                })

        # Assess model performance on test data
        _, test_acc = self.__evaluate(testing, model, criterion, device)
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