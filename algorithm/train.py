"""Training utilities."""

import torch
import sys
import time
import datetime

def train(model, device, num_epochs, train_dataloader, optimizer, loss_func, validation, val_dataloader=None):
    """Train the model and save the best."""
    # Training
    start = time.time()
    best_accuracy = 0.0 

    print("Start training...")
    for epoch in range(1,num_epochs+1):
        total_train_loss = 0
        total_val_loss = 0
        total_images = 0
        total_correct = 0

        # Training
        for batch in train_dataloader:           # Load batch
            images_train, labels_train = batch
            optimizer.zero_grad()
            images_train = images_train.to(device, dtype=torch.float)
            labels_train = labels_train.type(torch.LongTensor)
            labels_train = labels_train.to(device)

            preds_train = model(images_train)             # Process batch

            train_loss = loss_func(preds_train, labels_train) # Calculate loss

            train_loss.backward()                 # Calculate gradients
            optimizer.step()                # Update weights
            total_train_loss += train_loss.item()

            if not validation:
                output = preds_train.argmax(dim=1)
                total_images += int(labels_train.size(0))
                total_correct += int(output.eq(labels_train).sum().item())

        # Get train loss
        train_loss = total_train_loss/len(train_dataloader) 

        # Validation
        if validation:
            with torch.no_grad():
                model.eval()
                for batch in val_dataloader:
                    images_val, labels_val = batch
                    images_val = images_val.to(device, dtype=torch.float)
                    labels_val = labels_val.type(torch.LongTensor)
                    labels_val = labels_val.to(device)

                    preds_val = model(images_val)
                    val_loss = loss_func(preds_val, labels_val) 
                    total_val_loss += val_loss.item()

                    output = preds_val.argmax(dim=1)
                    total_images += int(labels_val.size(0))
                    total_correct += int(output.eq(labels_val).sum().item())
                model.train()

            # Get val loss
            val_loss = total_val_loss/len(val_dataloader) 

        model_accuracy = total_correct / total_images

        if not validation:
            print(f"ep {epoch}, train loss: {train_loss:.2f}, Train Acc {model_accuracy*100:.2f}%")
        else:
            print(f"ep {epoch}, train loss: {train_loss:.2f}, val loss: {val_loss:.2f}, Val Acc {model_accuracy*100:.2f}%")

        if model_accuracy > best_accuracy:
            torch.save(model.state_dict(),'savedModel.pth')
            print("\tModel saved to savedModel.pth")
            best_accuracy = model_accuracy 

        if epoch % 2 == 0:
            save_path = 'checkModel{}.pth'.format(epoch)
            torch.save(model.state_dict(), save_path)
            print(f"\tModel saved to {save_path}")

        sys.stdout.flush()

    end = time.time()
    print(f"\nTotal training time: {str(datetime.timedelta(seconds=end-start))}")
    sys.stdout.flush()