import torch
from transformers import get_scheduler
import time
import evaluate as evaluate
from tqdm import tqdm
class Inference():
    def __init__(self, mymodel, device) -> None:
        self.mymodel=mymodel
        self.device=device

    
    def evaluate_model(self, model, dataloader, device):
        self.mymodel.to(self.device)

        # load metrics
        # dev_accuracy = evaluate.load('accuracy')

        # turn model into evaluation mode
        self.model.eval()

        # iterate over the dataloader
        for batch in tqdm(enumerate(dataloader)):

            input_ids = batch['input_ids']
            input_ids=input_ids.to(device)


            # forward pass
            # name the output as `output`
            output = model(input_ids, attention_mask=masks)
            predictions = output['logits']


            predictions = torch.argmax(predictions, dim=-1)
            ## Flatten predictions and labels to match evaluation's format
            predictions_flat = predictions.flatten()
            labels_flat = labels.flatten()
            dev_accuracy.add_batch(predictions=predictions_flat, references=labels_flat)

        # compute and return metrics
        return dev_accuracy.compute()
    
    def train(self, train_dataloader, val_dataloader):
        self.mymodel.to(self.device)

        # here, we use the AdamW optimizer. Use torch.optim.Adam.

        # we are only tuning parameters of MLP
        # Pretrained model parameters are left frozen here
        print(" >>>>>>>>  Initializing optimizer")
        if self.type=="head":
        
            non_premodel_params = []
            # We assume that all parameters not belonging to `model.model` are non-pretrained
            for name, param in self.mymodel.named_parameters():
                if not name.startswith('model.'):  # `model.model` refers to the pretrained model part
                    non_premodel_params.append(param)
        
            para=non_premodel_params
            #mymodel.classifier.parameters()
        # Full parameter tuning
        elif self.type=="full":
            para=self.mymodel.parameters()
        optimizer = torch.optim.AdamW(para, lr=self.lr)
    
        # now, we set up the learning rate scheduler
        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=50,
            num_training_steps=len(train_dataloader) * self.num_epochs
        )
        loss = torch.nn.CrossEntropyLoss().to(self.device)
        epoch_list = []
        train_acc_list = []
        dev_acc_list = []
        for epoch in range(self.num_epochs):
            epoch_start_time = time.time()
            # put the model in training mode (important that this is done each epoch,
            # since we put the model into eval mode during validation)
            self.mymodel.train()

            # load metrics
            train_accuracy = evaluate.load('accuracy')

            print("Epoch "+str(epoch + 1)+" training:")
            for i, batch in tqdm(enumerate(train_dataloader)):
                ## Get input_ids, attention_mask, labels
                labels, input_ids, masks = batch['labels'], batch['input_ids'], batch['attention_mask']
                ## Send to GPU
                labels=labels.to(self.device)
                input_ids=input_ids.to(self.device)
                masks=masks.to(self.device)
                ## Forward pass
                output = self.mymodel(input_ids, attention_mask=masks)
                predictions = output['logits']
                ## Cross-entropy loss
                centr = loss(predictions.view(-1, self.mymodel.num_labels), labels.view(-1))
                ## Backward pass
                centr.backward()
                ## Update Model
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                ## Argmax get real predictions
                predictions = torch.argmax(predictions, dim=-1)
                ## Flatten predictions and labels to match evaluation's format
                predictions_flat = predictions.flatten()
                labels_flat = labels.flatten()
                # print("Shape of predictions:", predictions_flat.shape, predictions_flat.dtype)  
                # print("Shape of labels:", labels_flat.shape, labels_flat.dtype)  
                ## Update metrics
                train_accuracy.add_batch(predictions=predictions_flat, references=labels_flat)
            # print evaluation metrics
            print(" ===> Epoch "+str(epoch + 1))
            train_acc = train_accuracy.compute()
            print(" - Average training metrics: accuracy="+str(train_acc))
            train_acc_list.append(train_acc['accuracy'])

            # normally, validation would be more useful when training for many epochs
            val_accuracy = self.evaluate_model(self.mymodel, val_dataloader, self.device)
            print(f" - Average validation metrics: accuracy="+str(val_accuracy))
            dev_acc_list.append(val_accuracy['accuracy'])
        
            epoch_list.append(epoch)
        
            # test_accuracy = evaluate_model(mymodel, test_dataloader, device)
            # print(f" - Average test metrics: accuracy={test_accuracy}")

            epoch_end_time = time.time()
            print("Epoch "+str(epoch + 1)+" took "+str(epoch_end_time - epoch_start_time)+" seconds")