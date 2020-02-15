from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import torch
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm, trange
import transformers
from transformers import get_linear_schedule_with_warmup
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import time
import datetime




'''
USAGE:

dataset_embeddings = pd.read_hdf('/content/drive/My Drive/dataset/mm_imdb_embeddings_concat_lastfour.h5', 'embeddings')

Data_train, Data_test, Data_val, Labels_train_tensor, Labels_test_tensor, Labels_val_tensor, Label_names = Train_Test_Val_Split(dataset_embeddings)

Data_train_tensor_text = torch.tensor(Data_train['bert_embeddings'])
Data_test_tensor_text = torch.tensor(Data_test['bert_embeddings'])
Data_val_tensor_text = torch.tensor(Data_val['bert_embeddings'])

Data_train_tensor_image = torch.tensor(Data_train['vgg16_embeddings'])
Data_test_tensor_image = torch.tensor(Data_test['vgg16_embeddings'])
Data_val_tensor_image = torch.tensor(Data_val['vgg16_embeddings'])

'''


def Train_Test_Val_Split(data , test_data_fraction = 0.2, val_data_fraction = 0.1) :
    
    mlb = MultiLabelBinarizer()
    data_genres_one_hot_encoding = mlb.fit_transform(data['genres'])
    Label_names = mlb.classes_
    data_genres_one_hot_encoding = pd.DataFrame(data_genres_one_hot_encoding, columns = mlb.classes_)
    Data_train, Data_test, Labels_train, Labels_test = train_test_split(data, data_genres_one_hot_encoding, test_size = test_data_fraction)
    Labels_train = torch.tensor(Labels_train.values)
    Labels_test = torch.tensor(Labels_test.values)
    
    Data_train, Data_val, Labels_train, Labels_val = train_test_split(Data_train, Labels_train, test_size = val_data_fraction)

    Data_train = Data_train.reset_index(drop=True)
    Data_test = Data_test.reset_index(drop=True)
    Data_val = Data_val.reset_index(drop=True)
    

    return (Data_train, Data_test, Data_val, Labels_train, Labels_test, Labels_val, Label_names)
    






class Training_Testing():

    def __init__(self, Data_train_tensor, Labels_train_tensor, Data_test_tensor, Labels_test_tensor, Data_val_tensor, Labels_val_tensor,
                 hidden_layer_size = 512, Label_names = None, freeze_base_model = True, image_unimodal = False, text_unimodal = False, weight_decay = 0.1, scheduler_step_size = 100, scheduler_lr_fraction = 0.9,
                 hidden_activation = "tanh", batch_size = 32, epochs = 10, sigmoid_thresh = 0.2, learning_rate = 2e-5, num_labels = 23, dropout = 0.1):

      

      if(text_unimodal) :
        #self.config = BertConfig.from_pretrained('bert-base-uncased')
        self.model = BertMultiLabelClassifier(hidden_layer_size = hidden_layer_size, hidden_activation = hidden_activation, dropout = dropout).cuda()

      
      elif(image_unimodal) :
        self.model = VGG16MultiLabelClassifier(hidden_layer_size = hidden_layer_size, hidden_activation = hidden_activation, dropout = dropout).cuda()
 

      if(freeze_base_model) :
        self.model.freeze_base_model()
      self.label_names = Label_names
      self.num_labels = num_labels
      self.batch_size = batch_size
      self.learning_rate = learning_rate
      self.epochs = epochs
      self.sigmoid_thresh = sigmoid_thresh
      self.scheduler_step_size = scheduler_step_size
      self.scheduler_lr_fraction = scheduler_lr_fraction
      self.weight_decay = weight_decay
      self.optimizer = self.SetOptimizer()
      self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      self.results = pd.DataFrame(0, index=['Recall','Precision','F_Score'], columns=['micro', 'macro', 'weighted', 'samples']).astype(float)
      self.epoch_loss_set = []
      self.train_dataloader = self.SetTrainDataloader(Data_train_tensor, Labels_train_tensor)
      self.test_dataloader = self.SetTestDataloader(Data_test_tensor, Labels_test_tensor)
      self.scheduler = self.SetScheduler()

      self.val_accuracy_set = [] 
      self.val_dataloader = self.SetValDataloader(Data_val_tensor, Labels_val_tensor)
      self.class_wise_metrics = None
      self.predictions = None


    def SetOptimizer(self) :

      optimizer = AdamW(self.model.parameters(), lr=self.learning_rate, eps = 1e-6, weight_decay=self.weight_decay)
      return(optimizer)

    

    def SetScheduler(self) :

      scheduler = StepLR(self.optimizer, step_size = self.scheduler_step_size, gamma = self.scheduler_lr_fraction)
      return(scheduler)



    def Get_Metrics(self, actual, predicted) :

      #acc = metrics.accuracy_score(actual, predicted)
      #hamming = metrics.hamming_loss(actual, predicted)
      #(metrics.roc_auc_score(actual, predicted, average=average)
      averages = ('micro', 'macro', 'weighted', 'samples')
      for average in averages:
          precision, recall, fscore, _ = metrics.precision_recall_fscore_support(actual, predicted, average=average)
          self.results[average]['Recall'] += recall
          self.results[average]['Precision'] += precision
          self.results[average]['F_Score'] += fscore


    #source: https://mccormickml.com/2019/07/22/BERT-fine-tuning/
    def Plot_Training_Epoch_Loss(self) :

      sns.set(style='darkgrid')
      sns.set(font_scale=1.5)
      plt.rcParams["figure.figsize"] = (12,6)
      plt.plot(self.epoch_loss_set, 'b-o')
      plt.title("Training loss")
      plt.xlabel("Epoch")
      plt.ylabel("Loss")
      plt.savefig('Training_Epoch_Loss.png',bbox_inches='tight')
      plt.show()


    def Plot_Training_Epoch_Accuracy(self) :

      sns.set(style='darkgrid')
      sns.set(font_scale=1.5)
      plt.rcParams["figure.figsize"] = (12,6)
      plt.plot(self.val_accuracy_set, 'b-o')
      plt.title("Micro F1 Score")
      plt.xlabel("Epoch")
      plt.ylabel("Validation Accuracy")
      plt.savefig('Training_Validation_Accuracy.png',bbox_inches='tight')
      plt.show()


    #source: https://mccormickml.com/2019/07/22/BERT-fine-tuning/
    def format_time(self, elapsed):
      '''
      Takes a time in seconds and returns a string hh:mm:ss
      '''
      # Round to the nearest second.
      elapsed_rounded = int(round((elapsed)))
      return str(datetime.timedelta(seconds=elapsed_rounded))


    def SetTrainDataloader(self, Data_train_tensor, Labels_train_tensor) :

      train_dataset = TensorDataset(Data_train_tensor, Labels_train_tensor)
      train_sampler = RandomSampler(train_dataset)
      train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size = self.batch_size)
      return(train_dataloader)


    def SetTestDataloader(self, Data_test_tensor, Labels_test_tensor) :
      
      test_dataset = TensorDataset(Data_test_tensor, Labels_test_tensor)
      test_sampler = SequentialSampler(test_dataset)
      test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size = Data_test_tensor.shape[0])
      return(test_dataloader)


    def SetValDataloader(self, Data_val_tensor, Labels_val_tensor) :
      
      val_dataset = TensorDataset(Data_val_tensor, Labels_val_tensor)
      val_sampler = SequentialSampler(val_dataset)
      val_dataloader = DataLoader(val_dataset, sampler=val_sampler, batch_size = Data_val_tensor.shape[0])
      return(val_dataloader)



    def Train(self) :

      for _ in trange(self.epochs, desc="Epoch"):
        
        self.model.train()
        epoch_loss = 0

        # Measure how long the training epoch takes.
        t0 = time.time()
    
        for step_num, batch_data in enumerate(self.train_dataloader):

          # Progress update every 30 batches.
          if step_num % 30 == 0 and not step_num == 0:
            elapsed = self.format_time(time.time() - t0)
            print('  Batch : ',step_num, ' , Time elapsed : ',elapsed)

          samples, labels = tuple(t.to(self.device) for t in batch_data)
          self.optimizer.zero_grad()
          logits = self.model(samples.float())
          loss_fct = BCEWithLogitsLoss()
          batch_loss = loss_fct(logits.view(-1, self.num_labels).float(), labels.view(-1, self.num_labels).float())
          batch_loss.backward()
          self.optimizer.step()
          self.scheduler.step()
          epoch_loss += batch_loss.item()

        avg_epoch_loss = epoch_loss/len(self.train_dataloader)
        print("\nTrain loss for epoch: ",avg_epoch_loss)
        print("\nTraining epoch took: {:}".format(self.format_time(time.time() - t0)))
        self.epoch_loss_set.append(avg_epoch_loss)

        #Validation on the epoch
        self.model.eval()
        epoch_f1_score = 0

        for batch_data in self.val_dataloader:
          samples, labels = tuple(t.to(self.device) for t in batch_data)
          with torch.no_grad():
            output = self.model(samples.float())

          threshold = torch.Tensor([self.sigmoid_thresh]).to(self.device)
          predictions = (output > threshold).int()

          predictions = predictions.detach().cpu().numpy()
          labels = labels.to('cpu').numpy()
      
          micro_f_score = metrics.f1_score(labels,predictions,average="micro")
          epoch_f1_score += micro_f_score

        avg_val_f1_score = epoch_f1_score/len(self.val_dataloader)
        print("\n Micro F1 score for epoch: ",avg_val_f1_score,"\n")
        self.val_accuracy_set.append(avg_val_f1_score)

      torch.save(self.model.state_dict(), "/content/drive/My Drive/dataset/model.pt")
      self.Plot_Training_Epoch_Loss()
      self.Plot_Training_Epoch_Accuracy()

    

    def Test(self) :

      # Put model in evaluation mode to evaluate loss on the test set
      self.model.eval()

      for batch_data in self.test_dataloader:
  
        samples, labels = tuple(t.to(self.device) for t in batch_data)
      
        # Telling the model not to compute or store gradients, saving memory and speeding up validation
        # Forward pass, calculate logit predictions
        with torch.no_grad():
          output = self.model(samples)

        threshold = torch.Tensor([self.sigmoid_thresh]).to(self.device)
        predictions = (output > threshold).int()

        # Move preds and labels to CPU
        predictions = predictions.detach().cpu().numpy()
        labels = labels.to('cpu').numpy()
      
        self.Get_Metrics(labels, predictions)
        self.class_wise_metrics = metrics.classification_report(labels, predictions, target_names= list(self.label_names))
        self.predictions = predictions

      self.results = self.results/len(self.test_dataloader)
      #print("Test data metrics : \n")
      
      print("\nGenres with no predicted samples : ", self.label_names[np.where(np.sum(predictions, axis=0) == 0)[0]])

      return(self.results)
            
      
      
      
      
      


class Training_Testing_MM_Cat():

    def __init__(self, Data_train_tensor_text, Data_train_tensor_image, Labels_train_tensor, 
                 Data_test_tensor_text, Data_test_tensor_image, Labels_test_tensor,
                 Data_val_tensor_text, Data_val_tensor_image, Labels_val_tensor, 
                 Label_names = None, hidden_layer_size = 512, weight_decay = 0.1, scheduler_step_size = 100, scheduler_lr_fraction = 0.9,
                 hidden_activation = "tanh", batch_size = 32, epochs = 10, sigmoid_thresh = 0.2, learning_rate = 2e-5, num_labels = 23, dropout = 0.1):


      #self.dropout = dropout
      #self.hidden_layer_size = hidden_layer_size
      #self.hidden_activation = hidden_activation
      self.model = MM_MultiLabelClassifier(hidden_layer_size = hidden_layer_size, hidden_activation = hidden_activation, dropout = dropout).cuda()
      self.label_names = Label_names
      self.num_labels = num_labels
      self.batch_size = batch_size
      self.learning_rate = learning_rate
      self.epochs = epochs
      self.sigmoid_thresh = sigmoid_thresh
      self.scheduler_step_size = scheduler_step_size
      self.scheduler_lr_fraction = scheduler_lr_fraction
      self.weight_decay = weight_decay      
      self.optimizer = self.SetOptimizer()
      self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      self.results = pd.DataFrame(0, index=['Recall','Precision','F_Score'], columns=['micro', 'macro', 'weighted', 'samples']).astype(float)
      self.epoch_loss_set = []
      self.train_dataloader = self.SetTrainDataloader_MM(Data_train_tensor_text, Data_train_tensor_image, Labels_train_tensor)
      self.test_dataloader = self.SetTestDataloader_MM(Data_test_tensor_text, Data_test_tensor_image, Labels_test_tensor) 
      self.scheduler = self.SetScheduler()

      self.val_accuracy_set = [] 
      self.val_dataloader = self.SetValDataloader_MM(Data_val_tensor_text, Data_val_tensor_image, Labels_val_tensor)
      self.class_wise_metrics = None
      self.predictions = None
      

    def SetOptimizer(self) :

      optimizer = AdamW(self.model.parameters(), lr=self.learning_rate, eps = 1e-6, weight_decay=self.weight_decay)
      return(optimizer)

    

    def SetScheduler(self) :

      scheduler = StepLR(self.optimizer, step_size = self.scheduler_step_size, gamma = self.scheduler_lr_fraction)
      return(scheduler)



    def Get_Metrics(self, actual, predicted) :

      #acc = metrics.accuracy_score(actual, predicted)
      #hamming = metrics.hamming_loss(actual, predicted)
      #(metrics.roc_auc_score(actual, predicted, average=average)
      averages = ('micro', 'macro', 'weighted', 'samples')
      for average in averages:
          precision, recall, fscore, _ = metrics.precision_recall_fscore_support(actual, predicted, average=average)
          self.results[average]['Recall'] += recall
          self.results[average]['Precision'] += precision
          self.results[average]['F_Score'] += fscore


    #source: https://mccormickml.com/2019/07/22/BERT-fine-tuning/
    def Plot_Training_Epoch_Loss(self) :

      sns.set(style='darkgrid')
      sns.set(font_scale=1.5)
      plt.rcParams["figure.figsize"] = (12,6)
      plt.plot(self.epoch_loss_set, 'b-o')
      plt.title("Training loss")
      plt.xlabel("Epoch")
      plt.ylabel("Loss")
      plt.savefig('Training_Epoch_Loss.png',bbox_inches='tight')
      plt.show()


    def Plot_Training_Epoch_Accuracy(self) :

      sns.set(style='darkgrid')
      sns.set(font_scale=1.5)
      plt.rcParams["figure.figsize"] = (12,6)
      plt.plot(self.val_accuracy_set, 'b-o')
      plt.title("Micro F1 Score")
      plt.xlabel("Epoch")
      plt.ylabel("Validation Accuracy")
      plt.savefig('Training_Validation_Accuracy.png',bbox_inches='tight')
      plt.show()


    #source: https://mccormickml.com/2019/07/22/BERT-fine-tuning/
    def format_time(self, elapsed):
      '''
      Takes a time in seconds and returns a string hh:mm:ss
      '''
      # Round to the nearest second.
      elapsed_rounded = int(round((elapsed)))
      return str(datetime.timedelta(seconds=elapsed_rounded))


    def SetTrainDataloader_MM(self, Data_train_tensor_text, Data_train_tensor_image, Labels_train_tensor) :

      train_dataset = TensorDataset(Data_train_tensor_text, Data_train_tensor_image, Labels_train_tensor)
      train_sampler = RandomSampler(train_dataset)
      train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size = self.batch_size)
      return(train_dataloader)


    def SetTestDataloader_MM(self, Data_test_tensor_text, Data_test_tensor_image, Labels_test_tensor) :
      
      test_dataset = TensorDataset(Data_test_tensor_text, Data_test_tensor_image, Labels_test_tensor)
      test_sampler = SequentialSampler(test_dataset)
      test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size = Data_test_tensor_text.shape[0])
      return(test_dataloader)


    def SetValDataloader_MM(self, Data_val_tensor_text, Data_val_tensor_image, Labels_val_tensor) :
      
      val_dataset = TensorDataset(Data_val_tensor_text, Data_val_tensor_image, Labels_val_tensor)
      val_sampler = SequentialSampler(val_dataset)
      val_dataloader = DataLoader(val_dataset, sampler=val_sampler, batch_size = Data_val_tensor_text.shape[0])
      return(val_dataloader)


    def Train(self) :

      for _ in trange(self.epochs, desc="Epoch"):
        
        self.model.train()
        epoch_loss = 0

        # Measure how long the training epoch takes.
        t0 = time.time()
    
        for step_num, batch_data in enumerate(self.train_dataloader):

          # Progress update every 30 batches.
          if step_num % 30 == 0 and not step_num == 0:
            elapsed = self.format_time(time.time() - t0)
            print('  Batch : ',step_num, ' , Time elapsed : ',elapsed)

          samples_text, samples_image, labels = tuple(t.to(self.device) for t in batch_data)
          self.optimizer.zero_grad()
          logits = self.model(samples_image.float(), samples_text.float())
          loss_fct = BCEWithLogitsLoss()
          batch_loss = loss_fct(logits.view(-1, self.num_labels).float(), labels.view(-1, self.num_labels).float())
          batch_loss.backward()
          self.optimizer.step()
          self.scheduler.step()
          epoch_loss += batch_loss.item()

        avg_epoch_loss = epoch_loss/len(self.train_dataloader)
        print("\nTrain loss for epoch: ",avg_epoch_loss)
        print("\nTraining epoch took: {:}".format(self.format_time(time.time() - t0)))
        self.epoch_loss_set.append(avg_epoch_loss)

        #Validation on the epoch
        self.model.eval()
        epoch_f1_score = 0

        for batch_data in self.val_dataloader:
          samples_text, samples_image, labels = tuple(t.to(self.device) for t in batch_data)
          with torch.no_grad():
            output = self.model(samples_image.float(), samples_text.float())

          threshold = torch.Tensor([self.sigmoid_thresh]).to(self.device)
          predictions = (output > threshold).int()

          predictions = predictions.detach().cpu().numpy()
          labels = labels.to('cpu').numpy()
      
          micro_f_score = metrics.f1_score(labels,predictions,average="micro")
          epoch_f1_score += micro_f_score

        avg_val_f1_score = epoch_f1_score/len(self.val_dataloader)
        print("\n Micro F1 score for epoch: ",avg_val_f1_score,"\n")
        self.val_accuracy_set.append(avg_val_f1_score)

      torch.save(self.model.state_dict(), "/content/drive/My Drive/dataset/model.pt")
      self.Plot_Training_Epoch_Loss()
      self.Plot_Training_Epoch_Accuracy()

    

    def Test(self) :

      # Put model in evaluation mode to evaluate loss on the test set
      self.model.eval()

      for batch_data in self.test_dataloader:
  
        samples_text, samples_image, labels = tuple(t.to(self.device) for t in batch_data)
      
        # Telling the model not to compute or store gradients, saving memory and speeding up validation
        # Forward pass, calculate logit predictions
        with torch.no_grad():
          output = self.model(samples_image.float(), samples_text.float())

        threshold = torch.Tensor([self.sigmoid_thresh]).to(self.device)
        predictions = (output > threshold).int()

        # Move preds and labels to CPU
        predictions = predictions.detach().cpu().numpy()
        labels = labels.to('cpu').numpy()
      
        self.Get_Metrics(labels, predictions)
        self.class_wise_metrics = metrics.classification_report(labels, predictions, target_names= list(self.label_names))
        self.predictions = predictions
    
      self.results = self.results/len(self.test_dataloader)
      #print("Test data metrics : \n")
      
      print("\nGenres with no predicted samples : ", self.label_names[np.where(np.sum(predictions, axis=0) == 0)[0]])

      return(self.results)
      
      
      
      
      
      

'''
USAGE:


train_test = Training_Testing_MM_GMU( Data_train_tensor_text, Data_train_tensor_image, Labels_train_tensor, 
                                  Data_test_tensor_text, Data_test_tensor_image, Labels_test_tensor, 
                                  Data_val_tensor_text, Data_val_tensor_image, Labels_val_tensor, Label_names=Label_names, 
                                  hidden_layer_size = 512, epochs = 70, batch_size= 256, learning_rate = 0.01, dropout = 0.7, scheduler_step_size = 30, 
                                  scheduler_lr_fraction = 0.85, sigmoid_thresh = 0.3, num_maxout_units = 10, weight_decay = 0.1, max_norm = 10)
train_test.Train()
train_test.Test()

'''

class Training_Testing_MM_GMU():

    def __init__(self, Data_train_tensor_text, Data_train_tensor_image, Labels_train_tensor, 
                 Data_test_tensor_text, Data_test_tensor_image, Labels_test_tensor, 
                 Data_val_tensor_text, Data_val_tensor_image, Labels_val_tensor,
                 Label_names = None, hidden_layer_size = 512, num_maxout_units = 2, weight_decay= 0.1, scheduler_step_size = 30, scheduler_lr_fraction = 0.8,
                 hidden_activation = "tanh", batch_size = 32, epochs = 10, sigmoid_thresh = 0.2, learning_rate = 2e-5, num_labels = 23, dropout = 0.1, max_norm = 5):


      self.model = GMU(num_maxout_units = num_maxout_units, hidden_layer_size = hidden_layer_size, hidden_activation = hidden_activation, dropout = dropout).cuda()
      self.label_names = Label_names
      self.num_labels = num_labels
      self.batch_size = batch_size
      self.learning_rate = learning_rate
      self.max_norm = max_norm
      self.epochs = epochs
      self.sigmoid_thresh = sigmoid_thresh
      self.scheduler_step_size = scheduler_step_size
      self.scheduler_lr_fraction = scheduler_lr_fraction
      self.weight_decay = weight_decay
      self.optimizer = self.SetOptimizer()
      self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      self.results = pd.DataFrame(0, index=['Recall','Precision','F_Score'], columns=['micro', 'macro', 'weighted', 'samples']).astype(float)
      self.epoch_loss_set = []
      self.train_dataloader = self.SetTrainDataloader_MM(Data_train_tensor_text, Data_train_tensor_image, Labels_train_tensor)
      self.test_dataloader = self.SetTestDataloader_MM(Data_test_tensor_text, Data_test_tensor_image, Labels_test_tensor) 
      self.scheduler = self.SetScheduler()

      self.val_accuracy_set = [] 
      self.val_dataloader = self.SetValDataloader_MM(Data_val_tensor_text, Data_val_tensor_image, Labels_val_tensor)
      self.class_wise_metrics = None
      self.predictions = None



    def SetOptimizer(self) :

      optimizer = AdamW(self.model.parameters(), lr=self.learning_rate,  eps = 1e-6, weight_decay=self.weight_decay)
      #optimizer = Adam(self.model.parameters(), lr=self.learning_rate,  eps = 1e-6, weight_decay=self.weight_decay)
      return(optimizer)

    

    def SetScheduler(self) :

      '''
      scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps = 10, 
                                                 num_training_steps = self.epochs*len(self.train_dataloader))
      '''
      scheduler = StepLR(self.optimizer, step_size = self.scheduler_step_size, gamma = self.scheduler_lr_fraction)
      return(scheduler) 



    def Get_Metrics(self, actual, predicted) :

      #acc = metrics.accuracy_score(actual, predicted)
      #hamming = metrics.hamming_loss(actual, predicted)
      #(metrics.roc_auc_score(actual, predicted, average=average)
      averages = ('micro', 'macro', 'weighted', 'samples')
      for average in averages:
          precision, recall, fscore, _ = metrics.precision_recall_fscore_support(actual, predicted, average=average)
          self.results[average]['Recall'] += recall
          self.results[average]['Precision'] += precision
          self.results[average]['F_Score'] += fscore



    #source: https://mccormickml.com/2019/07/22/BERT-fine-tuning/
    def Plot_Training_Epoch_Loss(self) :

      sns.set(style='darkgrid')
      sns.set(font_scale=1.5)
      plt.rcParams["figure.figsize"] = (12,6)
      plt.plot(self.epoch_loss_set, 'b-o')
      plt.title("Training loss")
      plt.xlabel("Epoch")
      plt.ylabel("Loss")
      plt.savefig('Training_Epoch_Loss.png',bbox_inches='tight')
      plt.show()

    
    def Plot_Training_Epoch_Accuracy(self) :

      sns.set(style='darkgrid')
      sns.set(font_scale=1.5)
      plt.rcParams["figure.figsize"] = (12,6)
      plt.plot(self.val_accuracy_set, 'b-o')
      plt.title("Micro F1 Score")
      plt.xlabel("Epoch")
      plt.ylabel("Validation Accuracy")
      plt.savefig('Training_Validation_Accuracy.png',bbox_inches='tight')
      plt.show()


    #source: https://mccormickml.com/2019/07/22/BERT-fine-tuning/
    def format_time(self, elapsed):
      '''
      Takes a time in seconds and returns a string hh:mm:ss
      '''
      # Round to the nearest second.
      elapsed_rounded = int(round((elapsed)))
      return str(datetime.timedelta(seconds=elapsed_rounded))


    def SetTrainDataloader_MM(self, Data_train_tensor_text, Data_train_tensor_image, Labels_train_tensor) :

      train_dataset = TensorDataset(Data_train_tensor_text, Data_train_tensor_image, Labels_train_tensor)
      train_sampler = RandomSampler(train_dataset)
      train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size = self.batch_size)
      return(train_dataloader)


    def SetTestDataloader_MM(self, Data_test_tensor_text, Data_test_tensor_image, Labels_test_tensor) :
      
      test_dataset = TensorDataset(Data_test_tensor_text, Data_test_tensor_image, Labels_test_tensor)
      test_sampler = SequentialSampler(test_dataset)
      #test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size = self.batch_size)
      test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size = Data_test_tensor_text.shape[0])
      return(test_dataloader)

    
    def SetValDataloader_MM(self, Data_val_tensor_text, Data_val_tensor_image, Labels_val_tensor) :
      
      val_dataset = TensorDataset(Data_val_tensor_text, Data_val_tensor_image, Labels_val_tensor)
      val_sampler = SequentialSampler(val_dataset)
      #test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size = self.batch_size)
      val_dataloader = DataLoader(val_dataset, sampler=val_sampler, batch_size = Data_val_tensor_text.shape[0])
      return(val_dataloader)

   
    def Train(self) :

      for _ in trange(self.epochs, desc="Epoch"):
        
        self.model.train()
        epoch_loss = 0

        # Measure how long the training epoch takes.
        t0 = time.time()
    
        for step_num, batch_data in enumerate(self.train_dataloader):

          # Progress update every 30 batches.
          if step_num % 30 == 0 and not step_num == 0:
            elapsed = self.format_time(time.time() - t0)
            print('  Batch : ',step_num, ' , Time elapsed : ',elapsed)

          samples_text, samples_image, labels = tuple(t.to(self.device) for t in batch_data)
          self.optimizer.zero_grad()
          logits = self.model(samples_image.float(), samples_text.float())
          loss_fct = BCEWithLogitsLoss()
          batch_loss = loss_fct(logits.view(-1, self.num_labels).float(), labels.view(-1, self.num_labels).float())
          batch_loss.backward()
          clip_grad_norm_(self.model.parameters(), norm_type = 2, max_norm = self.max_norm)
          self.optimizer.step()
          self.scheduler.step()
          epoch_loss += batch_loss.item()

        avg_epoch_loss = epoch_loss/len(self.train_dataloader)
        print("\nTrain loss for epoch: ",avg_epoch_loss)
        print("\nTraining epoch took: {:}".format(self.format_time(time.time() - t0)))
        self.epoch_loss_set.append(avg_epoch_loss)

        #Validation on the epoch
        self.model.eval()
        epoch_f1_score = 0

        for batch_data in self.val_dataloader:
          samples_text, samples_image, labels = tuple(t.to(self.device) for t in batch_data)
          with torch.no_grad():
            output = self.model(samples_image.float(), samples_text.float())

          threshold = torch.Tensor([self.sigmoid_thresh]).to(self.device)
          predictions = (output > threshold).int()

          predictions = predictions.detach().cpu().numpy()
          labels = labels.to('cpu').numpy()
      
          micro_f_score = metrics.f1_score(labels,predictions,average="micro")
          epoch_f1_score += micro_f_score

        avg_val_f1_score = epoch_f1_score/len(self.val_dataloader)
        print("\n Micro F1 score for epoch: ",avg_val_f1_score,"\n")
        self.val_accuracy_set.append(avg_val_f1_score)

      #torch.save(self.model.state_dict(), "/content/drive/My Drive/dataset/model.pt")
      self.Plot_Training_Epoch_Loss()
      self.Plot_Training_Epoch_Accuracy()
   

    def Test(self) :

      # Put model in evaluation mode to evaluate loss on the test set
      self.model.eval()

      for batch_data in self.test_dataloader:
  
        samples_text, samples_image, labels = tuple(t.to(self.device) for t in batch_data)
      
        # Telling the model not to compute or store gradients, saving memory and speeding up validation
        # Forward pass, calculate logit predictions
        with torch.no_grad():
          output = self.model(samples_image.float(), samples_text.float())

        threshold = torch.Tensor([self.sigmoid_thresh]).to(self.device)
        predictions = (output > threshold).int()

        # Move preds and labels to CPU
        predictions = predictions.detach().cpu().numpy()
        labels = labels.to('cpu').numpy()

        self.predictions = predictions
        self.Get_Metrics(labels, predictions)
        self.class_wise_metrics = metrics.classification_report(labels, predictions, target_names= list(self.label_names))
        
    
      self.results = self.results/len(self.test_dataloader)
      #print("Test data metrics : \n")

      #print("\nGenres with no predicted samples : ", self.label_names[np.where(np.sum(predictions, axis=0) == 0)[0]])
      
      return(self.results)