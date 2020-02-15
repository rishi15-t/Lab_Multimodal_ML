import torch
from torch.utils.data import Dataset
from torchvision import transforms
import nltk
from transformers import BertTokenizer
from nltk import tokenize




class ImageTransformation_VGG16(Dataset) :

    def __init__(self, PIL_images_list, train = True):

        self.PIL_images_list = PIL_images_list
        self.train = train

        self.transformations_train = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        self.transformations_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def __len__(self):
        return len(self.PIL_images_list)

    def __getitem__(self, idx):

        if self.train:
            sample = self.transformations_train(self.PIL_images_list[idx])
        else:
            sample = self.transformations_test(self.PIL_images_list[idx])
        return sample




class TextTransformation_Bert(Dataset) :

    def __init__(self, text_list, max_input_length = 512):

        self.text_list = text_list
        self.indexed_tokens = []
        self.segment_ids = []
        self.masked_ids = []
        self.max_input_length = max_input_length

    def GetIndexedTokens(self, text):
        
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        tokenized_text = tokenizer.tokenize(text)
        tokenized_text.append("[SEP]")
        tokenized_text.insert(0,"[CLS]")
        self.indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        

    def GetSegmentIds(self) :
        
        self.segment_ids = [1] * len(self.indexed_tokens)

 
    def GetMaskedIds(self) :
        
        self.masked_ids = [1] * len(self.indexed_tokens)


    def Padding(self) :

        if(len(self.indexed_tokens) < self.max_input_length) :
           padding = [0]*(self.max_input_length - len(self.indexed_tokens))
           self.indexed_tokens += padding
           self.segment_ids += padding
           self.masked_ids += padding
        else :
           del self.indexed_tokens[self.max_input_length:]
           del self.segment_ids[self.max_input_length:]
           del self.masked_ids[self.max_input_length:]

    def __len__(self):
        return len(self.text_list)


    def __getitem__(self, idx):

        text = self.text_list[idx]

        self.GetIndexedTokens(text)
        self.GetSegmentIds()
        self.GetMaskedIds()
        self.Padding()
        
        self.indexed_tokens = torch.tensor(self.indexed_tokens)
        self.segment_ids = torch.tensor(self.segment_ids)
        self.masked_ids = torch.tensor(self.masked_ids)
        
        return self.indexed_tokens, self.segment_ids , self.masked_ids