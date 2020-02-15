import torch
from torch import nn
from torchvision import models
import transformers
from transformers import BertModel, BertConfig



#source: https://github.com/Duncanswilson/maxout-pytorch/blob/master/maxout_pytorch.ipynb
'''
Required for Maxout_MLP
'''
class ListModule(object):
    def __init__(self, module, prefix, *args):
        self.module = module
        self.prefix = prefix
        self.num_module = 0
        for new_module in args:
            self.append(new_module)

    def append(self, new_module):
        if not isinstance(new_module, nn.Module):
            raise ValueError('Not a Module')
        else:
            self.module.add_module(self.prefix + str(self.num_module), new_module)
            self.num_module += 1

    def __len__(self):
        return self.num_module

    def __getitem__(self, i):
        if i < 0 or i >= self.num_module:
            raise IndexError('Out of bound')
        return getattr(self.module, self.prefix + str(i))




'''
Required for GMU
'''
class Maxout_MLP(nn.Module):
    
    def __init__(self, hidden_layer_size1, hidden_layer_size2, dropout, num_maxout_units=2):
        
        super(Maxout_MLP, self).__init__()
        self.fc1_list = ListModule(self, "fc1_")
        self.fc2_list = ListModule(self, "fc2_")
        self.hidden_layer_size1 = hidden_layer_size1
        self.hidden_layer_size2 = hidden_layer_size2
        for _ in range(num_maxout_units):
            self.fc1_list.append(nn.Linear(self.hidden_layer_size1, self.hidden_layer_size2))
            self.fc2_list.append(nn.Linear(self.hidden_layer_size2, self.hidden_layer_size2))
        self.dropout = nn.Dropout(p=dropout)
        self.bn0 = nn.BatchNorm1d(self.hidden_layer_size1)
        self.bn1 = nn.BatchNorm1d(self.hidden_layer_size2)
        self.bn2 = nn.BatchNorm1d(self.hidden_layer_size2)

    def forward(self, x): 
        
        x = x.view(-1, self.hidden_layer_size1)
        x = self.bn0(x)
        x = self.maxout(x, self.fc1_list)
        x = self.bn1(x)
        x = self.dropout(x)
        x = self.maxout(x, self.fc2_list)
        x = self.bn2(x)
        return x

    def maxout(self, x, layer_list):
        
        max_output = layer_list[0](x)
        for _, layer in enumerate(layer_list, start=1):
            max_output = torch.max(max_output, layer(x))
        return max_output




'''
GMU based multimodal classifier
'''
class GMU(nn.Module):

    def __init__(self, num_maxout_units = 2, hidden_layer_size = 512, text_embeddings_size = 3072, img_embeddings_size = 4096, num_labels = 23, hidden_activation = None, dropout = 0.1):

        super(GMU, self).__init__()
        self.num_labels = num_labels
        self.hidden_layer_size = hidden_layer_size

        self.linear_h_text = torch.nn.Linear(text_embeddings_size, self.hidden_layer_size, bias = False)
        self.linear_h_image = torch.nn.Linear(img_embeddings_size, self.hidden_layer_size, bias = False)
        self.linear_z = torch.nn.Linear(text_embeddings_size + img_embeddings_size, self.hidden_layer_size, bias = False)

        self.tanh = torch.nn.Tanh()
        self.sigmoid = torch.nn.Sigmoid()

        self.dropout = nn.Dropout(p=dropout)
        self.linear = torch.nn.Linear(self.hidden_layer_size, self.num_labels)
        
        self.maxout = Maxout_MLP(self.hidden_layer_size, self.hidden_layer_size, dropout, num_maxout_units=num_maxout_units)

    def forward(self, image_embeddings, text_embeddings):
        
        image_h = self.linear_h_image(image_embeddings)
        image_h = self.tanh(image_h)
        text_h = self.linear_h_text(text_embeddings)
        text_h = self.tanh(text_h)
        concat = torch.cat((image_embeddings, text_embeddings), 1)
        z = self.linear_z(concat)
        z = self.sigmoid(z)
        gmu_output = z*image_h + (1-z)*text_h
        
        maxout_mlp_output = self.maxout(gmu_output)
        #dropped_layer = self.dropout(gmu_output)

        logits = self.linear(maxout_mlp_output)
        if(self.training) :
            return logits
        else :
            output = self.sigmoid(logits)
            return output




'''
VGG16 based image unimodal classifier 
'''

class VGG16MultiLabelClassifier(nn.Module):
  
  def __init__(self, hidden_layer_size = 512, hidden_activation = "tanh", input_size = 4096, num_labels = 23, dropout = 0.1, gen_embeddings = False):

        super(VGG16MultiLabelClassifier,self).__init__()
        self.num_labels = num_labels
        self.input_size = input_size
        self.base_model = models.vgg16()
        self.hidden_layer_size = hidden_layer_size
        self.gen_embeddings = gen_embeddings
        self.embedding_layers = list(self.base_model.classifier.children())[:-2] 
        self.base_model.classifier = nn.Sequential(*self.embedding_layers)

        self.hidden_layer1 = torch.nn.Linear(self.input_size, self.hidden_layer_size)        
        if(hidden_activation == "tanh") :
          self.hidden_activation1 = torch.nn.Tanh()
          self.hidden_activation2 = torch.nn.Tanh()
        elif(hidden_activation == "relu") :
          self.hidden_activation1 = torch.nn.ReLU()
          self.hidden_activation2 = torch.nn.ReLU()
        elif(hidden_activation == "sigmoid") :
          self.hidden_activation1 = torch.nn.Sigmoid()
          self.hidden_activation2 = torch.nn.Sigmoid()
        elif(hidden_activation == "leaky_relu") :
          self.hidden_activation1 = torch.nn.LeakyReLU()
          self.hidden_activation2 = torch.nn.LeakyReLU()
        else :
          return("Invalid hidden_activation parameter value.")

        self.hidden_layer2 = torch.nn.Linear(self.hidden_layer_size, self.hidden_layer_size)

        self.output_layer = torch.nn.Linear(self.hidden_layer_size, self.num_labels)
        self.output_activation = torch.nn.Sigmoid()

        self.dropout = nn.Dropout(p=dropout)


  def forward(self, input):
        
        if(self.gen_embeddings):
            embeddings = self.base_model(input)
            return embeddings
        else :
            embeddings = input
        
        logits1 = self.hidden_layer1(embeddings)
        activation1 = self.hidden_activation1(logits1)

        dropped1 = self.dropout(activation1)

        logits2 = self.hidden_layer2(dropped1)
        activation2 = self.hidden_activation2(logits2)

        dropped2 = self.dropout(activation2)

        logits3 = self.output_layer(dropped2)
        if(self.training) :
            return logits3
        else :
            output = self.output_activation(logits3)
            return output


  def freeze_base_model(self):
        for param in self.base_model.features.parameters():
            param.requires_grad = False


  def unfreeze_base_model(self):
        for param in self.base_model.features.parameters():
            param.requires_grad = True





'''
Bert based text unimodal classifier 
'''

class BertMultiLabelClassifier(nn.Module):

    def __init__(self, hidden_layer_size = 512, hidden_activation = "tanh", input_size = 3072, num_labels = 23, dropout = 0.1, gen_embeddings = False, use_pooled_output = False):

        super(BertMultiLabelClassifier, self).__init__()
        self.num_labels = num_labels
        self.input_size = input_size
        self.gen_embeddings = gen_embeddings
        self.hidden_layer_size = hidden_layer_size
        self.use_pooled_output = use_pooled_output

        if(self.use_pooled_output) :
            self.base_model = BertModel.from_pretrained('bert-base-uncased') #to generate embedding size of 768
        else :
            config = BertConfig.from_pretrained('bert-base-uncased')
            config.output_hidden_states=True
            self.base_model = BertModel.from_pretrained('bert-base-uncased', config=config) #to generate embedding size of 3072 by concatenating last four layers
        
        self.hidden_layer1 = torch.nn.Linear(self.input_size, self.hidden_layer_size)        
        if(hidden_activation == "tanh") :
          self.hidden_activation1 = torch.nn.Tanh()
          self.hidden_activation2 = torch.nn.Tanh()
        elif(hidden_activation == "relu") :
          self.hidden_activation1 = torch.nn.ReLU()
          self.hidden_activation2 = torch.nn.ReLU()
        elif(hidden_activation == "sigmoid") :
          self.hidden_activation1 = torch.nn.Sigmoid()
          self.hidden_activation2 = torch.nn.Sigmoid()
        elif(hidden_activation == "leaky_relu") :
          self.hidden_activation1 = torch.nn.LeakyReLU()
          self.hidden_activation2 = torch.nn.LeakyReLU()
        else :
          return("Invalid hidden_activation parameter value.")

        self.hidden_layer2 = torch.nn.Linear(self.hidden_layer_size, self.hidden_layer_size)

        self.output_layer = torch.nn.Linear(self.hidden_layer_size, self.num_labels)
        self.output_activation = torch.nn.Sigmoid()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, indexed_tokens, segment_ids=None, masked_ids=None):
        
        if(self.gen_embeddings):
            if(self.use_pooled_output) :
                pooled_output = self.base_model(indexed_tokens, segment_ids, masked_ids)
                embeddings = pooled_output[1]
                return embeddings
            else:
                output = self.base_model(indexed_tokens, segment_ids, masked_ids)
                embeddings = torch.cat((output[-1][-1],output[-1][-2],output[-1][-3],output[-1][-4]), dim = 2).mean(1)
                return embeddings
        else :
            embeddings = indexed_tokens
        
        logits1 = self.hidden_layer1(embeddings)
        activation1 = self.hidden_activation1(logits1)

        dropped1 = self.dropout(activation1)

        logits2 = self.hidden_layer2(dropped1)
        activation2 = self.hidden_activation2(logits2)

        dropped2 = self.dropout(activation2)

        logits3 = self.output_layer(dropped2)
        if(self.training) :
            return logits3
        else :
            output = self.output_activation(logits3)
            return output


    def freeze_base_model(self):
        for param in self.base_model.parameters():
            param.requires_grad = False


    def unfreeze_base_model(self):
        for param in self.base_model.named_parameters():
            param.requires_grad = True

'''
Concatenation based multimodal classifier
'''

class MM_MultiLabelClassifier(nn.Module):

    def __init__(self, hidden_layer_size = 512, hidden_activation = "tanh", input_size = 7168, num_labels = 23, dropout = 0.1):

        super(MM_MultiLabelClassifier, self).__init__()
        self.num_labels = num_labels
        self.hidden_layer_size = hidden_layer_size
        self.input_size = input_size

        self.hidden_layer1 = torch.nn.Linear(self.input_size, self.hidden_layer_size)
        
        if(hidden_activation == "tanh") :
          self.hidden_activation1 = torch.nn.Tanh()
          self.hidden_activation2 = torch.nn.Tanh()
        elif(hidden_activation == "relu") :
          self.hidden_activation1 = torch.nn.ReLU()
          self.hidden_activation2 = torch.nn.ReLU()
        elif(hidden_activation == "sigmoid") :
          self.hidden_activation1 = torch.nn.Sigmoid()
          self.hidden_activation2 = torch.nn.Sigmoid()
        elif(hidden_activation == "leaky_relu") :
          self.hidden_activation1 = torch.nn.LeakyReLU()
          self.hidden_activation2 = torch.nn.LeakyReLU()
        else :
          return("Invalid hidden_activation parameter value.")

        self.hidden_layer2 = torch.nn.Linear(self.hidden_layer_size, self.hidden_layer_size)

        self.output_layer = torch.nn.Linear(self.hidden_layer_size, self.num_labels)
        self.output_activation = torch.nn.Sigmoid()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, image_embeddings, text_embeddings):
        
        embeddings = torch.cat((image_embeddings, text_embeddings), 1)

        logits1 = self.hidden_layer1(embeddings)
        activation1 = self.hidden_activation1(logits1)

        dropped1 = self.dropout(activation1)

        logits2 = self.hidden_layer2(dropped1)
        activation2 = self.hidden_activation2(logits2)

        dropped2 = self.dropout(activation2)

        logits3 = self.output_layer(dropped2)
        if(self.training) :
            return logits3
        else :
            output = self.output_activation(logits3)
            return output