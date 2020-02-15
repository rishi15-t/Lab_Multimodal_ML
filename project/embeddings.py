import torch
from torch.utils.data import DataLoader, SequentialSampler




def GetImageEmbeddings_VGG16(dataset, batch_size = 32):
  
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  image_tensors = ImageTransformation_VGG16(dataset['image'])
  sampler = SequentialSampler(image_tensors)
  dataloader = DataLoader(image_tensors, sampler=sampler, batch_size=batch_size)
  
  results = torch.Tensor().to(device)

  model = VGG16MultiLabelClassifier(gen_embeddings = True)
  model.to(device)
  model.eval()

  for num, batch_data in enumerate(dataloader): 
      print(num)   
      data = batch_data.to(device)
      with torch.no_grad():
        emdeddings = model(data)
      results = torch.cat((results, emdeddings))

  return results




def GetTextEmbeddings_Bert(dataset, batch_size = 32):
  
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  input_ids = TextTransformation_Bert(dataset['plot'])
  sampler = SequentialSampler(input_ids)
  dataloader = DataLoader(input_ids, sampler=sampler, batch_size=batch_size)
  
  results = torch.Tensor().to(device)

  model = BertMultiLabelClassifier(gen_embeddings = True)
  model.to(device)
  model.eval()

  for num, batch_data in enumerate(dataloader):
      print(num) 
      indexed_tokens, segment_ids , masked_ids = tuple(t for t in batch_data)   
      data = indexed_tokens.to(device)
      with torch.no_grad():
        emdeddings = model(data)
      results = torch.cat((results, emdeddings))

  return results