# Husker-Analytics
Machine learning for Nebraska football

This page is meant to merge passions: Nebraska football, statistics, and machine learning. This page is not affiliated with the University of Nebraska.

### Go Big Red!

## Click Below To Browse Topics

[Home](index.md)  |  [Neural Networks](NeuralNet.md)  |  [Hierarchical Bayesian Analysis](HierarchicalBayes.md)  |     

## Deep Learning/Neural Networks
An artificial neural network is designed to function like the human brain where neurons (or nodes) are connected from input to output. In deep learning models, there are hidden layers between the input and output layers that allow the model to more effectively recognize complex patterns from the input to produce the desired output. Like the human brain, these models learn to recognize these patterns with training, allowing them to later be used with data that the model has yet to encounter. A general diagram of an artificial neural network is shown below:
![General Neural Net](https://github.com/user-attachments/assets/a2815c58-7194-4dbc-b17c-6446b6b8bcdc)

Here, the nodes between the input and output are assigned weights that transform the data at the node before being passed on to the next layer. For the model to learn, these weights are repeatedly adjusted to push the output closer to the desired results. To teach the model, a loss function is selected that provides feedback to the network as to whether the weights produce a result that is closer to the desired output. The final piece to this process is backpropagation. Here, the error is passed back through the neural network and the weights are adjusted using gradients. (We will dive more into the specifics on these concepts when we are building the neural network). 

Neural networks provide a lot of potential in modern sports to help recognize patterns and predict outcomes before they occur, giving teams a potentially decisive edge before and during a game. Below, we will attempt to create a neural network to predict Nebraska opponent play calls from pre-snap data. Accurate predictions would offer a clear advantage, allowing the Blackshirts to adjust their play calling to counter specific play types. 


## Opponent Predictions 
(***Data was gathered from collegefootballdata.com***)

The raw data is generally categorized into four different play types: pass, rush, punt, or field goal. We will attempt to create a neural network that is able to predict these play calls from limited pre-snap information. From general knowledge of football, I have selected features from the data as well as calculated new features that I believe will have the most impact on opponent play calling, shown in the table below:

![image](https://github.com/user-attachments/assets/e4193b9e-4509-413a-90e8-c915843f2cc0)


To create our network, there are multiple hyperparameters that need to be selected that can all have varying degrees of impact on our ability to accurately predict play calling. Just to name a few, these hyperparameters include the number of layers, number of nodes, optimizer to use, and the learning rate.  Choosing hyperparameters is commonly the most difficult task in machine learning, and neural networks are no exception. Initially we will choose our hyperparameters somewhat arbitrarily, mainly trying to avoid too much complexity. A diagram of our initial neural network is shown below:

![image](https://github.com/user-attachments/assets/549130f7-2036-4009-a042-0a0d54964e89)




Here, we have one hidden layer with the number of hidden nodes close to the average nodes between the input and output layers. You can see that for the hidden layer and output layers are activation functions (ReLU and softmax, respectively). The rectified linear unit (ReLU) activation function introduces non-linearity into our model and the softmax activation function transforms our output data into four values that all total to 1 (effectively mapping to the probability of each play). The loss function we will use is cross entropy. 

The highly used/cited Adam optimizer is chosen for our neural net. If you want to read more about how this optimizer functions, the citation is here: Kingma, D. P., & Ba, J. (2014). *Adam: A method for stochastic optimization*. 2014. arXiv preprint arXiv:1412.6980). Our learning rate (α) is set to 0.001 as suggested by Kingma et al. 

We will be using PyTorch to build our neural network so all other hyperparameters for the optimizer are set to the default that the library provides. This model in Python is shown below:
```
class NeuralNetwork(torch.nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(11, 7)
        self.linear2 = nn.Linear(7, 4)
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x
```
### Training the model
For training, validating, and testing the model we need to split the data into sets. For training and validation, we will use all 2023 data from 2024 Nebraska opponents. We will split the data randomly, training on 80% of the data and validating on the other 20%:
```
data_train, data_test, play_train, play_test = train_test_split(Pre_snap_data, play_type, train_size=0.8, shuffle=True)
```

To run the model, we need to convert the data into PyTorch tensors and put them into the DataLoader. The DataLoader will allow us to train in batches (more on this later). To start we will train on all the data at once:
```
data_train_tensor = torch.tensor(data_train, dtype=torch.float32)
play_train_tensor = torch.tensor(play_train, dtype=torch.float32)
data_test_tensor = torch.tensor(data_test, dtype=torch.float32)
play_test_tensor = torch.tensor(play_test, dtype=torch.float32)

class Dataset_Class(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.len = self.X.shape[0]  
        
    def __getitem__(self, index):
        return self.X[index], self.y[index]
        
    def __len__(self):
        return self.len
   
batch_size = len(data_train)

train_data = Dataset_Class(data_train_tensor, play_train_tensor)
train_dataloader = DataLoader(dataset=train_data, batch_size = batch_size, shuffle=True)

test_data = Dataset_Class(data_test_tensor, play_test_tensor)
test_dataloader = DataLoader(dataset=test_data, batch_size = batch_size, shuffle=True)
```

Once we have the model defined and the DataLoader ready, we can train and validate our model with the following code:
```
model = NeuralNetwork()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 50

loss_vals = []
loss = 0
for epoch in range(num_epochs):
    for X, y in train_dataloader:
        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_vals.append(loss.item())

acc_list = []
with torch.no_grad():
    for X, y in test_dataloader:
        pred= model(X)  # Get model outputs
        loss = loss_fn(pred, y)

        pred_int = torch.max(pred, axis = 1)[1]
        gt_int = torch.max(y, axis = 1)[1]

        acc_list.append(np.vstack([cat_int, gt_int]).T)
acc_arr = np.vstack(accuracy_list)
print('Accuracy:', sum(acc_arr[:,0]==acc_arr[:,1])/(len(acc_arr)),'%')
```

Plotting the loss values as a function of the epoch, we see an initial steep decline in loss followed by a slow decline in the first 50 epochs. Running the trained model on the validation data yields an accuracy of 58%, we can use a confusion matrix below to more fully understand where the model is going wrong:

![image](https://github.com/user-attachments/assets/59873a5f-8caa-48b3-8013-094be2c8b5ff)


Here, we can see that the model is assigning most play types to "Rush" since it is the most common play type in the set. From the confusion matrix we can clearly see that our model is performing poorly. 

### Mini-batch
Currently we train on the full dataset all at one. Instead, we might want to try stochastic gradient descent (SGD) or mini-batch approaches to help the model better generalize on our data. Here we will use the same architecture as above but vary the batch size ranging from 32 to the size of our dataset. Below we can see the effect of the batch size on our validation set when we average over 5 separate random training/validation splits for each data point:
![image](https://github.com/user-attachments/assets/7afb6191-0a35-4000-bf87-bf7904d09e06)

We can see that batching can help us generalize and potentially increase our accuracy. However, not shown on this graph is SGD (batch size = 1) which showed very poor performance overall. Thus, we will retrain the model with a batch size of 32 where we can see that the model is learning more frequently through each pass of the data:
```
batch_size = 32

train_data = Dataset_Class(data_train_tensor, play_train_tensor)
train_dataloader = DataLoader(dataset=train_data, batch_size = batch_size, shuffle=True)

test_data = Dataset_Class(data_test_tensor, play_test_tensor)
test_dataloader = DataLoader(dataset=test_data, batch_size = batch_size, shuffle=True)
```
![image](https://github.com/user-attachments/assets/af501504-8460-4a19-b45c-6b2095930066)

### Bayesian optimization
We will try to improve our model further by adjusting hyperparameters. We could do this fully explorative, randomly changing parameters in hopes of finding some better performing architecture. We will instead leverage Bayesian optimization to converge to a maximum more efficiently. Briefly, the Bayesian optimization method we will employ is a gaussian process that will maximize hyperparameter performance while balancing exploration and exploitation. The library we will use is Bayesian-optimization (Fernando Nogueira, *Bayesian Optimization: Open source constrained global optimization tool for Python*, 2014, https://github.com/bayesian-optimization/BayesianOptimization). We will attempt to optimize a more complex architecture with three hidden layers attempting to set the number of nodes at each layer and the learning rate (α). The results of the optimization are shown in the parallel plot below:

![image](https://github.com/user-attachments/assets/98a8d627-ab82-4655-842b-9b379a9eb2fc)

The optimal architecture for our neural net is much more complex than before, potentially allowing for more complex feature extracting resulting in better predictions:
![image](https://github.com/user-attachments/assets/213553ae-3244-4e53-a18b-293583ef55cb)

We can now build the model based in PyTorch choosing the best performing model above:
```
class NeuralNetwork(torch.nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(11, 165)
        self.linear2 = nn.Linear(165, 153)
        self.linear3 = nn.Linear(153, 68)
        self.linear4 = nn.Linear(68, 4)
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear4(x)
        return x
```


The loss from this model and the confusion matrix are plotted below, here we get an accuracy of 74.5%, an improvement of 16.5% compared to the initial model!
![image](https://github.com/user-attachments/assets/33ecb281-6bcd-408e-88b2-230ef329bccc)



### Predicting 2024 Opponent play calls
Until this point, we have only been training and validation on data from 2023. The goal is to now use our model to try to predict play calling in 2024. We can run our model on the 2024 data and get the confusion matrix below:
![image](https://github.com/user-attachments/assets/cd7c5fc7-8ae7-4554-b274-81f9abe45fdf)

We predict opponent play calling in the 2024 season with 71.7% accuracy. Assuming you were to take a blind approach and just select all plays as "Rush", this accuracy represents a 20% improvement from the baseline. One could argue that a punt and field goal should both be highly predictable in a game setting. Accounting for only our ability to predict Rush and Pass play calls, we maintain an accuracy of 69.7% representing a 12.7% improvement from the baseline. 

#### Additional thoughts on what could improve our model in the future
 - Currently we train our model on 2023 data only. We could also include all 2024 data that occurs prior to the snap when training, potentially offering more up to date knowledge that our model can learn from.

 - Many variables are not present in our data that could have a large impact on play calling. These might include weather conditions, offensive formation, and/or personnel to name a few. Training our model with these variables might improve our ability to predict play calling in real time. 

