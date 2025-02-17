# Husker-Analytics
Machine learning for Nebraska football

This page is meant to merge passions: Nebraska football, statistics, and machine learning. This page is not affiliated with the University of Nebraska.

### Go Big Red!

## Deep Learning/Neural Networks
An artificial neural network is designed to function similar to the human brain where neurons (or nodes) are connected from input to output. In deep learning models, there are hidden layers between the input and output layers that allow the model to more effectively recognize complex patterns from the input in order to produce the desired output. Similar to the human brain, these models learn to recognize these patterns with training data, allowing them to then be used on data that the model has yet to encounter. A diagram of an artificial neural network is shown below:
![General Neural Net](https://github.com/user-attachments/assets/a2815c58-7194-4dbc-b17c-6446b6b8bcdc)

Here, the nodes between the input and output are assigned weights that transform the data at the node before being passed on to the next layer. In order for the model to learn, these weights are repeatidly adjusted to push the output closer to the desired results. To teach the model, a loss function is selected that provides feedback to the network as to whether the weights produce a result that is closer to the desired output. The final piece to this process is backpropogation; here, ...... (We will dive more into the specifics on these concepts when we are building the neural network). 

Neural networks provide a lot of potential in modern sports to help recognize patterns and predict outcomes before they occur, giving teams a potentially decisive edge before and during a game. Below, we will attempt to create a neural network to predict Nebraska opponent playcalls from pre-snap data. Accurate predictions would offer a clear advantage, allowing the Blackshirts to adjust their playcalling in order to counter specific play types. 


## Opponent Predictions 
(***Data was gathered from collegefootballdata.com***)

The raw data is generally categoriezed into four different play types: pass, run, punt, or field goal. We will attempt to create a neural network that is able to predict these play calls from limited pre-snap information shown in the table below:







To create our network, there are multiple hyperparameters that need to be selected that can all have varying degrees of impact on our ability to accurately predict play calling. Choosing hyperparameters is commonly the most difficult task in machine learning, and neural networks are no exception. 

The hyperparameters we will need to initially choose to build a functioning neural network are shown in the table below:







Initially we will choose our hyperparameters somewhat arbitrarily, mainly trying to avoid too much complexity. A diagram of our initial neural network is shown below:

![Basic Neural Net](https://github.com/user-attachments/assets/234e0258-eae9-4c47-9bcd-4e5d77edc7b7)



Here, we have one hidden layer with a number of hidden nodes close to the average nodes between the input and output layers. You can see that for the hidden layer and output layers are activation functions (ReLU and Softmax, respectively). The rectified linear unit (ReLU) activation function introduces non-linearity into our model and the softmax activation function transforms our output data into four values that all total to 1 (effectively mapping to the probability of each play). 

The loss function we will use is cross entropy. 

The highly used/cited Adam optimizer is chosen for our neural net. If you want to read more about how this optimizer funcitons, the citation is here(). Our learning rate (alpha) is set to 0.001 as suggested by et al. (cite). We will be using PyTorch to build our neural network so all other hyperparameters for the optimizer are set to the default that the library provides.

This model in Python is shown below:
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
For training, validating, and testing the model we need to split the data into sets. For training and validation we will use all 2023 data from 2024 Nebraska opponents. We will split the data randomly, training on 80% of the data and validating on the other 20%:
```
data_train, data_test, play_train, play_test = train_test_split(Pre_snap_data, play_type, train_size=0.8, shuffle=True)
```

To run the model, we need to convert the data into pytorch tensors and put them into the dataloader. The dataloader will allow us to train in batches (more on this later). To start we will train on all of the data at once:
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

Plotting the loss as a funciton of epoch we see an initial steep decline in loss followed by a slow decline in the first 50 epochs. Running the trained model on the validation data yields an accuracy of 58%, we can use a confusion matrix below to more fully understand where the model is going wrong:

![image](https://github.com/user-attachments/assets/59873a5f-8caa-48b3-8013-094be2c8b5ff)


Here, we can see that the model is assigning most play types to "Rush" since it is the most common play type in the set. From the confusion matrix we can clearly see that our model is performing poorly. 

Currently we train on the full dataset all at one. Instead, we might want to try stochastic gradient descent (SGD) or mini-batch approaches to help the model better generalize on our data. Here we will use the same architecture as above but vary the batch size ranging from 1 to the size of our dataset. Below we can see the effect of the batch size on our validation set when we average over 5 separate random training/validation splits for each data point:
![image](https://github.com/user-attachments/assets/7afb6191-0a35-4000-bf87-bf7904d09e06)

We can see that mini-batching can help us generalize and potentially increase our accuracy. However, SGD (batch size = 1) results in very poor performance overall. We will retrain the model with a batch size of 32 where we can see that the model is learning more frequently through each pass of the data:
```
batch_size = 32

train_data = Dataset_Class(data_train_tensor, play_train_tensor)
train_dataloader = DataLoader(dataset=train_data, batch_size = batch_size, shuffle=True)

test_data = Dataset_Class(data_test_tensor, play_test_tensor)
test_dataloader = DataLoader(dataset=test_data, batch_size = batch_size, shuffle=True)
```
![image](https://github.com/user-attachments/assets/af501504-8460-4a19-b45c-6b2095930066)


We will try to improve our model further by adjusting hyperparameters. We could do this fully explorative, randomly changing parameters in hopes of finding some better performing architecture. We will instead leverage Bayesian optimization to converge to a maximum more efficiently. Briefly, the Bayesian optimization method we will employ is a gaussian process that will maximize hyperparameter performance while balancing exploration and exploitation. The libray we will use is bayesian-optimization (github:). We will attempt to optimize a more complex architecture with three hidden layers attempting to set the number of nodes at each layer and the learning rate (alpha). The results of the optimization are shown in the parallel plot below:

![image](https://github.com/user-attachments/assets/98a8d627-ab82-4655-842b-9b379a9eb2fc)

The optimal architecture for our neural net is much more complex than before, potentially allowing for more complex feature extracting resulting in better predictions:
![image](https://github.com/user-attachments/assets/213553ae-3244-4e53-a18b-293583ef55cb)

We can now build the model based in Pytorch choosing the best performing model above:
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

We predict opponent play calling in the 2024 season with 71.7% accuracy. Assuming you were to take a blind approach and just select all plays as "Rush", this accuracy represents a 20% improvement from the baseline. One could argue that a punt and field goal plays should both be highly predictable in a game setting. Accounting for only our ability to predict Rush and Pass play calls, we maintain an accuracy of 69.7% representing a 12.7% improvement from the baseline. 

#### A few thoughts on what could improve our model in the future:


-Currently we train our model on 2023 data only. We could also include all 2024 data that occurs prior to the snap when training, potentially offering more up to date knowledge that our model can learn from.

-Many variables are not present in our data that could have a large impact on play calling. These might include weather condition, offense formation, and/or personel to name a few. Training our model with these variables might improve our ability to predict play calling in real time. 



