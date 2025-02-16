# Husker-Analytics
Machine learning for Nebraska football

This page is meant to merge passions: Nebraska football, statistics, and machine learning. This page is not affiliated with the University of Nebraska.

### Go Big Red!

## Deep Learning/Neural Networks
An artificial neural network is designed to function similar to the human brain where neurons (or nodes) are connected from input to output. In deep learning models, there are hidden layers between the input and output layers that allow the model to more effectively recognize complex patterns from the input in order to produce the desired output. Similar to the human brain, these models learn to recognize these patterns with training data, allowing them to then be used on data that the model has yet to encounter. A diagram of an artificial neural network is shown below:
![General Neural Net](https://github.com/user-attachments/assets/a2815c58-7194-4dbc-b17c-6446b6b8bcdc)

Here, the nodes between the input and output are assigned weights that transform the data at the node before being passed on to the next layer. In order for the model to learn, these weights are repeatidly adjusted to push the output closer to the desired results. To teach the model, a loss function is selected that provides feedback to the network as to whether the weights produce a result that is closer to the desired output. The final piece to this process is backpropogation; here, ...... (We will dive more into the specifics on these concepts when we are building the neural network). 

Neural networks provide a lot of potential in modern sports to help recognize patterns and predict outcomes before they occur, giving teams a potentially decisive edge before and during a game. Below, we will attempt to create a neural network to predict Nebraska opponent playcalls from pre-snap data. Accurate predictions would offer a clear advantage, allowing the Blackshirts to adjust their playcalling in order to counter specific play types. 


### Opponent Predictions 
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

#### Training the model
For training, validating, and testing the model we need to split the data into sets. For training and validation we will use all 2023 data from 2024 Nebraska opponents. We will split the data randomly, training on 80% of the data and validating on the other 20%:


To run the model, we need to convert the data into pytorch tensors and put them into the dataloader. The dataloader will allow us to train in batches (more on this later). To start we will train on all of the data at once:


Plotting the loss as a funciton of epoch we see an initial steep decline in loss followed by a slow decline in the first 50 epochs. Running the trained model on the validation data yields an accuracy of 58%, we can use a confusion matrix below to more fully understand where the model is going wrong:

![image](https://github.com/user-attachments/assets/59873a5f-8caa-48b3-8013-094be2c8b5ff)


Here, we can see that the model is assigning most play types to "rush" since it is the most common play type in the set. From the confusion matrix we can clearly see that our model is performing poorly. 

We will try to improve our model first by adjusting hyperparameters. We could do this naively, randomly changing parameters in hopes of finding some better performing architecture. To converge to a maximum more efficiently, we will use Bayesian optimization. Briefly, the Bayesian optimization method we will use is a gaussian process that will maximize hyperparameter performance while balancing exploration and exploitation. The libray we will use is bayesian-optimization (github:). We will attempt to optimize a more complex architecture with three hidden layerswhere we will try to set the number of nodes at each layer and the learning rate (alpha). The results of the optimization are shown in the parallel plot below:




We can now build the model based on the best performing model:



The loss from this model and the confusion matrix are plotted below, here we get an accuracy of 68.5%, improving our model by 10.5%!
![image](https://github.com/user-attachments/assets/c4c2e0bd-8395-4c21-b3c1-76d055abfe11)



Another issue we have yet to discuss is the effect of the batch size while training. Currently we train on the full dataset all at one. Instead, we might want to try stochastic gradient descent (SGD) or mini-batch approaches to help the model better generalize on our data. Here we will use the same architecture as above but vary the batch size ranging from 1 to the size of our dataset. Below we can see the effect of the batch size on our validation set when we average over 5 separate random training/validation splits for each data point:

![image](https://github.com/user-attachments/assets/60ace6cf-af92-4d02-8525-7722066d07d3)


We can see that mini-batching can help us generalize and potentially increase our accuracy by a few percent. However, SGD (batch size = 1) results in very poor performance overall. We will retrain the model with a batch size of 32 where we can see that the number of Epochs needed to reach a plateau in the step-wise loss is much less since the model is learning more frequently through each pass of the data:
![image](https://github.com/user-attachments/assets/e74eec99-ceb9-4041-9b22-9c44ff9a61d0)


We get 74.1% accuracy overall now that we mini-batch. From the confusion matrix we can still see that plays that should be easily predicted (FG, Punt) are sometimes not predicted well. This may be due to the lack of balcance in our training dataset. In other words, we are training on many more Pass/Rush plays than Punt/FG plays such that the model may not be able to fully learn to identify the latter. TO address this we will simply try to duplicate the FG and Punt plays in our training dataset.



Running our model on the test data produces:




