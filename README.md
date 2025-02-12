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








Here, we have one hidden layer with a number of hidden nodes close to the average nodes between the input and output layers. Our learning rate is arbitrarily set between 0.1 and 0.00001 as commonly suggested by et al. (cite). 






