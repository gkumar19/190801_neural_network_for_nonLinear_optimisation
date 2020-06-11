# 190801_neural_network_for_nonLinear_optimisation

## context
under the usual scenerio of optimisation problems, Linear programming is a method which can be used to achieve the best outcome (such as maximum profit or lowest cost) in a mathematical model whose requirements are represented by linear relationships using linear models and restricted by certain boundary conditions

The problem in my hand is to find input [features] in a way that the output [Targets] fulfills some complicated requirements, with some constraints over the inputs [features]

The input to output correlation also being non-linear, thus a linear programming was not an option, thus i have used neural network to model up the input to output relationship [lets call it model_a] and extended the model_a to make a new neural network model [lets call it model_b] in a manner which can be used to optimise over model_a

Additionally, the benefit with this approach being, it was not very difficult and became trivial to further introduce some really complicated boundary conditions [How the optuput should reach to become, Constraints over the inputs, etc]

In my opinion the idea is powerful and can be extended over wide ranges of optimisation problem statement

The above was acheived using: custom tensorflow loss functions, custom tensorflow layers, custom tensorflow constraints functions, pandas, scikit-learn and in general scipy ecosystem
