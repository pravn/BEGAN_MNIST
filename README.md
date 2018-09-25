#BEGAN 
Experiments on MNIST

It seems that this code works, but the parameters/architecture might not be optimal.

It prints out D_loss, G_loss and convergence metric in notebook. 

Architecture - 
  - Conv, elu for Generator (Decoder of AE)
  - Conv, elu for Encoder of AE. 
  - Use a fully connected layer in the end
  - Use BatchNorm for the Decoder 'in places' (this is clearly an indication that we don't have the right setup).

LR halved every 25 iterations.

Important: Make sure to detach the parameter k_t from the graph - this is easy to miss. 

What did not work: 
  - Weights init as in WGAN code from MartinArjrovsky
  - DCGAN architecture [Conv, BatchNorm, ReLU, LeakyReLU] 
    (although in my case, batch norm actually seems to help) 
  - noise init with normal (I can't be sure of this, but initializing with
    uniform from (-1,1) definitely does work). 

I should try the 'upsampling' as in other BEGAN codes. 


