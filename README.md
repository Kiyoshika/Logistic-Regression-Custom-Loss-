# Logistic Regression (Custom Loss)
A custom implementation of logistic regression in Python with a custom loss function

### Drawbacks
I'm using BASE Python; the speed is very slow. I plan on creating a C++ equivalent of this code later.

### Advantages
* I use numerical derivatives, meaning you can swap any loss function without having to compute its derivative by hand.
* The custom loss function I'm using seems to do better than cross entropy, but this would need more experimentation.
