from grnn_qsar import *

# unlabeled training set (X) to classify
X = np.array([[1.0,2.0,3.0,4.0,5.0],
              [1.0,1.0,1.0,1.0,1.0],
              [2.0,3.0,1.5,2.0,1.9],
              [1.5,2.5,2.0,2.0,2.5]]);

# observed target y
y = np.array([90.0,1.3,2.4,1.8]);

sample_size, feature_size = X.shape
# array to hold predicted values
yHat = np.zeros(sample_size);

# Multiple-sigma model, one weight per descriptor
# sigmas = np.random.randint(10, 50, feature_size);
# single sigma
sigmas = np.random.randint(10, 50, 1);

distance = computeDistances(X, sigmas)
print("Distances:")
print distance

numerator = getSummationLayerNumerator(y, distance)
print("\nNumerators:")
print numerator

denominator = getSummationLayerDenominator(distance)
print("\nDenominators:")
print denominator

yHat = outputLayer(numerator, denominator)
print("\nActual:")
print y
print("\nPredicted:")
print yHat

