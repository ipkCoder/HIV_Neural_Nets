import GRNN_Class as grnn
import numpy as np

def main():
  
  # data
  X = np.array([[1.0,2.0,3.0,4.0,5.0],
                [1.0,1.0,1.0,1.0,1.0],
                [2.0,3.0,1.5,2.0,1.9],
                [1.5,2.5,2.0,2.0,2.5]])
  
  # y values
  y = np.array([6.0,1.3,2.4,1.8])

  model = grnn.GRNN()

  model.fit(X, y)

  for i in range(y.shape[0]):
    yh = model.predict(i, X, y)
    print yh



main()