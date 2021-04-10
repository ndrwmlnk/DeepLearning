import numpy as np

operator = 'and'

if operator == 'and':
	labels = np.array([0,0,0,1]).reshape(-1,1)
elif operator == 'or':
	labels = np.array([0,1,1,1]).reshape(-1,1)
elif operator == 'xor':
	labels = np.array([0,1,1,0]).reshape(-1,1)

input=np.vstack(([0,0],[0,1],[1,0],[1,1]))
alpha=0.5
W1=np.random.rand(2,16)
W2=np.random.rand(16,1)
def sigmoid(x):
  return (1/(1+np.exp(-x)))
Loss=[]
print('---------------')
print('operator =', operator)
print('---------------')
epochs = 5000
for i in range(5000):
  z=sigmoid(np.dot(input,W1))
  prediction=sigmoid(np.dot(z,W2))
  loss=np.sum((prediction-labels)**2)/len(labels)
  grad_W2=2*(np.dot(prediction.T,(prediction-labels)*prediction*(1-prediction)))
  grad_W1=2*np.dot(input.T,np.dot((prediction-labels)*prediction*(1-prediction),W2.T)*z*(1-z))
  W2=W2-alpha*grad_W2
  W1=W1-alpha*grad_W1
  Loss.append(loss)
  if i % 1000 == 0 or i == epochs-1:
      print('\nepoch = ', str(i).zfill(4), ' loss =', "{0:.3f}".format(loss), '\noperator =', operator, '\ninput =', [[ii[0], ii[1]] for ii in input], '\nprediction =', ["{0:.3f}".format(p[0]) for p in prediction], '\ntarget =', [l[0] for l in labels])
print('---------------')
print("DONE")
