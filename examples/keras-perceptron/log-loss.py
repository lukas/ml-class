from sklearn.metrics import log_loss

truth = [0,0,0,0,1]
pred = [0.2,0.2,0.2,0.2,0.2]

print(log_loss(truth, pred))
