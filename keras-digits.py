from keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

digit = X_train[0]
print(digit.shape)
str = ""
for i in range(digit.shape[0]):
    for j in range(digit.shape[1]):
        if digit[i][j] == 0:
            str += " "
        elif digit[i][j] < 128:
            str += "."
        else:
            str += "X"
    str += "\n"

print(str)
print("Label: ", y_train[0])
