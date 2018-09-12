from keras.callbacks import Callback
import numpy as np
import wandb

class Images(Callback):
    def on_epoch_end(self, epoch, logs):
        indices = np.random.randint(self.validation_data[0].shape[0], size=8)
        test_data = self.validation_data[0][indices]
        pred_data = self.model.predict(test_data)
        wandb.log({
             "examples": [
                   wandb.Image(np.hstack([data, pred_data[i]]), caption=str(i))
                   for i, data in enumerate(test_data)]
        }, commit=False)