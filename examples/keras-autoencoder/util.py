from tensorflow.keras.callbacks import Callback
import numpy as np
import wandb


class Images(Callback):
    def __init__(self, val_data, **kwargs):
        super(Images, self).__init__(**kwargs)
        self.validation_data = (np.array(val_data), )

    def on_epoch_end(self, epoch, logs):
        indices = np.random.randint(self.validation_data[0].shape[0], size=8)
        test_data = self.validation_data[0][indices]
        pred_data = np.clip(self.model.predict(test_data), 0, 1)
        wandb.log({
            "examples": [
                wandb.Image(np.hstack([data, pred_data[i]]), caption=str(i))
                for i, data in enumerate(test_data)]
        }, commit=False)
