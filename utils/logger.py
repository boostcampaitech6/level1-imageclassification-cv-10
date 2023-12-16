from typing import Any
import os
import wandb
from data.datasets import MaskBaseDataset
class Logger:
    def __init__(self, save_path: str, file_name :str='log.txt'):
        self.save_path = save_path
        self.file_name = file_name
        self.file = open(os.path.join(self.save_path, self.file_name), 'w')
        
    def update(self, contents: dict[str:Any]):
        logging_content = ''
        for key in contents:
            logging_content += (key + ': ' + str(contents[key]) + "\n")
        self.file.write(logging_content)
    
    def update_string(self, contents: str):
        if isinstance(contents, str):
            self.file.write(contents + "\n")
        else:
            raise TypeError("Args is not string type.")
        
    def close(self):
        self.file.close()

class WeightAndBiasLogger():
    def __init__(self, config: dict, exp_name: str):
        if isinstance(config, dict) or hasattr(config, '__dict__'):
            run = wandb.init(
                project='Boost Camp Lv1',
                dir=config.save_path,
            )
            assert run is wandb.run
            run.name = exp_name
            run.save()
            wandb.config.update(config)
        else:
            raise TypeError('config must be dictionary or convertible to dictionary type.')
        
    def update(self, args: dict):
        if not isinstance(args, dict):
            raise TypeError('Argument must be dictionary')
        wandb.update(args)
    
    def update_image_with_label(self, image, prediction, truth):
        return wandb.Image(image, caption = f"Pred: {prediction}, {MaskBaseDataset.class_name[prediction]}" + "\n" + f"Truth: {truth}, {MaskBaseDataset.class_name[truth]}")
        # combined_image = self.combine_image_and_label(image, label)
        # self.update_image(combined_image)

    
    # @staticmethod
    # def combine_image_and_label(self, image, label):
    #     pass
    
    # def update_image(self, image):
    #     wandb.update(wandb.Image(image))
    
    def log(self, log: dict):
        if isinstance(log, dict) or hasattr(log, '__dict__'):
            wandb.log(log)
        else:
            raise TypeError('log must be dictonary or convertible to dictionary type.')
    
    def log_confusion_matrix(self, targets, results):
        wandb.log({"confusuion_matrix" : wandb.plot.confusion_matrix(
                                        probs = None,
                                        y_true = targets,
                                        preds = results,
                                        class_names = MaskBaseDataset.class_name)})

