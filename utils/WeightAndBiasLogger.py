import wandb

class WeightAndBiasLogger():
    def __init__(self, config: dict, exp_name: str):
        if isinstance(config, dict) or hasattr(config, '__dict__'):
            run = wandb.init(
                project='Boost Camp Lv1',
                dir=config.save_dir,
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
    
    def update_image_with_label(self, image, label):
        pass
    
    def combine_image_and_label(self, image, label):
        pass
    
    def update_image(self, image):
        wandb.update(wandb.Image(image))
    
    def log(self, log: dict):
        if isinstance(log, dict) or hasattr(log, '__dict__'):
            wandb.log(log)
        else:
            raise TypeError('log must be dictonary or convertible to dictionary type.')