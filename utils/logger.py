from typing import Any
import os
import wandb
from data.datasets import MaskBaseDataset
class Logger:
    """
    로그 파일을 생성하고 관리하는 클래스.

    Attributes:
        save_path (str): 로그 파일을 저장할 경로.
        file_name (str): 로그 파일의 이름. 기본값은 'log.txt'.
        file: 로그 파일 객체.

    Methods:
        update: 주어진 내용을 로그 파일에 기록한다.
        update_string: 문자열 형태의 내용을 로그 파일에 기록한다.
        close: 로그 파일을 닫는다.
    """
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
    """
    Weights & Biases를 사용하여 실험 로그를 기록하는 클래스.

    Attributes:
        config (dict): 실험 설정 정보.
        exp_name (str): 실험 이름.
        project_name (str): Weights & Biases 프로젝트 이름. 기본값은 'Boost Camp Lv1'.

    Methods:
        update: 주어진 내용을 Weights & Biases 로그에 기록한다.
        update_image_with_label: 레이블이 포함된 이미지를 Weights & Biases에 기록한다.
        log: 딕셔너리 형태의 로그를 Weights & Biases에 기록한다.
        log_confusion_matrix: 혼동 행렬을 Weights & Biases에 기록한다.
    """

    def __init__(self, config: dict, exp_name: str, project_name: str='Boost Camp Lv1'):
        if isinstance(config, dict) or hasattr(config, '__dict__'):
            run = wandb.init(
                project=project_name,
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

