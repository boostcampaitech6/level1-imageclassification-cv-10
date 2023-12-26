import pandas as pd
import os

class EnsemblePredictor:
    def __init__(self, submit_csv_path, ensemble_dir):
        """
        EnsemblePredictor 클래스 초기화.

        Args:
            submit_csv_path (str): 제출할 CSV 파일의 경로.
            ensemble_dir (str): 앙상블할 모델의 예측 결과가 저장된 디렉토리 경로.
        """
        self.submit_csv_path = submit_csv_path
        self.ensemble_dir = ensemble_dir

    def get_csv_list(self):
        """
        앙상블 디렉토리 내의 모든 CSV 파일의 경로를 반환한다.

        Returns:
            list[str]: CSV 파일 경로 목록.
        """
        return [os.path.join(self.ensemble_dir, f) for f in os.listdir(self.ensemble_dir)]

    def load_predictions(self):
        """
        모든 CSV 파일에서 예측값을 로드한다.

        Returns:
            pd.DataFrame: 각 모델의 예측값을 포함하는 데이터프레임.
        """
        csv_list = self.get_csv_list()
        temp = pd.DataFrame()
        for index, csv in enumerate(csv_list):
            temp[f'{index}'] = pd.read_csv(csv, index_col=False)['ans']
        return temp

    def ensemble_predictions(self):
        """
        여러 모델의 예측값을 앙상블하여 최종 예측 결과를 생성한다.

        Returns:
            pd.DataFrame: 앙상블된 최종 예측 결과를 포함하는 데이터프레임.
        """
        submit = pd.read_csv(self.submit_csv_path, index_col=False)
        temp = self.load_predictions()
        submit['ans'] = temp.mode(axis=1)[0].astype('int')
        return submit

    def save_ensemble_result(self, save_path):
        """
        앙상블된 결과를 CSV 파일로 저장한다.

        Args:
            save_path (str): 저장할 CSV 파일 경로.
        """
        submit = self.ensemble_predictions()
        submit.to_csv(save_path, index=False)
        print('Ensemble result saved to:', save_path)

if __name__ == "__main__":
    submit_csv_path = '/data/ephemeral/home/input/eval/info.csv'
    ensemble_dir = '/data/ephemeral/home/level1-imageclassification-cv-10/output/reallast'
    save_path = f'{ensemble_dir}/voting.csv'

    predictor = EnsemblePredictor(submit_csv_path, ensemble_dir)
    predictor.save_ensemble_result(save_path)