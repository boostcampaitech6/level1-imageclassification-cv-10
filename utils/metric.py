def calculate_metrics(true_labels: list[int], predicted_labels: list[int], num_classes: int) -> dict:
    """
    주어진 실제 레이블과 예측 레이블을 기반으로 메트릭을 계산한다.

    이 함수는 정확도, 정밀도, 재현율, F1 점수를 계산하며, 각 클래스별로 이러한 메트릭을 계산한다.
    또한 예측이 잘못된 이미지의 인덱스도 반환한다.

    Args:
        true_labels (list[int]): 실제 레이블의 리스트.
        predicted_labels (list[int]): 예측된 레이블의 리스트.
        num_classes (int): 클래스의 총 개수.

    Returns:
        dict: 계산된 메트릭을 포함하는 사전. 정확도, 정밀도, 재현율, F1 점수 및 잘못된 예측의 인덱스를 포함한다.
    """
    tp = [0] * num_classes
    fp = [0] * num_classes
    fn = [0] * num_classes
    fp_image = []

    accuracy = 0
    precision = [0] * num_classes
    recall = [0] * num_classes
    f1_score = [0] * num_classes

    for i in range(len(true_labels)):
        if true_labels[i] == predicted_labels[i]:
            tp[true_labels[i]] += 1
        else:
            fp[predicted_labels[i]] += 1
            fn[true_labels[i]] += 1
            fp_image.append(i)

    accuracy = sum(tp) / len(true_labels)

    for i in range(num_classes):
        precision[i] = tp[i] / (tp[i] + fp[i]) if (tp[i] + fp[i]) > 0 else 0
        recall[i] = tp[i] / (tp[i] + fn[i]) if (tp[i] + fn[i]) > 0 else 0
        f1_score[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i]) if (precision[i] + recall[i]) > 0 else 0

    return {"Total Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1 Score": f1_score, 
            "Total Precision": sum(precision) / num_classes, "Total Recall": sum(recall) / num_classes,
            "Total F1 Score": sum(f1_score) / num_classes, "False Image Indexes": fp_image}
    
def parse_metric(metrics: dict, label_name: list[str]=None) -> str:
    """
    계산된 메트릭을 문자열로 변환하여 반환한다.

    이 함수는 각 클래스별 및 전체적인 정밀도, 재현율, F1 점수 및 정확도를 포함하는 문자열을 생성한다.

    Args:
        metrics (dict): calculate_metrics 함수로부터 계산된 메트릭이 담긴 사전.
        label_name (list[str], optional): 클래스 레이블의 이름. 기본값은 None이며, None인 경우 숫자로 레이블을 나타낸다.

    Returns:
        str: 메트릭의 요약을 담은 문자열.
    """
    class_num = len(metrics["F1 Score"])
    if label_name is None:
        label_name = []
        for idx in range(class_num):
            label_name.append(str(idx))
    
    parsed_string = ""
    zipped_metrics = zip(metrics["Precision"], metrics["Recall"], metrics["F1 Score"])
    parse_format = "{:>25} Precision {:3.4f}, Recall {:3.4f}, F1 Score {:3.4f}"
    for idx, (precision, recall, f1_score) in enumerate(zipped_metrics):
        parsed_string += parse_format.format(label_name[idx], precision, recall, f1_score) + "\n"
    
    parse_format += " Acc. {:3.4f}"
    parsed_string += parse_format.format("Total", metrics["Total Precision"], \
        metrics["Total Recall"], metrics["Total F1 Score"], metrics["Total Accuracy"]) + "\n"
    
    return parsed_string
