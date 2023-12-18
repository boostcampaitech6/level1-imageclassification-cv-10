import numpy as np

def calculate_metrics(true_labels: list[int], predicted_labels: list[int], num_classes: int) -> dict:
    tp = [0] * num_classes
    fp = [0] * num_classes
    fn = [0] * num_classes
    fp_image = []

    accuracy = 0
    precision = [0] * num_classes
    recall = [0] * num_classes
    f1_score = [0] * num_classes
    
    true_labels = np.array(true_labels, np.int64)
    predicted_labels = np.array(predicted_labels, np.int64)

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
