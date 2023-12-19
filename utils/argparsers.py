import argparse

class Parser(object):
    def __init__(self, description=""):
        self.description = description
        self.parser = argparse.ArgumentParser(description=self.description)
    
    def create_parser(self):
        self.parser.add_argument('--train-data-dir', default="/input", help="Path to the training data directory.")
        self.parser.add_argument('--val-data-dir', default="/input", help="Path to the validation data directory.")
        self.parser.add_argument('--test-data-dir', default="/input", help="Path to the test data directory.")
        self.parser.add_argument('--save-dir', default="./results", help="Directory for saving training outputs like models and logs.")
        self.parser.add_argument('--output-dir', default="./output", help="Directory to store additional output files.")
        self.parser.add_argument('--project-name', default="exp", help="Name of the Weights & Biases project for tracking experiments.")
        self.parser.add_argument('--exp-name', default="exp", help="Name of this specific experiment run.")
        self.parser.add_argument('--test-exp-num', default=None, help="Experiment number for the test run.")
        self.parser.add_argument('--config', default='./config/base.yml', help='Path to the configuration file in YAML format.')
        self.parser.add_argument('--dataset', default="BaseDataset", help="Type of dataset to be used (e.g., 'ProfileDataset').")
        self.parser.add_argument('--augmentation', default="BaseAugmentation", help="Augmentation method to be applied to the dataset.")
        self.parser.add_argument('--resize', default=[384, 288], help="Dimensions to which the input images will be resized.")
        self.parser.add_argument('--valid-ratio', default=0.2, help="Ratio for splitting the dataset into training and validation sets.")
        self.parser.add_argument('--balanced-split', default=True, help="Whether to split the dataset in a balanced manner.")
        self.parser.add_argument('--sampler', default="", help="Type of sampler to use for DataLoader.")
        self.parser.add_argument('--seed', default=777, help="Seed for random number generation to ensure reproducibility.")
        self.parser.add_argument('--max-epochs', default=10, help="Maximum number of training epochs.")
        self.parser.add_argument('--batch-size', default=32, help="Number of samples per training batch.")
        self.parser.add_argument('--valid-batch-size', default=128, help="Number of samples per validation batch.")
        self.parser.add_argument('--optimizer', default="Adam", help="Optimization algorithm to be used.")
        self.parser.add_argument('--lr', default=0.0001, help="Learning rate for the optimizer.")
        self.parser.add_argument('--scheduler', default="cosine", help="Learning rate scheduler to be used.")
        self.parser.add_argument('--lr-decay-step', default=777, help="Step size for learning rate decay.")
        self.parser.add_argument('--criterion', default="focal", help="Loss function to be used during training.")
        self.parser.add_argument('--log-interval', default=10, help="Interval at which to log training progress.")
        self.parser.add_argument('--model', default="EfficientnetB4", help="Model architecture to be used.")
        self.parser.add_argument('--hpo', default=True, help="Enable Hyperparameter Optimization (HPO).")
        self.parser.add_argument('--mix', default=False, help="Use Cutmix or Mixup data augmentation techniques during training.")
        self.parser.add_argument('--hardmix', default=False, help="Use hard label or soft label for cutmix/mixup")
        self.parser.add_argument('--age-drop', default=False, help="Drop age when training")

    def print_args(self, args):
        print("Arguments:")
        for arg in vars(args):
            print("\t{}: {}".format(arg, getattr(args, arg)))
