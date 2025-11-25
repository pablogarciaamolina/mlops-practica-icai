import torch

from core.data import load_pediatric_pneumonia, load_pediatric_pneumonia_mixed
from core.models import AlexNet
from core.pipelines import Classifier_Pipeline, Pipeline_Config, load_model
from config import NUM_CLASSES, BATCH_SIZE, PIPELINE_CONFIG

device: torch.device = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)

def main() -> None:
    """
    This function is the main program for the training.
    """

    print("Loading data...", end="")
    train_data, val_data, test_data = load_pediatric_pneumonia(batch_size=BATCH_SIZE)
    # train_data, val_data, test_data = load_pediatric_pneumonia_mixed(batch_size=BATCH_SIZE, test_size=0.2, val_size=0.1)
    print("DONE")

    # inputs: torch.Tensor = next(iter(train_data))[0]
    # model = AlexNet(inputs.shape[1], NUM_CLASSES)
    model = load_model("AlexNet_1764097347.8264112_mixed_data")

    config = Pipeline_Config(**PIPELINE_CONFIG)
    print("Using device:", config.device)
    pipeline = Classifier_Pipeline(model, config=config)
    
    # pipeline.train(
    #     train_data,
    #     val_data,
    #     add_to_name="_mixed_data"
    # )

    test_accuracy = pipeline.evaluate(
        test_data
    )
    print("Test accuracy:", test_accuracy)

    return None

if __name__ == "__main__":
    main()
