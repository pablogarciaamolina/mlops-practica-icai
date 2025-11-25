import os
import json
import torch
import mlflow

from core.data import load_pediatric_pneumonia, load_pediatric_pneumonia_mixed
from core.models import AlexNet
from core.pipelines import Classifier_Pipeline, Pipeline_Config, load_model
from config import NUM_CLASSES, BATCH_SIZE, PIPELINE_CONFIG

device: torch.device = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)

def save_model(model: torch.nn.Module, name: str) -> None:
    """
    This function saves a model as a torch.jit.

    Args:
        model: pytorch model.
        name: name of the model (without the extension, e.g. name.pt).
    """

    model_scripted = torch.jit.script(model.cpu())
    os.makedirs(os.path.join("results", name), exist_ok=True)
    model_scripted.save(os.path.join("results", name, name + ".pt"))

    return None

def main() -> None:
    
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    print(tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)

    print("Loading data...", end="")
    # train_data, val_data, test_data = load_pediatric_pneumonia(batch_size=BATCH_SIZE)
    train_data, val_data, test_data = load_pediatric_pneumonia_mixed(batch_size=BATCH_SIZE, test_size=0.2, val_size=0.1)
    print("DONE")

    with mlflow.start_run():


        inputs: torch.Tensor = next(iter(train_data))[0]
        model = AlexNet(inputs.shape[1], NUM_CLASSES)
        # model = load_model("AlexNet_1764097347.8264112_mixed_data")

        config = Pipeline_Config(**PIPELINE_CONFIG)
        print("Using device:", config.device)
        pipeline = Classifier_Pipeline(model, config=config)
        
        pipeline.train(
            train_data,
            val_data,
            add_to_name="_mixed_data",
            save_model=False
        )
        save_model(model, pipeline.name)

        test_accuracy = pipeline.evaluate(
            test_data
        )
        print("Test accuracy:", test_accuracy)


        mlflow.pytorch.log_model(model, pipeline.name.replace(".", ""))
        for k,v in PIPELINE_CONFIG.items():
            mlflow.log_param(k, v)
        mlflow.log_metric("accuracy", test_accuracy)
        print("Experimento registrado con MLflow.")

        # # --- Sección de Reporte para CML ---
        # # 1. Generar la matriz de confusión
        # cm = confusion_matrix(y_test, y_pred)
        # plt.figure(figsize=(8, 6))
        # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        # plt.title('Matriz de Confusión')
        # plt.xlabel('Predicciones')
        # plt.ylabel('Valores Reales')
        # plt.savefig('results/confusion_matrix.png')
        # print("Matriz de confusión guardada como 'confusion_matrix.png'")
        # # --- Fin de la sección de Reporte ---
        # mlflow.log_artifact("results/confusion_matrix.png")
        metrics = {
            "accuracy": test_accuracy
        }
        with open(os.path.join("results", pipeline.name, "mlflow_metrics.json"), "w") as f:
            json.dump(metrics, f)

    return None

if __name__ == "__main__":
    main()
