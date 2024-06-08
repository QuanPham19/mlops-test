if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter

import mlflow
import mlflow.sklearn
import pickle

@data_exporter
def export_data(output, *args, **kwargs):
    """
    Exports data to some source.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Output (optional):
        Optionally return any object and it'll be logged and
        displayed when inspecting the block run.
    """
    mlflow.set_tracking_uri("http://mlflow:5000")
    with mlflow.start_run():
        dv, lr = output[0], output[1]
        mlflow.sklearn.log_model(lr, "linear_regression_model")

        dv_path = "dict_vectorizer.pkl"
        with open(dv_path, "wb") as f:
            pickle.dump(dv, f)
        mlflow.log_artifact(dv_path)

        print(f"Model and DictVectorizer saved in run {mlflow.active_run().info.run_id}")
    # Specify your data exporting logic here
