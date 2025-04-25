from mlflow.tracking import MlflowClient
import mlflow

def register_model(model_uri: str, model_name: str, tags: dict):
    client = MlflowClient()

    result = mlflow.register_model(model_uri=model_uri, name=model_name)

    # Thêm tag cho phiên bản model đã được đăng ký
    for key, value in tags.items():
        client.set_model_version_tag(name=model_name, version=result.version, key=key, value=value)

    return result.version



def compare_models(new_run_id: str, experiment_name: str, metric_key: str = "mape") -> bool:
    """
    So sánh model mới với model tốt nhất trong cùng experiment.

    Args:
        new_run_id (str): Run ID của model mới vừa huấn luyện.
        experiment_name (str): Tên experiment.
        metric_key (str): Tên metric để so sánh (ví dụ: "mape").

    Returns:
        bool: True nếu model mới tốt hơn, False nếu không.
    """
    client = mlflow.tracking.MlflowClient()

    # Lấy thông tin model mới
    new_metrics = client.get_run(new_run_id).data.metrics
    new_metric_value = new_metrics.get(metric_key, None)
    if new_metric_value is None:
        raise ValueError(f"Metric '{metric_key}' không tồn tại trong run {new_run_id}")

    # Lấy top model trước đó trong cùng experiment
    experiment = client.get_experiment_by_name(experiment_name)
    all_runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"attributes.status = 'FINISHED'",
        order_by=[f"metrics.{metric_key} DESC"],
        max_results=5,
    )

    all_runs = [run for run in all_runs if metric_key in run.data.metrics and run.info.run_id != new_run_id]

    if not all_runs:
        print("Không có model trước đó để so sánh — dùng model mới.")
        return True

    best_old_run = all_runs[0]
    best_old_value = best_old_run.data.metrics[metric_key]

    print(f"[Model cũ] {metric_key}: {best_old_value:.4f}")
    print(f"[Model mới] {metric_key}: {new_metric_value:.4f}")

    return new_metric_value < best_old_value