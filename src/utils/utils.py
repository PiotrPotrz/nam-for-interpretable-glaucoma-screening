import pandas as pd

def parse_model_path(path):
    name_arr = path.split('$')
    path_dict = {
        "model_save_name": name_arr[0],
        "model": name_arr[1],
        "loss": name_arr[2],
        "optimizer": name_arr[3],
        "epochs": name_arr[4],
        "augmentation": name_arr[5],
        "scheduler": name_arr[6],
        "lr": name_arr[7],
        "pretrained": name_arr[8],
        "batch_size": name_arr[9],
        "dataset": name_arr[10],
        "type": name_arr[-1]
    }
    return path_dict

def add_result(results_df, model_path, result_dict):
    path_dict = parse_model_path(model_path)

    new_row = {"SaveName": path_dict["model_save_name"], "Model": path_dict["model"],
               "Loss": path_dict["loss"], "Optimizer": path_dict["optimizer"], "Epochs": path_dict["epochs"],
               "Augmentation": path_dict["augmentation"], "Scheduler": path_dict["scheduler"], "lr": path_dict["lr"],
               "Pretrained": path_dict["pretrained"], "Batch size": path_dict["batch_size"], "Dataset": path_dict["dataset"]}

    new_row.update({k: v for k, v in result_dict.items()})
    new_row.update({"type": path_dict["type"], "path": model_path})

    if results_df.empty:
        results_df = pd.DataFrame([new_row])
    else:
        results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)
    return results_df
