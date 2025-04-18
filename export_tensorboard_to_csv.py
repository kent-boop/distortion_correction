import os
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator


def export_scalars_to_csv(logdir, output_dir="csv_output"):
    """将 TensorBoard 中的标量数据导出为 CSV"""
    os.makedirs(output_dir, exist_ok=True)

    # 遍历日志目录中的所有事件文件
    for root, _, files in os.walk(logdir):
        for file in files:
            if "events.out" in file:
                event_path = os.path.join(root, file)
                ea = event_accumulator.EventAccumulator(event_path)
                ea.Reload()

                # 导出每个标量标签
                for tag in ea.Tags()["scalars"]:
                    df = pd.DataFrame(ea.Scalars(tag))
                    csv_path = os.path.join(output_dir, f"{tag.replace('/', '_')}.csv")
                    df.to_csv(csv_path, index=False)
                    print(f"Exported {tag} to {csv_path}")


if __name__ == "__main__":
    export_scalars_to_csv(logdir="./logs_distill")  # 修改为你的日志目录