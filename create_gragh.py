import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # CSVファイルの読み込み
    csv_file = '/home/uni/kaneko_ws/isaac/IsaacGymEnvs/statistic_data/output_20240523_192741.csv'
    data = pd.read_csv(csv_file, header=None, names=['Index', 'Success'])

    # 実験ラベルを計算
    data['Experiment'] = data['Index'] % 4

    # 統計データの計算
    stats = data.groupby('Experiment')['Success'].agg(['sum', 'count'])
    stats['Success Rate'] = stats['sum'] / stats['count']

    # 結果の表示
    print(stats)

    # グラフの作成
    plt.figure(figsize=(10, 6))
    plt.bar(stats.index, stats['Success Rate'], color='blue', alpha=0.7)
    plt.xlabel('Experiment')
    plt.ylabel('Success Rate')
    plt.title('Success Rate per Experiment')
    plt.ylim(0, 1)  # 成功率は0から1の範囲であるため
    plt.xticks(stats.index)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()