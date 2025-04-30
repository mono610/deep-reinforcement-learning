# 深層強化学習の実装

深層強化学習の基本的なアルゴリズムを実装するリポジトリです．<br>
実装中です :construction:


## 実行環境の準備
環境は uv で管理します．uv の詳細については [公式ドキュメント](https://docs.astral.sh/uv/guides/install-python/) や [GitHub](https://github.com/astral-sh/uv) をご確認ください．

### uv のインストール
#### macOS, Linux
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```
#### Windows
```
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### リポジトリの作成
```
git clone https://github.com/mono610/deep-reinforcement-learning.git
cd deep-reinforcement-learning
```

### 仮想環境の作成
```
uv sync
```

## 実行方法
```
uv run python main.py
```

