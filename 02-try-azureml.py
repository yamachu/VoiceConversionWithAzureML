# %% [markdown]
# ### Azure上にWorkspaceを作ってみよう
#
# 適当なResourceGroupとWorkspaceの名前を付けてWorkspaceを作ります。
# ResourceGroupもまだ作られていない場合は、以下のようなコードでWorkspaceを作成することが出来ます。

# %%
from azureml.core import Experiment, Workspace  # Experiment は後で使う

ws = Workspace.create(name='try-azureml',
                      subscription_id='ここに自分のSubscriptionId',
                      resource_group='try-azureml',
                      create_resource_group=True,
                      location='eastus2'  # 日本リージョンはないので適当に
                      )

# %% [markdown]
# 作成されたWorkspaceに接続するためのconfigファイルを以下のコードで保存しましょう。

# %%
ws.write_config()

# %% [markdown]
# 設定ファイルからWorkspaceを復旧するには以下のコードを使用します。

# %%
ws = Workspace.from_config()

# %% [markdown]
# ### 動くかどうかの確認をしよう
# https://docs.microsoft.com/ja-jp/azure/machine-learning/service/quickstart-create-workspace-with-python#use-the-workspace
# のコードを参考に、以下のコードを実行してみます。


# %%

# create a new experiment
exp = Experiment(workspace=ws, name='myexp')

# start a run
run = exp.start_logging()

# log a number
run.log('my magic number', 42)

# log a list (Fibonacci numbers)
run.log_list('my list', [1, 1, 2, 3, 5, 8, 13, 21, 34, 55])

# finish the run
run.complete()

# %% [markdown]
# ### ログを見てみる
# ログはAzureのポータルで見たり、VSCodeから見たりすることが出来ます。
# Azureのポータルで見る場合は、以下のコードを実行しURLを取得します。
# そのURLにアクセスすると二次曲線的なグラフが見ることが出来ます。
#
# ref: https://docs.microsoft.com/ja-jp/azure/machine-learning/service/quickstart-create-workspace-with-python#view-logged-results

# %%
print(run.get_portal_url())

# %% [markdown]
# VSCodeで見る場合はVSCodeの設定が必要です。
#
# VSCodeのPython: Select Intrepreterで先程作ったPythonのenvであるazuremlを選択し、インストールしてないのであれば
# https://marketplace.visualstudio.com/items?itemName=ms-toolsai.vscode-ai
# からVisual Studio Code Tools for AIの拡張をインストールします。
#
# その後VSCodeのAzureのTabから自分のサブスクリプションを選択し、Workspaceを開きます。
# するとポータルで見ることが出来るものと同様のものが見ることが出来ます。
