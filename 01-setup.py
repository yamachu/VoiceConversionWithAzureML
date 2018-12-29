# %% [markdown]
# ### Pythonの環境を作ろう
# 今回はmini-condaを使ってAzureMLを使うための環境を作っていきます。
# mini-condaの3系をダウンロードしてインストールしてある前提で進めていきます。
# 今回は azureml というenvをPython3.6を使って作ります。
# ```sh
# conda create -n azureml -y Python=3.6
# conda activate azureml
# ```


# %% [markdown]
# ### AzureMLのSDKをインストール
# AzureMLのSDKをインストールしていきましょう。
# 執筆時（2018年11月末日）ではpipでAzureMLのSDKをインストールする際に関連パッケージのバージョンの問題で正しくパッケージがインストールできない状態でした。
# そのためSDKのインストール前に依存パッケージのバージョン固定を行います。
# ```sh
# pip install -U requests
# pip install numpy==1.14.5
# pip install pytz==2018.5
# ```
# 環境によっては不要ですが、備忘録として載せておきます。
# その後SDKを導入します。
# 全部が全部必要なパッケージではないですが、とりあえず以下のコマンドでパッケージをインストールしましょう。
# ```sh
# pip install --upgrade "azureml-sdk[notebooks,automl]" azureml-dataprep
# ```
# 他のパッケージは以下のリンクで確認することが出来ます。
#
# https://docs.microsoft.com/ja-jp/azure/machine-learning/service/quickstart-create-workspace-with-python#next-steps


# %% [markdown]
# ### AzureML SDKが入っているのか確認


#%%
import azureml.core

print(azureml.core.VERSION)

# %% [markdown]
# importが出来て、Versionっぽい数字が出たらひとまずSDKはインストールできたと思って良さそうです。
