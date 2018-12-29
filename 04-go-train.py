# %% [markdown]
# ### 実際にTrainしてみよう
#
# Azure Machine Learning Serviceで使用するある一つの実行単位であるExperimentを作っていきましょう。


# %%
from azureml.core import Experiment, Workspace

ws = Workspace.from_config()
experiment_name = 'voice-conversion-sample'

exp = Experiment(workspace=ws, name=experiment_name)


# %% [markdown]
# 以上のコードでは `voice-convresion-sample` という名前でExperimentを作成しました。
# 次にこのExperimentを実行する環境を作っていきます。
# Azure Machine Learning Serviceをクラウド上で行うためにはAzure Managed Compute クラスタを作る必要があります。
# 以下のコードではCPUクラスタを作る例を示しています。
# [元のコード](https://docs.microsoft.com/ja-jp/azure/machine-learning/service/tutorial-train-models-with-aml#set-up-your-development-environment) はこちらを参照してください。


# %%
from azureml.core.compute import AmlCompute
from azureml.core.compute import ComputeTarget
import os

# If you want to accelerate training, set this flag True (this will be more costly)
use_GPU = False

# choose a name for your cluster
compute_name = os.environ.get("BATCHAI_CLUSTER_NAME", "gpucluster" if use_GPU else "cpucluster")
compute_min_nodes = os.environ.get("BATCHAI_CLUSTER_MIN_NODES", 0)
compute_max_nodes = os.environ.get("BATCHAI_CLUSTER_MAX_NODES", 4)

# This example uses CPU VM. For using GPU VM, set SKU to STANDARD_NC6
vm_size = os.environ.get("BATCHAI_CLUSTER_SKU", "STANDARD_NC6" if use_GPU else "STANDARD_D2_V2")


if compute_name in ws.compute_targets:
    compute_target = ws.compute_targets[compute_name]
    if compute_target and type(compute_target) is AmlCompute:
        print('found compute target. just use it. ' + compute_name)
else:
    print('creating a new compute target...')
    provisioning_config = AmlCompute.provisioning_configuration(vm_size = vm_size,
                                                                min_nodes = compute_min_nodes,
                                                                max_nodes = compute_max_nodes)

    # create the cluster
    compute_target = ComputeTarget.create(ws, compute_name, provisioning_config)

    # can poll for a minimum number of nodes and for a specific timeout.
    # if no min node count is provided it will use the scale settings for the cluster
    compute_target.wait_for_completion(show_output=True, min_node_count=None, timeout_in_minutes=20)

     # For a more detailed view of current BatchAI cluster status, use the 'status' property
    print(compute_target.status.serialize())


# %% [markdown]
# これでCPUクラスタの作成が出来ました。
# この作成したCPUクラスタで実際に学習を行っていきます。


# %% [markdown]
# 次は `prepare-env` で作成したデータをクラウド上にアップロードしていきましょう。
# 先程のスクリプトを使用した場合、 `./data/feature` 以下にデータが保存されているので、以下のようなコードを実行し転送してみましょう。


# %%
ds = ws.get_default_datastore()
print(ds.datastore_type, ds.account_name, ds.container_name)

ds.upload(src_dir='./data/feature', target_path='vsvc', overwrite=True, show_progress=True)


# %% [markdown]
# 上記のコードでAzureの今回のプロジェクトに紐付けられたFile Storageの `vsvc` ディレクトに学習データが保存されます。
# 学習スクリプト内で対象のファイルにアクセスするには、上記のコードで宣言した `ds` オブジェクトから取得できるファイルパスでないとアクセスできないので注意しましょう（出来ないこともないですが、環境変数などを見て自分でアクセスするのは若干手間です）。


# %% [markdown]
# 本題の学習フェーズです。
# ここではganttsのベースラインとして採用（？）されているMGE trainingを行ってみます。
# AzureML SDKのEstimatorに学習に必要なスクリプトや依存ライブラリ、また引数を格納しExperimentにsubmitすることで、Azure Machine Learning Service上で学習を行います。


# %%
from azureml.train.estimator import Estimator


i_dir = ds.path('vsvc').path('X')
o_dir = ds.path('vsvc').path('Y')

script_params = {
    '--hparams_name': 'vc',
    '--max_files': -1,
    '--w_d': 0,
    '--hparams': "nepoch=200",
    '--checkpoint-dir': 'outputs/baseline',
    '--log-event-path': "logs/vc_baseline_log",
    '--disable-slack': '',
    '--inputs-dir': i_dir.as_mount(),
    '--outputs-dir': o_dir.as_mount(),
}

est = Estimator(source_directory="./gantts",
                script_params=script_params,
                compute_target=compute_target,
                entry_script='train.py',
                conda_packages=[
                    'numpy',
                    'tensorflow',
                    'docopt',
                    'tqdm',
                    'cython'
                ],
                pip_packages=[
                    'torch',
                    'tensorboard_logger',
                    'nnmnkwii'
                ])


run = exp.submit(config=est)
run

# %% [markdown]
# 上記のコードは実際には簡略化すると以下のような処理を行うのに使われます。
#
# ```sh
# conda install numpy tensorflow docopt tqdm cython
# pip install torch tensorboard_logger nnmnkwii
# cd gantts
# python train.py --hparams_name=vc --max_files=-1 --w_d=0 \
#     --hparams="nepoch=200" --checkpoint_dir=outputs/baseline \
#     --log-event-path=logs/vc_baseline_log --disable-slack \
#     --inputs-dir="ここはリモートのinputのPATH" \
#     --outputs-dir="ここはリモートのoutputのPATH"
# ```
#
# そのためtrain.pyの中身をうまいこと書き換えることで、更に生成モデルの性能の良いGANを使った学習を行うことが出来ます。
# またこの学習の進捗はAzureのダッシュボードからアクセスできる、今回作成したExperimentのページか、VSCodeの拡張を入れている場合は対象のExperimentsを右クリックしてView Experimentを押すことで見ることが可能です。

# %% [markdown]
# ### 学習したモデルを使ってみよう
# 学習したモデルはAzureのダッシュボードからアクセスできるExperimentのページや、またMicrosoft Azure Storage Explorerからダウンロードを行うことが出来ます。
# 学習を行ったファイル名は `checkpoint_epoch200_Generator.pth` のような名前になっているはずなので、それをダウンロードします。
# また入力ファイルの特徴をまとめた `data_mean.npy` と `data_var.npy` もダウンロードしておきます。
# `stat` と言うディレクトリを作って、ダウンロードしたファイルを移動します（このディレクトリ名は何でも構いません）。
#　そして以下のコードを実行することで特定話者から対象話者への声質変換を試すことが出来ます。
#
# ```sh
# python gantts/evaluation_vc.py ./stat/checkpoint_epoch200_Generator.pth ./stat \
# [変換元のwavファイルがあるPATH] outputs --diffvc
# ```


# %% [markdown]
# 環境構築から学習、また学習結果を使用して声質変換を行うところまでを行いました。
# 今回は用意されたデータを使用するものでしたが、これを自分のファイルに置き換えることで自分の声を他者の声にする声質変換モデルを作ることが出来ます。
#
# Azure Machine Learning Serviceの使い方掴んで、快適なクラウド上での機械学習をお楽しみください。