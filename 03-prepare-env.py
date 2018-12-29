# %% [markdown]
# ### 学習データの準備
# それでは本題の声質変換を行うためのモデルを作ってみましょう。
# 作るためにはまずデータが必要です。
# 実際に自分のデータで行うには対象話者と同じ内容の文を呼んだ音声が必要ですが、今この記事を読んでいる人でそういうデータを持っている人は少ないと思います。
# なので一旦自分ではなく、コーパスとして公開されているデータを使用してみます。
# 今回使用するのは同人サークルである[日本声優統計学会](https://voice-statistics.github.io/) が配布している声優統計コーパスを使用します。
# 今回はその中の tsuchiya と uemura を使用してみます。
# [声優統計コーパス](https://voice-statistics.github.io/#dataset) のページから通常感情の音声である `tsuchiya_normal.tar.gz` と `uemura_normal.tar.gz` をダウンロードします。
# ダウンロードした2つのファイルを解凍し、その中身を `./data/voice_statistics/tsuchiya_normal/tsuchiya_normal_[ここに数字].wav` と `./data/voice_statistics/uemura_normal/uemura_normal_[ここに数字].wav` となるように配置します。
#
# 以上のwavファイルから声質変換使用する特徴量を抽出します。
# 特徴量抽出には[r9y9/gantts](https://github.com/r9y9/gantts) を使用します。
# 抽出した特徴量で学習を行う際もganttsを使用するため、カレントディレクトリで
#
# ```sh
# $ git submodule update -i
# ```
#
# を叩いて予め手元に落としてください。
# なお今回はオリジナルのganttsをフォークし、Azure Machine Learningで使いやすいように一部改変したものを使用しています。
# それでは実際に特徴量を抽出していきましょう。
#
# ```sh
# pip install pyworld pysptk # ganttsの特徴量抽出が依存しているため
# cd gantts
# pip install -e ".[train]"
# cd ..
# python gantts/prepare_features_vc.py --max_files=100 ./data/voice_statistics \
#    uemura tsuchiya --emotion=normal --dst_dir=./data/feature
# ```
#
# これで学習に必要なデータの抽出が完了しました。
# このデータを使って学習を行っていきます。