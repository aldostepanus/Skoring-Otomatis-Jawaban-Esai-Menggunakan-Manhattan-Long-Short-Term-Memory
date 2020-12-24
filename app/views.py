from app import app
from app import ml_server as ml
from sklearn.model_selection import train_test_split
from flask import render_template, json, request, jsonify
from pandas.tseries.offsets import DateOffset
import pandas as pd
import numpy as np
from pandas import DataFrame
from werkzeug.utils import secure_filename
import os
import csv

import math


# Inisialisasi Kelas
load = ml.LoadData()
mlPrep = ml.Preproses()
plot = ml.Plot()
mlTrain = ml.Train()
mlTest = ml.Testing()
# Akhir Inisialisasi Kelas


# ini lokasi file baik untuk disimpan maupun untuk di buka
# seusaikan dengan direktori
DATASET_LOC = 'app/static/file/dataset'
TEST_LOC = 'app/static/file/test_dataset'
PREDICT_LOC = 'app/static/file/predict/'

ALLOWED_EXTENSIONS = {'csv'}


def set_default(obj):
    if isinstance(obj, set):
        return list(obj)
    raise TypeError


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/train", methods=["GET", "POST"])
def training():
    if request.method == "POST":

        file = request.files["trainDataset"]

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(DATASET_LOC, filename))
        else:
            return jsonify(message='error'), 500

        # Get Input Form

        epoch = int(request.form['epoch'])
        neuron = int(request.form['neuron'])
        loc = os.path.join(DATASET_LOC, filename)
        bs = int(request.form['bs'])

        data = load.load_train_dataset(loc)
        data = mlPrep.preprocessing(data)
        data = mlPrep.text_to_number(data)

        validation_data = load.load_validation_dataset('app/static/file/dataset/validation_dataset.csv')
        validation_data = mlPrep.preprocessing(validation_data)
        validation_data = mlPrep.text_to_number(validation_data)

        kalimat_cols = ['jawaban_siswa', 'kunci_jawaban']
        X_train = data[kalimat_cols]
        Y_train = data['value']
        X_validation = validation_data[kalimat_cols]
        Y_validation = validation_data['value']

        #padding
        X_train = mlPrep.zero_padding(X_train)
        X_validation = mlPrep.zero_padding(X_validation)

        mlTrain.parameter(bs,neuron,epoch)
        model = mlTrain.training(X_train, Y_train, X_validation, Y_validation, mlPrep)
            
        grafikLoss = plot.plot(mlTrain.malstm_trained.history['loss'],mlTrain.malstm_trained.history['val_loss'],'Model Loss','Train','Validation','Loss','Epoch','model_loss')
        early_epoch = len(mlTrain.malstm_trained.history['loss'])
        mlTrain.save_model(mlTrain.model)

        validation_loss = model.evaluate([X_train['left'], X_train['right']], Y_train)
        validation_loss = round(validation_loss, 4)
        hasil = {"message": f"{file.filename} uploaded",
                 "epoch": {epoch},
                 "neuron": {neuron},
                 "gl": {grafikLoss},
                 "loss" : {validation_loss},
                 "early_epoch" : {early_epoch}
                 }

        res = json.dumps(hasil, default=set_default), 200
        return res
    return render_template("index.html")


@app.route("/test", methods=["GET", "POST"])
def test():
    if request.method == "POST":

        file = request.files["testDataset"]

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(TEST_LOC, filename))
        else:
            return jsonify(message='error'), 500

        # Get Input Form
        loc = os.path.join(TEST_LOC, filename)
        # END Get Input

        #! Letak Proses Test
        data = load.load_test_dataset(loc)
        mlTest.set_model(mlTrain)
        mlTest.testing(data, mlPrep, mlTrain)
        pr_value, sp_value = mlTest.korelasi()
        pr_value = round(pr_value, 3)
        sp_value = round(sp_value, 3)

        #! END Letak Proses Test

        hasil = {"message": f"{file.filename} uploaded",
                 "pr": {pr_value},
                 "sp": {sp_value}
                 }

        res = json.dumps(hasil, default=set_default), 200
        return res
    return render_template("index.html")


@app.route("/hpred", methods=["GET"])
def tampil():
    data = []

    # ini mengambil file hasil prediksi
    # jadi file hasil prediksi mu di eksport dulu ke csv
    # tentukan nama filenya jangan yang berubah-ubah mis "hasilPrediksi" atau yang lain
    # tapi kalau mau bikin yang dinamis jg Gaskeun
    with open(PREDICT_LOC + 'hasil_prediksi.csv', encoding='utf-8') as csvfile:
        data_csv = csv.DictReader(csvfile, delimiter=',')
        for row in data_csv:
            data.append(dict(row))
    data = {"data": data}
    return jsonify(data)
