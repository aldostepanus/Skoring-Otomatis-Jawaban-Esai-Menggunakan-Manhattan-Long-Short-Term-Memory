// Fungsi
var ID = function (elID) {
  return document.getElementById(elID);
};

var hide = function (id) {
  return id.classList.add("d-none");
};

var show = function (id) {
  return id.classList.remove("d-none");
};

var inHtm = function (el, content) {
  return (el.innerHTML = content);
};

// Akhir Fungsi

// Deklarasi semua variabel
// yang dibutuhkan menggunakan id elemen

var emptyTraining = ID("trainkosong");
var emptyPrediksi = ID("testkosong");

var btnTrainReset = ID("btnTrainReset");
var btnTestReset = ID("btnTestReset");
var btnTrain = ID("btnTraining");
var btnTest = ID("btnTest");

var spinner = ID("spinner");
var testSpinner = ID("test_spinner");

var hasilTraining = ID("hasilTraining");
var hasilTest = ID("hasilTest")

var grafikLoss = ID("gl");

var formTrain = ID("formTrain");
var formTest = ID("formTest");

// VARIABEL UNTUK DETAIL

// contohnya untuk nampilin neuron
// jadi kita ambil id kolom neuron (lihat index html cari s_neuron)
var s_neuron = ID("s_neuron");
var model_loss = ID("model_loss");
var early_epoch = ID("early_epoch");
var pearson = ID("pearson");
var spearman = ID("spearman");
// untuk yang lain bikin lagi kolom dan id nya terus
// panggil lagi di custom.js samaa seperti diatas


// Akhir Deklarasi Variabel

//! Train
// eslint-disable-next-line no-undef
$(formTrain).submit(function (e) {
  e.preventDefault();
  show(emptyTraining);
  hide(hasilTraining);
  hide(btnTrain);

  show(btnTrainReset);
  show(spinner);

  var formData = new FormData(this);
  // eslint-disable-next-line no-undef
  var xhr = $.ajax({
    url: "/train",
    type: "POST",
    cache: false,
    contentType: false,
    processData: false,
    data: formData,
    success: function (data) {
      // eslint-disable-next-line no-undef
      obj = $.parseJSON(data);

      hide(spinner);
      hide(emptyTraining);
      show(hasilTraining);

      // eslint-disable-next-line no-undef
      $(grafikLoss).attr("src", "./static/img/grafik/" + obj["gl"]);

      console.log(obj["message"]);

      // menampilkan skor model
      inHtm(s_neuron, obj["neuron"] + " neuron");
      inHtm(model_loss, obj["loss"]);
      inHtm(early_epoch, obj["early_epoch"]);

    },
    error: function (xhr, ajaxOption, thrownError) {
      // eslint-disable-next-line no-undef
      Swal.fire({
        icon: "error",
        title: "Proses Dibatalkan",
        confirmButtonColor: "#577EF4",
      });
      location.reload();
    },
  });

  btnTrainReset.onclick = function () {
    xhr.abort();
    $(formTrain)[0].reset();
    hide(btnTrainReset);
    show(btnTrain);
  };
});
//? Akhir Proses Train

//! Akhir Train

//! Get Hasil Prediksi
function setupData() {

  $('#tabelPrediksi').DataTable({
    "ajax": {
      "url": '/hpred',
      "dataType": "json",
      "dataSrc": "data",
      "contentType": "application/json"
    },
    "columns": [{
        "data": "nama"
      },
      {
        "data": "jawaban_siswa"
      },
      {
        "data": "kunci_jawaban"
      },
      {
        "data": "value"
      },
      {
        "data": "prediction"
      }
    ],

    "columnDefs": [{
        "className": "text-center bolded",
        "targets": -1
      },
      {
        "className": "text-center",
        "targets": "_all"
      },
    ],
  });
}
//! END Hasil Prediksi

//! Test
// eslint-disable-next-line no-undef
$(formTest).submit(function (e) {
  e.preventDefault();

  show(emptyPrediksi);
  hide(btnTest);
  hide(hasilTest);

  show(btnTestReset);
  show(testSpinner);
  var formData = new FormData(this);
  // eslint-disable-next-line no-undef
  var xhr = $.ajax({
    url: "/test",
    type: "POST",
    cache: false,
    contentType: false,
    processData: false,
    data: formData,
    success: function (data) {
      // eslint-disable-next-line no-undef
      obj = $.parseJSON(data);
      hide(testSpinner);
      hide(emptyPrediksi);
      show(hasilTest);
      setupData();

      inHtm(pearson, "Pearson Correlation : "+ obj["pr"]);
      inHtm(spearman, "Spearman Correlation : "+ obj["sp"]);
      
    },
    error: function (xhr, ajaxOption, thrownError) {
      // eslint-disable-next-line no-undef
      Swal.fire({
        icon: "error",
        title: 'Terjadi Masalah :(',
        text: 'Periksa Server',
        confirmButtonText: `Oke`
      }).then((result) => {
        if (result.isConfirmed) {
          location.reload();
        }
      })
    },
  });

  btnTestReset.onclick = function () {
    xhr.abort();
    $(formTest)[0].reset();
    hide(btnTestReset);
    show(btnTest);
  };

});
//! Akhir Test