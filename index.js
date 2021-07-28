const dataset = new Dataset();
var isPredicting = false;

let mobilenet;
let model;

const video = document.getElementById('wc');
const webcam = new Webcam(video);
const images_array = new Array;

var c = 0;
var cl = 0;

images_array[0] = "forward.jpg";
images_array[1] = "backward.jpg";
images_array[2] = "upload.jpg";
images_array[3] = "download.jpg";
images_array[4] = "like.jpg";
images_array[5] = "done.jpg";

function start() {
    document.getElementById("sample_image").src = images_array[0];
    document.getElementById('count').innerHTML = "Count: " + c;
}

function capture() {
//    console.log(dataset.labels)
    if (c < 50) {
        const img = webcam.capture();
        const canvas = document.getElementById('canvas');
        var ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0 ,0);

        dataset.addExample(mobilenet.predict(img), cl);
        
        c ++;
        if (c % 10 == 0) {
            document.getElementById("sample_image").src = images_array[Math.floor(c/10)];
            cl ++;
        }
        document.getElementById('count').innerHTML = "Count: " + c;
    } else {document.getElementById("sample_image").src = images_array[5];}

}

async function train() {
    dataset.ys = null;
    dataset.encodeLabels(5);
    model = tf.sequential({layers:[
        tf.layers.flatten({inputShape: mobilenet.outputs[0].shape.slice(1)}),
        tf.layers.dense({units: 100, activation: 'relu'}),
        tf.layers.dense({units: 5, activation: 'softmax'})
    ]})
    model.compile({optimizer: tf.train.adam(0.0001), loss: "categoricalCrossentropy"})
    model.fit(dataset.xs, dataset.ys, {
        epochs: 20,
        callbacks: {
            onBatchEnd: async (batch, logs) => {
                loss = logs.loss.toFixed(5);
                console.log("Loss:" , loss);
            } 
        }
    })
}

async function predict() {
    while (isPredicting) {
        const pred_class = tf.tidy(() => {
            const img = webcam.capture();
            const activation = mobilenet.predict(img);
            const prediction = model.predict(activation);
            return prediction.as1D().argMax();
        });
        const classID = (await pred_class.data())[0];
        document.getElementById('prediction').innerHTML = "Prediction:" + images_array[classID]
    
        pred_class.dispose();
        await tf.nextFrame();
    }
}

function startPrediction() {
    isPredicting = true;
    predict();
}

function stopPrediction() {
    isPredicting = false;
    predict();
}

async function loadMobilenet() {
    const mobilenet = await tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json');
    const layer = mobilenet.getLayer('conv_pw_13_relu');
    return tf.model({inputs:mobilenet.inputs, outputs:layer.output});
}


window.addEventListener('keydown', function(e) {
    if (e.keyCode == 12) {
        capture();
    }
})

async function init() {
    await webcam.setup();
    mobilenet = await loadMobilenet();
    console.log(mobilenet)
}

init();
