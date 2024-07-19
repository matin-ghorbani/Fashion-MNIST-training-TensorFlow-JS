import * as tf from '@tensorflow/tfjs-node';
import Jimp from 'jimp';

const labels = [
    'T-shirt/Top',
    'Trouser',
    'Pullover',
    'Dress',
    'Coat',
    'Sandal',
    'Shirt',
    'Sneaker',
    'Bag',
    'Ankle boot'
];

const imageWidth = 28;
const imageHeight = 28;
const imageChannels = 1;
const imageShape = [imageWidth, imageHeight, imageChannels];

const toPixelData = async (imgPath) => {
    const pixelData = [];
    const img = await Jimp.read(imgPath);

    img
        .resize(imageWidth, imageHeight)
        .grayscale()
        .invert()
        .scan(0, 0, imageWidth, imageHeight, (x, y, idx) => {
            let v = img.bitmap.data[idx + 0];
            pixelData.push(v / 255);
        });

    return pixelData;
};

const predict = async (model, imgPath) => {
    const pixelData = await toPixelData(imgPath);
    const imgTensor = tf.tensor(pixelData, imageShape);
    const inputTensor = imgTensor.expandDims();
    const prediction = model.predict(inputTensor);

    const scores = prediction.arraySync()[0];
    const maxScore = prediction.max().arraySync();
    const maxScoreIndex = scores.indexOf(maxScore);

    const labelScores = {};

    scores.forEach((s, i) => {
        labelScores[labels[i]] = parseFloat(s.toFixed(4));
    });

    return {
        prediction: `${labels[maxScoreIndex]} (${parseInt(maxScore * 100)}%)`,
        scores: labelScores
    };
};

let run = async () => {
    if (process.argv.length < 3) {
        console.log('Please pass an image to process. ex:');
        console.log('  node test_model.js /path/to/image.jpg');
    } else {
        const imgPath = process.argv[2];
        const modelPath = 'file://./models/fashion-mnist-tfjs/model.json';

        console.log('Loading the model...');
        const model = await tf.loadLayersModel(modelPath);
        model.summary();

        console.log('Running prediction...');
        const prediction = await predict(model, imgPath);
        console.log(prediction);
    }
};

run().catch(error => {
    console.error('Error during prediction:', error);
});
