import * as tf from '@tensorflow/tfjs-node';
import * as fs from 'fs';
import * as path from 'path';
import * as readline from 'readline';

const trainDataPath = path.resolve('./data/fashion-mnist_train.csv');
const testDataPath = path.resolve('./data/fashion-mnist_test.csv');

const numOfClasses = 5;
const imageWidth = 28;
const imageHeight = 28;
const imageChannels = 1;
const imageShape = [imageWidth, imageHeight, imageChannels];
const numOfEpochs = 5;
const batchSize = 100;

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

const loadData = (dataPath, batches = batchSize) => {
    const fileStream = fs.createReadStream(dataPath);
    const rl = readline.createInterface({
        input: fileStream,
        crlfDelay: Infinity
    });

    const data = [];

    rl.on('line', (line) => {
        const cols = line.split(',');
        const label = parseInt(cols.pop(), 10);
        const xs = cols.map(Number).map(x => x / 255);
        data.push({ xs, ys: label });
    });

    return new Promise((resolve, reject) => {
        rl.on('close', () => {
            const normalize = ({ xs, ys }) => ({ xs, ys });
            const transform = ({ xs, ys }) => {
                const zeros = new Array(numOfClasses).fill(0);
                return {
                    xs: tf.tensor(xs, imageShape),
                    ys: tf.tensor1d(zeros.map((z, i) => (i === (ys - numOfClasses) ? 1 : 0)))
                };
            };

            const dataset = tf.data
                .array(data)
                .map(normalize)
                .filter(f => f.ys >= (labels.length - numOfClasses))
                .map(transform)
                .batch(batchSize);

            resolve(dataset);
        });
        rl.on('error', reject);
    });
};

const buildModel = (baseModel) => {
    // Remove the last layer
    const newBaseModel = tf.sequential({
        layers: baseModel.layers.slice(0, -1)
    });

    for (const layer of newBaseModel.layers) {
        layer.trainable = false;
    }

    const model = tf.sequential();
    model.add(newBaseModel);
    model.add(tf.layers.dense({
        units: numOfClasses,
        activation: 'softmax'
    }));

    model.compile({
        optimizer: 'adam',
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    });

    return model;
};

const trainModel = async (model, trainingData, epochs = numOfEpochs) => {
    const options = {
        epochs: epochs,
        verbose: 0,
        callbacks: {
            onEpochBegin: async (epoch, logs) => {
                console.log(`Epoch ${epoch + 1} of ${epochs} ...`);
            },
            onEpochEnd: async (epoch, logs) => {
                console.log(`  train-set loss: ${logs.loss.toFixed(4)}`);
                console.log(`  train-set accuracy: ${logs.acc.toFixed(4)}`);
            }
        }
    };

    return await model.fitDataset(trainingData, options);
};

const evaluateModel = async (model, testingData) => {
    const result = await model.evaluateDataset(testingData);
    const testLoss = result[0].dataSync()[0];
    const testAccuracy = result[1].dataSync()[0];

    console.log(`  test-set loss: ${testLoss.toFixed(4)}`);
    console.log(`  test-set accuracy: ${testAccuracy.toFixed(4)}`);
};

const run = async () => {
    try {
        const trainData = await loadData(trainDataPath);
        const testData = await loadData(testDataPath);

        const amount = Math.floor(3000 / batchSize);
        const trainDataSubset = testData.take(amount);

        const baseModelPath = 'file://./models/fashion-mnist-tfjs/model.json';
        const saveModel = 'file://./models/fashion-mnist-tfjs-transfer-learning';

        console.log('Loading the base model...');
        const baseModel = await tf.loadLayersModel(baseModelPath);
        
        const model = buildModel(baseModel);
        model.summary();

        console.log('Training model...');
        const info = await trainModel(model, trainDataSubset);
        console.log(info);

        console.log('Evaluating model...');
        await evaluateModel(model, testData);

        console.log('Saving model...');
        // Ensure the directory exists
        fs.mkdirSync(path.dirname(saveModel.replace('file://', '')), { recursive: true });

        await model.save(saveModel);
        console.log('Model saved successfully.');

    } catch (error) {
        console.error('Error during training:', error);
    }
}

console.log('Running...');
run();
