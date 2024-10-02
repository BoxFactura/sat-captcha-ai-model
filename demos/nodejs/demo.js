const ort = require('onnxruntime-node');
const sharp = require('sharp');
const fs = require('fs');

const imagePath = '../../dataset/L3SVSZ.png';
const modelPath = '../../model/model.onnx';
const alphabet = 'Y65WRD98SMBG3NJ21CP4KF7ZXHVTQL'.split('');

const softmax = (logits) => {
  const exps = logits.map(Math.exp);
  const sumExps = exps.reduce((a, b) => a + b);
  return exps.map((exp) => exp / sumExps);
};

const processCaptcha = async () => {
  const imageBuffer = fs.readFileSync(imagePath);

  const img = sharp(imageBuffer);
  const pixels = await img.raw().toBuffer();
  const input = Array.from(pixels);

  const model = await ort.InferenceSession.create(modelPath);
  const tensor = new ort.Tensor('float32', Float32Array.from(input), [1, 60, 160, 3]);
  const outputs = await model.run({ input: tensor });
  const output = outputs.output;

  const { cpuData, dims } = output;
  const [, , logitsSize] = dims;
  const numOfLogits = cpuData.length / logitsSize;

  let processedOutput = [];
  for (let i = 0; i < numOfLogits; i++) {
    const logitsArray = cpuData.slice(i * logitsSize, (i + 1) * logitsSize);
    processedOutput.push(logitsArray);
  }

  let ocrText = [];
  processedOutput.forEach((logits) => {
    const probs = softmax(logits);
    const maxIndex = probs.indexOf(Math.max(...probs));

    if (maxIndex !== -1 && alphabet[maxIndex] !== ocrText[ocrText.length - 1]) {
      ocrText.push(alphabet[maxIndex]);
    }
  });

  console.log('Resultado:', ocrText.join(''));
};

processCaptcha();
