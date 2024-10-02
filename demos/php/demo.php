<?php

require 'vendor/autoload.php';

use OnnxRuntime\InferenceSession;
use OnnxRuntime\Model;

use Imagine\Gd\Imagine;
use Imagine\Image\Box;
use Imagine\Image\Palette\RGB;
use Imagine\Image\Point;

$imagePath = '../../dataset/L3SVSZ.png';
$modelPath = '../../model/model.onnx';
$alphabet = str_split('Y65WRD98SMBG3NJ21CP4KF7ZXHVTQL');

function softmax($logits) {
    $exp_logits = array_map('exp', $logits);
    $sum_exp_logits = array_sum($exp_logits);
    return array_map(function($exp_logit) use ($sum_exp_logits) {
        return $exp_logit / $sum_exp_logits;
    }, $exp_logits);
}


$imagine = new Imagine();
$img = $imagine->open($imagePath);
$palette = new RGB();
$white = $palette->color('ffffff', 0);
$size = $img->getSize();
$flattenedImage = $imagine->create($size, $white);
$flattenedImage->paste($img, new Point(0, 0));
$resizedImage = $flattenedImage->resize(new Box(160, 60));
$width = $resizedImage->getSize()->getWidth();
$height = $resizedImage->getSize()->getHeight();

$input = [];
for ($y = 0; $y < $height; $y++) {
    for ($x = 0; $x < $width; $x++) {
        $pixel = $resizedImage->getColorAt(new Point($x, $y));

        $red = $pixel->getRed();
        $green = $pixel->getGreen();
        $blue = $pixel->getBlue();

        $input[] = $red;
        $input[] = $green;
        $input[] = $blue;
    }
}

$height = 60;
$width = 160;
$channels = 3;

$reshapedInput = [];
for ($y = 0; $y < $height; $y++) {
    $row = [];
    for ($x = 0; $x < $width; $x++) {
        $index = ($y * $width + $x) * $channels;
        $pixel = array_slice($input, $index, $channels);
        $row[] = $pixel;
    }
    $reshapedInput[] = $row;
}

$reshapedInput = [$reshapedInput];
$session = new InferenceSession($modelPath);
$output = $session->run(null, ['input' => $reshapedInput])[0];

$ocr_text = [];
foreach ($output as $logits_array) {
    foreach ($logits_array as $logits) {
        $probs = softmax($logits);

        $max_index = array_search(max($probs), $probs);
        $predicted = isset($alphabet[$max_index]) ? $alphabet[$max_index] : "";
        if (!end($ocr_text) || $predicted != end($ocr_text)) {
            $ocr_text[] = $predicted;
        }
    }
}
$ocr_result = implode('', $ocr_text);

echo "OCR Result: " . $ocr_result;
