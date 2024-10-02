# frozen_string_literal: true

require 'onnxruntime'
require 'mini_magick'
require 'numo/narray'

def softmax(arr)
  arr = arr.flatten
  exps = arr.map { |x| Math.exp(x) }
  sum = exps.sum
  exps.map { |x| x / sum }
end

alphabet = 'Y65WRD98SMBG3NJ21CP4KF7ZXHVTQL'.chars
model = OnnxRuntime::Model.new('../../model/model.onnx')

img = MiniMagick::Image.open('../../dataset/L3SVSZ.png')
img.resize '160x60^', '-gravity', 'center', '-extent', '160x60'
pixels = img.get_pixels
pixels = Numo::NArray.cast(pixels).cast_to(Numo::DFloat)
pixels = pixels.expand_dims(0)

result = model.predict({ input: pixels })
output = result['output']
ocr_text = []

output.each do |logits_array|
  logits_array.each do |logits|
    probs = softmax(logits)
    max_index = probs.index(probs.max)
    ocr_text << alphabet[max_index] if max_index && alphabet[max_index] != ocr_text.last
  end
end

ocr_text = ocr_text.join

puts "Resultado: #{ocr_text}"
