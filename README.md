SAT Captcha AI Model
===

Un modelo personalizado que resuelve captchas de SAT en fracciones de segundo.

**[Demo](https://www.boxfactura.com/sat-captcha-ai-model/)** • [Ejemplos de implementación](/demos)

### Entrena tu modelo

Primero asegúrate de tener Python 3.x instalado localmente, después clona el repo.

```bash
git clone git@github.com:BoxFactura/sat-captcha-ai-model.git
```

Ahora instala los paquetes requeridos.

```bash
pip install -r requirements.txt
```

Entrena tu modelo. Este proceso es intensivo y podría tomar horas en terminar.

```bash
python train.py
```

Una vez terminado, podrás verificar el resultado de tu modelo.

```bash
python inferenceModel.py
```

#### Siguientes pasos

1. Afina tu modelo incluyendo más captchas en `dataset/`, asegúrate que el nombre del archivo sea la solución del captcha.
2. Ajusta los parámetros en `train.py` para soportar diferentes captchas de distintas dimensiones.
3. Consume el modelo en tu lenguaje de preferencia mediante [Onnx Runtime](https://onnxruntime.ai/).

**Importante**: Al entrenar un nuevo modelo, además de generar el binario `model.onnx`, creará un archivo de configuración en `configs.py` de donde es necesario extraer el vocabulario y ponerlo como variable en nuestra implementación para poder ser utilizado, recomendamos revisar los demos.

### Créditos

- **[Pylessons](https://pylessons.com/tensorflow-ocr-captcha)** • Por su excelente tutorial, que es la base de esta implementación.
- **[@eclipxe13](https://github.com/eclipxe13/)** • Por su ayuda al revisar y corregir la implementación de PHP.
