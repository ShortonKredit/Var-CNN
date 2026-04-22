import os

replacements = {
    'from keras.layers.convolutional import': 'from tensorflow.keras.layers import',
    'from keras.layers.normalization import': 'from tensorflow.keras.layers import',
    'from keras.utils.np_utils import': 'from tensorflow.keras.utils import',
    'from keras.models import': 'from tensorflow.keras.models import',
    'from keras.layers import': 'from tensorflow.keras.layers import',
    'from keras.optimizers import': 'from tensorflow.keras.optimizers import',
    'from keras.callbacks import': 'from tensorflow.keras.callbacks import',
    'from keras import': 'from tensorflow.keras import',
    'import keras': 'import tensorflow.keras as keras'
}

for root, _, files in os.walk('.'):
    for f in files:
        if f.endswith('.py') and f != 'fix_imports.py':
            path = os.path.join(root, f)
            with open(path, 'r', encoding='utf-8') as file:
                content = file.read()
            original = content
            for old, new in replacements.items():
                content = content.replace(old, new)
            if original != content:
                with open(path, 'w', encoding='utf-8') as file:
                    file.write(content)
                print(f"Updated {path}")
