import matplotlib.pyplot as plt
import os
import cv2
import random
import numpy as np

DATADIR = "C:\Users\fish-\デスクトップ\geeksalon\develop\HairImages"
CATEGORIES = ["1", "2", "3", "4", "5"]
IMG_SIZE = 50
training_data = []


def create_training_data():
    for class_num, category in enumerate(CATEGORIES):
        path = os.path.join(DATADIR, category)
        for image_name in os.listdir(path):
            try:
                img_array = cv2.imread(
                    os.path.join(path, image_name),
                )  # 画像読み込み
                img_resize_array = cv2.resize(
                    img_array, (IMG_SIZE, IMG_SIZE)
                )  # 画像のリサイズ
                training_data.append([img_resize_array, class_num])  # 画像データ、ラベル情報を追加
            except Exception as e:
                pass


create_training_data()
random.shuffle(training_data)  # データをシャッフル
X_train = []  # 画像データ
y_train = []  # ラベル情報
# データセット作成
for feature, label in training_data:
    X_train.append(feature)
    y_train.append(label)
# numpy配列に変換
X_train = np.array(X_train)
y_train = np.array(y_train)
# データセットの確認
for i in range(0, 4):
    print("学習データのラベル：", y_train[i])
    plt.subplot(2, 2, i + 1)
    plt.axis("off")
    plt.title(label="Dog" if y_train[i] == 0 else "Cat")
    img_array = cv2.cvtColor(X_train[i], cv2.COLOR_BGR2RGB)
    plt.imshow(img_array)
plt.show()



from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np

#認識させたい画像の読み込み
input_filename = input('画像のパスを入力してください')

input_image = image.load_img(input_filename, target_size=(224,224))

#画像の前処理
input_image = image.img_to_array(input_image)

input_image = np.expand_dims(input_image,axis=0)

input_image = preprocess_input(input_image)

#既存モデルの導入
model = VGG16(weights='imagenet')

#画像を予測
results = model.predict(input_image)

#予測結果とクラス名を紐付け（上位５クラスまで）
decode_results = decode_predictions(results, top=5)[0]

#上位５個の候補を順番に出力
for decode_result in decode_results:
    print(decode_result)