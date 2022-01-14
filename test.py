#openCV をインポート
import cv2

# 顔を変えるための関数
def effect_face_func(img, rect):
    # 黒い画像(顔面上に乗せたい)をセット
    image_file  = "//Users/satoutakumi/Documents/my-product/face-secret/face-black/img/face-black.jpg" #絶とした画像の絶対パスに変更
    #画像の読み込み
    marge_image = cv2.imread(image_file,-1)
    #画像をグレーに加工(画像のデータ量の圧縮)
    marge_gray = cv2.cvtColor(marge_image, cv2.COLOR_BGR2GRAY)

    #検出した顔の座標 + 10 でタプルを作成
    (x1, y1, x2, y2) = rect

    #検出した顔の大きさの横幅と高さを計算
    w = x2 - x1
    h = y2 - y1

    #グレー化した画像を検出した顔の面積にリサイズ
    img_face = cv2.resize(marge_gray, (w, h))

    #カメラの映像(画像)をコピー
    img2 = img.copy()
    #検出された顔面の面積分をリサイズしたグレー画像に上書き
    img2[y1:y2, x1:x2]      = img_face
    #編集された画像を返す
    return img2

# 定数定義
WINDOW_NAME = "face-black"
DEVICE_ID   = 0     # カメラのデバイスID(0:Macに内蔵のカメラ)

# 分類器の指定
cascade_file = "/Users/satoutakumi/Documents/my-product/face-secret/face-black/haarcascade_frontalface_default.xml" #haarcascade_frontalface_default.xmlのある場所の絶対パスに変更
#haarcascade_frontalface_default.xml は https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_alt.xml より引用
face_cascade = cv2.CascadeClassifier(cascade_file)

# カメラから映像を取得
cap = cv2.VideoCapture(DEVICE_ID)

# 初期フレームの読込
end_flag, c_frame = cap.read()
height, width, channels = c_frame.shape

# 画像を表示するウィンドウを準備
cv2.namedWindow(WINDOW_NAME)

# 映像処理のループ
while end_flag == True:

    # 画像の取得
    img      = c_frame
    #取得した画像をグレー化
    gray     = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #画像から顔を検出
    faces    = face_cascade.detectMultiScale(gray, minSize=(100, 100))

    # 検出した顔の座標を利用して加工
    for (x, y, w, h) in faces:
        ## 顔に画像をつける
        gray = effect_face_func(gray, (x+10, y+10, x+w+10, y+h+10))

    # フレームに表示をする
    cv2.imshow(WINDOW_NAME, gray)

    # qキーが押されたら途中終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # 次のフレーム読み込み
    end_flag, c_frame = cap.read()

# 終了処理
cv2.destroyAllWindows()
cap.release()



