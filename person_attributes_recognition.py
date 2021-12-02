#!/usr/bin/env python
"""
 Copyright (C) 2018-2020 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.

 Intel社のサンプルを元にy.fukuharaが簡略化と日本語コメントの追記（2020/06/14）

"""
import sys
import os
from argparse import ArgumentParser, SUPPRESS
import cv2
import numpy as np
import logging as log
from openvino.inference_engine import IENetwork, IECore
import time

# コマンド実行時の引数を読む
def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument("-p", "--person", help="Required. Path to an .xml file with a trained model.", required=True,  type=str)
    args.add_argument("-a", "--attr", help="Required. Path to an .xml file with a trained model.", required=True,  type=str)
    args.add_argument("-i", "--input", help="Path to a movie file", type=str)

    return parser


# 実行プログラムのメイン部分
def main():
    # ログ出力の設定
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)

    # コマンドライン引数を読み込む
    args = build_argparser().parse_args()

    # 初期化
    log.info("推論エンジンの設定")
    ie = IECore()

    # PersonDetectionモデルの読み込み
    model_xml_1 = args.person
    model_bin_1 = os.path.splitext(model_xml_1)[0] + ".bin"
    log.info("PersonDetectモデルファイルを確認:\n\t{}\n\t{}".format(model_xml_1, model_bin_1))
    if os.name == 'posix':
        net_1 = ie.read_network(model=model_xml_1, weights=model_bin_1)
    else:
        net_1 = IENetwork(model=model_xml_1, weights=model_bin_1)    
    exec_net_1 = ie.load_network(network=net_1, device_name="CPU")

    # Attrモデルの読み込み
    model_xml_2 = args.attr
    model_bin_2 = os.path.splitext(model_xml_2)[0] + ".bin"
    log.info("Attrモデルファイルを確認:\n\t{}\n\t{}".format(model_xml_2, model_bin_2))
    if os.name == 'posix':
        net_2 = ie.read_network(model=model_xml_2, weights=model_bin_2)
    else:
        net_2 = IENetwork(model=model_xml_2, weights=model_bin_2)    
    exec_net_2 = ie.load_network(network=net_2, device_name="CPU")    
  
    # personモデルのネットワークに関するデータを取得
    log.info("入出力層の情報を取得")
    input_blob_1 = next(iter(net_1.inputs))
    out_blob_1 = next(iter(net_1.outputs))
    net_1.batch_size = 1
    # attrモデルのネットワークに関するデータを取得
    input_blob_2 = next(iter(net_2.inputs))
    out_2 = iter(net_2.outputs)
    out_blob_2_attr = next(out_2) #
    out_blob_2_top = next(out_2)#
    out_blob_2_bottom = next(out_2)#
    net_2.batch_size = 1    

    # 入力層への入力形式を取得する
    n1, c1, h1, w1 = net_1.inputs[input_blob_1].shape
    log.info(str(net_1.inputs[input_blob_1].shape))
    log.info(str(net_1.outputs[out_blob_1].shape))

    n2, c2, h2, w2 = net_2.inputs[input_blob_2].shape
    log.info(str(net_2.inputs[input_blob_2].shape))
    log.info(str(net_2.outputs[out_blob_2_attr].shape))
    log.info(str(net_2.outputs[out_blob_2_top].shape))
    log.info(str(net_2.outputs[out_blob_2_bottom].shape))


    # カメラ（もしくは映像）をオープン ##########################################################
    if args.input: # 映像ファイルを指定した場合
        log.info("映像ファイルを読み込みます。")
        cap = cv2.VideoCapture(args.input)
    else:
        # システムの状態によってはVideoCaputure(0)を1や2に指定する必要がある。
        log.info("カメラを起動します。")
        cap = cv2.VideoCapture(0)

    # 映像を読み込みながら、キー入力があるまでフレームごとに処理をする ###############################
    while True:     
        # VideoCaptureから1フレーム読み込む
        ret, image = cap.read()
        if ret==False:
            break
        original_image = image

        ih, iw = image.shape[:-1]
        # 画像を入力層に合わせてリサイズする
        if (ih, iw) != (h1, w1):    
            image = cv2.resize(image, (w1, h1))
        # 画像データ形式をモデルに合わせて変更（H,W,CからC,H,Wの並び順にする）
        image = image.transpose((2, 0, 1))

        # 推論実行
        res = exec_net_1.infer(inputs={input_blob_1: image})

        # 結果の出力
        res = res[out_blob_1] # 人物検知の出力
        objects = {}
        data = res[0][0]
        for number, proposal in enumerate(data):
            if proposal[2] > 0:
                # imid = np.int(proposal[0])
                imid = number
                # ラベルIDの取得
                label = np.int(proposal[1])
                # 確信度
                confidence = proposal[2]
                # 座標の取得(0-1の範囲なので、縦横ピクセル倍する)
                xmin = np.int(iw * proposal[3])
                ymin = np.int(ih * proposal[4])
                xmax = np.int(iw * proposal[5])
                ymax = np.int(ih * proposal[6])
                # 信頼度が0.5以上なら出力する
                if proposal[2] > 0.5:
                    try:
                        # 元画像から人物部分を切り出して、新しい画像を作る
                        person_image = original_image[ymin:ymax, xmin:xmax]

                        # 画像をattrモデルの入力層に合わせてリサイズする
                        fh, fw = person_image.shape[:-1]
                        if (fh, fw) != (h2, w2):    
                            person_image = cv2.resize(person_image, (w2, h2))
                        # 画像データ形式をモデルに合わせて変更（H,W,CからC,H,Wの並び順にする）
                        person_image = person_image.transpose((2, 0, 1))

                        # 属性の推論実行
                        attrs = exec_net_2.infer(inputs={input_blob_2: person_image})

                        # 属性を取得
                        attr = np.squeeze(attrs[out_blob_2_attr])
                        # topとbottomの座標を取得
                        top = np.squeeze(attrs[out_blob_2_top])
                        bottom = np.squeeze(attrs[out_blob_2_bottom])
                        topx = np.int(fw * top[0])
                        topy = np.int(fh * top[1])
                        bottomx = np.int(fw * bottom[0])
                        bottomy = np.int(fh * bottom[1])

                        # 各情報を配列に格納
                        array =[xmin, ymin, xmax, ymax, xmin + topx, ymin + topy, xmin + bottomx, ymin + bottomy]
                        array.append(attr)
                        objects[imid] = array
                    #　try以降で問題があった場合の処理
                    except Exception as e:
                        # print("error:", e.args)
                        continue

        # 描画処理 ######################################################################
        font = cv2.FONT_HERSHEY_SIMPLEX
        if len(objects)>0:
            for box in objects.values():
                width_center = int((box[0] + box[2]) / 2)
                thickness = 2

                # 四角い枠を描く
                cv2.rectangle(original_image, (box[0], box[1]), (box[2], box[3]), (232, 35, 244), 2)

                # 上半身、下半身のポイントを描画
                cv2.circle(original_image, (box[4], box[5]), 3, (232, 25, 244), -1)
                cv2.circle(original_image, (box[6], box[7]), 3, (232, 25, 244), -1)

                # 各属性を色をつけて表示
                # is_male
                if box[8][0] > 0.5:
                    color = (0, 255, 0)
                else:
                    color = (152, 145, 234)
                cv2.putText(original_image, "male" ,(width_center-200,box[1]-30), font, 0.5, color, thickness)

                # has_bag
                if box[8][1] > 0.5:
                    color = (0, 255, 0)
                else:
                    color = (152, 145, 234)
                cv2.putText(original_image, "bag" ,(width_center-100,box[1]-30), font, 0.5, color, thickness)

                # has_backpack
                if box[8][2] > 0.5:
                    color = (0, 255, 0)
                else:
                    color = (152, 145, 234)
                cv2.putText(original_image, "backpack" ,(width_center,box[1]-30), font, 0.5, color, thickness)

                # has_hat
                if box[8][3] > 0.5:
                    color = (0, 255, 0)
                else:
                    color = (152, 145, 234)
                cv2.putText(original_image, "hat" ,(width_center+100,box[1]-30), font, 0.5, color, thickness)

                # has_longsleeves
                if box[8][4] > 0.5:
                    color = (0, 255, 0)
                else:
                    color = (152, 145, 234)
                cv2.putText(original_image, "longsleeves" ,(width_center-200,box[1]-10), font, 0.5, color, thickness)                

                # has_longpants
                if box[8][5] > 0.5:
                    color = (0, 255, 0)
                else:
                    color = (152, 145, 234)
                cv2.putText(original_image, "longpants" ,(width_center-100,box[1]-10), font, 0.5, color, thickness)                

                # has_longhair
                if box[8][6] > 0.5:
                    color = (0, 255, 0)
                else:
                    color = (152, 145, 234)
                cv2.putText(original_image, "longhair" ,(width_center,box[1]-10), font, 0.5, color, thickness)                

                # has_coat_jacket
                if box[8][7] > 0.5:
                    color = (0, 255, 0)
                else:
                    color = (152, 145, 234)
                cv2.putText(original_image, "coat jacket" ,(width_center+100,box[1]-10), font, 0.5, color, thickness)                

        cv2.imshow('result', original_image)

        # Speceキーで画面キャプチャ
        # ESCキーで終了
        k = cv2.waitKey(1)
        if k == 32:
            cv2.imwrite('out.bmp', original_image)
        if k == 27:
            break

    # 終了処理

    
    cap.release()
    cv2.destroyAllWindows()        


if __name__ == '__main__':
    sys.exit(main() or 0)
