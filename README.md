# Testing Non-Autoregressive Image Captioning Considering CRF (Conditional Random Field) Loss

# Reference

## Paper

Non-Autoregressive Text Generation with Pre-trained Language Models

https://arxiv.org/pdf/2102.08220

## Implement

NAG-BERT

https://github.com/yxuansu/NAG-BERT/tree/main

# Differences from general Transformer Encoder Decoder

## Feature Extractor + Encoder
We use a pre-trained Clip for Feature Extractor + Encoder.

## Decoder
The decoder uses pre-trained Bert.

## Loss
Loss is the sum of loss_crf, which takes CRF into account, and cross-entropy loss loss_ce.

```
loss = loss_crf + 0.5 * loss_ce
```


For both loss_crf and loss_ce, I used the calculation code from the GitHub source mentioned above. The loss was calculated by taking the sum of the sequence and the average of the batch size.

In the initial calculations, when the loss_ce coefficient was set to 1.0 as stated in the paper instead of 0.5, there were many repeated phrases in the inferred captions.

The parameters for calculating loss_crf are low_rank = 32, beam_size=k=256, dropout = 0.0, pad_idx = tokenizer.pad_token_id.

### Characteristics of CRF

Cross Entropy Loss maximizes the likelihood by comparing each generated token with the training token. CRF Loss maximizes the likelihood by taking into account the relationship between the token and the next token.

It seems that CRF loss is unrelated to repetition, and that Cross-Entropy Loss is what causes repetition.

## Differences between the paper and the implementation

In the paper, the loss is the sum of loss_crf, loss_ce, and lca. lca is a term that considers a window around the token of interest and reduces the probability that the same token will appear within the window.
However, this term is not taken into account in the implementation. In this calculation, lca was not included.

## Inference Function

The inference function uses dynamic programming, taking into account CRF. I used the source code on GitHub above as a reference.

## Caption EOS

There is no padding in the input image. Unlike the paper, the teacher captions have a fixed length of 97 tokens and are padded, but the padding_key_attention_mask is set to the same value as a normal token even at the padding position. In addition, the pad_token_id is not ignored in embedding and loss calculation. It is treated the same as a normal token.

# Calculation Result

Training was performed for 10 epochs on the v7 dataset. Since epoch 9 seemed to have better accuracy than epoch 10, we will post the calculation results for epoch 9. 'CTCLoss' is included for comparison (epoch 10). 'CRFLoss + CELoss' is the calculation result this time. The calculation results for epoch 10 will be posted in the captions folder.


```
CTCLoss
hypo: in this image we can see straw on the surface .
refe: in this i can see there are red colored strawberries.
this pic. WER : 0.6666666666666666
this pic. BLEU: 0.5409165243982964
test number = 1 average, WER = 0.6666666865348816, BLEU = 0.5409165024757385

CRFLoss + CELoss
hypo: in this image there is apberries there are placed on the table.
refe: in this i can see there are red colored strawberries.
this pic. WER : 0.75
this pic. BLEU: 0.5437880605638222
test number = 1 average, WER = 0.75, BLEU = 0.5437880754470825
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/2958180/72e86a2a-7f1a-41ae-8bf5-b3771e7ab7c7.png)
　
```
CTCLoss
hypo: in this image we can people sitting on chair and screens .
refe: there are few persons sitting on the chairs. here we can see monitors, keyboards, tables, and devices.
this pic. WER : 0.8181818181818182
this pic. BLEU: 0.3198485665828107
test number = 2 average, WER = 0.7424242496490479, BLEU = 0.43038254976272583

CRFLoss + CELoss
hypo: in this image we can see a group of people sitting on the chair. in front of the background we can see a table.
refe: there are few persons sitting on the chairs. here we can see monitors, keyboards, tables, and devices.
this pic. WER : 1.0454545454545454
this pic. BLEU: 0.5318584318989124
test number = 2 average, WER = 0.8977272510528564, BLEU = 0.5378232598304749
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/2958180/ba295e00-927d-472c-9358-27bd90975243.png)
　
```
CTCLoss
hypo: in this image can are three people standing and . in the background there can buildings building .
refe: this image is taken outdoors. at the bottom of the there is a floor. in the background there are a few buildings with walls, windows and balconies. in the middle of the image two men and a woman are standing on the floor and they are with smiling faces.
this pic. WER : 0.8571428571428571
this pic. BLEU: 0.1759781025429421
test number = 3 average, WER = 0.780663788318634, BLEU = 0.3455810546875

CRFLoss + CELoss
hypo: in this image there are two persons standing and a woman standing on the background there is a building.
refe: this image is taken outdoors. at the bottom of the there is a floor. in the background there are a few buildings with walls, windows and balconies. in the middle of the image two men and a woman are standing on the floor and they are with smiling faces.
this pic. WER : 0.8214285714285714
this pic. BLEU: 0.23074816150042202
test number = 3 average, WER = 0.8722943663597107, BLEU = 0.43546488881111145
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/2958180/7ab532c2-7ec8-4e50-80dc-a78f4d6c0919.png)
　
```
CTCLoss
hypo: in this image we can see plants flower . there is an insect . the there is sky .
refe: in this image in front there are plants. in the background of the image there is sky.
this pic. WER : 0.5789473684210527
this pic. BLEU: 0.6441286229283673
test number = 4 average, WER = 0.730234682559967, BLEU = 0.42021793127059937

CRFLoss + CELoss
hypo: in this image we can see some plants and there are flowers. in the background we can see the sky.
refe: in this image in front there are plants. in the background of the image there is sky.
this pic. WER : 0.631578947368421
this pic. BLEU: 0.7117591157301676
test number = 4 average, WER = 0.812115490436554, BLEU = 0.5045384168624878
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/2958180/8f697685-d364-4234-8507-9f6e976e3f7f.png)
　

```
CTCLoss
hypo: in this image we can see a woman wearing black scarf .
refe: this is a black and white image. in this image we can see women wearing spectacles.
this pic. WER : 0.631578947368421
this pic. BLEU: 0.5294512659233979
test number = 5 average, WER = 0.7105035185813904, BLEU = 0.4420646131038666

CRFLoss + CELoss
hypo: this is a black and white image. here we can see a woman wearing goggles.
refe: this is a black and white image. in this image we can see women wearing spectacles.
this pic. WER : 0.3684210526315789
this pic. BLEU: 0.7891547975845697
test number = 5 average, WER = 0.7233766317367554, BLEU = 0.5614616870880127
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/2958180/c28eb13b-cdc8-4c2b-bd0b-fab54cc6853a.png)
　
```
CTCLoss
hypo: in this image we can see a food in plate .
refe: as we can see in the image there is a white color plate. in plate there is a dish.
this pic. WER : 0.7619047619047619
this pic. BLEU: 0.32737356029156794
test number = 6 average, WER = 0.7190704345703125, BLEU = 0.4229494333267212

CRFLoss + CELoss
hypo: in this image we can see some food item plate.
refe: as we can see in the image there is a white color plate. in plate there is a dish.
this pic. WER : 0.8095238095238095
this pic. BLEU: 0.33591874058510646
test number = 6 average, WER = 0.7377344965934753, BLEU = 0.5238712430000305
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/2958180/29778bf9-1412-437c-8aef-a16887cfacc6.png)
　
```
CTCLoss
hypo: in this image a woman guitar holding a guitar there is a wall .
refe: this image is clicked in a musical concert where there is a woman standing and she is holding a guitar in her hand. she is wearing black color dress. there is a mic in front of her and there is a bottle. she is holding a stick. there are speakers back side and there are some musical instruments on the bottom left corner.
this pic. WER : 0.8529411764705882
this pic. BLEU: 0.014975101608451207
test number = 7 average, WER = 0.7381948232650757, BLEU = 0.3646673858165741

CRFLoss + CELoss
hypo: in this image i can see a woman standing and holding a guitar and she is a guitar.
refe: this image is clicked in a musical concert where there is a woman standing and she is holding a guitar in her hand. she is wearing black color dress. there is a mic in front of her and there is a bottle. she is holding a stick. there are speakers back side and there are some musical instruments on the bottom left corner.
this pic. WER : 0.8088235294117647
this pic. BLEU: 0.05045683678811388
test number = 7 average, WER = 0.747890055179596, BLEU = 0.45624059438705444
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/2958180/27408da8-df9d-48e7-9919-6cd9de322462.png)
　
```
CTCLoss
hypo: in this image we can see a dog . in the background there is blur snow .
refe: in this image, we can see a black color dog, there is a blurred background.
this pic. WER : 0.5
this pic. BLEU: 0.7534088300060641
test number = 8 average, WER = 0.70842045545578, BLEU = 0.4132600426673889

CRFLoss + CELoss
hypo: in this image we can see a dog which is a black color.
refe: in this image, we can see a black color dog, there is a blurred background.
this pic. WER : 0.3888888888888889
this pic. BLEU: 0.6182981666016651
test number = 8 average, WER = 0.7030149102210999, BLEU = 0.4764977693557739
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/2958180/9e5c5e7a-edd3-43c7-82ec-eef53715efcf.png)
　
```
CTCLoss
hypo: in this image we can see a food on food in bowl . spoon on on there can table .
refe: in this picture there is a bowl and a plate in the center of the image, which contains food items in it.
this pic. WER : 0.8333333333333334
this pic. BLEU: 0.4239862704579267
test number = 9 average, WER = 0.7222996354103088, BLEU = 0.41445186734199524

CRFLoss + CELoss
hypo: in this image we can see some food items on the table there is a plate, we can see a plate.
refe: in this picture there is a bowl and a plate in the center of the image, which contains food items in it.
this pic. WER : 0.8333333333333334
this pic. BLEU: 0.5944948414682107
test number = 9 average, WER = 0.7174947261810303, BLEU = 0.4896085560321808
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/2958180/f03fce68-d5c3-4d8e-a92d-5de5dad61d7f.png)
　
```
CTCLoss
hypo: in this image we can see a food on blur .
refe: in this picture i can observe some food places in the plate. the food is in brown, orange, green and red colors. it is looking like a burger. the background is completely blurred.
this pic. WER : 0.8717948717948718
this pic. BLEU: 0.0243064396862738
test number = 10 average, WER = 0.7372491955757141, BLEU = 0.37543731927871704

CRFLoss + CELoss
hypo: in this image we can see a burger is a plate.
refe: in this picture i can observe some food places in the plate. the food is in brown, orange, green and red colors. it is looking like a burger. the background is completely blurred.
this pic. WER : 0.8205128205128205
this pic. BLEU: 0.039677394219209836
test number = 10 average, WER = 0.7277966141700745, BLEU = 0.4446154534816742
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/2958180/e99d4a70-9e0f-418c-9cfd-ab948431ce64.png)
　
```
CTCLoss
hypo: in this image we can people a on there a table the table .
refe: in this image i can see a person standing wearing a black shirt, blue jeans and glasses. he is holding a electronic gadget in his hand. in the background i can see few people standing, and the ceiling of the building.
this pic. WER : 0.8297872340425532
this pic. BLEU: 0.049831660115363975
test number = 11 average, WER = 0.745661735534668, BLEU = 0.3458368182182312

CRFLoss + CELoss
hypo: in this image we can see a group of people standing on the table. in front of him there is a table.
refe: in this image i can see a person standing wearing a black shirt, blue jeans and glasses. he is holding a electronic gadget in his hand. in the background i can see few people standing, and the ceiling of the building.
this pic. WER : 0.8085106382978723
this pic. BLEU: 0.23652621810060495
test number = 11 average, WER = 0.7351343035697937, BLEU = 0.42569825053215027
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/2958180/b7a008a5-d5c3-4cc0-81e0-5889be4bb80e.png)
　
```
CTCLoss
hypo: in this image we can see car there are banners banners banners . background there can buildings the background and sky .
refe: in this image we can see cars, people, banners, hoardings, tent, pole, trees, boards, and buildings. in the background there is sky.
this pic. WER : 0.6470588235294118
this pic. BLEU: 0.6637801792014791
test number = 12 average, WER = 0.7374448776245117, BLEU = 0.3723320960998535

CRFLoss + CELoss
hypo: in this image we can see a car on the road. i can see the right side of the right side of the background we can see a camera.
refe: in this image we can see cars, people, banners, hoardings, tent, pole, trees, boards, and buildings. in the background there is sky.
this pic. WER : 0.7941176470588235
this pic. BLEU: 0.5253497533691487
test number = 12 average, WER = 0.7400495409965515, BLEU = 0.434002548456192
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/2958180/0b66890d-1f53-4bd0-ade6-2cf276bb8879.png)
　
```
CTCLoss
hypo: in this image there are two men standing on is holding a gun . the background there are trees , . the there are and trees .
refe: in this image we can see two persons standing and holding the objects, there are some stones, grass, plants and trees, also we can see the sky.
this pic. WER : 0.75
this pic. BLEU: 0.575617142838839
test number = 13 average, WER = 0.7384106516838074, BLEU = 0.38796937465667725

CRFLoss + CELoss
hypo: in this image we can see two persons standing on the ground and holding a gun in his hand. in his hand.
refe: in this image we can see two persons standing and holding the objects, there are some stones, grass, plants and trees, also we can see the sky.
this pic. WER : 0.65625
this pic. BLEU: 0.526560920723452
test number = 13 average, WER = 0.7336034178733826, BLEU = 0.4411224126815796
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/2958180/789ec1ac-1aba-42af-a4b6-20ae28a5d90b.png)
　
```
CTCLoss
hypo: in this image we can see a heads and there are . in the there are trees .
refe: in front of the image there are some engravings on the headstone, around the headstone on the surface there are green leaves and dry leaves and sticks, behind the headstone there are trees and a wall.
this pic. WER : 0.7619047619047619
this pic. BLEU: 0.14851496865804886
test number = 14 average, WER = 0.7400888204574585, BLEU = 0.37086552381515503

CRFLoss + CELoss
hypo: in this image we can see a stone with some text on the bottom there are some dried leaves.
refe: in front of the image there are some engravings on the headstone, around the headstone on the surface there are green leaves and dry leaves and sticks, behind the headstone there are trees and a wall.
this pic. WER : 0.8095238095238095
this pic. BLEU: 0.19891352851567473
test number = 14 average, WER = 0.7390262484550476, BLEU = 0.4238217771053314
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/2958180/a54e4a4f-690b-4595-bd42-f1ca02c00fef.png)
　
```
CTCLoss
hypo: in this image can see a wall trees . the there is sky .
refe: in this picture i can see building and few trees and a cloudy sky.
this pic. WER : 0.6
this pic. BLEU: 0.48914025574620656
test number = 15 average, WER = 0.7307495474815369, BLEU = 0.3787505030632019

CRFLoss + CELoss
hypo: in this image we can see a brick wall. in the sky.
refe: in this picture i can see building and few trees and a cloudy sky.
this pic. WER : 0.6
this pic. BLEU: 0.409768330450839
test number = 15 average, WER = 0.7297578454017639, BLEU = 0.42288488149642944
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/2958180/f0bf2d78-df94-4de0-b80e-479bf4609ed4.png)
　
```
CTCLoss
hypo: in this image we can people . there buildings the sky .
refe: in this image there are many people in front of the building. some of them are holding camera. in the background there are buildings. there is a banner over here.
this pic. WER : 0.7647058823529411
this pic. BLEU: 0.12524073602103492
test number = 16 average, WER = 0.7328717708587646, BLEU = 0.36290615797042847

CRFLoss + CELoss
hypo: in this image we can see a group of people standing on the right side of the background we can see a camera.
refe: in this image there are many people in front of the building. some of them are holding camera. in the background there are buildings. there is a banner over here.
this pic. WER : 0.7647058823529411
this pic. BLEU: 0.4262970491809168
test number = 16 average, WER = 0.7319420576095581, BLEU = 0.4230981469154358
```

![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/2958180/61496715-a69c-4caf-9c4d-9deee0f76f0c.png)

　
```
CTCLoss
hypo: in this image we can see a snake the . there see and grass .
refe: in this image i can see a snake on the ground. it is in black color. i can see few wooden sticks, few stones and grass.
this pic. WER : 0.5666666666666667
this pic. BLEU: 0.32647241788622017
test number = 17 average, WER = 0.7230949997901917, BLEU = 0.36076298356056213

CRFLoss + CELoss
hypo: in this image we can see a snake on the ground.
refe: in this image i can see a snake on the ground. it is in black color. i can see few wooden sticks, few stones and grass.
this pic. WER : 0.6333333333333333
this pic. BLEU: 0.22736643766922282
test number = 17 average, WER = 0.7261415123939514, BLEU = 0.4115845263004303
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/2958180/51e0639b-ff3d-4814-b85a-0b241f0b1844.png)

 
　

```
CTCLoss
hypo: in this image we can see a woman holding a paper there is a podium is a laptop the there is a laptop . in the background there is curtain .
refe: in this image we can see a person standing and holding a book and to the side we can see a podium with mic and there is a laptop and some other objects on the table. we can see a person standing in the bottom right.
this pic. WER : 0.6458333333333334
this pic. BLEU: 0.450435022276757
test number = 18 average, WER = 0.7188026905059814, BLEU = 0.36574479937553406

CRFLoss + CELoss
hypo: in this image we can see a woman standing on the podium. in front of the podium. in front of the right side of the background there is a podium.
refe: in this image we can see a person standing and holding a book and to the side we can see a podium with mic and there is a laptop and some other objects on the table. we can see a person standing in the bottom right.
this pic. WER : 0.75
this pic. BLEU: 0.45172358121189515
test number = 18 average, WER = 0.7274670004844666, BLEU = 0.4138144850730896
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/2958180/4f6ad1e7-6036-47f2-b89e-4a32bdc9d0f2.png)

 
```
CTCLoss
hypo: in this image there see two woman on the face is the background .
refe: in this picture i can see 2 women in front and the women right is holding a brush in her hand and i see the paint on the face of the woman on the left and in the background i see the grass and on the top left of this image i see the blue color things.
this pic. WER : 0.8448275862068966
this pic. BLEU: 0.05620873821487133
test number = 19 average, WER = 0.7254354953765869, BLEU = 0.3494534194469452

CRFLoss + CELoss
hypo: in this image we can see two women are two makeup we can see the right side of the right side.
refe: in this picture i can see 2 women in front and the women right is holding a brush in her hand and i see the paint on the face of the woman on the left and in the background i see the grass and on the top left of this image i see the blue color things.
this pic. WER : 0.8275862068965517
this pic. BLEU: 0.1364724599106979
test number = 19 average, WER = 0.7327364087104797, BLEU = 0.39921751618385315
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/2958180/8f3d3259-31e7-4723-9ad9-d4a240b9c968.png)

　

　
```
CTCLoss
hypo: in this image can see a polar water there is a there tree .
refe: in this image we can see an animal, water, rocks, and leaves. at the bottom of the image we can see a person who is truncated.
this pic. WER : 0.7419354838709677
this pic. BLEU: 0.2525684277567393
test number = 20 average, WER = 0.7262605428695679, BLEU = 0.34460917115211487

CRFLoss + CELoss
hypo: in this image we can see a close polar bear. in the background there is a water.
refe: in this image we can see an animal, water, rocks, and leaves. at the bottom of the image we can see a person who is truncated.
this pic. WER : 0.6774193548387096
this pic. BLEU: 0.4027525037534245
test number = 20 average, WER = 0.7299705743789673, BLEU = 0.39939427375793457
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/2958180/d7423c3e-6202-4141-a8e8-11da69634d24.png)

　
　
```
CTCLoss
hypo: in this black and white image can see three persons standing and are on . in the background there can .
refe: in this image we can see three people, one of them is wearing a backpack, in front of them, we can see some bags, box, also we can see some plants, grass, and trees.
this pic. WER : 0.8571428571428571
this pic. BLEU: 0.37845386865337
test number = 21 average, WER = 0.7324930429458618, BLEU = 0.34622082114219666

CRFLoss + CELoss
hypo: this is a black and white image. in this image there are three persons standing on the background we can see some objects.
refe: in this image we can see three people, one of them is wearing a backpack, in front of them, we can see some bags, box, also we can see some plants, grass, and trees.
this pic. WER : 0.8095238095238095
this pic. BLEU: 0.46526049699693484
test number = 21 average, WER = 0.733758807182312, BLEU = 0.4025307595729828
```

![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/2958180/3f145d18-638c-47a7-8f68-f55602a60612.png)



```
CRFLoss + CELoss
test 21 average WER : 0.7337588657323613
test 21 average BLEU: 0.40253075365823865
```
