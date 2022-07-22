# Shadow_Removal_Tool

## 이미지 저장

```
img_predicted.png: LaMa Shadow로 예측한 영상
img_inpainted.png: img_predicted의 non-shadow 영역을 원본 영상으로 대체
img_edgePredicted.png: img_inpainted의 edge 부분에 LaMa를 이용해 inpainting한 영상
img_edgeInpainted.png: img_edgePredicted의 non-shadow영역을 원본 영상으로 대체
img_mask.png: 원본 마스크에 dilation을 조금 첨가한 mask
img_maskEdge.png: img_mask의 edge를 추출한 mask, LaMa로 inpainting할 때 쓰임
```