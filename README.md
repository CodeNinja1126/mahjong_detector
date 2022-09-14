# mahjong_detector
- 마작 패를 감지하는 detector입니다.
- 리치마작에 사용되는 패 34개를 감지합니다. 그 외의 패는 others로 분류합니다.
    - 1~9만
    
    ![image](https://user-images.githubusercontent.com/56903243/188040861-09214feb-d507-44c2-b3ea-82d5aa0e74d9.png)
    
    - 1~9통
    
    ![image](https://user-images.githubusercontent.com/56903243/188040966-3edff3e0-ccc0-46d9-ba1b-cd0b5986b2d6.png)
    
    - 1~9삭
    
    ![image](https://user-images.githubusercontent.com/56903243/188040990-8a75b11e-d636-4e52-ba52-3630230b1523.png)
    
    - 자패
    
    <img width="290" alt="스크린샷 2022-09-02 오전 10 46 00" src="https://user-images.githubusercontent.com/56903243/188041336-eb31929b-3884-48c6-8aac-5916960d461b.png">

    - 그 외
        - 꽃 패, 인식 안 되는 패 등등...
# 개요
## YOLO v3
- yolo v3 416 모델을 사용했습니다. coco dataset에 학습시킨 모델을 이용해 classifier를 바꾸고 마작 데이터에 fine tuning했습니다.
- [튜토리얼](https://github.com/ayooshkathuria/YOLO_v3_tutorial_from_scratch) 코드를 기반으로 학습과 loss함수를 구현해 추가하였습니다.

## 진행 상황

<img src ="https://user-images.githubusercontent.com/56903243/188042920-fbe9943e-468c-4426-8f26-ef485f9ee74f.jpeg" width="50%" height="50%"/>
<img src ="https://user-images.githubusercontent.com/56903243/188042932-b57c558a-fdaf-4a1d-b1a5-885075b2648e.jpeg" width="50%" height="50%"/>

- 약 100장 정도의 데이터로 학습을 수행한 결과입니다.
- 적은 데이터로도 높은 분류 성능을 확인할 수 있었습니다.
- 적은 데이터로도 그럴듯하게 bounding box들을 위치시켰지만 아직 개선의 여지가 있어보입니다.
- 이후 데이터를 증가시키고, 하이퍼 파라미터 튜닝, augmentation, 등의 기술들을 시도해 성능을 향상시킬 예정입니다.
- 자세한 개발 과정은 [링크](https://thrilling-gate-29a.notion.site/4e89b09fbdf94720be83701b6a8c4a1f)를 참고해 주세요. 
