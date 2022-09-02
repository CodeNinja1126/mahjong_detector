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
- yolo v3 416 모델을 사용했습니다. coco dataset에 학습시킨 모델을 이용해 classifier를 바꾸고 마작 데이터에 fine_tuning시켰습니다.
- https://github.com/ayooshkathuria/YOLO_v3_tutorial_from_scratch 위 repo를 기반으로 학습과 loss함수를 구현하였습니다.

# 진행 상황
<img src = "https://user-images.githubusercontent.com/56903243/188042920-fbe9943e-468c-4426-8f26-ef485f9ee74f.jpeg  width="50" height="50">
![det_0114](https://user-images.githubusercontent.com/56903243/188042920-fbe9943e-468c-4426-8f26-ef485f9ee74f.jpeg)
![det_0100](https://user-images.githubusercontent.com/56903243/188042932-b57c558a-fdaf-4a1d-b1a5-885075b2648e.jpeg)
