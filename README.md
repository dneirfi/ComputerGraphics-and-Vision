# 1. cascade

  -화상회의를 돕는 프로젝트
  
  -얼굴인식은 이미 학습된 모델을 다운 받아 사용
  
  -손바닥 인식은 positive image 1500개, negative image 4500개로 학습시켜 사용
  
  -positive image는 웹캠으로 다양한 손 이미지 캡쳐 + 구글 이미지 크롤링 
  
  -negative image는 구글 이미지 크롤링
  
  -학습시키는 방법은 http://darkpgmr.tistory.com/73?category=460964 이용
  
  ***-누군가 손을 들면 손바닥을 인식하여 발표자로 간주하고 그 사람의 얼굴을 확대한다***
  
  
# 2. object detecting and tracking

  -6개의 tracker 비교, 분석 (Boosting, MIL, KCF, TLD, MEDIANFLOW, GOTURN)
  {Boosting은 tracking이 굉장히 잘 됬지만, tracking을 하던 물체가 가려져도 fail이 뜨지 않기 때문에 적합하지 않았다. 
    MIL 또한 비슷한 이유인 트래킹이 fail되는 경우가 정확하게 나타나지 않았기 때문에 적합하지 않았다.
    TLD는 성능은 좋았으나 속도가 굉장히 느려 좋지 못하였다. 
    GOTURN은 디렉토리 문제와 내장함수의 문제가 겹쳐져서 사용해 보지 못했다.
  }
  
  -성능, 속도 모두 만족스럽고 tracking이 언제 실패했는지 찾기 쉬운 KCF 선택
  
  -Tracker 감시 보완 알고리즘으로 SURF Detection 이용
  
  -Object가 화면 밖으로 나가거나 장애물에 가려져서 tracker 가 지정한 물체를 찾지 못하면 SURF Detection 사용해 물체 찾음
  
  -<물체를 놓쳤을 때 시나리오 -  SURF Detection>
    {
      Tracker의 status가 Detection failed일 때 이 알고리즘을 사용하도록 구현하였다.
      우선 SURF 를 사용하기 위해 이미지를 Grey-Scale로 바꾼다.
      SURF Detection을 사용하여 Match들을 찾아낸다.
      Detect된 Match 중에 좋은 Match들의 개수가 Minimum match-count 인 10 보다 크다면, 물체를 인식한 것으로 간주한다.
      인식한 물체에 대해 FindHomography 함수를 돌리고, 그결과를 활용해  PerspectiveTransform함수를 돌려 물체의 회전을 맞춘다.
    }
  
