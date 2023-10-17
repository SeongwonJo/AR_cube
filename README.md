## AR cube

### required
- python
  - opencv-python
  - PyYAML
- windows



### 코드 실행 예시

```python
python AR_cube.py --grid 4 4 --image_size 320 240
```



### 옵션 설명

- grid
  - 사용할 체스보드 이미지의 (가로칸수 - 1, 세로칸수 - 1)
  <br>
- image_size
  - 출력할 영상의 크기
  - 캠 영상의 크기가 크면 끊김이 발생할 수 있으므로 원본 크기보다 줄이기 위함
  <br>
- real_time_calibrate
  - vanishing point 를 구하는 방법을 사용해서 30프레임마다 focal length 값 계산해서 갱신
  - 미리 calibration 값을 구해놓을 필요없고 카메라 auto focus 기능을 켜놔도 된다는 장점이 있음
  - 불안정함
  <br>
- intrinsic_params
  - 미리 구한 calibration 값 (focal length 값)을 저장해둔 yaml 파일 경로 입력
  <br>
- calib_method
  - yaml 파일에 저장한 parameter 선택용
  - DarkCamCalibrator 기준 3가지 사용 (zhang, fOV_coupled, fOV_decoupled)
