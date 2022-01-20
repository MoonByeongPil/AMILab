import torch
import numpy as np
import skimage.data
from PIL import Image, ImageDraw, ImageFont
import math
#from tensorflow.python.platform import gfile
import scipy.misc

IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256

# *****************************************************************************************************

def calc_loss(output, y, z_r):  #y와 output의 행렬 shape가 같아야 할 것으로 보임, 결국 y와 output의 차이를 비교.
#tf.where(  bool type 텐서,   True일 때 출력값,   False일 때 출력값  )
    # y refine
    y_masked = torch.where(z_r, y, 0*torch.ones_like(y))  #bool 타입 z_r, 이 행렬에서 true값이 들어가 있는 곳은 y를 참조하고 false가 들어가 있는 곳은 0값으로 넣어버림.//tf.ones_like(y)는 y배열의 shape와 같은데 내용물이 1인 행렬
    y_masked_flat_refined = torch.reshape(y_masked,[-1, IMAGE_HEIGHT*IMAGE_WIDTH]) #위 마스크된 결과 reshape할 것. 높이축은 임의로, 가로축 넓이는 이미지 입력 높이*너비
    
    # output refine
    o_masked = torch.where(z_r, output, 0*torch.ones_like(y)) #bool 타입z_r에 의해서 true인 부분은 output 넣고 false면 0넣음
    o_masked_flat_refined = torch.reshape(o_masked,[-1, IMAGE_HEIGHT*IMAGE_WIDTH]) #위에서 만든 행렬의 모양을 변경, 너비를 높이*너비로 고정. (y_masked와 동일)
    
    # mask refine ==> binary하게 refine즉, 바이너리 마스크다.
    mask_one_refined =torch.where(z_r, torch.ones_like(y), 0*torch.ones_like(y))    #z_r true면 1, false면 0.
    mask_one_flat_refined = torch.reshape(mask_one_refined,[-1, IMAGE_HEIGHT*IMAGE_WIDTH]) #마찬가지로, 너비를 높이*너비로 고정한다. (reshape)
    
    # num of pixels
    numOfPix = torch.sum(mask_one_flat_refined,1)   #mask_one_flat_refined를 행 단위로 sum, 즉, numOfPix의 각 행은 하나의 이미지 넓이에 대해서 1의 개수를 센다. 
    
    d = torch.subtract(o_masked_flat_refined, y_masked_flat_refined)   #z_r이 true인 부분에 대해서는 output-y를 대입, false인 부분에 대해서는 0 대입.
    d_sum = torch.sum(torch.square(d),1)   #위의 각 요소별 제곱을 한 후에 행 단위로 합침. 각 행별로 더하므로 
    
    cost = torch.mean(torch.div(d_sum, numOfPix))
#여기서 torch는 mean함수를 나오면 tensor로 반환함. 상수 아닌데 괜찮나. 
#truediv는 파이썬3에서 하듯이 나눗셈시 실수형으로 답 나옴. python2에서는 정수형으로 내림. tf나 torch나 div하면 tensor 형으로 나옴.
    return cost

# *****************************************************************************************************

def calc_loss_normal(output, y_normal,z_refined):   #Normal에 대한 loss를 계산 :4차원 텐서인가? ==> 아래에 reduce_sum이 axis=3에 대해 수행됨.

    # gives mean angle error for given output tensor and its ref y
    output_mask = torch.abs(output) < 1e-5 #output의 절댓값이 10의 -5승보다 작으면 mask에 bool로 저장.
    output_no0 = torch.where(output_mask, 1e-5*torch.ones_like(output), output)   #위의 마스크가 true인 경우에는 10 -5승을 입력으로 넣어줌. false면 output의 값을 넣어줌.
    output_mag = torch.unsqueeze(torch.sqrt(torch.sum(torch.square(output_no0),3)),-1) #위의 행렬 요소 제곱 -> 3이라는 축(4번째로 큰 축=1차원)으로 더해줌 최소 4차원-> 모든 요소에 root. -1은 가장 안쪽 차원 추가(가장 작은 차원)
    output_unit = torch.divide(output_no0,output_mag)  #요소별 나눗셈 진행 ==> 말 그대로 normalize: 가장 낮은 차원의 요소들을 하나의 벡터로 보면 가장 낮은 차원의 벡터는 크기가 모두 1로 만들어줌.
                                                    #output이 작은 요소의 경우 입력을 10^-5로 통일해버림.
    z_mask = z_refined[...,0]   #가장 낮은 차원을 없애면서 가장 낮은 차원의 수들 중 가장 앞에 있는 수 추출.
    a11 = torch.masked_select(torch.sum(torch.square(output_unit),3),z_mask)   #4차원 벡터의 각 요소 제곱합.제곱하고 reduce_sum하면 4번째 차원이 제거 되면서 모든 요소 1, 10^-5로 퉁친 부분은 값 커짐--> 여기다 z_mask를 씌우면 True인 부분만 남음
    a22 = torch.masked_select(torch.sum(torch.square(y_normal),3),z_mask)  #4차원 벡터의 각 요소 제곱합.제곱하고 reduce_sum하면 4번째 차원이 제거되면서 모든 요소 ==> a11과 동일한 결과??? = y_normal의 가장 낮은 차원의 벡터의 크기가 1인 경우 && y-normal과 ouput_unit의 차원읻 동일하다면 같은 결과 나올 것.
    a12 = torch.masked_select(torch.sum(torch.multiply(output_unit,y_normal),3),z_mask)    #제곱하는 대신 output_unit과 y_normal을 곱한다. 이후 제곱후 합을 진행. z_mask 씌워 a12.
#이해안됨!!!!! a11과 a22는 같은 위치에서 같은 1이라는 값을 갖지 않나?
    cos_angle = a12/torch.sqrt(torch.multiply(a11,a22))   #cos각을 값으로 갖는 행렬 cos 공식과 동일. 메모장에 기록.
    cos_angle_clipped = torch.clamp(torch.where(torch.isnan(cos_angle),-1*torch.ones_like(cos_angle),cos_angle),-1,1)  #true(nan)일 때는 -1, false(nan 아님)일 때는 cos각도 나옴. (-1~1)사이 값만 가지도록 clip
    # MAE, using tf.acos() is numerically unstable, here use Taylor expansion of "acos" instead
    loss = torch.mean(3.1415926/2-cos_angle_clipped-torch.pow(cos_angle_clipped,3)/6-torch.pow(cos_angle_clipped,5)*3/40-torch.pow(cos_angle_clipped,7)*5/112-torch.pow(cos_angle_clipped,9)*35/1152)
    return loss

def calc_loss_normal2(output, y_normal,z_refined):
    
    # gives mean angle error for given output tensor and its ref y
    output_mask = torch.abs(output) < 1e-5 #output의 절댓값이 10의 -5승보다 작으면 mask에 bool로 저장.
    output_no0 = torch.where(output_mask, 1e-5*torch.ones_like(output), output)   #위의 마스크가 true인 경우에는 10 -5승을 입력으로 넣어줌. false면 output의 값을 넣어줌.
    output_mag = torch.unsqueeze(torch.sqrt(torch.sum(torch.square(output_no0),3)),-1) #위의 행렬 요소 제곱 -> 3이라는 축(4번째로 큰 축=1차원)으로 더해줌 최소 4차원-> 모든 요소에 root. -1은 가장 안쪽 차원 추가(가장 작은 차원)
    output_unit = torch.divide(output_no0,output_mag)  #요소별 나눗셈 진행 ==> 말 그대로 normalize: 가장 낮은 차원의 요소들을 하나의 벡터로 보면 가장 낮은 차원의 벡터는 크기가 모두 1로 만들어줌.
                                                    #output이 작은 요소의 경우 입력을 10^-5로 통일해버림.
    z_mask = z_refined[...,0]   #가장 낮은 차원을 없애면서 가장 낮은 차원의 수들 중 가장 앞에 있는 수 추출.
    
    z_mask = z_refined[...,0]
    a11 = torch.masked_select(torch.sum(torch.square(output_unit),3),z_mask)    #4차원 벡터의 각 요소 제곱합.제곱하고 reduce_sum하면 4번째 차원이 제거 되면서 모든 요소 1, 10^-5로 퉁친 부분은 값 커짐--> 여기다 z_mask를 씌우면 True인 부분만 남음
    a22 = torch.masked_select(torch.sum(torch.square(y_normal),3),z_mask)  #4차원 벡터의 각 요소 제곱합.제곱하고 reduce_sum하면 4번째 차원이 제거되면서 모든 요소 ==> a11과 동일한 결과??? = y_normal의 가장 낮은 차원의 벡터의 크기가 1인 경우 && y-normal과 ouput_unit의 차원읻 동일하다면 같은 결과 나올 것.
    a12 = torch.masked_select(torch.sum(torch.multiply(output_unit,y_normal),3),z_mask)    #제곱하는 대신 output_unit과 y_normal을 곱한다. 이후 제곱후 합을 진행. z_mask 씌워 a12.
#이해안됨!!!!! a11과 a22는 같은 위치에서 같은 1이라는 값을 갖지 않나?
    cos_angle = a12/(a11+0.00001)   #cos각을 값으로 갖는 행렬 cos 공식 대신 이것을 사용.
    loss = torch.mean(torch.acos(cos_angle))   #cos값들의 map에서 loss를 구함. 딱히 axis가 지정되지 않았으므로 모든 세타에 대해 평균을 구함. 
    return loss



# ***************************************************************************************************

def calc_loss_d_refined_mask(output, y, z_refined):
    
    multiply = torch.tensor([IMAGE_HEIGHT*IMAGE_WIDTH])  #상수 = 256*256
    
    # mask nonrefine
    mask_one = torch.where(z_refined, torch.ones_like(y), 0*torch.ones_like(y))  #true이면 1 false면 0
    mask_one_flat = torch.reshape(mask_one,[-1, IMAGE_HEIGHT*IMAGE_WIDTH]) #위의 행렬을 reshape. 너비만 256*256으로 고정. (높이는 고정된 h라 하자)
    
    # y refine
    y_masked = torch.where(z_refined, y, 0*torch.ones_like(y))    #z_refined 기준으로 true면 y출력, false면 0
    y_masked_flat_refined = torch.reshape(y_masked,[-1, IMAGE_HEIGHT*IMAGE_WIDTH]) #같은 모양으로 reshape. 너비를 256*256고정
    
    max_y = torch.max(y_masked_flat_refined,1)  #axis 1에 대해 가장 큰 값을 추출. 즉, 너비 256*256당 1개의 max값 도출
    matrix_max_y = torch.transpose(torch.reshape(max_y.repeat(multiply), [ multiply[0], torch.numel(max_y)]))   #tile=max_y를 복붙하는데 256*256 길이로 복붙. 각 픽셀마다 max_y 벡터 존재.
    #Max_y(h)가 동일한 형태로 256*256으로 배열되어 있던거를 reshape: [256*256, max_y벡터크기]
    #max_y(h)가 가로로 256*256개 늘어서 있는 모양 max_y.repeat(multiply,1)일수도##====> axis 결정해주기!!!!! 혹시 안맞을 경우
    
    # normalize depth
    output_flat = torch.reshape(output,[-1, IMAGE_HEIGHT*IMAGE_WIDTH]) #output을 reshape: 너비를 256*256으로, 높이는 모름.(H=정해진 높이)
    output_flat_masked = torch.multiply(output_flat, mask_one_flat)    #z_refined가 true면 reshape된 output 결과 그대로, false면 0
    
    output_max = torch.max(output_flat_masked,1)    #output masked를 1차원에 대해 max값 취함. 
    matrix_max = torch.transpose(torch.reshape(output_max.repeat(multiply), [ multiply[0], torch.numael(output_max)]))   #위 행렬을 256*256
    #multiply[0] = 행렬의 전체 요소 개수를 의미 (크기 의미)
    #output_max(H)가 가로로 256*256개 늘어서 있는 모양###output_max.repeat(multiply)##tf.tile(output_max, multiply)
    output_min = torch.min(output_flat_masked,1)    #output masked를 1차원에 대해 min값 취함. 
    matrix_min = torch.transpose(torch.reshape(output_min.repeat(multiply), [ multiply[0], torch.numel(output_min)]))
    #output_min(H)이 가로로 256*256개 늘어서 있는 모양##output_min.repeat(multiply,1)일수도
    output_unit_flat = torch.div(torch.subtract(output_flat_masked,matrix_min),torch.subtract(matrix_max,matrix_min))    #(원본-min)/(max-min)
    output_unit_flat = torch.multiply(output_unit_flat,matrix_max_y)   #스스로를 max_y에 곱함. 
    
    # mask refine
    mask_one_refined = torch.where(z_refined, torch.ones_like(y), 0*torch.ones_like(y))  #다시, z_refined에 의해 true=1, false=0
    mask_one_flat_refined = torch.reshape(mask_one_refined,[-1, IMAGE_HEIGHT*IMAGE_WIDTH]) #어떤 높이 H'가지는 넓이 256*256 되도록 위 행렬 reshape
    
    # output refine
    output_unit_masked_flat_refined = torch.multiply(output_unit_flat, mask_one_flat_refined)  #두 개를 곱함.

    # y refine
    y_masked = torch.where(z_refined, y, 0*torch.ones_like(y))    #z_refined가 true=y, false=0
    y_masked_flat_refined = torch.reshape(y_masked,[-1, IMAGE_HEIGHT*IMAGE_WIDTH]) #위 행렬을 너비 256*256되도록 reshape
    
    
    numOfPix = torch.sum(mask_one_flat_refined,1)   #H' 높이 가지는 이 행렬에 대해 1차원에 대해 합해줌.
    
    d = torch.subtract(output_unit_masked_flat_refined, y_masked_flat_refined)
    a1 = torch.sum(torch.square(d),1)
    a2 = torch.square(torch.sum(d,1))

    cost = torch.mean(torch.div(a1, numOfPix) - (0.5 * torch.div(a2, torch.square(numOfPix))))
    return cost

#꼭 확인할 부분
#tile
#torch.shape()[0] 이상한거 나옴
