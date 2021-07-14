# UDP-TokenPose Debug笔记

![image-20210713102150934](C:\Users\86138\AppData\Roaming\Typora\typora-user-images\image-20210713102150934.png)  



## config

### config/default.py

+ 以`TokenPose/lib/config/default.py`为主，添加UDP相关的参数

```python
_C.CONTINUE_FROM_BEST = False
_C.MODEL_BEST = ''

_C.LOSS.REDUCTION = 'mean'

# for UDP
_C.LOSS.KPD = 4.0

_C.TEST.SHIFT_HEATMAP = False
```



## core

### core/function.py

+ 主要的作用是训练和评估；以我修改的UDP-TransPose为主；

### core/inference.py

+ 主要的作用是heatmap->keypoint；以我修改的UDP-TransPose为主，加入TokenPose中的处理方式备用；

## dataset

### dataset/coco.py

+ coco数据集处理，以我修改的UDP-TransPose为主；

### dataset/JointsDataset.py

+ 训练过程中heatmap生成处理，以我修改的UDP-TransPose为主；

## models

+ 注意根据UDP进行修改，特别是**神经网络输出heatmap的地方**，UDP是[B, 17*3, H, W]；是不是可以考虑修改`lib/models/tokenpose_base.py`中的`heatmap_dim`；

```python
self.mlp_head = nn.Sequential(
    nn.LayerNorm(dim),
    nn.Linear(dim, hidden_heatmap_dim),
    nn.LayerNorm(hidden_heatmap_dim),
    nn.Linear(hidden_heatmap_dim, heatmap_dim)
) if (dim <= hidden_heatmap_dim*0.5 and apply_multi) else  nn.Sequential(
    nn.LayerNorm(dim),
    nn.Linear(dim, heatmap_dim)
)
```



```python
if cfg.MODEL.TARGET_TYPE=='offset':
	self.factor=3
else:
	self.factor=1
heatmap_dim = heatmap_dim*self.factor
```



```python
##[1, 17, 3072*3] -> [1, 17*3, 64, 48]
x = rearrange(x,'b c (p1 p2 factor) -> b (c factor) p1 p2',p1=self.heatmap_size[0],p2=self.heatmap_size[1])
```



但是要注意offset heatmap编解码的处理方式会产生什么样的影响？

```python
target[joint_id, 0, keep_pos] = 1
target[joint_id, 1, keep_pos] = x_offset[keep_pos]
target[joint_id, 2, keep_pos] = y_offset[keep_pos]
```



```python
batch_heatmaps = net_output[:,::3,:]
offset_x = net_output[:,1::3,:] * kps_pos_distance_x
offset_y = net_output[:,2::3,:] * kps_pos_distance_y
```



是没有问题的，上面的`rearrange`的方式是将原来的`[B, 17, heatmap_size * 3]`的tensor按照各个关键点*3的方式转换为`[B, 17 * 3, heatmap_size]`。



## nms

## utils

+ TokenPose与DarkPose完全一致，因此直接采用UDP中的处理方式



## tools

+ 以TokenPose为主，根据UDP-Pose修改offset的loss；