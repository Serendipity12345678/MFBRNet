
import torch, time
import numpy as np
from thop import profile, clever_format
from config import Config

from Model.MFBRNet import Net
cfg = Config()
model = Net().to(cfg.device)
#model = FINet(backbone='SMT', channels=(8, 24, 32, 64))

# 定义numpy输入矩阵
bs =1
test_img = np.random.random((bs, 3, 384, 384)).astype('float32')
test_depth = np.random.random((bs, 3, 384, 384)).astype('float32')

pytorch_test_img = torch.Tensor(test_img).cuda()
pytorch_test_depth = torch.Tensor(test_depth).cuda()


turns = 100

pytorch_model = model.cuda()

pytorch_model.eval()

for i in range(10):
	pytorch_result = pytorch_model(pytorch_test_img,pytorch_test_depth)  # Pytorch热身
torch.cuda.synchronize()
sta = time.time()
for i in range(turns):
	pytorch_result = pytorch_model(pytorch_test_img,pytorch_test_depth)
torch.cuda.synchronize()
end = time.time()
tc_time = round((end - sta) / turns, 5)
tc_fps = round(bs * turns / (end - sta), 0)
print(f"- Pytorch forward average time cost: {tc_time}, Batch Size: {bs}, FPS: {tc_fps}")


# 计算 FLOPs 和参数量
flops, params = profile(pytorch_model,
                      inputs=(pytorch_test_img, pytorch_test_depth),
                      verbose=False)

# 格式化输出
flops, params = clever_format([flops, params], "%.3f")
print(f"Model FLOPs: {flops}, Params: {params}")


