"""
EMA (Exponential Moving Average) Teacher 工具
SoftTeacher 中: Student 训练, Teacher = Student 的 EMA 影子模型
"""
import math
import torch
import torch.nn as nn


class EMATeacher:
    """
    维护 teacher 模型, 每次 student 更新后:
        teacher = alpha * teacher + (1 - alpha) * student
    alpha 从 0.999 cosine warmup 到 0.9999 (早期用小 alpha, teacher 跟 student 快)
    """
    def __init__(self, student: nn.Module, alpha_init=0.999, alpha_final=0.9999,
                 warmup_steps=2000, device='cuda'):
        self.student = student
        # 深拷贝 student 作为 teacher 初始权重
        self.teacher = self._copy_module(student).to(device)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)

        self.alpha_init = alpha_init
        self.alpha_final = alpha_final
        self.warmup_steps = warmup_steps
        self.device = device

    @staticmethod
    def _copy_module(m: nn.Module) -> nn.Module:
        import copy
        return copy.deepcopy(m)

    @torch.no_grad()
    def update(self, global_step: int):
        """按当前 step 计算 alpha, 平滑更新 teacher 权重"""
        alpha = self._cosine_alpha(global_step)
        for s_p, t_p in zip(self.student.parameters(), self.teacher.parameters()):
            t_p.data.mul_(alpha).add_(s_p.data, alpha=1 - alpha)
        # 同步 BN running stats
        for s_b, t_b in zip(self.student.buffers(), self.teacher.buffers()):
            t_b.data.copy_(s_b.data)

    def _cosine_alpha(self, step):
        if step < self.warmup_steps and self.warmup_steps > 0:
            # cosine warmup: alpha 从 alpha_init 平滑升到 alpha_final
            frac = step / self.warmup_steps
            return self.alpha_init + (self.alpha_final - self.alpha_init) * \
                (1.0 - math.cos(math.pi * frac)) / 2.0
        return self.alpha_final

    @torch.no_grad()
    def predict(self, imgs):
        return self.teacher(imgs)


if __name__ == "__main__":
    from .md_yolo import MDYOLO
    stu = MDYOLO(num_classes=10)
    ema = EMATeacher(stu, device='cpu')
    x = torch.randn(1, 3, 640, 640)
    out_s = stu(x)
    out_t = ema.predict(x)
    print(f"Student head0 cls: {out_s[0][0].shape}, Teacher head0 cls: {out_t[0][0].shape}")
    ema.update(step=0)
    ema.update(step=1000)
    print("EMATeacher OK")
