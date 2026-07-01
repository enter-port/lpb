# LPB Pipeline 数据流总结图

```
zarr 磁盘数据
  keypoint (T, 9, 2)  +  state (T, 2)  +  action (T, 2)
         │                    │                  │
         ▼                    ▼                  ▼
    Sampler 滑动窗口采样 (horizon=16, 含padding)
         │                    │                  │
    (16, 9, 2)            (16, 2)           (16, 2)
         │                    │                  │
         ▼ reshape            ▼                  │
    (16, 18)  ──concat──  (16, 2)                │
                │                                │
            obs (16, 20)                    action (16, 2)
                │                                │
           to Tensor + DataLoader batch
                │                                │
          (B, 16, 20)                      (B, 16, 2)
                │                                │
           归一化 (形状不变)
                │                                │
          取前2步展平为全局条件              作为扩散目标 trajectory
                │                                │
          global_cond (B, 40)             trajectory (B, 16, 2)
                │                                │
                │          加噪                   │
                │     noisy_trajectory (B, 16, 2) │
                │                │                │
                └───→ ConditionalUnet1D ←─────────┘
                         (B, 2, 16) 内部
                              │
                     pred_noise (B, 16, 2)
                              │
                      MSE loss → scalar
```
