# 运行分析
ncu build/thread_block_divergence

# 1. 收集更详细的指标（包括 warp 发散详情）
ncu --set full build/thread_block_divergence

# 2. 将结果导出为报告文件（方便后续查看或传给本地电脑）
ncu --export divergence_report.ncu-rep build/thread_block_divergence

# 3. 只分析特定的 Kernel (如果你的程序有多个 Kernel)
ncu --kernel-name "your_kernel_name" build/thread_block_divergence

# 运行分析，会自动生成一个 .nsys-rep 文件
nsys profile build/thread_block_divergence

# 运行并直接在终端打印统计摘要
nsys profile --stats=true build/thread_block_divergence