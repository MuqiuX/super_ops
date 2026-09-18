# 001-线程束分化测试（course001.cu）
sum_kernel 分化严重，线程对应内存地址，导致一个warp中有很多线程没有执行  
sum_kernel_better 经过优化，让连续的线程去找内存地址，产生更多满的warp，更多全空的warp   
sum_kernel_better_plus 在防止线程分化的基础上，同时还保持了每轮的内存访问是连续的，但是相邻归约只有第一轮的内存访问是连续的，所以交错归约比相邻归约更快
```bash
nsys profile --stats=true build/course001
```

![图片](../images/course001-1.png)