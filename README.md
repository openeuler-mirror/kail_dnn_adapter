# kail_dnn_adapter

#### 介绍
Adapter for Kunpeng Deep Neural Network Library

#### 软件架构
kail_dnn_adapter包含了ACL和oneDNN两个软件以及kail_dnn的适配层，kail_dnn通过算子库插件形式集成进开源软件oneDNN。


#### 安装教程

1.  cd kail_dnn_adapter
2.  sh build.sh

#### 使用说明

1.  编译后，单独链接libdnnl.so可以得到oneDNN v3.4.0的全部接口功能，可替换原先的libdnnl.so使用。
libdnnl.so路径为：“out/oneDNN-3.4/build/src/”
其依赖库的路径如下：
ACL库相关so路径：“out/ComputeLibrary-23.11/build/”
数学库相关so路径：
“/usr/local/kml/lib/kblas/omp/libkblas.so”
“/usr/local/kml/lib/libkfft_omp.so”
“/usr/local/kml/lib/libkfftf_omp.so”
“/usr/local/kml/lib/libkffth_omp.so”
“/usr/local/kml/lib/kvml/multi/libkvml.so”
“/usr/local/kml/lib/libkm.so”


#### 参与贡献

1.  Fork 本仓库
2.  新建 Feat_xxx 分支
3.  提交代码
4.  新建 Pull Request


#### 特技

1.  使用 Readme\_XXX.md 来支持不同的语言，例如 Readme\_en.md, Readme\_zh.md
2.  Gitee 官方博客 [blog.gitee.com](https://blog.gitee.com)
3.  你可以 [https://gitee.com/explore](https://gitee.com/explore) 这个地址来了解 Gitee 上的优秀开源项目
4.  [GVP](https://gitee.com/gvp) 全称是 Gitee 最有价值开源项目，是综合评定出的优秀开源项目
5.  Gitee 官方提供的使用手册 [https://gitee.com/help](https://gitee.com/help)
6.  Gitee 封面人物是一档用来展示 Gitee 会员风采的栏目 [https://gitee.com/gitee-stars/](https://gitee.com/gitee-stars/)
