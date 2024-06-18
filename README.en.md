# kail_dnn_adapter

#### Description
Adapter for Kunpeng Deep Neural Network Library

#### Software Architecture
The kail_dnn_adapter contains the ACL and oneDNN software and the kail_dnn adaptation layer. The kail_dnn is integrated into the open-source software oneDNN as an operator library plug-in.

#### Installation

1. Install dependencies: Math Library, AI Library  (You can obtain the corresponding installation package and installation guide from the [Kunpeng community](https://www.hikunpeng.com/zh/developer/boostkit/library/math).)
2. cd kail_dnn_adapter
3. sh build.sh

#### Instructions

After compilation, libdnnl.so is linked separately to obtain all interface functions of oneDNN v3.4.0 and can replace the original libdnnl.so.

The path of libdnnl.so is as follows: “out/oneDNN-3.4/build/src/”

The path of the dependent library is as follows:

1. The so path related to the ACL library is:

* “out/ComputeLibrary-23.11/build/”.

2. The so paths related to the ai library are as follows:

* “/usr/local/kail/lib/libkdnn.so”

3. The so paths related to the math library are as follows:

* “/usr/local/kml/lib/kblas/omp/libkblas.so”

* “/usr/local/kml/lib/libkfft_omp.so”

* “/usr/local/kml/lib/libkfftf_omp.so”

* “/usr/local/kml/lib/libkffth_omp.so”

* “/usr/local/kml/lib/kvml/multi/libkvml.so”

* “/usr/local/kml/lib/libkm.so”

#### Contribution

1.  Fork the repository
2.  Create Feat_xxx branch
3.  Commit your code
4.  Create Pull Request


#### Gitee Feature

1.  You can use Readme\_XXX.md to support different languages, such as Readme\_en.md, Readme\_zh.md
2.  Gitee blog [blog.gitee.com](https://blog.gitee.com)
3.  Explore open source project [https://gitee.com/explore](https://gitee.com/explore)
4.  The most valuable open source project [GVP](https://gitee.com/gvp)
5.  The manual of Gitee [https://gitee.com/help](https://gitee.com/help)
6.  The most popular members  [https://gitee.com/gitee-stars/](https://gitee.com/gitee-stars/)
