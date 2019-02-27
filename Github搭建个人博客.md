Github搭建个人博客

<!-- MarkdownTOC levels="1,2,3,4" autolink="true" style="ordered" -->

1. [新建Github仓库](#%E6%96%B0%E5%BB%BAgithub%E4%BB%93%E5%BA%93)
1. [Hexo博客框架安装](#hexo%E5%8D%9A%E5%AE%A2%E6%A1%86%E6%9E%B6%E5%AE%89%E8%A3%85)
1. [安装NexT等个性化主题](#%E5%AE%89%E8%A3%85next%E7%AD%89%E4%B8%AA%E6%80%A7%E5%8C%96%E4%B8%BB%E9%A2%98)
1. [添加NexT主题插件](#%E6%B7%BB%E5%8A%A0next%E4%B8%BB%E9%A2%98%E6%8F%92%E4%BB%B6)
1. [多设备支持方案](#%E5%A4%9A%E8%AE%BE%E5%A4%87%E6%94%AF%E6%8C%81%E6%96%B9%E6%A1%88)
1. [其他](#%E5%85%B6%E4%BB%96)

<!-- /MarkdownTOC -->




前几天终于搭建好了自己的博客站点，中间折腾了很多地方，权当学习，整理一下搭建过程的资料，写一个搭建的教程。

教程可能不同于网上单一的搭建教程，在网上搜集到的搭建教程要么是比较老的教程（=_=，弄得我走了很多弯路！），要么是一些比较零散的教程，或缺少一些部分。总而言之，缺少一种较为完整的搭建教程，从入手搭建到博客在不同设备之间正常迁移，并正常配置是教程的所有。本教程分为：
+ 1）新建Github仓库
+ 2）Hexo博客框架安装
+ 3）安装NexT等个性化主题
+ 4）添加NexT主题插件
+ 5）多设备支持方案
+ 6）其他

搭建流程为：
<!-- 甘地图 -->



----------


## 新建Github仓库

请确认在该步骤前已经注册了一个[Github](https://github.com/)账号。

+ 进入https://github.com/{your github username}/页面，确保已经登陆账号；
+ 点击右上角的**+**，会跳转到[新建界面](https://github.com/new)，然后必须以**username.github.io**名称新建一个人仓库，其他设置如下图所示：
![](https://github.com/leaguecn/leenotes/raw/master/img/github-build-blog-create-repo.png)

+ 拷贝新建仓库的git地址，如：https://github.com/{username}/{username}.github.io.git
到文本备存。

## Hexo博客框架安装
**依赖环境：Node.js+npm**
如果没有安装依赖环境，可以使用教程：

```
#安装Node.js
wget -qO- https://raw.github.com/creationix/nvm/v0.33.11/install.sh | sh
#使用命令行vim ~/.profile添加环境变量 & source ~/.profile使得nvm生效
#然后利用nvm安装特定版本的node.js
nvm install 8.0.0
#安装npm
apt-get install npm

```

+ 请确保已经正常安装Node.js+npm，然后使用命令行安装hexo框架：
```
npm install -g hexo-cli
```

+ 初始化Hexo项目和安装项目模块： 
```
#hexo初始化
hexo init <folder>
#安装项目模块
cd <folder> && npm install

```
+ 项目文件分析

```
.
├── _config.yml#网站配置文件
├── package.json#项目的应用信息
├── scaffolds#模版文件夹。当您新建文章时，Hexo 会根据 scaffold 来建立文件。
├── source#资源文件夹，新建博客的存储文件夹
|   ├── _drafts
|   └── _posts
└── themes#主题文件夹，可以在文件夹下安装不同的主题，并修改相关主题风格


```



## 安装NexT等个性化主题
其实就是将主题文件夹next拷贝到上述项目的主题文件夹theme中。[HexT主题](https://theme-next.org/)风格简洁、优雅，目前已经更新到6.0.6版本（2019-02）。


+ 进入上述Hexo新建项目文件夹并下载主题文件到主题theme文件夹下：
```
$ cd hexo
$ git clone https://github.com/theme-next/hexo-theme-next themes/next
```



## 添加NexT主题插件



## 多设备支持方案





## 其他



**参考**

+ 1 https://hexo.io/zh-cn/docs/
+ 2 https://theme-next.org/
+ 3
