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

```mermaid
graph LR

step1(新建仓库)-->step2(安装博客框架)
step2-->step3(安装主题)
step3-->step4(添加主题插件)
step4-->step5(多设备支持方案)
```

----------


## 新建Github仓库

请确认在该步骤前已经注册了一个[Github](https://github.com/)账号。

+ 进入https://github.com/{your github username}/页面，确保已经登陆账号；
+ 点击右上角的+，会跳转到[新建界面](https://github.com/new)，然后必须以**username.github.io*名称新建一个人仓库，其他设置如下图所示：
![](https://github.com/leaguecn/leenotes/raw/master/img/github-build-blog-create-repo.png)

+ 拷贝新建仓库的git地址，如：https://github.com/{username}/{username}.github.io.git
到文本备存。

## Hexo博客框架安装

**依赖环境：Node.js+npm**     

+ 依赖环境安装教程
```
#安装Node.js
wget -qO- https://raw.github.com/creationix/nvm/v0.33.11/install.sh | sh
#使用命令行vim ~/.profile添加环境变量 & source ~/.profile使得nvm生效
#然后利用nvm安装特定版本的node.js
nvm install 8.0.0
#安装npm
apt-get install npm

```

+ 安装hexo框架
```
#请确保已经正常安装Node.js+npm，然后使用命令行安装
npm install -g hexo-cli
```

+ 初始化Hexo项目和安装项目模块 
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
可以根据需要和要求对上述文件进行修改。

+ 生成静态网页
```
hexo g/genarate

```
需要一秒左右的时间对整个项目工程进行编译，从而生成静态的网页。

+ 调试网页

```
hexo server

```

然后在浏览器中访问http://localhost:40000，将会看到类似如下的页面：     
![](https://github.com/leaguecn/leenotes/raw/master/img/hexo-server.png)      
如果没有看到，那就是环境出了问题，根据不同的问题自行debug。

+ 发布到Github仓库
如果调试没有出现问题，那就可以发布到Github上的仓库
```
#首先安装hexo配置到git上的插件
npm install hexo-deployer-git --save
#之后修改_config.yml中的deploy字段属性
deploy:
     type: git
     repo: https://github.com/{username}/{username}.github.io.git
     branch: master

#配置到Github
hexo d/deploy
```
通过https://{username}.github.io访问验证配置是否正常，看到的界面应和调试所看到的界面一致。

+ 写博客

```
#新建博客postname
hexo new postname

#编辑博客
vim source/_posts/postname.md

#生成静态网页并配置到仓库

hexo clean && hexo g/generate && hexo d/deploy

```

+ Hexo常用命令
    + hexo init <folder> #新建hexo项目工程
    + hexo new postname #新建博客
    + hexo generate/g #生成静态网页
    + hexo deploy/d #配置静态网页到指定的仓库
    + hexo server #建立本地服务
    + hexo help #显示hexo帮助


## 安装NexT等个性化主题 

其实就是将主题文件夹next拷贝到上述项目的主题文件夹theme中。[HexT主题](https://theme-next.org/)风格简洁、优雅，目前已经更新到6.0.6版本（2019-02）。


+ 进入上述Hexo新建项目文件夹并下载主题文件到主题theme文件夹下：

```
cd hexo
git clone https://github.com/theme-next/hexo-theme-next themes/next
```

+ 修改配置文件的主题属性
```
#打开_config.yml配置文件，查找theme字段，注释掉原有属性，修改为next
theme: next
```

+ 应用新的主题配置

```
#参照上一节，生成静态网页然后配置到仓库
hexo clean && hexo g/generate && hexo d/deploy
```

效果如下：       
![](https://github.com/leaguecn/leenotes/raw/master/img/hexo-server.png)

## 添加NexT主题插件

+ 修改主题风格

```
#修改next文件夹下的_config.yml文件中的scheme字段属性

# Schemes
scheme: Muse
#scheme: Mist
#scheme: Pisces
```


+ 设定菜单

```
#修改next文件夹下的_config.yml文件中的menu字段属性
menu:
  home: / #主界面
  categories: /categories #分类
  about: /about #关于
  archives: /archives #存档
  tags: /tags #标签
  #commonweal: /404.html

#同时修改主目录下的_config.yml文件的language字段属性为zh-CN.yml
language: zh-CN

```

+ 修改布局






+ 添加部件







+ 添加页面

```
#添加标签页面
hexo new page tags
#编辑source/tags/index.md
---
title: tags
date: 2019-02-21 19:10:05
type: "tags"
---

#添加分类页面
hexo new page categories
#编辑source/categories/index.md
---
title: categories
date: 2019-02-21 19:11:13
type: "categories"
---

#添加关于页面
hexo new page about
#source/about/index.md
---
title: about
date: 2019-02-21 19:13:13
---
## 关于我

一只学习中的小菜鸟，欢迎留言讨论。

From Xuzhou

QQ：10001
Email: 10001@qq.com

```

+ 编辑博客的标签和分类

```
---
title: Github搭建个人博客
date: 2019-02-27
tags: 
  - npm
  - hexo
  - github
  - 博客
categories: 教程
---
...

```


+ 使更改生效/发布更改配置

```
hexo clean && hexo g/generate && hexo d/deploy
```

## 多设备支持方案

网络上的方案较为混乱，模糊不清。看了几个较为清晰的博客后大致理解了方案的原理。
多设备支持是为了方便在多种设备中更新博客，方案的思路如下图所示：



+ 新建分支对hexo的工程文件进行备份在原来的部署仓库中新建另一分支backup/hexo等分支，将分支设为默认分支


+ 在原部署设备上clone部署仓库的默认分支backup/hexo

```
git clone https://github.com/{username}/{username}.github.io.git
```

+ 合并部署文件

```
#将原部署文件全部拷贝到{username}.github.io文件夹中
cp hexoprjs {username}.github.io/

```

+ 推送所有的部署文件

```
#移除theme文件夹下的.git文件夹
rm -rf {username}.github.io/theme/.git

#git命令行推送部署文件到backup/hexo分支
git add . && git commit -m '- update' && git push

```

+ 其他设备同步

```
#直接拷贝部署仓库的默认分支backup/hexo
git clone https://github.com/{username}/{username}.github.io.git

#安装npm插件
npm install

#新建博客
hexo new postname

#发布博客
hexo clean && hexo g/generate && hexo d/deploy

#同步配置文件到部署仓库
git add . && git commit -m '- update' && git push

```
总体思路是：拷贝默认分支中的hexo部署文件，安装npm插件，新建博客，每次发布博客后最好随手同步配置文件到分支backup/hexo。



## 其他



**参考**

+ 1 https://hexo.io/zh-cn/docs/
+ 2 https://theme-next.org/
+ 3 http://www.cnblogs.com/syd192/p/6074323.html
