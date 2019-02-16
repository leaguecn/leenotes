
## Android Dev Notes


### design lib error in using android.support.design.widget.FloatingActionButton class
使用design库时没有使用配套的android support V7 appcompat可能会出现以下的非法类访问异常:     
```
Illegal class access: 'android.support.design.widget.FloatingActionButton' attempting to access 'android.support.v7.widget.AppCompatImageHelper'
```    

**解决办法：使用同一support lib下的design和v7 appcompat两个库**

**Reference**       

+ https://stackoverflow.com/questions/35675855/android-studio-floatingactionbutton-error

*2019-02-16*


### More



