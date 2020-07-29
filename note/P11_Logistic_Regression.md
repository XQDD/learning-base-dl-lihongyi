# Logistic Regression

## 分类概率函数(或者说model，由P10分类得到)

<img src="/note/tex/eef72266574dd9dba30cc016b0b3fb36.svg?invert_in_darkmode&sanitize=true" align=middle width=103.60811009999999pt height=24.65753399999998pt/>
<img src="/note/tex/63390cc113f2c5e4bcb2f47142cc4607.svg?invert_in_darkmode&sanitize=true" align=middle width=186.14417909999997pt height=27.77565449999998pt/>
<img src="/note/tex/700376d61142d1851af980e46d1acdbe.svg?invert_in_darkmode&sanitize=true" align=middle width=115.10443394999999pt height=24.657735299999988pt/>

## 损失函数

1. 训练数据

| <img src="/note/tex/2b0a082e488e8534f9a4facc3b1f04e3.svg?invert_in_darkmode&sanitize=true" align=middle width=15.94753544999999pt height=26.76175259999998pt/> | <img src="/note/tex/6177db6fc70d94fdb9dbe1907695fce6.svg?invert_in_darkmode&sanitize=true" align=middle width=15.94753544999999pt height=26.76175259999998pt/> | <img src="/note/tex/3c63d4517a41fc372162eaa29bc7d970.svg?invert_in_darkmode&sanitize=true" align=middle width=15.94753544999999pt height=26.76175259999998pt/> | ... | <img src="/note/tex/561cb7f0320be6f51d73630253421c13.svg?invert_in_darkmode&sanitize=true" align=middle width=21.04114979999999pt height=27.6567522pt/> |
| ----- | ----- | ----- | --- | ----- |
| <img src="/note/tex/d81a84099e7856ffa4484e1572ceadff.svg?invert_in_darkmode&sanitize=true" align=middle width=18.30139574999999pt height=22.465723500000017pt/> | <img src="/note/tex/d81a84099e7856ffa4484e1572ceadff.svg?invert_in_darkmode&sanitize=true" align=middle width=18.30139574999999pt height=22.465723500000017pt/> | <img src="/note/tex/85f3e1190907b9a8e94ce25bec4ec435.svg?invert_in_darkmode&sanitize=true" align=middle width=18.30139574999999pt height=22.465723500000017pt/> | ... | <img src="/note/tex/d81a84099e7856ffa4484e1572ceadff.svg?invert_in_darkmode&sanitize=true" align=middle width=18.30139574999999pt height=22.465723500000017pt/> |

2. 设<img src="/note/tex/126440ed2fbbb61f3380d30c9cb0dae2.svg?invert_in_darkmode&sanitize=true" align=middle width=149.2219146pt height=24.65753399999998pt/>
3. 则得到损失函数<img src="/note/tex/2c658cc35dfbe3b05dd654bc7bc98881.svg?invert_in_darkmode&sanitize=true" align=middle width=364.06595115pt height=27.6567522pt/>
4. 最适合的<img src="/note/tex/31dd81634e85e6307a1ad98007005174.svg?invert_in_darkmode&sanitize=true" align=middle width=26.571525749999992pt height=22.831056599999986pt/>是<img src="/note/tex/a15d7e9ef3d64d058a5f03b4ce852d68.svg?invert_in_darkmode&sanitize=true" align=middle width=373.51374884999996pt height=24.65753399999998pt/>
5. 设<img src="/note/tex/cb25640543808c2825cd39bcbede9b71.svg?invert_in_darkmode&sanitize=true" align=middle width=16.77522989999999pt height=22.831056599999986pt/>{1,0}，<img src="/note/tex/4ad68122df7d97f68a20028b88d43947.svg?invert_in_darkmode&sanitize=true" align=middle width=98.52029879999999pt height=22.465723500000017pt/>（即当前为二分类问题）：

| <img src="/note/tex/2b0a082e488e8534f9a4facc3b1f04e3.svg?invert_in_darkmode&sanitize=true" align=middle width=15.94753544999999pt height=26.76175259999998pt/>         | <img src="/note/tex/6177db6fc70d94fdb9dbe1907695fce6.svg?invert_in_darkmode&sanitize=true" align=middle width=15.94753544999999pt height=26.76175259999998pt/>         | <img src="/note/tex/3c63d4517a41fc372162eaa29bc7d970.svg?invert_in_darkmode&sanitize=true" align=middle width=15.94753544999999pt height=26.76175259999998pt/>         | ... | <img src="/note/tex/561cb7f0320be6f51d73630253421c13.svg?invert_in_darkmode&sanitize=true" align=middle width=21.04114979999999pt height=27.6567522pt/>             |
| ------------- | ------------- | ------------- | --- | ----------------- |
| <img src="/note/tex/1587183060a699708af256d523c1bbc6.svg?invert_in_darkmode&sanitize=true" align=middle width=46.16050064999999pt height=26.76175259999998pt/> | <img src="/note/tex/f3ff1fc4d381d267a734c2aa592e59c5.svg?invert_in_darkmode&sanitize=true" align=middle width=46.16050064999999pt height=26.76175259999998pt/> | <img src="/note/tex/cbd21b4a239fc016f8f03a2af19caeef.svg?invert_in_darkmode&sanitize=true" align=middle width=46.16050064999999pt height=26.76175259999998pt/> | ... | <img src="/note/tex/212d5988c8c7eea0aaef8f624a2f713c.svg?invert_in_darkmode&sanitize=true" align=middle width=38.46878474999998pt height=27.6567522pt/>{1,0} |

6. 最终的损失函数为：<img src="/note/tex/b36ac5e84cf40ed445b4763d911de272.svg?invert_in_darkmode&sanitize=true" align=middle width=700.2742642499999pt height=48.85840079999999pt/>

## 梯度下降更新函数

将最终的损失函数通过一系列微分计算得到
<img src="/note/tex/19cecbc4f19cfa48a5c630987e05eebd.svg?invert_in_darkmode&sanitize=true" align=middle width=251.73440819999996pt height=24.657735299999988pt/>
