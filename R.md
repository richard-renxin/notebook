# 文件导入、分隔符和查看列名

```
pf <- read.csv('pseudo_facebook.tsv', sep = '\t')
names(pf)
```
# 琢面包裹和琢面网格（3列）
```
qplot(x = dob_day, data = pf) +
  scale_x_continuous(breaks=1:31) +
  facet_wrap(~dob_month, ncol = 3)
```
# 限制轴（使用下面的方法）
```
qplot(x = friend_count, data = pf, xlim = c(0, 1000))
```
# 限制轴(此方法利于排除空值)
```
qplot(x = friend_count, data = pf) +
  scale_x_continuous(limits = c(0, 1000))
```
# 调整组距 binwidth
```
qplot(x = friend_count, data = pf, binwidth = 25) +
  scale_x_continuous(limits = c(0, 1000))
```
# 调整间断 breaks
```
qplot(x = friend_count, data = pf, binwidth = 25) +
  scale_x_continuous(limits = c(0, 1000), breaks = seq(0, 1000, 50))
```
# 性别分类，利用琢面包裹
```
qplot(x = friend_count, data = pf) +
  facet_wrap(~gender)
```
# 性别分类，利用琢面网格,两种图不一样
```
qplot(x = friend_count, data = pf) +
  facet_grid(gender ~ .)
```
# 去除性别中的空值,用subset和！is.na
```
qplot(x = friend_count, data = subset(pf, !is.na(gender)),
      binwidth = 10) +
  scale_x_continuous(lim = c(0, 1000), breaks = seq(0, 1000, 50)) +
  facet_wrap(~gender)
```
# 去除数据集所有空值，用na.omit
```
qplot(x = friend_count, data = na.omit(pf), binwidth = 10) +
  scale_x_continuous(lim = c(0, 1000), breaks = seq(0, 1000, 50)) +
  facet_wrap(~gender)
```
# 性别计数（表命令）
```
table(pf$gender)
```
# 分组统计
```
by(pf$friend_count, pf$gender, summary)
```
# 彩色（天蓝）填充,color轮廓线，fill是区域颜色
```
qplot(x = tenure, data = pf,
      color = I('black'), fill = I('#099DD9'))
```
# 彩色（橙黄）填充，并考虑组距
```
qplot(x = tenure/365, data = pf,binwidth = 0.25,
      color = I('black'), fill = I('#F79420'))
```
# 添加x轴图层，限制x轴长度并改变间隔
```
qplot(x = tenure/365, data = pf,binwidth = 0.25,
      color = I('black'), fill = I('#F79420'))+
  scale_x_continuous(breaks = seq(1, 7, 1), limits = c(0, 7))
```
# 添加坐标轴标签(别忘了逗号)
```
qplot(x = tenure/365, data = pf,binwidth = 0.25,
      xlab = 'Number of years using Facebook',
      ylab = 'Number of users in sample',
      color = I('black'), fill = I('#F79420'))+
  scale_x_continuous(breaks = seq(1, 7, 1), limits = c(0, 7))
```
# 对长尾数据进行对数变换(+1是为了防止对0取对数)log10是10的量级，比较好
```
qplot(x = friend_count, data = pf)
summary(log10(pf$friend_count + 1))
```
# 对长尾数据取平方差
```
summary(sqrt(pf$friend_count))
```
# 三联图（安装gridExtra包）
```
p1 <- qplot(x = friend_count, data = pf)
p2 <- qplot(x = log10(friend_count +1), data = pf)
p3 <- qplot(x = sqrt(friend_count), data = pf)

grid.arrange(p1, p2, p3, ncol = 1)
```
# 创建以10为底的对数变换直方图的三种方法(同一幅图展现，第一幅图x轴标签不同)
```
logScale <- qplot(x = log10(friend_count), data = pf)

logScale2 <- qplot(x = friend_count, data = pf) +
  scale_x_log10()

countScale <- ggplot(aes(x = friend_count), data = pf) +
  geom_histogram() +
  scale_x_log10()

grid.arrange(logScale, logScale2, countScale, ncol = 3)
```
# 将两张按性别分类的直方图变成频率多边形（折线图？）
```
qplot(x = friend_count, data = subset(pf, !is.na(gender)),
      binwidth = 10) +
  scale_x_continuous(lim = c(0, 1000), breaks = seq(0, 1000, 50)) +
  facet_wrap(~gender)

qplot(x = friend_count, data = subset(pf, !is.na(gender)),
      binwidth = 10, geom = 'freqpoly', color = gender) +
  scale_x_continuous(lim = c(0, 1000), breaks = seq(0, 1000, 50))
```
# 将上图频率多边形y轴变成百分比，添加坐标轴标签
```
qplot(x = friend_count, y = ..count../sum(..count..),
      data = subset(pf, !is.na(gender)),
      xlab = 'Friend Count',
      ylab = 'Proportion of User with that friend count',
      binwidth = 10, geom = 'freqpoly', color = gender) +
  scale_x_continuous(lim = c(0, 1000), breaks = seq(0, 1000, 50))
```
# 使用频率多边形查看男女点赞数,因为长尾又做对数变换
```
qplot(x = www_likes, data = subset(pf, !is.na(gender)),
      geom = 'freqpoly', color = gender) +
  scale_x_continuous() +
  scale_x_log10()
```  
# 按性别分类统计点赞数，最后一行是计数
```
qplot(x = www_likes, data = subset(pf, !is.na(gender)),
      geom = 'freqpoly', color = gender) +
  scale_x_continuous() +
  scale_x_log10()
  table(pf$gender)
by(pf$www_likes, pf$gender, sum)
```
# 按性别分类，查看好友数量0-250箱线图,
#### 使用scale_y_continueous图层（删数据）
#### 或coord_cartesian图层（不删数据），
#### 最后一种不使用图层简单不灵活（删数据）
```
qplot(x = gender, y = friend_count,
      data = subset(pf, !is.na(gender)),
      geom = 'boxplot') +
  scale_y_continuous(lim = c(0, 1000))

qplot(x = gender, y = friend_count,
      data = subset(pf, !is.na(gender)),
      geom = 'boxplot') +
  coord_cartesian(ylim = c(0, 1000))

qplot(x = gender, y = friend_count,
      data = subset(pf, !is.na(gender)),
      geom = 'boxplot', ylim = c(0 , 1000))
```
# ifelse逻辑函数，计算用户是否登陆过手机
```
summary(pf$mobile_likes > 0)
pf$mobile_check_in <- NA
pf$mobile_check_in <- ifelse(pf$mobile_likes > 0, 1, 0)
pf$mobile_check_in <- factor(pf$mobile_check_in)
summary(pf$mobile_check_in)
```
# 计算手机登陆比例
#### mobile_check_in是因子变量，sum（）函数无法运行
#### 使用length（）函数来确定向量中的值数量
```
summary(pf$mobile_check_in)
sum(pf$mobile_check_in == 1)/length(pf$mobile_check_in)
```
# 查看四分位间距
```
IQR(subset(diamonds, price <1000)$price) 
```
# ggplot语法:散点图
```
ggplot(aes(x = age, y = friendships_initiated), data = pf) +
  geom_point()
```
# ggplot y轴取平方根
```
ggplot(aes(x = age, y = friendships_initiated), data = pf) +
  geom_point() + 
  coord_trans(y = 'sqrt')
```
# 抖动两种方法(离散变量变成连续变量)alpha和jitter
### 第二种方法
```
ggplot(aes(x = age, y = friendships_initiated), data = pf) +
  goem_jitter(alpha = a/10)

ggplot(aes(x = age, y = friendships_initiated), data = pf) +
  geom_point(alpha = 1/20, position = jitter)
```
# 抖动与y轴平方根配合,注意0的平方根是负值,结果为虚数.
```
ggplot(aes(x = age, y = friendships_initiated), data = pf) +
  geom_point(alpha = 1/20, position = position_jitter(h = 0)) + xlim(13,90) +
  coord_trans(y = 'sqrt')
```
# 条件均值 dployr包(均值\中位数\数量)
```
age_groups <- group_by(pf, age)
pf.fc_by_age <- summarise(age_groups,
          friend_count_mean = mean(friend_count),
          friend_count_median = median(friend_count),
          n = n())
pf.fc_by_age <- arrange(pf.fc_by_age)
head(pf.fc_by_age)
```
# ggplot加颜色(橙色)
```
ggplot(aes(x = age, y = friend_count), data = pf) +
  xlim(13, 90) +
  geom_point(alpha = 0.05, position = position_jitter(h = 0),
             color = 'orange') +
  coord_trans(y = 'sqrt')
```
# ggplot叠加一条平均线
```
ggplot(aes(x = age, y = friend_count), data = pf) +
  xlim(13, 90) +
  geom_point(alpha = 0.05, position = position_jitter(h = 0),
             color = 'orange') +
  coord_trans(y = 'sqrt') +
  geom_line(stat = 'summary', fun.y = mean)
```
# ggplot再叠加一条10%分位数线,虚线,蓝色
```
ggplot(aes(x = age, y = friend_count), data = pf) +
  xlim(13, 90) +
  geom_point(alpha = 0.05, position = position_jitter(h = 0),
             color = 'orange') +
  coord_trans(y = 'sqrt') +
  geom_line(stat = 'summary', fun.y = mean) +
  geom_line(stat = 'summary', fun.y = quantile, fun.args = list(probs = .1),
            linetype = 2, color = 'blue')
```
# 叠加四条线,10%\平均数\50%\90%
```
ggplot(aes(x = age, y = friend_count), data = pf) +
  xlim(13, 90) +
  geom_point(alpha = 0.05, position = position_jitter(h = 0),
             color = 'orange') +
  coord_trans(y = 'sqrt') +
  geom_line(stat = 'summary', fun.y = mean) +
  geom_line(stat = 'summary', fun.y = quantile, fun.args = list(probs = .1),
            linetype = 2, color = 'blue') +
  geom_line(stat = 'summary', fun.y = quantile, fun.args = list(probs = .5),
            color = 'blue') +
  geom_line(stat = 'summary', fun.y = quantile, fun.args = list(probs = .9),
            linetype = 2, color = 'blue')
```
# 限制x\y轴
```
ggplot(aes(x = age, y = friend_count), data = pf) +
  coord_cartesian(xlim = c(13, 70), ylim = c(0, 1000)) +
  geom_point(alpha = 0.05, position = position_jitter(h = 0),
             color = 'orange') +
  geom_line(stat = 'summary', fun.y = mean) +
  geom_line(stat = 'summary', fun.y = quantile, fun.args = list(probs = .1),
            linetype = 2, color = 'blue') +
  geom_line(stat = 'summary', fun.y = quantile, fun.args = list(probs = .5),
            color = 'blue') +
  geom_line(stat = 'summary', fun.y = quantile, fun.args = list(probs = .9),
            linetype = 2, color = 'blue')
```
# 相关系数两种方法,使用cor.test()函数,默认皮尔森,不加method也行
```
cor.test(pf$age, pf$friend_count, method = 'pearson')

with(pf, cor.test(age, friend_count, method = 'pearson'))
```
# 等级相关度量,单调
```
method = 'spearman'
```
#  限制x/y轴,用0-95%百分比
```
ggplot(aes(x = www_likes_received, y = likes_received), data = pf) +
  geom_point() +
  xlim(0, quantile(pf$www_likes_received, 0.95)) +
  ylim(0, quantile(pf$likes_received, 0.95))
```
# 最佳拟合直线斜率,就是相关性(线性平滑)
### 相关系数在x或y的线性转换下是不变的,并且当x和y都被转换为z分数时,回归线的斜率就是相关系数
```
ggplot(aes(x = www_likes_received, y = likes_received), data = pf) +
  geom_point() +
  xlim(0, quantile(pf$www_likes_received, 0.95)) +
  ylim(0, quantile(pf$likes_received, 0.95)) +
  geom_smooth(method = 'lm', color = 'red')
```
# 将连续的月份数据打散变成每12月循环
```
range(Mitchell$Month)
ggplot(aes(x = Month, y = Temp), data = Mitchell) +
  geom_point() +
scale_x_continuous(breaks = seq(0, 203, 12))
```
# 将203个连续月,按年份叠加,同比
```
ggplot(aes(x=(Month%%12),y=Temp),data=Mitchell)+ 
  geom_point()
```
# 将年龄变为带月份的年龄
### 比如36岁零3个月=36.25岁,注意3月出生和9月出生的差别
```
pf$age_with_months <- pf$age + (12 - pf$dob_month) / 12
```
# 带有月均值的年龄
```
pf.fc_by_age_months <- pf %>%
  group_by(age_with_months) %>%
  summarise(friend_count_mean = mean(friend_count),
            friend_count_median = median(friend_count),
            n = n()) %>%
  arrange(age_with_months)
head(pf.fc_by_age_months)


age_with_months_groups <- group_by(pf, age_with_months)
pf.fc_by_age_months2 <- summarise(age_with_months_groups, 
              friend_count_mean = mean(friend_count), 
              friend_count_median = median(friend_count), 
              n = n()) 
pf.fc_by_age_months2 <- arrange(pf.fc_by_age_months2, age_with_months)
head(pf.fc_by_age_months2)
```
# 条件均值中的噪声
```
ggplot(aes(x = age_with_months, y = friend_count_mean),
       data = subset(pf.fc_by_age_months, age_with_months < 71)) +
  geom_line()
```
# 平滑化条件均值,原图
### 需要gridExtra包支持
```
p1 <- ggplot(aes(x = age, y = friend_count_mean),
             data = subset(pf.fc_by_age, age < 71)) +
  geom_line()

p2 <- ggplot(aes(x = age_with_months, y = friend_count_mean),
             data = subset(pf.fc_by_age_months, age_with_months < 71)) +
  geom_line()

grid.arrange(p2, p1, ncol = 1)
```
### 平滑后图:
```p1 <- ggplot(aes(x = age, y = friend_count_mean),
             data = subset(pf.fc_by_age, age < 71)) +
  geom_line() +
  geom_smooth()

p2 <- ggplot(aes(x = age_with_months, y = friend_count_mean),
             data = subset(pf.fc_by_age_months, age_with_months < 71)) +
  geom_line() +
  geom_smooth()

p3 <- ggplot(aes(x = round(age / 5) * 5, y = friend_count),
             data = subset(pf, age < 71)) +
  geom_line(stat = 'summary', fun.y = mean)
grid.arrange(p2, p1, p3, ncol = 1)
```
#钻石价格和克拉,不包括前1%的值
```
summary(diamonds$price)
summary(diamonds$carat)

ggplot(aes(x = carat, y = price), data = diamonds) +
  geom_point() +
  xlim(quantile(diamonds$carat, 0.01), 5.1) +
  ylim(quantile(diamonds$price, 0.01), 18830)
```
# 创造新变量至x轴,限制x轴做相关性分析,体积
```
data <- diamonds
data$volume <- with(diamonds, x * y * z)
sub_data <- subset(data,volume < 800 & volume > 0)
cor.test(sub_data$volume,sub_data$price)
```
# 创建新变量净度并连接,查看均值等
### 平均价格和净度
```
diamondsByClarity <- diamonds %>%
  group_by(clarity) %>%
  summarise(mean_price = mean(as.numeric(price)),
            median_price = median(as.numeric(price)),
            min_price = min(as.numeric(price)),
            max_price = max(as.numeric(price)),
            n = n()) %>%
  arrange(clarity)
head(diamondsByClarity)
```
# 创建两张柱状图
### grid和gridExtra包
```
p1 <- ggplot(aes(x=clarity,y=mean_price),data=diamonds_mp_by_clarity)+
geom_bar(stat = "identity")
p2 <- ggplot(aes(x=color,y=mean_price),
data=diamonds_mp_by_color)+
geom_bar(stat = "identity")
grid.arrange(p1,p2,ncol=1)
```
# 性别和年龄好友数箱线图
### 性别的平均值添加到箱线图 shape = 4
```
ggplot(aes(x = gender, y = age),
       data = subset(pf, !is.na(gender))) + geom_boxplot() +
  stat_summary(fun.y = mean, geom = 'point', shape = 4)
```
# 性别年龄好友数折线图
```
ggplot(aes(x = age, y = friend_count),
       data = subset(pf, !is.na(gender))) +
  geom_line(aes(color = gender), stat = 'summary', fun.y = median)
```
# 第三个定性变量
### 删除一些性别空值
### 总结将在运行时删除一层分组,所以我们要删除一些性别层,我们需要再一次运行分组已删除年龄层.
### 新版本ungroup()已经失效,可以去掉
### 按年龄安排数据框
```
pf.fc_by_age_gender <- pf %>%
  filter(!is.na(gender)) %>%
  #filter(age<71) %>%
  group_by(age, gender) %>%
  summarise(friend_count_mean = mean(friend_count),
            friend_count_median = median(friend_count),
            n = n()) %>%
  #ungroup() %>%
  arrange(age) 

head(pf.fc_by_age_gender)
```
# 将数据框从长格式变为宽格式（整洁格式）
### 使用tidyr包
```
spread(subset(pf.fc_by_age_gender,
       select = c('gender', 'age', 'median_friend_count')),
       gender, median_friend_count)
```
# 比率图
### 带基准线,女性比男性多多少倍
```
ggplot(aes(x = age, y = female/male), data = pf.fc_by_age_gender.wide) +
  geom_line() +
  geom_hline(yintercept = 1, alpha = 0.3, linetype = 2)
```
# 创建一个截至2014年使用时间的变量
### floor向下取整,省略小数, ceiling() 向上取整
```
pf$year_joined <- floor(2014 - pf$tenure/365)
```
# 切割函数cut,把不同年份分割成几段
###连续变量变成分类变量
```
summary(pf$year_joined)
table(pf$year_joined)

pf$year_joined.bucket <- cut(pf$year_joined,
                             c(2004, 2009, 2011, 2012, 2014))
```
# table()函数还有一个参数
```
table(pf$year_joined.bucket, useNA = 'ifany')
```
#按使用年限,性别,对好友数作图,并添加一条总平均线.
### linesype=2是虚线
```
ggplot(aes(x = age, y = friend_count),
           data = subset(pf, !is.na(year_joined.bucket))) +
  geom_line(aes(color = year_joined.bucket), stat = 'summary', fun.y = mean) +
  geom_line(stat = 'summary', fun.y = mean, linetype = 2)
  ```
# 申请好友数 
### 使用颜色标记使用时间分类,做好友数与使用时间关系图,时间必须大于1
```
ggplot(aes(x = tenure, y = friendships_initiated/tenure),
           data = subset(pf, tenure >= 1)) +
  geom_line(aes(color = year_joined.bucket))
```
# 申请好友数
### 这个比上面的不同
```
ggplot(aes(x = tenure, y = friendships_initiated/tenure),
           data = subset(pf, tenure >= 1)) +
  geom_line(aes(color = year_joined.bucket),
            stat = 'summary', fun.y = mean)
```
# 偏差方差折衷
### 四幅图联合
```
ggplot(aes(x = tenure, y = friendships_initiated / tenure),
       data = subset(pf, tenure >= 1)) +
  geom_line(aes(color = year_joined.bucket),
            stat = 'summary',
            fun.y = mean)

ggplot(aes(x = 7 * round(tenure / 7), y = friendships_initiated / tenure),
       data = subset(pf, tenure > 0)) +
  geom_line(aes(color = year_joined.bucket),
            stat = "summary",
            fun.y = mean)

ggplot(aes(x = 30 * round(tenure / 30), y = friendships_initiated / tenure),
       data = subset(pf, tenure > 0)) +
  geom_line(aes(color = year_joined.bucket),
            stat = "summary",
            fun.y = mean)

ggplot(aes(x = 90 * round(tenure / 90), y = friendships_initiated / tenure),
       data = subset(pf, tenure > 0)) +
  geom_line(aes(color = year_joined.bucket),
            stat = "summary",
            fun.y = mean)
```
### 平滑其中一个图像
```
ggplot(aes(x = 7 * round(tenure / 7), y = friendships_initiated / tenure),
       data = subset(pf, tenure > 0)) +
  geom_smooth(aes(color = year_joined.bucket))
```
# 将其中一个变量转换为因子factor
```
yo <- read.csv('yogurt.csv')
str(yo)
yo$id <- factor(yo$id)
str(yo)
```
# 做酸奶价格柱状图
```
ggplot(aes(x = price), data = yo) +
  geom_histogram()
```
# 描述酸奶数据集和价格数据规律
```
summary(yo)
length(unique(yo$price))
table(yo$price)
```
# 传递函数,两种
```
yo <- transform(yo, all.purchases = strchases = strawberry + blueberry + pina.colada + plain + mixed.berry)


yo$all.purchases <- yo$strawberry + yo$blueberry + yo$ pina.colada + yo$plain + yo$mixed.berry
```
# 做酸奶价格与时间散点图
### 原图
```
ggplot(aes(x = time, y = price), data = yo) +
  geom_point()
```
### 橙色抖动美化版
```
ggplot(aes(x = time, y = price), data = yo) +
  geom_jitter(alpha = 1/4, shape = 21, fill = I('#F79420'))
```
# 家庭样本
### 每个家庭购买酸奶的频率
###设置种子保证得到可重复结果
```
set.seed(4230)
sample.ids <- sample(levels(yo$id), 16)
table(sample.ids)


ggplot(aes(x = time, y = price),
        data = subset(yo, id %in% sample.ids)) +
   facet_wrap( ~ id) +
   geom_line() +
   geom_point(aes(size = all.purchases), pch = 1)
```
# 散点图矩阵
### 安装GGally包
```
theme_set(theme_minimal(20))

set.seed(1836)
pf_subset <- pf[, c(2:15)]
names(pf_subset)
ggpairs(pf_subset[sample.int(nrow(pf_subset), 1000), ])
```
# 将癌症数据集的颜色名称编号改为1-64,
### 以便通过在x轴贴标签优化图表
```
nci <- read.table("nci.tsv")
colnames(nci) <- c(1:64)
```
# 热图
### 安装reshape包
```
nci.long.samp <- melt(as.matrix(nci[1:200,]))
names(nci.long.samp) <- c("gene", "case", "value")
head(nci.long.samp)

ggplot(aes(y = gene, x = case, fill = value),
  data = nci.long.samp) +
  geom_tile() +
  scale_fill_gradientn(colours = colorRampPalette(c("blue", "red"))(100))
```
# 钻石,用颜色分面,做切割和价格直方图
### x轴是价格,y轴是切割,分面是颜色,讲师注释说用log10
### http://i.imgur.com/b5xyrOu.jpg
```
ggplot(aes(x = price, fill=cut), data = diamonds) + geom_histogram() +
         facet_wrap(~ color) +
         scale_x_log10() +
         scale_fill_brewer(type = 'qual')
```
# table,价格和切割做散点图
## x轴是table, y轴是价格,切割用不同颜色分类
### http://i.imgur.com/rQF9jQr.jpg
```
ggplot(aes(x=table,y=price,color=cut),data=diamonds)+
  geom_point()+
  xlim(c(50,80))+
  scale_color_brewer(type = 'qual')
```
# 价格体积与净度
### 体积x轴,价格y轴,净度用颜色分类.删除体积前1%.
### http://i.imgur.com/excUpea.jpg
```
ggplot(aes(x=volume,y=price,color=clarity),data=diamonds)+
  geom_point()+
  xlim(quantile(diamonds$volume,0.01), 400) +
  scale_y_log10()+
  scale_color_brewer(type = 'div')
  ```
# geom_line中设置中位数
### 两种
```
ggplot(aes(x = tenure, y = prop_initiated, color = year_joined.bucket),
       data = subset(pf, !is.na(year_joined.bucket))) +
  geom_line(aes(color = year_joined.bucket), stat = 'summary', fun.y = median) +
  xlim(10, 3200)


ggplot(pf, aes(x=tenure, y=prop_initiated, color=year_joined.bucket)) +
  geom_line(stat='summary', fun.y=median, na.rm=TRUE) +
  xlim(10, 3200)
```
# 分组中的一个变量描述统计
### 使用with
```
pf.nan <- subset(pf,prop_initiated != 'NaN')
with(subset(pf.nan ,year_joined.bucket=='(2012,2014]'),mean(prop_initiated))
```
### 或者by
```
by(pf$prop_initiated, pf$year_joined.bucket, summary)
```
# diamond数据集经过分组、分面和填色的价格/克拉 
### 用jitter代替point抖动让散点图变粗像柱状图
```
ggplot(aes(x= cut, y= price/carat),
       data = diamonds)+
  geom_jitter(aes(color= color ))+
  facet_wrap(~clarity)+
  scale_color_brewer(type = 'div')
```
# 散点图回顾
### 坐标轴截断scale_x_continuous
```
ggplot(diamonds, aes(x = carat, y = price)) +
  scale_x_continuous(lim = c(0, quantile(diamonds$carat, 0.99))) +
  scale_y_continuous(lim = c(0, quantile(diamonds$price, 0.99))) +
  geom_point(color = I('#F79420'), color = I('black'), shape = 21)
```
### 坐标轴xlim
```
ggplot(aes(x = carat, y = price), data = diamonds) +
  geom_point(color = I('#F79420'), shape = 21) +
  xlim(0, quantile(diamonds$carat, 0.99)) +
  ylim(0, quantile(diamonds$price, 0.99))
```
# 预测钻石 安装包大集合
### GGally  多图
### scales  标度
### memisc  汇总递归
### lattice 其他方面
### mass 用于各种函数
### car 用于重写变量代码
### reshape 整理数据
### plyr 汇总传输
```
# install these if necessary
install.packages('GGally')
install.packages('scales')
install.packages('memisc')
install.packages('lattice')
install.packages('MASS')
install.packages('car')
install.packages('reshape')
install.packages('plyr')

# load the ggplot graphics package and the others
library(ggplot2)
library(GGally)
library(scales)
library(memisc)

# sample 10,000 diamonds from the data set
set.seed(20022012)
diamond_samp <- diamonds[sample(1:length(diamonds$price), 10000), ]
ggpairs(diamond_samp,
  lower = list(continous = wrap("points", shape = I('.'))),
  upper = list(combo = wrap("box", outlier.shape = I('.'))))
```
# 对钻石的需求
### 并列两个图,两个颜色,一个log10
```
p1 <- ggplot(aes(x = price), data = diamonds) +
  geom_histogram(fill = I('#099DD9')) +
  ggtitle('Price')
p2 <- ggplot(aes(x = price), data = diamonds) +
  geom_histogram(fill = I('#F79420')) +
  ggtitle('Price (log10)') +
  scale_x_log10()
grid.arrange(p1,p2,ncol=2)
```
# 散点图转换
### log10在Y轴上
```
ggplot(aes(x = carat, y = price), data = diamonds) +
  geom_point() +
  scale_y_continuous(trans = log10_trans()) +
  ggtitle('Price (log10) by Carat')
```
# 转换克拉变量
### trans函数是立方根转换函数,trans_new是撤销运算的反函数.
```
cuberoot_trans = function() trans_new('cuberoot', transform = function(x) x^(1/3), inverse = function(x) x^3)
```
### 转换散点图
```
ggplot(aes(carat, price), data = diamonds) + 
  geom_point() + 
  scale_x_continuous(trans = cuberoot_trans(), limits = c(0.2, 3),
                     breaks = c(0.2, 0.5, 1, 2, 3)) + 
  scale_y_continuous(trans = log10_trans(), limits = c(350, 15000),
                     breaks = c(350, 1000, 5000, 10000, 15000)) +
  ggtitle('Price (log10) by Cube-Root of Carat')
```
# 找出最大的几个点
```
head(sort(table(diamonds$carat), decreasing = T))
head(sort(table(diamonds$price), decreasing = T))
```
# 通过抖动和添加透明度\大小来让点小一些,调节疏密程度
```
ggplot(aes(carat, price), data = diamonds) + 
  geom_point(alpha = 0.5, size = 0.75, position = 'jitter') + 
  scale_x_continuous(trans = cuberoot_trans(), limits = c(0.2, 3),
                     breaks = c(0.2, 0.5, 1, 2, 3)) + 
  scale_y_continuous(trans = log10_trans(), limits = c(350, 15000),
                     breaks = c(350, 1000, 5000, 10000, 15000)) +
  ggtitle('Price (log10) by Cube-Root of Carat')
  ```
# 价格克拉与净度
### 安装RColorBrewer
```
ggplot(aes(x = carat, y = price, color = clarity), data = diamonds) + 
  geom_point(alpha = 0.5, size = 1, position = 'jitter') +
  scale_color_brewer(type = 'div',
    guide = guide_legend(title = 'Clarity', reverse = T,
    override.aes = list(alpha = 1, size = 2))) +  
  scale_x_continuous(trans = cuberoot_trans(), limits = c(0.2, 3),
    breaks = c(0.2, 0.5, 1, 2, 3)) + 
  scale_y_continuous(trans = log10_trans(), limits = c(350, 15000),
    breaks = c(350, 1000, 5000, 10000, 15000)) +
  ggtitle('Price (log10) by Cube-Root of Carat and Clarity')
  ```
  # 价格克拉与切工
  ### 颜色分级是D -> J,因此去掉倒序参数reverse = T
  ```
  ggplot(aes(x = carat, y = price, color = color), data = diamonds) + 
  geom_point(alpha = 0.5, size = 1, position = 'jitter') +
  scale_color_brewer(type = 'div',
                     guide = guide_legend(title = colors(),
                                          override.aes = list(alpha = 1, size = 2))) +  
  scale_x_continuous(trans = cuberoot_trans(), limits = c(0.2, 3),
                     breaks = c(0.2, 0.5, 1, 2, 3)) + 
  scale_y_continuous(trans = log10_trans(), limits = c(350, 15000),
                     breaks = c(350, 1000, 5000, 10000, 15000)) +
  ggtitle('Price (log10) by Cube-Root of Carat and Cut')
  ```
  # 构建线性模型
  ###
```
  m1 <- lm(I(log(price)) ~ I(carat^(1/3)), data = diamonds)
m2 <- update(m1, ~ . + carat)
m3 <- update(m2, ~ . + cut)
m4 <- update(m3, ~ . + color)
m5 <- update(m4, ~ . + clarity)
mtable(m1, m2, m3, m4, m5)
```
# 更大更好数据集(未运行)
```
diamondsbig <- load("BigDiamonds.rda")

diamondsbig$logprice = log(diamondsbig$price)



m1 <- lm(logprice ~ I(carat^(1/3)),
  data=diamondsbig[diamondsbig$price < 10000 &
                     diamondsbig$cert == "GIA",])
m2 <- update(m1, ~ . + carat)
m3 <- update(m2, ~ . + cut)
m4 <- update(m3, ~ . + color)
m5 <- update(m4, ~ . + clarity)
mtable(m1, m2, m3, m4, m5)
```
# 