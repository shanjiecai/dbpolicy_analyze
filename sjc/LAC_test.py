from LAC import LAC

# 装载LAC模型
lac = LAC(mode='lac')

# 单个样本输入，输入为Unicode编码的字符串
text = u"2016年1月20日，内蒙古自治区通信管理局组织召开2016年度电信行业工作电视电话会议。内蒙古管局党组书记、局长、安全分中心主任刘宝钧作工作报告。党组成员、副局长、纪检组组长乔伟主持会议。党组成员、巡视员耿利君，党组成员、专用通信局局长张俊生，党组成员、副局长赵荣贵出席会议。自治区联通、移动、电信、铁塔公司负责人参加会议并作交流发言。会议在分析“十三五”面临的新形势新任务以及行业发展和管理中存在问题的基础上，安排部署了明年的四项重点工作。会议以电视电话会议形式举行，自治区各盟市设分会场。区管局机关、直属单位副处级以上干部、自治区和盟市两级通信建设管理办公室全体成员和自治区、盟市两级基础电信运营企业、铁塔公司领导班子及主要部门负责人参加了会议。"
lac_result = lac.run(text)
print(lac_result)

# 批量样本输入, 输入为多个句子组成的list，平均速率更快
texts = [u"LAC是个优秀的分词工具", u"百度是一家高科技公司"]
lac_result = lac.run(texts)
print(lac_result)
