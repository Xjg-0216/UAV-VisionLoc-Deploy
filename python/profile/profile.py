import pstats

# 读取性能分析结果
p = pstats.Stats('/home/xujg/code/UAV-VisionLoc-Deploy/python/profile/profile_output.prof')

# 排序并打印统计信息
p.sort_stats('cumulative').print_stats(10)
# 按照时间排序
p.sort_stats('time').print_stats(10)

# 过滤特定模块或函数
# p.sort_stats('cumulative').print_stats('your_module')
