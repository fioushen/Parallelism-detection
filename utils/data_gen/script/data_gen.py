import time
import os
# import utils.data_gen.script.data_gen_util as du
import utils.data_gen.script.simple_data_util as du
import pickle
import pandas as pd

"""
1、gen_new_code：     source_code         --->    after_extract
2、pluto_data_gen：   after_extract       --->    after_pluto            ---> pluto_error.log
3、classify：         after_pluto         --->    parallel/unparallel
4、compile_data：     parallel/unparallel --->    1/2                    --->comp_error.log

"""
# du.clean_dir()
sum = 0
pluto_err = 0
classlist = [f for f in os.listdir("/parallel")]
print(classlist)
for i in range(0, len(classlist)):
    clazz = classlist[i]
    file_list = du.gen_new_code("/parallel" + "/", clazz)
    sum += len(file_list)
    pluto_list, error_list = du.pluto_data_gen(file_list)
    pluto_err += len(error_list)
    du.classify(pluto_list)
    if i % 10 == 0:
        time.sleep(0.1)
parallel_error_list, unparallel_error_list = du.compile_data()
du.process_ir()
parallel = len([f for f in os.listdir("/data/handled/parallel")])
unparallel = len([f for f in os.listdir("/data/handled/unparallel")])
comp_err = len(unparallel_error_list) + len(parallel_error_list)
after_extract = len([f for f in os.listdir("../data/pre_data/after_extract")])
after_pluto = len([f for f in os.listdir("../data/pre_data/after_pluto")])
ir1 = len([f for f in os.listdir("/data/IR/1")])
ir2 = len([f for f in os.listdir("/data/IR/2")])
print(parallel, unparallel, ir1, ir2)
print("=====================================")
print("共生成代码： %s 份" % sum)
print("=====================================")
print("开始数量验证：")
print("文件总量验证：", sum == after_extract)
print("分类过程验证：", after_pluto + pluto_err == after_extract, parallel + unparallel == after_pluto)
print("编译过程验证：", comp_err + ir1 + ir2 == parallel + unparallel)
print("=====================================")

