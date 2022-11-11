import re
import os
import time
import hashlib
from collections import namedtuple
import subprocess
import pandas as pd
from pycparser import c_parser
parser = c_parser.CParser()

# ===========================================================
#
# 所有正则表达式
#
# ===========================================================
pattern_dict = {
    'var_expr': r'[a-zA-Z_]+\w* *(?=[=<>\+\-\*/;\),\]&|])',
    'for_expr': r'(for\s*\(.*\) *\{[^}]*\}*?\})',
    'fun_expr': r'\w+(?=[(].*)',
    'anno_expr': r'(?://[^\n]*|/\*(?:(?!\*/).)*\*/)',
    'arr1_expr': r'[a-zA-Z_]+\w*(?=\[[^\]]+\][\=><\+\-\*\/;,\)&|\]])',
    'arr2_expr': r'[a-zA-Z_]+\w*(?=\[[^\]]+\]\[[^\]]+\][\=><\+\-\*\/;,\)&|\]])',
    'arr3_expr': r'[a-zA-Z_]+\w*(?=\[[^\]]+\]\[[^\]]+\]\[[^\]]+\][\=><\+\-\*\/;,\)&|\]])',
    'arr4_expr': r'[a-zA-Z_]+\w*(?=\[[^\]]+\]\[[^\]]+\]\[[^\]]+\]\[[^\]]+\][\=><\+\-\*\/;,\)&|\]])',
    'arr5_expr': r'[a-zA-Z_]+\w*(?=\[[^\]]+\]\[[^\]]+\]\[[^\]]+\]\[[^\]]+\]\[[^\]]+\][\=><\+\-\*\/;,\)&|\]])',
    'arr6_expr': r'[a-zA-Z_]+\w*(?=\[[^\]]+\]\[[^\]]+\]\[[^\]]+\]\[[^\]]+\]\[[^\]]+\]\[[^\]]+\][\=><\+\-\*\/;,\)&|\]])',
    'poi_expr': r'(?<=[=<>\+\-\*/\(,;]\*)[a-zA-Z_]+\w*\s*(?=[=\+\-\*/;,])'
}

type_list = ['void', 'int', 'double', 'float', 'bool', 'char', 'wchar_t', 'short', 'long', 'enum', 'const', 'signed',
             'unsigned', 'static', 'mutable', 'vector<', 'public', 'private']

keyword_list = ['true', 'false', 'for', 'if', 'break', 'while', 'else', 'switch', 'case', 'return', 'continue', 'class']
# 函数对象(函数名，定义行，函数参数,函数内容)
# void fun(int a,int b) ---> (fun,void,[int a,int b],"")
FUN = namedtuple('Function', 'fun_name fun_define fun_lines')
FOR_LINE = namedtuple('For_part', 'lines fun_name idx')


# ===========================================================
#
# 读取文件，删除注释,返回行列表,行首行尾没有空格
#
# ===========================================================
def read_file(file_path, filename):
    """从文件按行加载内容，返回字符串列表file_lines=['..',...''],不保留空行"""
    file_lines = []
    with open(file_path + filename, 'r', encoding='ISO-8859-1') as f:
        temp = f.read()
        f.close()
    pattern = re.compile(pattern_dict['anno_expr'], re.DOTALL)
    res = pattern.findall(temp)
    for s in res:
        temp = temp.replace(s, "", 1)
    for x in temp.split("\n"):
        if x.strip() == "" or x.strip() == "\n" or '#include' in x or '# include ' in x or 'using namespace' in x or '#pragma' in x:
            continue
        else:
            file_lines.append(x.strip())
    return file_lines


# ===============================================
#
#   头文件和宏定义
#   返回字符串header
#
# ===============================================
def gen_header():
    # 生成头文件
    with open("header.cpp", 'r', encoding='ISO-8859-1') as f:
        header = f.read()
        f.close()
    return header

def find_last(string, str):
    last_position = -1
    while True:
        position = string.find(str, last_position + 1)
        if position == -1:
            return last_position
        last_position = position

# ===============================================
#
#   查找方法定义
#
#
# ===============================================
def find_funs(file_lines):
    fun_list = []
    pattern = re.compile(pattern_dict['fun_expr'], re.DOTALL)
    fun_flag = False
    define_flag = False
    define_begin = 0
    begin = 0
    fun_lines = ""
    fun_name = ""
    fun_define = ""
    for line in file_lines:
        row = line.split(" ")
        if row[0] in type_list:
            try:
                rst = pattern.findall(str(row[1]))
            except IndexError:
                continue
            # 是方法定义
            if len(rst) != 0:
                fun_name = rst[0]
                define_flag = True
                fun_flag = True
        if define_flag:
            if '(' in line:
                define_begin += line.count('(')
            if ')' in line:
                define_begin -= line.count(')')
                if define_begin == 0:
                    define_flag = False
                    fun_define += line[0:find_last(line, ')') + 1].strip()
            if define_begin != 0:
                fun_define += line.strip()
        if fun_flag:
            if len(fun_lines) != 0:
                fun_lines += "\n" + line
            else:
                fun_lines += line
            if '{' in line:
                begin += line.count('{')
            if '}' in line:
                begin -= line.count('}')
                if begin == 0:
                    fun_flag = False
                    fun_list.append(FUN(fun_name, fun_define, fun_lines))
                    fun_lines = ""
                    fun_name = ""
                    fun_define = ""
    return fun_list

# ===========================================================
#
#   从代码提取for循环,针对代码不完整只有循环片段
#   for循环列表new_code
#
# ===========================================================
def extract_for(file_lines, fun_list):
    new_code = []
    new_file = "\n".join(file_lines[:])
    for_part = ""
    begin = 0
    flag = False
    if len(fun_list) != 0:
        for i in range(len(fun_list)):
            fun = fun_list[i]
            fun_lines = fun.fun_lines.split("\n")
            new_file = new_file.replace(fun.fun_lines, "\n" + fun.fun_define + ";\n")
            idx = 1000
            for line in fun_lines:
                if "for(" in line or 'for (' in line:
                    flag = True
                    if len(for_part) != 0:
                        for_part += "\n" + line
                    else:
                        for_part += line
                    if "{" in line:
                        begin += str(line).count('{', 0, len(line))
                    if "}" in line:
                        begin -= str(line).count('}', 0, len(line))
                        # 有打括号for一行就结束了
                        if begin == 0:
                            new_code.append(
                                FOR_LINE("#pragma scop\n" + for_part + "\n#pragma endscop", fun.fun_name, idx))
                            fun_list[i] = FUN(fun_list[i].fun_name, fun_list[i].fun_define,
                                              fun_list[i].fun_lines.replace(for_part, "//loop" + str(idx) + "\n"))
                            idx += 1
                            for_part = ""
                            flag = False
                    # 无大括号for一行结束
                    if begin == 0 and line[-1] == ';':
                        new_code.append(
                            FOR_LINE("#pragma scop\n" + for_part + "\n#pragma endscop", fun.fun_name, idx))
                        fun_list[i] = FUN(fun_list[i].fun_name, fun_list[i].fun_define,
                                          fun_list[i].fun_lines.replace(for_part, "//loop" + str(idx) + "\n"))
                        idx += 1
                        for_part = ""
                        flag = False
                    continue
                if flag:
                    if "{" in line:
                        begin += str(line).count('{', 0, len(line))
                    if "}" in line:
                        begin -= str(line).count('}', 0, len(line))
                if begin != 0:
                    if len(for_part) != 0:
                        for_part += "\n" + line
                    else:
                        for_part += line
                elif flag:
                    if len(for_part) != 0:
                        for_part += "\n" + line
                    else:
                        for_part += line
                    new_code.append(
                        FOR_LINE("#pragma scop\n" + for_part + "\n#pragma endscop", fun.fun_name, idx))
                    fun_list[i] = FUN(fun_list[i].fun_name, fun_list[i].fun_define,
                                      fun_list[i].fun_lines.replace(for_part, "//loop" + str(idx) + "\n"))
                    idx += 1
                    for_part = ""
                    flag = False
    else:
        for_idx = 1000
        for line in file_lines:
            if "for(" in line or 'for (' in line:
                flag = True
                for_part = for_part + line + "\n"
                if "{" in line:
                    begin += str(line).count('{', 0, len(line))
                if "}" in line:
                    begin -= str(line).count('}', 0, len(line))
                    # 有打括号for一行就结束了
                    if begin == 0:
                        new_code.append(
                            FOR_LINE("#pragma scop\n" + for_part + "\n#pragma endscop", None, for_idx))
                        new_file = new_file.replace(for_part, "//loop" + str(for_idx) + "\n")
                        for_idx += 1
                        for_part = ""
                        flag = False
                # 无大括号for一行结束
                if begin == 0 and line[-1] == ';':
                    new_code.append(
                        FOR_LINE("#pragma scop\n" + for_part + "\n#pragma endscop", None, for_idx))
                    new_file = new_file.replace(for_part, "//loop" + str(for_idx) + "\n")
                    for_idx += 1
                    for_part = ""
                    flag = False
                continue
            if flag:
                if "{" in line:
                    begin += str(line).count('{', 0, len(line))
                if "}" in line:
                    begin -= str(line).count('}', 0, len(line))
            if begin != 0:
                for_part = for_part + line + "\n"
            elif flag:
                for_part = for_part + line
                new_code.append(
                    FOR_LINE("#pragma scop\n" + for_part + "\n#pragma endscop", None, for_idx))
                new_file = new_file.replace(for_part, "//loop" + str(for_idx) + "\n")
                for_idx += 1
                for_part = ""
                flag = False
    return new_code, new_file, fun_list

# ===============================================
#
#   生成新文件
#   返回生成的文件名列表
# ===============================================
def gen_new_code(file_path, file_name):
    file_list = []
    # 读取文件
    file_lines = read_file(file_path, file_name)
    # 提取方法
    fun_list = find_funs(file_lines)
    # new_file是字符串
    for_part_list, new_file, fun_list = extract_for(file_lines, fun_list)
    if len(for_part_list) == 0:
        return file_list
    # 程序不完整情况handle
    if len(fun_list) == 0:
        for for_part in for_part_list:
            vars = get_for_var(for_part)
            header, var_define = get_define(vars)
            code = header + "int main(){\n" + var_define + new_file.replace("//loop" + str(for_part.idx),
                                                                            for_part.lines) + "\nreturn 0;\n}"
            filename = generate_file_name(code + str(time.time()))
            file_list.append(filename)
            with open('../data/pre_data/after_extract/' + filename, "w", encoding='utf-8') as f:
                f.write(code)
                f.close()
    else:
        # 生成代码头文件
        header = gen_header()
        # 遍历所有for
        for for_part in for_part_list:
            code = header + "\n"
            fun_name = for_part.fun_name
            for fun in fun_list:
                if fun.fun_name == fun_name:
                    # 替换main函数定义为main的方法体，并替换第idx个循环
                    code += new_file.replace(fun.fun_define + ";",
                                             fun.fun_lines.replace("//loop" + str(for_part.idx), for_part.lines))
                    filename = generate_file_name(code + str(time.time()))
                    file_list.append(filename)
                    with open('../data/pre_data/after_extract/' + filename, "w", encoding='utf-8') as f:
                        f.write(code)
                        f.close()
    print("文件：%s    共生成新文件：%s 个" % (file_name, len(file_list)))
    return file_list

# ===============================================
#
# 生成文件名
#
#
# ===============================================
def generate_file_name(instance_str):
    """generate filename according to instance_str"""
    byte_obj = bytes(instance_str, 'utf-8')
    fname = hashlib.shake_128(byte_obj).hexdigest(5)
    fname = "{}.cpp".format(fname)
    return fname

# ===============================================
#
# 生成的pluto文件
#
# ===============================================
def pluto_data_gen(file_list):
    error_list = []
    cmd = './polycc ../data/pre_data/after_extract/%s ' \
          '--noprevector --tile --parallel --innerpar --lbtile ' \
          '-o %s/%s'
    # cmd = './polycc ../data/pre_data/after_extract/%s ' \
    #       '--noprevector --tile --lbtile ' \
    #       '-o %s/%s'
    # cmd = './ppcg ../data/pre_data/after_extract/%s ' \
    #       '--target=c --tile --openmp ' \
    #       '-o %s/%s'
    pluto_list = []
    i = 0
    for clazz in file_list:
        cmd1 = cmd % (clazz, '../data/pre_data/after_pluto', clazz)
        a = subprocess.run(cmd1, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if a.returncode == 0:
            pluto_list.append(clazz)
        else:
            error_list.append(clazz + "\n" + bytes.decode(a.stderr) + "=" * 50 + "\n")
        i += 1
        print("分析代码：    %s / %s ..." % (str(i), str(len(file_list))))
    with open('../log/pluto_error.log', "a+", encoding='utf-8') as f:
        f.write("\n".join(error_list))
        f.close()
    return pluto_list, error_list

def pluto_data_gen_cant(file_list):
    error_list = []
    cmd = './polycc ../data/pre_data/after_extract/%s ' \
          '--noprevector --tile --parallel --innerpar --lbtile ' \
          '-o %s/%s'
    # cmd = './polycc ../data/pre_data/after_extract/%s ' \
    #       '--noprevector ' \
    #       '-o %s/%s'
    cmd2 = 'cp ../data/pre_data/after_extract/%s ../data/pre_data/pluto_cant/%s'
    # cmd = './ppcg ../data/pre_data/after_extract/%s ' \
    #       '--target=c --tile --openmp ' \
    #       '-o %s/%s'
    pluto_list = []
    i = 0
    for clazz in file_list:
        cmd1 = cmd % (clazz, '../data/pre_data/after_pluto', clazz)
        a = subprocess.run(cmd1, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if a.returncode == 0:
            pluto_list.append(clazz)
        else:
            cmd3 = cmd2 % (clazz, clazz)
            b = subprocess.run(cmd3, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            if b.returncode == 0:
                error_list.append(clazz + "\n" + bytes.decode(a.stderr) + "=" * 50 + "\n")
        i += 1
        print("分析代码：    %s / %s ..." % (str(i), str(len(file_list))))
    with open('../log/pluto_error.log', "a+", encoding='utf-8') as f:
        f.write("\n".join(error_list))
        f.close()
    return pluto_list, error_list

# ===============================================
#
# 划分pluto文件
#
# ===============================================
def classify(pluto_list):
    cmd = 'cp ../data/pre_data/after_extract/%s ../data/handled/%s/%s'
    for clazz in pluto_list:
        with open("../data/pre_data/after_pluto/" + clazz, "r") as f:
            code = f.read()
            f.close()
        if "#pragma omp parallel" in code:
            cmd1 = cmd % (clazz, "parallel", clazz)
        else:
            cmd1 = cmd % (clazz, "unparallel", clazz)
        subprocess.call(cmd1, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# ===============================================
#
# 编译生成的新文件
#
# ===============================================
def compile_data_llvm():
    cmds = [
        r"clang++ -S -emit-llvm -std=c++11 -O0 -march=native ../data/%s/%s -o ../data/IR/%s/%s.ll",
        # r"clang++ -S -emit-llvm -std=c++11 -O0 -ffast-math -march=native ../data/handled/%s/%s -o ../data/IR/%s/%s.ll"
    ]
    parallel_error_list = []
    unparallel_error_list = []
    classlist = [f for f in os.listdir("/data/handled/parallel")]
    for i in range(0, len(cmds)):
        for clazz in classlist:
            cmd = cmds[i] % ("parallel", clazz, "1", clazz + "_" + str(i))
            a = subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            if a.returncode == 0:
                pass
            else:
                parallel_error_list.append(clazz + "\n" + bytes.decode(a.stderr) + "=" * 50)
        if i % 10 == 0:
            print("编译可并行代码：\t%s / %s ..." % (str(i), str(len(classlist))))
            time.sleep(0.1)

    classlist = [f for f in os.listdir("/data/handled/unparallel")]
    for i in range(0, len(cmds)):
        for clazz in classlist:
            cmd = cmds[i] % ("unparallel", clazz, "2", clazz + "_" + str(i))
            a = subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            if a.returncode == 0:
                pass
            else:
                unparallel_error_list.append(clazz + "\n" + bytes.decode(a.stderr) + "=" * 50)
        if i % 10 == 0:
            print("编译不可并行代码：\t%s / %s ..." % (str(i), str(len(classlist))))
            time.sleep(0.1)
    with open('../log/comp_error.log', "a+", encoding='utf-8') as f:
        f.write("\n" + "*" * 50 + "ParallelError" + "*" * 50 + "\n" +
                "\n".join(parallel_error_list) + "\n" + "*" * 50 + "Unparallel" + "*" * 50 + "\n" + "\n".join(
            unparallel_error_list))
        f.close()
    return parallel_error_list, unparallel_error_list

def compile_data_llvm_astnn():
    command = 'cp ../data/handled/%s/%s /data/%s/%s'
    cmds = [
        r"clang++ -S -emit-llvm -std=c++11 -O0 -march=native ../data/handled/%s/%s -o ../data/IR/%s/%s.ll",
        # r"clang++ -S -emit-llvm -std=c++11 -O0 -ffast-math -march=native ../data/handled/%s/%s -o ../data/IR/%s/%s.ll"
    ]
    parallel_error_list = []
    unparallel_error_list = []
    classlist = [f for f in os.listdir("../data/handled/parallel")]
    for i in range(0, len(cmds)):
        for clazz in classlist:
            cmd = cmds[i] % ("parallel", clazz, "1", clazz + "_" + str(i))
            a = subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            if a.returncode == 0:
                cmd1 = command % ("parallel", clazz, "parallel", clazz)
                subprocess.call(cmd1, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                pass
            else:
                unparallel_error_list.append(clazz + "\n" + bytes.decode(a.stderr) + "=" * 50)

        if i % 10 == 0:
            print("编译可并行代码：\t%s / %s ..." % (str(i), str(len(classlist))))
            time.sleep(0.1)

    classlist = [f for f in os.listdir("../data/handled/unparallel")]
    for i in range(0, len(cmds)):
        for clazz in classlist:
            cmd = cmds[i] % ("unparallel", clazz, "2", clazz + "_" + str(i))
            a = subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            if a.returncode == 0:
                cmd1 = command % ("unparallel", clazz, "unparallel", clazz)
                subprocess.call(cmd1, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                pass
            else:
                unparallel_error_list.append(clazz + "\n" + bytes.decode(a.stderr) + "=" * 50)



        if i % 10 == 0:
            print("编译不可并行代码：\t%s / %s ..." % (str(i), str(len(classlist))))
            time.sleep(0.1)
    with open('../log/comp_error.log', "a+", encoding='utf-8') as f:
        f.write("\n" + "*" * 50 + "ParallelError" + "*" * 50 + "\n" +
                "\n".join(parallel_error_list) + "\n" + "*" * 50 + "Unparallel" + "*" * 50 + "\n" + "\n".join(
            unparallel_error_list))
        f.close()
    return parallel_error_list, unparallel_error_list


def compile_data_ast():
    cmds = [
        r"clang++ -emit-ast /home/syy/Code/data_gen/data/handled/%s/%s"
        # r"clang++ -S -emit-llvm -std=c++11 -O0 -march=native ../data/handled/%s/%s -o ../data/IR/%s/%s.ll",
        # r"clang++ -S -emit-llvm -std=c++11 -O0 -ffast-math -march=native ../data/handled/%s/%s -o ../data/IR/%s/%s.ll"
    ]
    parallel_error_list = []
    unparallel_error_list = []
    classlist = [f for f in os.listdir("../data/handled/parallel")]
    for i in range(0, len(cmds)):
        for clazz in classlist:
            # cmd = cmds[i] % ("parallel", clazz, "1", clazz + "_" + str(i))
            cmd = cmds[i] % ("parallel", clazz)
            a = subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            if a.returncode == 0:
                pass
            else:
                parallel_error_list.append(clazz + "\n" + bytes.decode(a.stderr) + "=" * 50)
        if i % 10 == 0:
            print("编译可并行代码：\t%s / %s ..." % (str(i), str(len(classlist))))
            time.sleep(0.1)

    classlist = [f for f in os.listdir("../data/handled/unparallel")]
    for i in range(0, len(cmds)):
        for clazz in classlist:
            # cmd = cmds[i] % ("unparallel", clazz, "2", clazz + "_" + str(i))
            cmd = cmds[i] % ("unparallel", clazz)
            a = subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            if a.returncode == 0:
                pass
            else:
                unparallel_error_list.append(clazz + "\n" + bytes.decode(a.stderr) + "=" * 50)
        if i % 10 == 0:
            print("编译不可并行代码：\t%s / %s ..." % (str(i), str(len(classlist))))
            time.sleep(0.1)
    with open('../log/comp_error.log', "a+", encoding='utf-8') as f:
        f.write("\n" + "*" * 50 + "ParallelError" + "*" * 50 + "\n" +
                "\n".join(parallel_error_list) + "\n" + "*" * 50 + "Unparallel" + "*" * 50 + "\n" + "\n".join(
            unparallel_error_list))
        f.close()
    return parallel_error_list, unparallel_error_list

def compile_data():

    pattern_dict = {
        'var_expr': r'[a-zA-Z_]+\w* *(?=[=<>\+\-\*/;\),\]&|])',
        'for_expr': r'(for\s*\(.*\) *\{[^}]*\}*?\})',
        'fun_expr': r'\w+(?=[(].*)',
        'anno_expr': r'(?://[^\n]*|/\*(?:(?!\*/).)*\*/)',
        'arr1_expr': r'[a-zA-Z_]+\w*(?=\[[^\]]+\][\=><\+\-\*\/;,\)&|\]])',
        'arr2_expr': r'[a-zA-Z_]+\w*(?=\[[^\]]+\]\[[^\]]+\][\=><\+\-\*\/;,\)&|\]])',
        'arr3_expr': r'[a-zA-Z_]+\w*(?=\[[^\]]+\]\[[^\]]+\]\[[^\]]+\][\=><\+\-\*\/;,\)&|\]])',
        'arr4_expr': r'[a-zA-Z_]+\w*(?=\[[^\]]+\]\[[^\]]+\]\[[^\]]+\]\[[^\]]+\][\=><\+\-\*\/;,\)&|\]])',
        'arr5_expr': r'[a-zA-Z_]+\w*(?=\[[^\]]+\]\[[^\]]+\]\[[^\]]+\]\[[^\]]+\]\[[^\]]+\][\=><\+\-\*\/;,\)&|\]])',
        'arr6_expr': r'[a-zA-Z_]+\w*(?=\[[^\]]+\]\[[^\]]+\]\[[^\]]+\]\[[^\]]+\]\[[^\]]+\]\[[^\]]+\][\=><\+\-\*\/;,\)&|\]])',
        'poi_expr': r'(?<=[=<>\+\-\*/\(,;]\*)[a-zA-Z_]+\w*\s*(?=[=\+\-\*/;,])'
    }

    # 将自己的文件转为astnn实验要求格式
    PARALLEL_DATA_FOLDER = '/utils/data_gen/data/parallel'
    UNPARALLEL_DATA_FOLDER = '/utils/data_gen/data/unparallel'
    parallel_file_list = [f for f in os.listdir(PARALLEL_DATA_FOLDER)]
    unparallel_file_list = [f for f in os.listdir(UNPARALLEL_DATA_FOLDER)]
    files_list = []
    print(len(parallel_file_list))
    # pattern = re.compile(pattern_dict['anno_expr'], re.DOTALL)
    sas = []
    for f1 in parallel_file_list:
        print(f1)
        file_lines = []
        with open(os.path.join(PARALLEL_DATA_FOLDER, f1), 'r', encoding='ISO-8859-1') as f:
            temp = f.read()
            f.close()
        # file_path = os.path.join(PARALLEL_DATA_FOLDER, f1)
        pattern = re.compile(pattern_dict['anno_expr'], re.DOTALL)
        res = pattern.findall(temp)
        for s in res:
            temp = temp.replace(s, "", 1)
        for x in temp.split("\n"):
            if x.strip() == "" or x.strip() == "\n" or '#include' in x or '# include ' in x or 'using namespace' in x or '#pragma' in x:
                # if x.strip() == "" or x.strip() == "\n" or 'using namespace' in x or '#pragma' in x:
                continue
            else:
                file_lines.append(x.strip())

        a = ''
        for i in range(len(file_lines)):
            a += file_lines[i]
            a += '\n'
        files_list.append(a)
        with open('/parallel/%s' % f1, 'w', encoding='utf-8') as f:
            f.write(a)
            f.close()

        print(a)
        # pp = parser.parse(a, file_path, debuglevel=0)
        # sas.append(pp)
    print(sas)
    for f2 in unparallel_file_list:
        file_lines = []
        with open(os.path.join(UNPARALLEL_DATA_FOLDER, f2), 'r') as f:
            temp = f.read()
            f.close()
        pattern = re.compile(pattern_dict['anno_expr'], re.DOTALL)
        res = pattern.findall(temp)
        for s in res:
            temp = temp.replace(s, "", 1)
        for x in temp.split("\n"):
            if x.strip() == "" or x.strip() == "\n" or '#include' in x or '# include ' in x or 'using namespace' in x or '#pragma' in x:
                continue
            else:
                file_lines.append(x.strip())
        b = ''
        for i in range(len(file_lines)):
            b += file_lines[i]
            b += '\n'
        files_list.append(b)
        with open('/unparallel/%s' % f2, 'w', encoding='utf-8') as f:
            f.write(b)
            f.close()
    labels = [0 for _ in range(len(parallel_file_list))]
    # print(len(unparallel_file_list))
    for i in range(len(unparallel_file_list)):
        labels.append(1)
    # print(len(labels))
    ids = []
    for i in range(len(files_list)):
        ids.append(i)

    d = {'id': ids, 'code': files_list, 'label': labels}
    sa = pd.DataFrame(data=d)
    order = ['id', 'code', 'label']
    sa = sa[order]
    return files_list

def parser_data():
    cmd = 'cp %s/%s %s/%s'
    PARALLEL_DATA_FOLDER = '/parallel'
    UNPARALLEL_DATA_FOLDER = '/unparallel'
    parallel_file_list = [f for f in os.listdir(PARALLEL_DATA_FOLDER)]
    unparallel_file_list = [f for f in os.listdir(UNPARALLEL_DATA_FOLDER)]
    parallel_parse_list = []
    unparallel_parse_list = []
    for f1 in parallel_file_list:
        file_path = os.path.join(PARALLEL_DATA_FOLDER,f1)
        with open(file_path, 'r', encoding='utf-8') as f:
            temp = f.read()
            f.close()
        try:
            sa_p = parser.parse(temp,file_path,debuglevel=0)
            parallel_parse_list.append(sa_p)
            cmd1 = cmd % ("parallel", f1, "parallel", f1)
            subprocess.call(cmd1, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            pass
        continue
    for f2 in unparallel_file_list:
        file_path = os.path.join(UNPARALLEL_DATA_FOLDER,f2)
        with open(file_path, 'r', encoding='utf-8') as f:
            temp = f.read()
            f.close()
        try:
            sa_unp = parser.parse(temp, file_path, debuglevel=0)
            unparallel_parse_list.append(sa_unp)
            cmd1 = cmd % ("unparallel", f2, "unparallel", f2)
            subprocess.call(cmd1, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            pass
        continue

    return parallel_parse_list, unparallel_parse_list
def parser_compiled_data():
    UNPARALLEL_DATA_FOLDER = '/unparallel'
    PARALLEL_DATA_FOLDER = '/parallel'

    parallel_file_list = [f for f in os.listdir(PARALLEL_DATA_FOLDER)]
    parallel_file_list.sort(key=lambda x: str.lower(x[:2]))
    unparallel_file_list = [f for f in os.listdir(UNPARALLEL_DATA_FOLDER)]
    unparallel_file_list.sort(key=lambda x: str.lower(x[:2]))
    parallel_parse_list = []
    unparallel_parse_list = []
    for f1 in parallel_file_list:
        file_path = os.path.join(PARALLEL_DATA_FOLDER,f1)
        with open(file_path, 'r', encoding='utf-8') as f:
            temp = f.read()
            f.close()
        try:
            sa_p = parser.parse(temp,file_path,debuglevel=0)
            parallel_parse_list.append(sa_p)
            sa_p.show(offset=4, attrnames=True, nodenames=True)
            print(sa_p)
            print(type(sa_p))
        except Exception as e:
            pass
        continue
    for f2 in unparallel_file_list:
        file_path = os.path.join(UNPARALLEL_DATA_FOLDER,f2)
        with open(file_path, 'r', encoding='utf-8') as f:
            temp = f.read()
            f.close()
        try:
            sa_unp = parser.parse(temp, file_path, debuglevel=0)
            unparallel_parse_list.append(sa_unp)
        except Exception as e:
            pass
        continue

    return parallel_parse_list, unparallel_parse_list

def dataframe_data_ast(parallel_parse_list, unparallel_parse_list):
    files_list = []
    for f1 in parallel_parse_list:
        files_list.append(f1)
    for f2 in unparallel_parse_list:
        files_list.append(f2)
    labels = [0 for _ in range(len(parallel_parse_list))]


    # print(len(unparallel_file_list))
    for i in range(len(unparallel_parse_list)):
        labels.append(1)
    # print(len(labels))
    ids = []
    for i in range(len(files_list)):
        ids.append(i)

    d = {'id': ids, 'code': files_list, 'label': labels}
    sa = pd.DataFrame(data=d)
    order = ['id', 'code', 'label']
    sa = sa[order]
    return sa



def clean_dir():
    # prefix = time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime(time.time()))
    # zip = "zip -r ../backup/%s_data.zip ../data" % prefix
    clean = [
        r"rm -rf ../data/handled/*/*",
        r"rm -rf ../data/IR/*/*",
        r"rm -rf ../data/IR_processed/*/*",
        r"rm -rf ../data/pre_data/*/*",
        r"rm -rf ../data/unhandle/*/*",
        r"rm -rf ../data/compiled/*/*",
        r"rm -rf ../data/simple/*/*",
        r"rm -rf ../log/*"
    ]
    # subprocess.call(zip, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for cl in clean:
        subprocess.call(cl, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# ===========================================================
#
# 提取for循环中的变量
# 参数for_part为处理后的for片段
#
# ===========================================================
def get_for_var(for_part):
    for_part = str(for_part).replace(" ", "")
    # 提取变量名
    vars = []
    # 提取普通变量
    pattern = re.compile(pattern_dict['var_expr'])
    vars.append(set(pattern.findall(str(for_part))))
    # 提取1维数组
    pattern = re.compile(pattern_dict['arr1_expr'])
    vars.append(set(pattern.findall(str(for_part))))
    # 提取2维数组
    pattern = re.compile(pattern_dict['arr2_expr'])
    vars.append(set(pattern.findall(str(for_part))))
    # 提取3维数组
    pattern = re.compile(pattern_dict['arr3_expr'])
    vars.append(set(pattern.findall(str(for_part))))
    # 提取4维数组
    pattern = re.compile(pattern_dict['arr4_expr'])
    vars.append(set(pattern.findall(str(for_part))))
    # 提取5维数组
    pattern = re.compile(pattern_dict['arr5_expr'])
    vars.append(set(pattern.findall(str(for_part))))
    # 提取6维数组
    pattern = re.compile(pattern_dict['arr6_expr'])
    vars.append(set(pattern.findall(str(for_part))))
    # 提取指针变量
    pattern = re.compile(pattern_dict['poi_expr'])
    vars.append(set(pattern.findall(str(for_part))))
    # 去除普通变量列表中错误提取的指针变量及关键字
    for var in vars[7]:
        if var in vars[0]:
            vars[0].remove(var)
            continue
    for var in keyword_list:
        if var in vars[0]:
            vars[0].remove(var)
            continue
    for var in type_list:
        if var in vars[0]:
            vars[0].remove(var)
            continue

    # 修补多维数组匹配bug
    for i in range(2, 7):
        for var in vars[i]:
            if var in vars[i - 1]:
                vars[i - 1].remove(var)
    return vars


# ===============================================
#
# 生成文件前半部分
# 参数vars为从for循环中提取到的变量表
#
# ===============================================
def get_define(vars):
    # 生成头文件
    with open("header.cpp", 'r', encoding='ISO-8859-1') as f:
        header = f.read()
        f.close()
    var_define = ""
    for var in vars[0]:
        var_define = var_define + 'int ' + var + '= 100' + ';\n'
    # 生成普通变量及数组定义
    for i in range(1, len(vars) - 1):
        if len(vars[i]) != 0:
            for var in vars[i]:
                var_define = var_define + 'int ' + var + '[101]' * i + ';\n'
    # 生成指针定义
    for var in vars[7]:
        var_define = var_define + 'int *' + var + ';\n'
    return header, var_define


# ===============================================
#
# 处理IR
#
#
# ===============================================
def process_ir():
    out_path = "../data/IR_processed/"
    for i in range(1, 3):
        classlist = [x for x in os.listdir("../data/IR/NPB_original_IR/" + str(i))]
        for clazz in classlist:
            ir = []
            with open("../data/IR/" + str(i) + "/" + clazz, 'r', encoding='ISO-8859-1') as f:
                temp = f.readlines()
                f.close()
            flag = False
            for line in temp:
                if "; <label>:" in line:
                    flag = True
                    ir.append(line)
                    continue
                if flag:
                    if line.strip() == "" or line.strip() == "\n" or 'ret' in line or '}' in line:
                        flag = False
                    else:
                        ir.append(line)
            if len(ir) == 0:
                continue
            with open(out_path + str(i) + "/" + clazz, 'w', encoding='utf-8') as f:
                f.write("".join(ir))
                f.close()
