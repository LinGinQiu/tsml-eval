import re
import subprocess

# 读取 requirements.txt
with open("requirements.txt") as f:
    requirements = [
        line.strip() for line in f if line.strip() and not line.startswith("#")
    ]


# 将 requirement 解析为 (package, version) 元组
def parse_requirement(req_line):
    match = re.match(r"([a-zA-Z0-9_\-]+)([=<>!~]+)([a-zA-Z0-9_.\-]+)", req_line)
    if match:
        return match.group(1).lower(), match.group(2), match.group(3)
    else:
        return req_line.lower(), None, None


requirements_parsed = [parse_requirement(r) for r in requirements]

# 获取当前环境中安装的包
installed = subprocess.check_output(["pip", "freeze"]).decode("utf-8").splitlines()
installed_dict = {}
for pkg in installed:
    if "==" in pkg:
        name, version = pkg.split("==")
        installed_dict[name.lower()] = version

# 开始检查
not_satisfied = []
for pkg, op, ver in requirements_parsed:
    if pkg not in installed_dict:
        not_satisfied.append(f"{pkg} NOT INSTALLED (required: {op}{ver})")
    elif op == "==" and installed_dict[pkg] != ver:
        not_satisfied.append(
            f"{pkg} version mismatch: installed {installed_dict[pkg]}, required {ver}"
        )

# 输出结果
if not not_satisfied:
    print("✅ All requirements are satisfied!")
else:
    print("❗ Unmet requirements:")
    for line in not_satisfied:
        print(" -", line)
